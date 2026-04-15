import logging
import time
import uuid
from dataclasses import dataclass, field

import litellm

import inference_toolkit.cache.semantic_cache as semantic_cache_module
import inference_toolkit.compression.compressor as compressor_module
from inference_toolkit.config import settings

_LOG = logging.getLogger(__name__)

# Type alias for a single chat message dict.
Message = dict[str, str]


class BudgetExceededError(Exception):
    """
    Raise when a conversation's cumulative cost exceeds the configured budget.
    """

    def __init__(self, cumulative_cost: float, max_cost: float) -> None:
        self.cumulative_cost = cumulative_cost
        self.max_cost = max_cost
        super().__init__(f"Conversation budget exceeded: ${cumulative_cost:.4f} >= ${max_cost:.4f}")


@dataclass
class ConversationTurn:
    """
    Capture the outcome of a single message sent to a conversation.
    """

    response: str
    cache_hit: semantic_cache_module.CacheHit | None
    cost_usd: float
    cumulative_cost_usd: float
    tokens_used: int
    compressed: bool


@dataclass
class Conversation:
    """
    Manage a stateful multi-turn LLM conversation with automatic caching,
    context compression, and cost guardrails.
    """

    id: str
    model: str
    messages: list[Message] = field(default_factory=list)
    cumulative_cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)

    async def send(
        self,
        prompt: str,
        cache: semantic_cache_module.SemanticCache,
        compressor: compressor_module.ContextCompressor,
    ) -> ConversationTurn:
        """
        Send a user message and return the assistant's response with full metadata.

        Check the semantic cache first; on a miss, apply cost guardrails and context
        compression before calling the LLM. Cache the response on every real LLM call.

        :param prompt: the user's message
        :param cache: shared semantic cache instance
        :param compressor: shared context compressor instance
        :return: ConversationTurn with response, cache metadata, cost, and compression flag
        """
        # Enforce budget before making any API call.
        self._check_budget()
        # Check semantic cache before calling the LLM.
        hit = await cache.get(prompt)
        if hit:
            self.messages.append({"role": "user", "content": prompt})
            self.messages.append({"role": "assistant", "content": hit.response})
            _LOG.debug(
                "[%s] Cache hit (score=%.4f) for: %s", self.id, hit.similarity_score, prompt[:60]
            )
            return ConversationTurn(
                response=hit.response,
                cache_hit=hit,
                cost_usd=0.0,
                cumulative_cost_usd=self.cumulative_cost_usd,
                tokens_used=0,
                compressed=False,
            )
        # Apply cost-aware compression if approaching the budget.
        self.messages.append({"role": "user", "content": prompt})
        compressed = False
        messages_before = len(self.messages)
        # Trigger compression when at guardrail threshold of budget (if budget is set).
        if self._should_compress_for_budget():
            self.messages = await compressor.compress(self.messages, self.model)
            compressed = len(self.messages) < messages_before
        else:
            # Fall back to token-based compression.
            self.messages = await compressor.compress(self.messages, self.model)
            compressed = len(self.messages) < messages_before
        # Call the LLM.
        response = await litellm.acompletion(model=self.model, messages=self.messages)
        assistant_content = str(response.choices[0].message.content)
        self.messages.append({"role": "assistant", "content": assistant_content})
        # Track cost.
        turn_cost = self._extract_cost(response)
        self.cumulative_cost_usd += turn_cost
        tokens_used = response.usage.total_tokens if response.usage else 0
        # Store in cache for future similar prompts.
        await cache.set(prompt, assistant_content, model=self.model, cost_usd=turn_cost)
        _LOG.info(
            "[%s] Turn complete — cost=$%.4f cumulative=$%.4f compressed=%s",
            self.id,
            turn_cost,
            self.cumulative_cost_usd,
            compressed,
        )
        return ConversationTurn(
            response=assistant_content,
            cache_hit=None,
            cost_usd=turn_cost,
            cumulative_cost_usd=self.cumulative_cost_usd,
            tokens_used=tokens_used,
            compressed=compressed,
        )

    def serialise(self) -> dict[str, object]:
        """
        Export the conversation state to a JSON-serialisable dict.

        :return: dict containing all conversation fields
        """
        return {
            "id": self.id,
            "model": self.model,
            "messages": self.messages,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "created_at": self.created_at,
        }

    @classmethod
    def deserialise(cls, data: dict[str, object]) -> "Conversation":
        """
        Restore a Conversation from a previously serialised dict.

        :param data: dict produced by `serialise()`
        :return: restored Conversation instance
        """
        raw_messages = data.get("messages", [])
        messages: list[dict[str, str]] = (
            list(raw_messages) if isinstance(raw_messages, list) else []
        )
        raw_cost = data.get("cumulative_cost_usd", 0.0)
        raw_ts = data.get("created_at", time.time())
        return cls(
            id=str(data["id"]),
            model=str(data["model"]),
            messages=messages,
            cumulative_cost_usd=float(raw_cost) if isinstance(raw_cost, (int, float, str)) else 0.0,
            created_at=float(raw_ts) if isinstance(raw_ts, (int, float, str)) else time.time(),
        )

    def _check_budget(self) -> None:
        """
        Raise BudgetExceededError if the cumulative cost has hit the configured limit.
        """
        max_cost = settings.max_cost_usd_per_conversation
        if max_cost > 0 and self.cumulative_cost_usd >= max_cost:
            raise BudgetExceededError(self.cumulative_cost_usd, max_cost)

    def _should_compress_for_budget(self) -> bool:
        """
        Return True when cumulative cost is at the guardrail threshold of the budget.
        """
        max_cost = settings.max_cost_usd_per_conversation
        if max_cost <= 0:
            return False
        threshold = max_cost * settings.cost_guardrail_threshold
        return self.cumulative_cost_usd >= threshold

    @staticmethod
    def _extract_cost(response: object) -> float:
        """
        Extract the cost of an LLM response using litellm's cost helper.

        Fall back to 0.0 if the model or response is not supported.

        :param response: raw litellm completion response
        :return: cost in USD
        """
        try:
            cost = litellm.completion_cost(completion_response=response)
            return float(cost) if cost else 0.0
        except Exception:
            return 0.0


class ConversationStore:
    """
    Manage the lifecycle of Conversation instances in memory.
    """

    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}

    async def create(self, model: str, system_prompt: str = "") -> Conversation:
        """
        Instantiate and register a new conversation.

        :param model: the LLM model identifier for this conversation
        :param system_prompt: optional system prompt to prepend
        :return: newly created Conversation
        """
        conversation_id = str(uuid.uuid4())
        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        conversation = Conversation(id=conversation_id, model=model, messages=messages)
        self._store[conversation_id] = conversation
        _LOG.info("Created conversation '%s' (model=%s).", conversation_id, model)
        return conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        """
        Retrieve a conversation by ID.

        :param conversation_id: UUID of the conversation
        :return: Conversation if found, else None
        """
        return self._store.get(conversation_id)

    async def save(self, conversation: Conversation) -> None:
        """
        Persist an updated conversation back into the store.

        :param conversation: the conversation to save
        """
        self._store[conversation.id] = conversation

    async def delete(self, conversation_id: str) -> bool:
        """
        Remove a conversation from the store.

        :param conversation_id: UUID of the conversation to delete
        :return: True if deleted, False if not found
        """
        if conversation_id in self._store:
            del self._store[conversation_id]
            _LOG.info("Deleted conversation '%s'.", conversation_id)
            return True
        return False

    def count(self) -> int:
        """
        Return the number of active conversations.

        :return: conversation count
        """
        return len(self._store)
