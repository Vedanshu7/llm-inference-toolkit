import logging
import time
from dataclasses import dataclass

import numpy as np

import inference_toolkit.cache.store as cache_store

_LOG = logging.getLogger(__name__)


@dataclass
class CacheEntryDetail:
    """
    Expose a single cache entry for the inspect endpoint, omitting the raw embedding.
    """

    prompt_preview: str
    response_preview: str
    model: str
    hits: int
    cost_usd: float
    estimated_savings_usd: float
    age_seconds: float
    created_at: float


@dataclass
class ClusterMember:
    """
    Represent one entry inside a semantic cluster.
    """

    prompt_preview: str
    hits: int
    similarity_to_centroid: float


@dataclass
class Cluster:
    """
    Group of semantically similar cached prompts.
    """

    cluster_id: int
    centroid_prompt: str
    member_count: int
    total_hits: int
    avg_similarity: float
    members: list[ClusterMember]


@dataclass
class SavingsReport:
    """
    Summarise estimated cost savings from the cache.
    """

    total_entries: int
    total_cache_hits: int
    total_cost_of_original_calls_usd: float
    estimated_savings_usd: float
    avg_cost_per_call_usd: float


class CacheAnalytics:
    """
    Provide analytical views over the semantic cache: entry inspection,
    semantic clustering, and cost savings estimation.
    """

    async def inspect(self, store: cache_store.CacheStore) -> list[CacheEntryDetail]:
        """
        Return a detailed view of every entry currently in the cache.

        :param store: the cache backend to inspect
        :return: list of entry details sorted by hit count descending
        """
        entries = await store.get_all()
        now = time.time()
        details = []
        for entry in entries:
            details.append(
                CacheEntryDetail(
                    prompt_preview=entry.prompt[:120],
                    response_preview=entry.response[:120],
                    model=entry.model,
                    hits=entry.hits,
                    cost_usd=entry.cost_usd,
                    # Each hit saved one API call worth cost_usd.
                    estimated_savings_usd=round(entry.cost_usd * entry.hits, 6),
                    age_seconds=round(now - entry.created_at, 2),
                    created_at=entry.created_at,
                )
            )
        return sorted(details, key=lambda d: d.hits, reverse=True)

    async def clusters(
        self,
        store: cache_store.CacheStore,
        threshold: float = 0.85,
    ) -> list[Cluster]:
        """
        Group cached entries into semantic clusters using greedy cosine similarity.

        Entries within `threshold` cosine similarity of a cluster centroid are merged
        into that cluster. New clusters are seeded by the next unassigned entry.

        :param store: the cache backend to cluster
        :param threshold: minimum cosine similarity to join an existing cluster
        :return: list of clusters sorted by total hits descending
        """
        assert 0.0 < threshold <= 1.0, f"threshold must be in (0, 1], got '{threshold}'"
        entries = await store.get_all()
        if not entries:
            return []
        # Sort by hit count descending so high-value prompts seed clusters.
        entries = sorted(entries, key=lambda e: e.hits, reverse=True)
        embeddings = [np.array(e.embedding) for e in entries]
        assigned: list[int | None] = [None] * len(entries)
        cluster_indices: list[list[int]] = []
        # Greedy assignment: each entry joins the nearest cluster above threshold.
        for i, emb_i in enumerate(embeddings):
            best_cluster = None
            best_score = -1.0
            for c_idx, members in enumerate(cluster_indices):
                centroid_emb = embeddings[members[0]]
                score = self._cosine(emb_i, centroid_emb)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_cluster = c_idx
            if best_cluster is not None:
                cluster_indices[best_cluster].append(i)
                assigned[i] = best_cluster
            else:
                assigned[i] = len(cluster_indices)
                cluster_indices.append([i])
        # Build Cluster objects from the assignment groups.
        result = []
        for c_idx, members in enumerate(cluster_indices):
            centroid_emb = embeddings[members[0]]
            similarities = [self._cosine(embeddings[m], centroid_emb) for m in members]
            cluster_members = [
                ClusterMember(
                    prompt_preview=entries[m].prompt[:120],
                    hits=entries[m].hits,
                    similarity_to_centroid=round(similarities[j], 4),
                )
                for j, m in enumerate(members)
            ]
            result.append(
                Cluster(
                    cluster_id=c_idx,
                    centroid_prompt=entries[members[0]].prompt[:120],
                    member_count=len(members),
                    total_hits=sum(entries[m].hits for m in members),
                    avg_similarity=round(float(np.mean(similarities)), 4),
                    members=cluster_members,
                )
            )
        return sorted(result, key=lambda c: c.total_hits, reverse=True)

    async def savings_report(self, store: cache_store.CacheStore) -> SavingsReport:
        """
        Compute estimated cost savings from all cache hits to date.

        :param store: the cache backend to analyse
        :return: savings report with totals and averages
        """
        entries = await store.get_all()
        total_entries = len(entries)
        total_hits = sum(e.hits for e in entries)
        total_cost = sum(e.cost_usd for e in entries)
        estimated_savings = sum(e.cost_usd * e.hits for e in entries)
        avg_cost = total_cost / total_entries if total_entries else 0.0
        return SavingsReport(
            total_entries=total_entries,
            total_cache_hits=total_hits,
            total_cost_of_original_calls_usd=round(total_cost, 6),
            estimated_savings_usd=round(estimated_savings, 6),
            avg_cost_per_call_usd=round(avg_cost, 6),
        )

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        :param a: first vector
        :param b: second vector
        :return: cosine similarity score in [-1.0, 1.0]
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
