"""Graphiti temporal-graph wrapper for mnemolith (Phase 1c).

Layers a time-aware knowledge graph on top of the vault. Each note becomes
an "episode" — Graphiti auto-extracts entities + relationships + facts and
tracks WHEN each fact was true. Enables queries like:

    "What did Maria say about her timeline in March?"
    "Has anything changed about the Highlands inventory since June?"

We default to Anthropic for the LLM (matches the rest of the LevelUp stack)
and OpenAI for embeddings (text-embedding-3-small, matches vault pgvector).
Both are swappable via env vars.

Async — graphiti-core is async-first. Call sites should `asyncio.run(...)`
or be inside an event loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.llm_client import AnthropicClient, LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType

from mnemolith.config import (
    get_graphiti_llm_model,
    get_graphiti_llm_provider,
    get_neo4j_password,
    get_neo4j_uri,
    get_neo4j_user,
)


@dataclass
class VaultEpisode:
    """One vault note translated into Graphiti's episode shape."""

    name: str  # stable id for idempotency — use the vault path
    body: str  # the note's markdown body
    reference_time: datetime  # when the knowledge in the note was true
    source_description: str  # e.g. "vault note · Maria meeting"


def _build_llm_client():
    provider = get_graphiti_llm_provider()
    model = get_graphiti_llm_model()
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY is required when GRAPHITI_LLM_PROVIDER=anthropic.")
        cfg = LLMConfig(api_key=api_key, model=model or "claude-sonnet-4-6")
        return AnthropicClient(config=cfg)
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY is required when GRAPHITI_LLM_PROVIDER=openai.")
        cfg = LLMConfig(api_key=api_key, model=model or "gpt-4o-mini")
        return OpenAIClient(config=cfg)
    raise OSError(f"Unsupported GRAPHITI_LLM_PROVIDER={provider!r}. Use 'anthropic' or 'openai'.")


def build_graphiti() -> Graphiti:
    """Construct a connected Graphiti client. Caller owns the lifecycle."""
    return Graphiti(
        uri=get_neo4j_uri(),
        user=get_neo4j_user(),
        password=get_neo4j_password(),
        llm_client=_build_llm_client(),
    )


async def ensure_indices(graphiti: Graphiti) -> None:
    """One-time per Neo4j db: create the Graphiti indices + constraints.

    Idempotent on the Graphiti side. Safe to call on every startup.
    """
    await graphiti.build_indices_and_constraints()


async def add_episode(
    graphiti: Graphiti,
    episode: VaultEpisode,
) -> None:
    """Ingest one vault note as a Graphiti episode.

    Graphiti dedupes entities + relationships across episodes by name, so
    re-ingesting the same note with updated body produces a clean diff
    rather than duplicates. The episode `name` should be a stable id
    (vault path is the canonical choice).
    """
    await graphiti.add_episode(
        name=episode.name,
        episode_body=episode.body,
        source=EpisodeType.text,
        source_description=episode.source_description,
        reference_time=episode.reference_time,
    )


async def temporal_search(
    graphiti: Graphiti,
    query: str,
    num_results: int = 10,
):
    """Hybrid temporal search across the graph.

    Returns Graphiti's ranked result set: relevant facts/entities ordered
    by a mix of semantic similarity, graph centrality, and recency.
    Callers should serialize results as needed for transport (MCP / JSON).
    """
    return await graphiti.search(query=query, num_results=num_results)
