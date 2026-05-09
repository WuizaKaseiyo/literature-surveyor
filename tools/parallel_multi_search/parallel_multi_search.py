"""parallel_multi_search — Concurrently query arxiv + Semantic Scholar + OpenAlex.

Dedupes results by (arxiv_id, doi, normalized_title). Returns a unified list with
source provenance per paper.

Each source's tool is imported lazily so this file works even when run standalone.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from langchain_core.tools import tool

DEFAULT_SOURCES = ["arxiv", "openalex"]  # SS off by default if no API key


def _normalize_title(t: str) -> str:
    """Lowercase + strip punct + collapse whitespace for fuzzy dedup."""
    if not t:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(cleaned.split())


def _paper_key(p: dict) -> tuple[str, str, str]:
    """Dedup key: (arxiv_id, doi, normalized_title). Empty fields ignored in match."""
    arxiv_id = p.get("arxiv_id") or ""
    if not arxiv_id and isinstance(p.get("paper_id"), str) and "/" not in p.get("paper_id", ""):
        # Some sources put arxiv id in paper_id field
        pass
    doi = p.get("doi") or ""
    title = _normalize_title(p.get("title", ""))
    return (arxiv_id, doi, title)


def _merge_dedupe(results_per_source: dict[str, list[dict]]) -> list[dict]:
    """Dedup across sources. Prefer the entry with most fields filled."""
    bucket: dict[tuple, dict] = {}

    for source, papers in results_per_source.items():
        for p in papers:
            # Annotate provenance
            sources = list(p.get("found_in_sources") or [])
            if source not in sources:
                sources.append(source)
            p = {**p, "found_in_sources": sources}

            keys_to_check = []
            arxiv_id = p.get("arxiv_id") or ""
            doi = p.get("doi") or ""
            ntitle = _normalize_title(p.get("title", ""))

            if arxiv_id:
                keys_to_check.append(("arxiv", arxiv_id))
            if doi:
                keys_to_check.append(("doi", doi))
            if ntitle:
                keys_to_check.append(("title", ntitle))

            # Find existing entry that shares any key
            existing_key = None
            for kt in keys_to_check:
                if kt in bucket:
                    existing_key = kt
                    break

            if existing_key is None:
                # New entry — register under all available keys
                first_key = keys_to_check[0] if keys_to_check else ("title", _normalize_title(p.get("title", "untitled")))
                for kt in keys_to_check:
                    bucket[kt] = p
                if not keys_to_check:
                    bucket[first_key] = p
            else:
                # Merge into existing
                cur = bucket[existing_key]
                merged = _merge_two(cur, p)
                # Update all alias keys
                for kt in keys_to_check:
                    bucket[kt] = merged
                # Also rebind original key
                bucket[existing_key] = merged

    # Deduplicate values (since same paper appears under multiple keys)
    seen_ids: set[int] = set()
    out: list[dict] = []
    for v in bucket.values():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        out.append(v)
    return out


def _merge_two(a: dict, b: dict) -> dict:
    """Merge two paper dicts; prefer non-empty values; union sources list."""
    merged = dict(a)
    for k, v in b.items():
        if k == "found_in_sources":
            existing = list(merged.get("found_in_sources") or [])
            for s in v:
                if s not in existing:
                    existing.append(s)
            merged["found_in_sources"] = existing
        elif k not in merged or not merged[k]:
            merged[k] = v
    return merged


@tool
def parallel_multi_search(
    query: str,
    sources: list[str] | None = None,
    max_results_per_source: int = 30,
) -> dict[str, Any]:
    """Concurrently query multiple academic sources for the same query, dedupe, return unified list.

    Args:
        query: Free-text query.
        sources: Subset of ["arxiv", "semantic_scholar", "openalex"]. Default ["arxiv", "openalex"]
            (Semantic Scholar excluded by default to avoid rate limits unless S2_API_KEY is set).
        max_results_per_source: Per-source cap. Default 30.

    Returns:
        {
          "results": [...],            # unique papers
          "total_unique": N,
          "per_source_counts": {"arxiv": 30, "openalex": 25},
          "query": "...",
          "sources_queried": [...]
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total_unique": 0}

    requested = sources or list(DEFAULT_SOURCES)
    if "semantic_scholar" in requested and not os.getenv("S2_API_KEY"):
        # Auto-include only if user explicitly requested it; warn but continue
        pass

    return asyncio.run(_async_search(query, requested, max_results_per_source))


async def _async_search(
    query: str, sources: list[str], max_results: int
) -> dict[str, Any]:
    tasks = []
    name_per_task = []
    for s in sources:
        if s == "arxiv":
            tasks.append(_run_arxiv(query, max_results))
            name_per_task.append("arxiv")
        elif s == "semantic_scholar":
            tasks.append(_run_ss(query, max_results))
            name_per_task.append("semantic_scholar")
        elif s == "openalex":
            tasks.append(_run_openalex(query, max_results))
            name_per_task.append("openalex")

    raw = await asyncio.gather(*tasks, return_exceptions=True)

    per_source: dict[str, list[dict]] = {}
    per_source_counts: dict[str, int] = {}
    errors: dict[str, str] = {}

    for name, result in zip(name_per_task, raw):
        if isinstance(result, Exception):
            errors[name] = str(result)
            per_source[name] = []
            per_source_counts[name] = 0
            continue
        papers = result.get("results", []) if isinstance(result, dict) else []
        per_source[name] = papers
        per_source_counts[name] = len(papers)

    unified = _merge_dedupe(per_source)

    out = {
        "results": unified,
        "total_unique": len(unified),
        "per_source_counts": per_source_counts,
        "query": query,
        "sources_queried": sources,
    }
    if errors:
        out["errors"] = errors
    return out


# Wrappers that run sync tools in thread (so we can asyncio.gather them)


async def _run_arxiv(query: str, max_results: int) -> dict:
    from tools.arxiv_search.arxiv_search import arxiv_search  # type: ignore

    return await asyncio.to_thread(
        arxiv_search.invoke, {"query": query, "max_results": max_results}
    )


async def _run_ss(query: str, max_results: int) -> dict:
    from tools.semantic_scholar_search.semantic_scholar_search import (  # type: ignore
        semantic_scholar_search,
    )

    return await asyncio.to_thread(
        semantic_scholar_search.invoke, {"query": query, "max_results": max_results}
    )


async def _run_openalex(query: str, max_results: int) -> dict:
    from tools.openalex_search.openalex_search import openalex_search  # type: ignore

    return await asyncio.to_thread(
        openalex_search.invoke, {"query": query, "max_results": max_results}
    )
