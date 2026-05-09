"""semantic_scholar_search — Query Semantic Scholar Graph API.

Adapted from SakanaAI/AI-Scientist-v2 (ai_scientist/tools/semantic_scholar.py),
Apache-2.0. Original used a custom BaseTool ABC; here repackaged as LangChain @tool.

Sorts results by citationCount descending (influence-weighted ranking).

Set S2_API_KEY env var for higher rate limits (otherwise heavily throttled).
"""

from __future__ import annotations

import functools
import os
import time
import warnings
from typing import Any, Callable

import requests
from langchain_core.tools import tool


def _exp_backoff_retry(
    exceptions: tuple[type[BaseException], ...],
    max_tries: int = 5,
    base_wait_s: float = 1.0,
) -> Callable:
    """Stdlib-only exponential backoff decorator (replaces `backoff` library).

    Retries `max_tries` times on the listed exceptions, sleeping
    base * 2^(attempt-1) seconds between attempts.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(1, max_tries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_tries:
                        break
                    wait = base_wait_s * (2 ** (attempt - 1))
                    print(f"[{fn.__name__}] backing off {wait:.1f}s after {attempt} tries: {e}")
                    time.sleep(wait)
            if last_exc:
                raise last_exc
        return wrapper
    return decorator

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TIMEOUT_S = 20
MAX_RESULTS_HARD_CAP = 100
DEFAULT_FIELDS = "title,authors,venue,year,abstract,citationCount,externalIds,openAccessPdf"


@tool
def semantic_scholar_search(
    query: str,
    max_results: int = 20,
) -> dict[str, Any]:
    """Search Semantic Scholar for papers. Sorted by citation count (most influential first).

    Use for citation-aware search and influence-weighted retrieval. Especially useful
    when you want to find foundational works in a sub-field.

    Set environment variable S2_API_KEY for higher rate limits (free at
    https://www.semanticscholar.org/product/api).

    Args:
        query: Free-text search query.
        max_results: Max papers to return. Default 20, hard cap 100.

    Returns:
        {
          "results": [
            {
              "paper_id": "<S2 paper ID>",
              "title": "...",
              "authors": ["First Last", ...],
              "venue": "...",
              "year": 2024,
              "abstract": "...",
              "citation_count": 123,
              "arxiv_id": "2401.12345",     # if cross-listed on arxiv
              "doi": "10.xxx/...",          # if available
              "pdf_url": "..."              # open access PDF if available
            },
            ...
          ],
          "total": N,
          "query": "...",
          "source": "semantic_scholar"
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total": 0}

    max_results = max(1, min(int(max_results), MAX_RESULTS_HARD_CAP))

    api_key = os.getenv("S2_API_KEY")
    if not api_key:
        warnings.warn(
            "S2_API_KEY not set — Semantic Scholar requests will be heavily rate-limited.",
            stacklevel=2,
        )

    try:
        papers = _search_with_backoff(query, max_results, api_key)
    except Exception as e:
        return {"error": f"Semantic Scholar request failed: {e}", "results": [], "total": 0}

    if not papers:
        return {
            "results": [],
            "total": 0,
            "query": query,
            "source": "semantic_scholar",
        }

    # Sort by citationCount descending — most influential first
    papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

    results = [_normalize(p) for p in papers]

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "source": "semantic_scholar",
    }


@_exp_backoff_retry(
    exceptions=(requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    max_tries=5,
    base_wait_s=1.0,
)
def _search_with_backoff(query: str, max_results: int, api_key: str | None) -> list[dict] | None:
    """Hit S2 API with exponential backoff on transient errors."""
    headers = {"X-API-KEY": api_key} if api_key else {}
    params = {"query": query, "limit": max_results, "fields": DEFAULT_FIELDS}

    rsp = requests.get(S2_API_URL, headers=headers, params=params, timeout=TIMEOUT_S)
    rsp.raise_for_status()
    body = rsp.json()

    # Be polite even when not rate-limited
    time.sleep(0.5 if api_key else 1.5)

    if not body.get("total", 0):
        return None
    return body.get("data", []) or []


def _normalize(p: dict) -> dict:
    """Flatten S2's response shape into our standard paper schema."""
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return {
        "paper_id": p.get("paperId", ""),
        "title": p.get("title", ""),
        "authors": [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")],
        "venue": p.get("venue", "") or "",
        "year": p.get("year"),
        "abstract": p.get("abstract", "") or "",
        "citation_count": p.get("citationCount", 0) or 0,
        "arxiv_id": ext.get("ArXiv", ""),
        "doi": ext.get("DOI", ""),
        "pdf_url": oa.get("url", "") or "",
    }
