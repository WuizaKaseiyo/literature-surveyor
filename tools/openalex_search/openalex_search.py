"""openalex_search — Query the OpenAlex API.

OpenAlex is the open-source replacement for Microsoft Academic Graph.
240M+ scholarly works, all fields. Free, no API key needed.

Joining the "polite pool" via OPENALEX_MAILTO env var gives faster + more
stable responses (https://docs.openalex.org/how-to-use-the-api/api-overview#the-polite-pool).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

import backoff
import requests
from langchain_core.tools import tool

OPENALEX_API_URL = "https://api.openalex.org/works"
TIMEOUT_S = 20
MAX_RESULTS_HARD_CAP = 200


def _on_backoff(details: dict) -> None:
    print(
        f"[openalex_search] backing off "
        f"{details['wait']:0.1f}s after {details['tries']} tries"
    )


@tool
def openalex_search(
    query: str,
    max_results: int = 30,
    since_year: int = 0,
    open_access_only: bool = False,
) -> dict[str, Any]:
    """Search OpenAlex (240M+ scholarly works, all fields, free).

    Best for: cross-domain search, older papers, venue/citation metadata,
    discovering work outside ML.

    Args:
        query: Free-text query.
        max_results: Max papers to return. Default 30, hard cap 200.
        since_year: Optional minimum publication year (e.g. 2020).
        open_access_only: If True, only return papers with free PDF.

    Returns:
        {
          "results": [
            {
              "openalex_id": "W123456789",
              "title": "...",
              "authors": ["First Last", ...],
              "venue": "...",
              "year": 2024,
              "abstract": "...",
              "cited_by_count": 87,
              "doi": "10.xxx/...",
              "arxiv_id": "",            # if cross-listed
              "pdf_url": "...",
              "is_open_access": true,
              "type": "article" | "preprint" | ...
            },
            ...
          ],
          "total": N,
          "query": "...",
          "source": "openalex"
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total": 0}

    max_results = max(1, min(int(max_results), MAX_RESULTS_HARD_CAP))

    mailto = os.getenv("OPENALEX_MAILTO")
    if not mailto:
        warnings.warn(
            "OPENALEX_MAILTO not set — joining polite pool gives faster responses. "
            "Set to your email address.",
            stacklevel=2,
        )

    try:
        works = _search_with_backoff(query, max_results, since_year, open_access_only, mailto)
    except Exception as e:
        return {"error": f"OpenAlex request failed: {e}", "results": [], "total": 0}

    results = [_normalize(w) for w in works]

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "source": "openalex",
    }


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.HTTPError, requests.exceptions.ConnectionError),
    max_tries=5,
    on_backoff=_on_backoff,
)
def _search_with_backoff(
    query: str, max_results: int, since_year: int, open_access_only: bool, mailto: str | None
) -> list[dict]:
    filters = []
    if since_year:
        filters.append(f"from_publication_date:{since_year}-01-01")
    if open_access_only:
        filters.append("is_oa:true")

    params: dict[str, str | int] = {
        "search": query,
        "per-page": max_results,
        "select": (
            "id,title,authorships,publication_year,primary_location,abstract_inverted_index,"
            "cited_by_count,doi,open_access,type,ids"
        ),
    }
    if filters:
        params["filter"] = ",".join(filters)
    if mailto:
        params["mailto"] = mailto

    rsp = requests.get(OPENALEX_API_URL, params=params, timeout=TIMEOUT_S)
    rsp.raise_for_status()
    body = rsp.json()

    time.sleep(0.3 if mailto else 1.0)
    return body.get("results", []) or []


def _normalize(w: dict) -> dict:
    """OpenAlex returns a baroque shape — flatten to our standard."""
    primary = w.get("primary_location") or {}
    venue = (primary.get("source") or {}).get("display_name") or ""
    oa = w.get("open_access") or {}
    ids = w.get("ids") or {}

    # OpenAlex stores abstracts as inverted-index for IP reasons. Reconstruct.
    abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))

    # arxiv ID detection from various places
    arxiv_id = ""
    doi = w.get("doi", "") or ""
    if doi and "arxiv" in doi.lower():
        arxiv_id = doi.split("/")[-1]

    return {
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "title": w.get("title", "") or "",
        "authors": [
            (a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])
        ],
        "venue": venue,
        "year": w.get("publication_year"),
        "abstract": abstract,
        "cited_by_count": w.get("cited_by_count", 0),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pdf_url": oa.get("oa_url", "") or "",
        "is_open_access": bool(oa.get("is_oa")),
        "type": w.get("type", ""),
    }


def _reconstruct_abstract(inv_idx: dict | None) -> str:
    """OpenAlex abstracts come as {word: [pos1, pos2, ...]}. Rebuild text."""
    if not inv_idx:
        return ""
    pos_to_word: dict[int, str] = {}
    for word, positions in inv_idx.items():
        for p in positions:
            pos_to_word[p] = word
    return " ".join(pos_to_word[i] for i in sorted(pos_to_word.keys()))
