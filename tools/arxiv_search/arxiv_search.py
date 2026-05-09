"""arxiv_search — Query the arxiv.org export API and return structured paper metadata.

Self-contained LangChain @tool. Uses only stdlib + langchain_core.
No API key required (arxiv is free + rate-limited to ~1 req/3s).
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from langchain_core.tools import tool

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
TIMEOUT_S = 20
MAX_RESULTS_HARD_CAP = 100


@tool
def arxiv_search(
    query: str,
    max_results: int = 30,
    since: str = "",
) -> dict[str, Any]:
    """Search arxiv.org for papers matching a query.

    Use this to find recent ML/AI/physics/math preprints. Free + no API key needed.

    Args:
        query: Free-text query. Supports field prefixes: ti:title, abs:abstract,
            au:author, cat:category. Default search is "all:".
            Example: "RLHF small language models" or "ti:attention au:vaswani".
        max_results: Max papers to return. Default 30, hard cap 100.
        since: Optional ISO date "YYYY-MM-DD". Filter to papers published on/after.

    Returns:
        {
          "results": [
            {
              "arxiv_id": "2401.12345",        # without version suffix
              "title": "...",
              "authors": ["First Last", ...],
              "abstract": "...",
              "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
              "published": "2024-01-25T...",
              "categories": ["cs.CL", "cs.LG"]
            },
            ...
          ],
          "total": N,
          "query": "...",
          "source": "arxiv"
        }

    Errors are returned as {"error": "...", "results": [], "total": 0}.
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total": 0}

    max_results = max(1, min(int(max_results), MAX_RESULTS_HARD_CAP))

    params = urllib.parse.urlencode(
        {
            "search_query": _build_search_query(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception as e:
        return {"error": f"arxiv API request failed: {e}", "results": [], "total": 0}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"error": f"arxiv XML parse failed: {e}", "results": [], "total": 0}

    results = []
    for entry in root.findall("atom:entry", NS):
        published = entry.findtext("atom:published", "", NS)
        if since and published and published < since:
            continue

        raw_id = entry.findtext("atom:id", "", NS)
        # http://arxiv.org/abs/2401.12345v2 → 2401.12345
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0] if "/abs/" in raw_id else raw_id

        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
        ]
        categories += [
            c.attrib.get("term", "")
            for c in entry.findall("{http://arxiv.org/schemas/atom}category")
            if c.attrib.get("term", "") not in categories
        ]

        results.append(
            {
                "arxiv_id": arxiv_id,
                "title": _clean(entry.findtext("atom:title", "", NS)),
                "authors": [
                    a.findtext("atom:name", "", NS) for a in entry.findall("atom:author", NS)
                ],
                "abstract": _clean(entry.findtext("atom:summary", "", NS)),
                "pdf_url": pdf_url,
                "published": published,
                "categories": [c for c in categories if c],
            }
        )

    # Polite pause to respect arxiv ToS
    time.sleep(1.0)

    return {
        "results": results,
        "total": len(results),
        "query": query,
        "source": "arxiv",
    }


def _build_search_query(q: str) -> str:
    """Wrap a bare query as `all:<q>` if it has no field prefix."""
    if ":" in q.split()[0] if q.split() else False:
        return q  # already has field prefix
    return f"all:{q}"


def _clean(s: str) -> str:
    """Collapse whitespace and trim — arxiv abstracts have ugly newlines."""
    return " ".join(s.split()).strip()
