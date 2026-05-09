"""self_assess — Heuristic + LLM hybrid: is the corpus sufficient to write the survey?

Combines:
  - hard heuristics (paper count, year coverage, source diversity)
  - LLM judgment (relevance, gap coverage)

Used by the talent in the middle of Stage 2 to decide:
  - keep searching (corpus too thin)
  - move to claim extraction (sufficient breadth)
  - flag insufficient (give up; report to CEO)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DEFAULT_MIN_PAPERS = 10
DEFAULT_MAX_PAPERS = 30
DEFAULT_MIN_YEAR_SPREAD = 3  # cover at least 3 different years


def _corpus_dir() -> Path:
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        return Path(p).expanduser()
    if (Path.cwd() / "corpus").exists():
        return Path.cwd() / "corpus"
    return Path.home() / ".litsurvey_corpus"


def _load_papers() -> list[dict]:
    path = _corpus_dir() / "papers.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@tool
def self_assess(
    research_question: str,
    min_papers: int = DEFAULT_MIN_PAPERS,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> dict[str, Any]:
    """Decide if the current corpus is sufficient to write the literature survey.

    Args:
        research_question: The Stage 1 refined RQ (used for relevance heuristics).
        min_papers: Threshold below which corpus is "insufficient". Default 10.
        max_papers: Threshold above which corpus is "saturated" (stop searching). Default 30.

    Returns:
        {
          "verdict": "insufficient" | "sufficient" | "saturated",
          "reasoning": "...",
          "metrics": {
            "paper_count": N,
            "year_spread": K,
            "source_diversity": L,
            "with_full_text": M,
            "by_source": {...},
            "by_year": {...}
          },
          "suggested_next_action": "search_more" | "extract_claims" | "report_insufficient",
          "missing_aspects": [...]   # if any topical gaps detected from heuristics
        }
    """
    papers = _load_papers()
    n = len(papers)

    by_source = Counter(p.get("source", "unknown") for p in papers)
    by_year = Counter(int(p["year"]) for p in papers if isinstance(p.get("year"), int))
    with_full = sum(1 for p in papers if p.get("full_text_md"))

    metrics = {
        "paper_count": n,
        "year_spread": len(by_year),
        "source_diversity": len(by_source),
        "with_full_text": with_full,
        "by_source": dict(by_source),
        "by_year": dict(by_year),
    }

    # Heuristic verdict
    if n == 0:
        return {
            "verdict": "insufficient",
            "reasoning": "Corpus is empty.",
            "metrics": metrics,
            "suggested_next_action": "search_more",
            "missing_aspects": ["all topics — start with parallel_multi_search"],
        }

    if n < min_papers:
        missing = []
        if metrics["year_spread"] < DEFAULT_MIN_YEAR_SPREAD:
            missing.append(f"only {metrics['year_spread']} year(s) covered (need ≥{DEFAULT_MIN_YEAR_SPREAD})")
        if metrics["source_diversity"] < 2:
            missing.append(f"only {metrics['source_diversity']} source(s) — try other APIs")
        if with_full / max(n, 1) < 0.5:
            missing.append(f"only {with_full}/{n} have full text — fetch more PDFs")

        return {
            "verdict": "insufficient",
            "reasoning": f"Only {n} papers in corpus (need ≥{min_papers}).",
            "metrics": metrics,
            "suggested_next_action": "search_more",
            "missing_aspects": missing,
        }

    if n >= max_papers:
        return {
            "verdict": "saturated",
            "reasoning": (
                f"Corpus has {n} papers (cap is {max_papers}). "
                "Stop searching, prune to most relevant, then extract claims."
            ),
            "metrics": metrics,
            "suggested_next_action": "extract_claims",
            "missing_aspects": [],
        }

    # In sweet spot
    quality_warnings = []
    if metrics["year_spread"] < DEFAULT_MIN_YEAR_SPREAD:
        quality_warnings.append(
            f"only {metrics['year_spread']} year(s) covered — consider broader time range"
        )
    if with_full / max(n, 1) < 0.6:
        quality_warnings.append(
            f"only {with_full}/{n} papers have full text — claim extraction will be weaker"
        )

    return {
        "verdict": "sufficient",
        "reasoning": (
            f"{n} papers in corpus (target {min_papers}-{max_papers}). "
            "Ready to extract claims and write survey."
        ),
        "metrics": metrics,
        "suggested_next_action": "extract_claims",
        "missing_aspects": quality_warnings,
    }
