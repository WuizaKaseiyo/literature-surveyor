"""corpus_store — Per-project paper corpus with simple BM25-lite search.

Five LangChain @tools in one file (each registered separately):
- corpus_add_paper
- corpus_search
- corpus_status
- corpus_list_papers
- corpus_get_paper

Storage: JSONL at <CORPUS_DIR>/papers.jsonl with a sidecar index.json.
CORPUS_DIR resolution order:
  1. LITSURVEY_CORPUS_DIR env var
  2. <CWD>/corpus/                       (when run inside an OMC project workspace)
  3. ~/.litsurvey_corpus/                (fallback)

The corpus is per-process-CWD by default — each OMC project gets its own corpus
because OMC sets CWD to the project workspace before invoking tools.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

CORPUS_FILENAME = "papers.jsonl"
INDEX_FILENAME = "corpus_index.json"
CLAIMS_FILENAME = "claims.jsonl"

# Tokenization regex — alphanumeric + apostrophe-aware
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
_STOP = frozenset(
    "a an the of in on at to for with from by and or but if then is are was were be been "
    "this that these those it its as we our you your they their he she them his her "
    "i me my so do does did have has had not no can may will would could should "
    "via using used use also more most less than which who what when where why how".split()
)


def _corpus_dir() -> Path:
    """Resolve corpus directory. Auto-creates if missing."""
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        path = Path(p).expanduser()
    elif (Path.cwd() / "corpus").exists() or _looks_like_workspace():
        path = Path.cwd() / "corpus"
    else:
        path = Path.home() / ".litsurvey_corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _looks_like_workspace() -> bool:
    """Heuristic: are we in an OMC project workspace?"""
    cwd = Path.cwd()
    # Indicators: stage1/stage2 markdown files exist, or task_tree.yaml at parent
    return (
        any(cwd.glob("stage*.md"))
        or (cwd / "task_tree.yaml").exists()
        or (cwd.parent / "task_tree.yaml").exists()
    )


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP]


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _load_papers() -> list[dict]:
    path = _corpus_dir() / CORPUS_FILENAME
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


def _append_paper(paper: dict) -> None:
    path = _corpus_dir() / CORPUS_FILENAME
    with path.open("a") as f:
        f.write(json.dumps(paper, ensure_ascii=False) + "\n")


def _rewrite_papers(papers: list[dict]) -> None:
    path = _corpus_dir() / CORPUS_FILENAME
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_index() -> dict:
    path = _corpus_dir() / INDEX_FILENAME
    if not path.exists():
        return {"doc_freq": {}, "doc_count": 0, "doc_tokens": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"doc_freq": {}, "doc_count": 0, "doc_tokens": {}}


def _save_index(idx: dict) -> None:
    path = _corpus_dir() / INDEX_FILENAME
    path.write_text(json.dumps(idx, ensure_ascii=False))


def _update_index_for(paper: dict, idx: dict) -> None:
    """Add a paper's tokens to the index."""
    pid = paper["id"]
    text = " ".join(
        [paper.get("title", ""), paper.get("abstract", ""), paper.get("full_text_md", "")[:5000]]
    )
    tokens = _tokenize(text)
    counts = Counter(tokens)
    idx["doc_tokens"][pid] = dict(counts)
    for tok in counts:
        idx["doc_freq"][tok] = idx["doc_freq"].get(tok, 0) + 1
    idx["doc_count"] = idx.get("doc_count", 0) + 1


# ---------------------------------------------------------------------------
# @tool: add
# ---------------------------------------------------------------------------


@tool
def corpus_add_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Add a paper to the project corpus. Idempotent (skips if id already present).

    Required `paper` fields:
        id: unique identifier (arxiv_id / DOI / S2 paper_id / openalex_id)
        title: str

    Recommended fields:
        authors: list[str], year: int, venue: str, abstract: str,
        full_text_md: str (from pdf_extract), source_query: str (which query found it),
        source: "arxiv" | "semantic_scholar" | "openalex" | "user_upload"

    Returns:
        {"status": "added" | "skipped_duplicate", "id": "...", "corpus_size": N}
    """
    if not isinstance(paper, dict):
        return {"error": "paper must be a dict", "status": "rejected"}
    pid = paper.get("id", "")
    if not pid:
        return {"error": "paper.id is required", "status": "rejected"}
    if not paper.get("title"):
        return {"error": "paper.title is required", "status": "rejected"}

    papers = _load_papers()
    if any(p.get("id") == pid for p in papers):
        return {"status": "skipped_duplicate", "id": pid, "corpus_size": len(papers)}

    paper.setdefault("added_at", time.time())
    _append_paper(paper)

    idx = _load_index()
    _update_index_for(paper, idx)
    _save_index(idx)

    return {"status": "added", "id": pid, "corpus_size": len(papers) + 1}


# ---------------------------------------------------------------------------
# @tool: search
# ---------------------------------------------------------------------------


@tool
def corpus_search(query: str, top_k: int = 10) -> dict[str, Any]:
    """Search the local corpus using BM25-lite scoring.

    Returns the most relevant papers from corpus that match the query terms.
    Use this BEFORE making external API calls — corpus may already have what
    you need from a previous round.

    Args:
        query: Free-text query.
        top_k: How many top hits to return. Default 10, max 50.

    Returns:
        {
          "results": [
            {"paper": {...}, "score": 12.3, "matched_terms": ["x", "y"]},
            ...
          ],
          "total_corpus_size": N,
          "query": "..."
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total_corpus_size": 0}

    top_k = max(1, min(int(top_k), 50))

    papers = _load_papers()
    if not papers:
        return {"results": [], "total_corpus_size": 0, "query": query}

    idx = _load_index()
    q_tokens = _tokenize(query)
    if not q_tokens:
        return {"results": [], "total_corpus_size": len(papers), "query": query}

    N = max(idx.get("doc_count", len(papers)), 1)
    scores: list[tuple[float, list[str], dict]] = []
    paper_by_id = {p["id"]: p for p in papers}

    # Average doc length for BM25 normalization (token count)
    doc_lens = {pid: sum(toks.values()) for pid, toks in idx.get("doc_tokens", {}).items()}
    avgdl = max(sum(doc_lens.values()) / max(len(doc_lens), 1), 1.0)
    k1, b = 1.5, 0.75

    for pid, tokens in idx.get("doc_tokens", {}).items():
        if pid not in paper_by_id:
            continue
        score = 0.0
        matched = []
        dl = doc_lens.get(pid, 1) or 1
        for q in q_tokens:
            if q not in tokens:
                continue
            df = idx["doc_freq"].get(q, 1)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            tf = tokens[q]
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)))
            matched.append(q)
        if score > 0:
            scores.append((score, matched, paper_by_id[pid]))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = [
        {"paper": p, "score": round(s, 3), "matched_terms": m}
        for s, m, p in scores[:top_k]
    ]

    return {
        "results": results,
        "total_corpus_size": len(papers),
        "query": query,
    }


# ---------------------------------------------------------------------------
# @tool: status
# ---------------------------------------------------------------------------


@tool
def corpus_status() -> dict[str, Any]:
    """Report the current corpus state — paper count, year distribution, sources.

    Use this at the START of a Stage 2 task to see if you already have relevant papers.

    Returns:
        {
          "corpus_size": N,
          "by_source": {"arxiv": 12, "semantic_scholar": 8, ...},
          "by_year": {"2024": 10, "2023": 8, ...},
          "with_full_text": K,
          "claims_count": M,
          "corpus_dir": "/path/to/corpus"
        }
    """
    papers = _load_papers()
    cd = _corpus_dir()
    claims_path = cd / CLAIMS_FILENAME
    claims_count = 0
    if claims_path.exists():
        with claims_path.open() as f:
            claims_count = sum(1 for line in f if line.strip())

    by_source = Counter(p.get("source", "unknown") for p in papers)
    by_year = Counter(str(p.get("year", "unknown")) for p in papers)
    with_full = sum(1 for p in papers if p.get("full_text_md"))

    return {
        "corpus_size": len(papers),
        "by_source": dict(by_source),
        "by_year": dict(by_year),
        "with_full_text": with_full,
        "claims_count": claims_count,
        "corpus_dir": str(cd),
    }


# ---------------------------------------------------------------------------
# @tool: list
# ---------------------------------------------------------------------------


@tool
def corpus_list_papers(
    limit: int = 50,
    source: str = "",
) -> dict[str, Any]:
    """List papers in the corpus. Optionally filter by source.

    Args:
        limit: Max papers to list. Default 50.
        source: Optional source filter ("arxiv" / "semantic_scholar" / etc).

    Returns:
        {"papers": [{"id": "...", "title": "...", "year": ..., "source": "..."}, ...]}
    """
    papers = _load_papers()
    if source:
        papers = [p for p in papers if p.get("source") == source]
    papers = papers[: max(1, min(int(limit), 200))]
    return {
        "papers": [
            {
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "year": p.get("year"),
                "source": p.get("source", ""),
                "has_full_text": bool(p.get("full_text_md")),
            }
            for p in papers
        ],
        "count": len(papers),
    }


# ---------------------------------------------------------------------------
# @tool: get
# ---------------------------------------------------------------------------


@tool
def corpus_get_paper(paper_id: str) -> dict[str, Any]:
    """Fetch the full record for a single paper by ID.

    Args:
        paper_id: The id passed to corpus_add_paper.

    Returns:
        The full paper dict, or {"error": "not_found", "id": ...} if absent.
    """
    if not paper_id:
        return {"error": "paper_id required"}
    for p in _load_papers():
        if p.get("id") == paper_id:
            return p
    return {"error": "not_found", "id": paper_id}
