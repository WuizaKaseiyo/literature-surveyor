"""claim_store — Read-side access to claims extracted from corpus papers.

Five LangChain @tools in one file (each registered separately):
- claim_search          BM25-lite over claim_text + evidence_quote
- claim_list_by_paper   all claims for a given paper
- claim_get             single claim by id
- claim_status          aggregate stats over claims.jsonl
- claim_find_evidence   given a sentence + paper, find best matching claims
                        (used by fact_check_rendered_survey to anchor sentences
                        to already-extracted claims instead of re-running BM25
                        on raw paper text)

Storage: reads `<CORPUS_DIR>/claims.jsonl` written by `extract_claims`. CORPUS_DIR
resolution mirrors `corpus_store` exactly so the two tools share the same files.

Index: a sidecar `claims_index.json` is built lazily — rebuilt whenever
claims.jsonl is newer than the index file. extract_claims does NOT need to know
about this index; the rebuild is on-demand.
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

CLAIMS_FILENAME = "claims.jsonl"
INDEX_FILENAME = "claims_index.json"

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
_STOP = frozenset(
    "a an the of in on at to for with from by and or but if then is are was were be been "
    "this that these those it its as we our you your they their he she them his her "
    "i me my so do does did have has had not no can may will would could should "
    "via using used use also more most less than which who what when where why how".split()
)


def _corpus_dir() -> Path:
    """Resolve corpus directory. Mirrors corpus_store._corpus_dir exactly.

    A2 layered mode: claims.jsonl lives in the global entity store when
    LITSURVEY_GLOBAL_CORPUS_DIR is set, so cross-project reuse of extracted
    claims is automatic.
    """
    g = os.getenv("LITSURVEY_GLOBAL_CORPUS_DIR")
    if g:
        path = Path(g).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
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
    cwd = Path.cwd()
    return (
        any(cwd.glob("stage*.md"))
        or (cwd / "task_tree.yaml").exists()
        or (cwd.parent / "task_tree.yaml").exists()
    )


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _claims_path() -> Path:
    return _corpus_dir() / CLAIMS_FILENAME


def _index_path() -> Path:
    return _corpus_dir() / INDEX_FILENAME


def _load_claims() -> list[dict]:
    p = _claims_path()
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _empty_index() -> dict:
    return {"doc_freq": {}, "doc_count": 0, "doc_tokens": {}}


def _build_index(claims: list[dict]) -> dict:
    idx = _empty_index()
    for c in claims:
        cid = c.get("id")
        if not cid:
            continue
        text = (c.get("claim_text", "") + " " + c.get("evidence_quote", ""))
        tokens = _tokenize(text)
        counts = Counter(tokens)
        idx["doc_tokens"][cid] = dict(counts)
        for tok in counts:
            idx["doc_freq"][tok] = idx["doc_freq"].get(tok, 0) + 1
        idx["doc_count"] += 1
    return idx


def _save_index(idx: dict) -> None:
    _index_path().write_text(json.dumps(idx, ensure_ascii=False))


def _get_index_and_claims() -> tuple[dict, list[dict]]:
    """Return (index, claims). Rebuilds index lazily when stale."""
    claims = _load_claims()
    cp, ip = _claims_path(), _index_path()
    if not cp.exists():
        return _empty_index(), claims
    if not ip.exists() or ip.stat().st_mtime < cp.stat().st_mtime:
        idx = _build_index(claims)
        _save_index(idx)
        return idx, claims
    try:
        return json.loads(ip.read_text()), claims
    except json.JSONDecodeError:
        idx = _build_index(claims)
        _save_index(idx)
        return idx, claims


def _bm25_score(
    q_tokens: list[str],
    doc_tokens: dict[str, int],
    doc_freq: dict[str, int],
    doc_count: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[float, list[str]]:
    """Return (score, matched_terms)."""
    score = 0.0
    matched: list[str] = []
    dl = sum(doc_tokens.values()) or 1
    n = max(doc_count, 1)
    for q in q_tokens:
        if q not in doc_tokens:
            continue
        df = doc_freq.get(q, 1)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        tf = doc_tokens[q]
        score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl)))
        matched.append(q)
    return score, matched


def _search_core(
    query: str,
    paper_ids: list[str] | None,
    claim_type: str,
    top_k: int,
) -> tuple[list[dict], int]:
    """Shared BM25 search used by claim_search and claim_find_evidence."""
    idx, claims = _get_index_and_claims()
    if not claims:
        return [], 0

    q_tokens = _tokenize(query)
    if not q_tokens:
        return [], len(claims)

    claim_by_id = {c["id"]: c for c in claims if c.get("id")}
    paper_filter = set(paper_ids) if paper_ids else None
    type_filter = claim_type.strip().lower() or None

    doc_lens = {cid: sum(toks.values()) for cid, toks in idx.get("doc_tokens", {}).items()}
    avgdl = max(sum(doc_lens.values()) / max(len(doc_lens), 1), 1.0) if doc_lens else 1.0

    scored: list[tuple[float, list[str], dict]] = []
    for cid, tokens in idx.get("doc_tokens", {}).items():
        claim = claim_by_id.get(cid)
        if not claim:
            continue
        if paper_filter and claim.get("paper_id") not in paper_filter:
            continue
        if type_filter and str(claim.get("claim_type", "")).lower() != type_filter:
            continue
        score, matched = _bm25_score(q_tokens, tokens, idx["doc_freq"], idx["doc_count"], avgdl)
        if score > 0:
            scored.append((score, matched, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(int(top_k), 50))]
    results = [
        {"claim": c, "score": round(s, 3), "matched_terms": m} for s, m, c in top
    ]
    return results, len(claims)


# ---------------------------------------------------------------------------
# @tool: search
# ---------------------------------------------------------------------------


@tool
def claim_search(
    query: str,
    paper_ids: list[str] | None = None,
    claim_type: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """BM25 search across all extracted claims (claim_text + evidence_quote).

    Use BEFORE writing a finding into stage2 markdown to confirm at least one
    extracted claim backs the assertion. If nothing comes back, either the
    corpus doesn't really support the claim or extract_claims missed it.

    Args:
        query: Free-text query (a candidate finding sentence works well).
        paper_ids: Optional. Restrict to claims from these paper IDs.
        claim_type: Optional. One of factual/methodological/negative_result/conjecture.
        top_k: Max results. Default 10, cap 50.

    Returns:
        {
          "results": [{"claim": {...}, "score": 12.3, "matched_terms": ["x", "y"]}, ...],
          "total_claims": N,
          "query": "..."
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total_claims": 0}
    results, total = _search_core(query, paper_ids, claim_type, top_k)
    return {"results": results, "total_claims": total, "query": query}


# ---------------------------------------------------------------------------
# @tool: list by paper
# ---------------------------------------------------------------------------


@tool
def claim_list_by_paper(paper_id: str) -> dict[str, Any]:
    """List every claim belonging to one paper.

    Args:
        paper_id: The id used when corpus_add_paper was called.

    Returns:
        {"paper_id": "...", "count": N, "claims": [{...}, ...]}
    """
    if not paper_id:
        return {"error": "paper_id required", "paper_id": "", "count": 0, "claims": []}
    claims = [c for c in _load_claims() if c.get("paper_id") == paper_id]
    return {"paper_id": paper_id, "count": len(claims), "claims": claims}


# ---------------------------------------------------------------------------
# @tool: get
# ---------------------------------------------------------------------------


@tool
def claim_get(claim_id: str) -> dict[str, Any]:
    """Fetch one claim by id (format: '<paper_id>#claim-N').

    Args:
        claim_id: The id assigned by extract_claims.

    Returns:
        The claim dict, or {"error": "not_found", "id": "..."}.
    """
    if not claim_id:
        return {"error": "claim_id required"}
    for c in _load_claims():
        if c.get("id") == claim_id:
            return c
    return {"error": "not_found", "id": claim_id}


# ---------------------------------------------------------------------------
# @tool: status
# ---------------------------------------------------------------------------


@tool
def claim_status() -> dict[str, Any]:
    """Aggregate stats: total count, type distribution, papers covered.

    Returns:
        {
          "claims_count": N,
          "papers_with_claims": K,
          "by_type": {"factual": 12, "methodological": 4, ...},
          "with_evidence_quote": M,
          "unverified_quote_count": U,  # placeholder until extract_claims v2 sets the flag
          "corpus_dir": "/path/to/corpus"
        }
    """
    claims = _load_claims()
    by_type = Counter(str(c.get("claim_type", "unknown")) for c in claims)
    papers_with_claims = len({c.get("paper_id") for c in claims if c.get("paper_id")})
    with_quote = sum(1 for c in claims if c.get("evidence_quote"))
    unverified = sum(1 for c in claims if c.get("evidence_quote_verified") is False)
    return {
        "claims_count": len(claims),
        "papers_with_claims": papers_with_claims,
        "by_type": dict(by_type),
        "with_evidence_quote": with_quote,
        "unverified_quote_count": unverified,
        "corpus_dir": str(_corpus_dir()),
    }


# ---------------------------------------------------------------------------
# @tool: find_evidence  (used by fact_check_rendered_survey in PR-3)
# ---------------------------------------------------------------------------


@tool
def claim_find_evidence(
    claim_text: str,
    paper_id: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Find best-matching extracted claims in one paper for a rendered sentence.

    fact_check_rendered_survey should call this BEFORE BM25-searching the raw
    paper text. If a high-scoring match exists, the LLM judge gets the matched
    claim's evidence_quote as authoritative context — much stronger anchor than
    re-deriving evidence from full_text_md every time.

    Args:
        claim_text: The sentence from the rendered survey (the candidate claim).
        paper_id: Restrict matches to this paper's claims.
        top_k: How many candidate matches to return. Default 3.

    Returns:
        {
          "paper_id": "...",
          "query": "...",
          "matches": [
            {
              "claim_id": "<paper_id>#claim-N",
              "claim_text": "...",
              "evidence_quote": "...",
              "source_section": "...",
              "score": 12.3,
              "matched_terms": ["x", "y"]
            },
            ...
          ]
        }
    """
    if not claim_text or not claim_text.strip():
        return {"error": "claim_text required", "matches": []}
    if not paper_id:
        return {"error": "paper_id required", "matches": []}

    results, _ = _search_core(
        query=claim_text,
        paper_ids=[paper_id],
        claim_type="",
        top_k=max(1, min(int(top_k), 20)),
    )
    matches = [
        {
            "claim_id": r["claim"].get("id", ""),
            "claim_text": r["claim"].get("claim_text", ""),
            "evidence_quote": r["claim"].get("evidence_quote", ""),
            "source_section": r["claim"].get("source_section", ""),
            "score": r["score"],
            "matched_terms": r["matched_terms"],
        }
        for r in results
    ]
    return {
        "paper_id": paper_id,
        "query": claim_text,
        "matches": matches,
    }


# Module-load timestamp (lets tests assert side effects didn't fire at import)
_LOADED_AT = time.time()
