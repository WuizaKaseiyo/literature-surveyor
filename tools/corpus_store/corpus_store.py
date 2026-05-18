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
REFS_FILENAME = "refs.jsonl"
PROJECT_META_FILENAME = "project_meta.json"
GLOBAL_LOCK_FILENAME = ".lock"

# Tokenization regex — alphanumeric + apostrophe-aware
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
_STOP = frozenset(
    "a an the of in on at to for with from by and or but if then is are was were be been "
    "this that these those it its as we our you your they their he she them his her "
    "i me my so do does did have has had not no can may will would could should "
    "via using used use also more most less than which who what when where why how".split()
)


def _in_layered_mode() -> bool:
    """A2: layered mode is opt-in via LITSURVEY_GLOBAL_CORPUS_DIR.

    When unset, behavior is legacy single-tenant (current users see no change).
    When set, entities (papers, claims, indices) live in the global dir and
    each project records its reference set in refs.jsonl.
    """
    return bool(os.getenv("LITSURVEY_GLOBAL_CORPUS_DIR"))


def _global_dir() -> Path:
    """A2: shared entity store (papers, claims, indices) across projects."""
    p = os.getenv("LITSURVEY_GLOBAL_CORPUS_DIR")
    if p:
        path = Path(p).expanduser()
    else:
        path = Path.home() / ".litsurvey_corpus_global"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_dir() -> Path | None:
    """A2: per-project ref store. Returns None when no project context is
    detectable (standalone scripts / global-only operation)."""
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        path = Path(p).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    if _looks_like_workspace():
        path = Path.cwd() / "corpus"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return None


def _corpus_dir() -> Path:
    """Resolve the canonical entity-store directory.

    - Legacy mode (no LITSURVEY_GLOBAL_CORPUS_DIR): returns the project dir
      (or fallback ~/.litsurvey_corpus/) — single-tenant, all data co-located.
      Existing tests + existing users are unaffected.
    - Layered mode: returns the global dir. Papers/claims/indices live here;
      project refs live in `_project_dir()`.
    """
    if _in_layered_mode():
        return _global_dir()
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


# ---------------------------------------------------------------------------
# A2/A3: project refs + write lock for the shared global store
# ---------------------------------------------------------------------------

import contextlib
import fcntl
import hashlib


def _project_id() -> str:
    """Stable project ID derived from the absolute project directory path."""
    pd = _project_dir()
    if pd is None:
        return ""
    abs_path = str(pd.resolve())
    return hashlib.sha256(abs_path.encode()).hexdigest()[:12]


def _ensure_project_meta() -> None:
    pd = _project_dir()
    if pd is None:
        return
    meta_path = pd / PROJECT_META_FILENAME
    if meta_path.exists():
        return
    meta_path.write_text(json.dumps({
        "project_id": _project_id(),
        "project_path": str(pd.resolve()),
        "created_at": time.time(),
        "retired_paper_ids": [],
    }, ensure_ascii=False, indent=2))


def _load_project_refs() -> list[dict]:
    pd = _project_dir()
    if pd is None:
        return []
    path = pd / REFS_FILENAME
    if not path.exists():
        return []
    out: list[dict] = []
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


def _project_ref_paper_ids() -> set[str]:
    return {r["paper_id"] for r in _load_project_refs() if r.get("paper_id")}


def _append_project_ref(ref: dict) -> None:
    pd = _project_dir()
    if pd is None:
        return
    _ensure_project_meta()
    path = pd / REFS_FILENAME
    with path.open("a") as f:
        f.write(json.dumps(ref, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def _global_write_lock():
    """fcntl-based exclusive write lock for the global store.

    Wraps (read existing → mutate → write back) so concurrent OMC processes
    don't tear lines or clobber each other's index updates. No-op in legacy
    mode (no global store contention to worry about).
    """
    if not _in_layered_mode():
        yield
        return
    lock_path = _global_dir() / GLOBAL_LOCK_FILENAME
    lock_path.touch(exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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
    """Atomic write — tmp + rename so a concurrent reader can never see a
    truncated JSON file mid-write. POSIX guarantees rename atomicity within
    the same filesystem; concurrent reads see either the old or the new
    complete file, never a torn version."""
    path = _corpus_dir() / INDEX_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False))
    tmp.replace(path)


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
    """Add a paper to the corpus. Idempotent (skips global add if id already present).

    Required `paper` fields:
        id: unique identifier (arxiv_id / DOI / S2 paper_id / openalex_id)
        title: str

    Recommended fields:
        authors: list[str], year: int, venue: str, abstract: str,
        full_text_md: str (from pdf_extract), source_query: str (which query found it),
        source: "arxiv" | "semantic_scholar" | "openalex" | "user_upload"

    In layered mode (LITSURVEY_GLOBAL_CORPUS_DIR set), paper entity goes to the
    global store; a per-project ref record is appended to refs.jsonl regardless
    of whether the paper was already in global. This means re-running searches
    that surface the same paper accumulates per-query audit trail in refs.

    Returns (legacy mode):
        {"status": "added" | "skipped_duplicate", "id": "...", "corpus_size": N}
    Returns (layered mode):
        {"status": "added" | "skipped_duplicate",
         "global_status": "added" | "skipped_duplicate",
         "project_status": "added_ref" | "rejected",
         "id": "...",
         "corpus_size": N,            # project refs count
         "global_corpus_size": M}
    """
    if not isinstance(paper, dict):
        return {"error": "paper must be a dict", "status": "rejected"}
    pid = paper.get("id", "")
    if not pid:
        return {"error": "paper.id is required", "status": "rejected"}
    if not paper.get("title"):
        return {"error": "paper.title is required", "status": "rejected"}

    layered = _in_layered_mode()

    with _global_write_lock():
        papers = _load_papers()
        already_global = any(p.get("id") == pid for p in papers)

        if not already_global:
            paper.setdefault("added_at", time.time())
            _append_paper(paper)
            idx = _load_index()
            _update_index_for(paper, idx)
            _save_index(idx)
            global_size_after = len(papers) + 1
            global_status = "added"
        else:
            global_size_after = len(papers)
            global_status = "skipped_duplicate"

    # In layered mode, always append a ref so the per-query audit trail is
    # captured (same paper might be found by multiple queries — that's a signal).
    project_status = "rejected"
    project_size_after = 0
    if layered and _project_dir() is not None:
        ref = {
            "paper_id": pid,
            "source_query": paper.get("source_query", ""),
            "found_via": paper.get("source", ""),
            "added_at": time.time(),
            "kept": True,
        }
        _append_project_ref(ref)
        project_status = "added_ref"
        project_size_after = len(_project_ref_paper_ids())

    if layered:
        return {
            "status": "added" if (global_status == "added" or project_status == "added_ref") else "skipped_duplicate",
            "global_status": global_status,
            "project_status": project_status,
            "id": pid,
            "corpus_size": project_size_after,
            "global_corpus_size": global_size_after,
        }

    # Legacy mode: original return shape preserved exactly.
    if already_global:
        return {"status": "skipped_duplicate", "id": pid, "corpus_size": global_size_after}
    return {"status": "added", "id": pid, "corpus_size": global_size_after}


# ---------------------------------------------------------------------------
# @tool: search
# ---------------------------------------------------------------------------


@tool
def corpus_search(query: str, top_k: int = 10, scope: str = "both") -> dict[str, Any]:
    """Search the corpus using BM25-lite scoring.

    Returns the most relevant papers from corpus that match the query terms.
    Use this BEFORE making external API calls — corpus may already have what
    you need from a previous round.

    Args:
        query: Free-text query.
        top_k: How many top hits to return. Default 10, max 50.
        scope: layered-mode scope. "project" = only papers this project has
            ref'd; "global" = all global papers; "both" (default) = all global
            with `from_project` tag on each result. Ignored in legacy mode.

    Returns:
        {
          "results": [
            {"paper": {...}, "score": 12.3, "matched_terms": ["x", "y"],
             "from_project": True}  # in layered mode only
            ...
          ],
          "total_corpus_size": N,
          "query": "...",
          # layered mode only:
          "project_corpus_size": K,
          "global_corpus_size": M,
        }
    """
    if not query or not query.strip():
        return {"error": "empty query", "results": [], "total_corpus_size": 0}

    top_k = max(1, min(int(top_k), 50))
    layered = _in_layered_mode()
    scope_norm = scope.strip().lower() if scope else "both"
    if scope_norm not in {"project", "global", "both"}:
        scope_norm = "both"

    papers = _load_papers()
    if not papers:
        return {"results": [], "total_corpus_size": 0, "query": query}

    idx = _load_index()
    q_tokens = _tokenize(query)
    if not q_tokens:
        return {"results": [], "total_corpus_size": len(papers), "query": query}

    project_refs = _project_ref_paper_ids() if layered else set()

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
        # Apply project-only filter early to skip scoring papers we'll drop.
        if layered and scope_norm == "project" and pid not in project_refs:
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
    results: list[dict[str, Any]] = []
    for s, m, p in scores[:top_k]:
        item: dict[str, Any] = {"paper": p, "score": round(s, 3), "matched_terms": m}
        if layered:
            item["from_project"] = p["id"] in project_refs
        results.append(item)

    out: dict[str, Any] = {
        "results": results,
        "total_corpus_size": len(papers),
        "query": query,
    }
    if layered:
        out["project_corpus_size"] = len(project_refs)
        out["global_corpus_size"] = len(papers)
        out["scope"] = scope_norm
    return out


# ---------------------------------------------------------------------------
# @tool: status
# ---------------------------------------------------------------------------


@tool
def corpus_status() -> dict[str, Any]:
    """Report the current corpus state — paper count, year distribution, sources.

    Use this at the START of a Stage 2 task to see if you already have relevant papers.

    In layered mode (LITSURVEY_GLOBAL_CORPUS_DIR set), `corpus_size` is the
    project's ref count (papers this project has touched); `global_corpus_size`
    is the cross-project shared pool. Use the latter to spot reuse opportunities.

    Returns:
        {
          "corpus_size": N,                # project view (refs) in layered mode, else all papers
          "by_source": {"arxiv": 12, ...},
          "by_year": {"2024": 10, ...},
          "with_full_text": K,
          "claims_count": M,
          "corpus_dir": "/path/to/corpus",
          # layered mode only:
          "global_corpus_size": M,
          "global_claims_count": K,
          "project_dir": "/path/to/project/corpus",
          "global_dir": "/path/to/global"
        }
    """
    layered = _in_layered_mode()
    cd = _corpus_dir()

    all_papers = _load_papers()
    claims_path = cd / CLAIMS_FILENAME
    global_claims_count = 0
    if claims_path.exists():
        with claims_path.open() as f:
            global_claims_count = sum(1 for line in f if line.strip())

    if layered:
        ref_ids = _project_ref_paper_ids()
        project_papers = [p for p in all_papers if p.get("id") in ref_ids]
        by_source = Counter(p.get("source", "unknown") for p in project_papers)
        by_year = Counter(str(p.get("year", "unknown")) for p in project_papers)
        with_full = sum(1 for p in project_papers if p.get("full_text_md"))
        # Project-scope claims count = claims attached to ref'd papers.
        project_claims_count = 0
        if claims_path.exists():
            with claims_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                        if c.get("paper_id") in ref_ids:
                            project_claims_count += 1
                    except json.JSONDecodeError:
                        continue
        pd = _project_dir()
        return {
            "corpus_size": len(project_papers),
            "by_source": dict(by_source),
            "by_year": dict(by_year),
            "with_full_text": with_full,
            "claims_count": project_claims_count,
            "corpus_dir": str(pd) if pd else str(cd),
            "global_corpus_size": len(all_papers),
            "global_claims_count": global_claims_count,
            "project_dir": str(pd) if pd else "",
            "global_dir": str(_global_dir()),
            "mode": "layered",
        }

    # Legacy: original shape preserved.
    by_source = Counter(p.get("source", "unknown") for p in all_papers)
    by_year = Counter(str(p.get("year", "unknown")) for p in all_papers)
    with_full = sum(1 for p in all_papers if p.get("full_text_md"))
    return {
        "corpus_size": len(all_papers),
        "by_source": dict(by_source),
        "by_year": dict(by_year),
        "with_full_text": with_full,
        "claims_count": global_claims_count,
        "corpus_dir": str(cd),
    }


# ---------------------------------------------------------------------------
# @tool: list
# ---------------------------------------------------------------------------


@tool
def corpus_list_papers(
    limit: int = 50,
    source: str = "",
    scope: str = "project",
) -> dict[str, Any]:
    """List papers in the corpus. Optionally filter by source.

    Args:
        limit: Max papers to list. Default 50.
        source: Optional source filter ("arxiv" / "semantic_scholar" / etc).
        scope: layered-mode scope. "project" (default) = papers this project
            has ref'd; "global" = all global papers; "both" = global with
            `from_project` tag. Ignored in legacy mode.

    Returns:
        {"papers": [{"id": "...", "title": "...", "year": ..., "source": "..."}, ...]}
    """
    layered = _in_layered_mode()
    scope_norm = scope.strip().lower() if scope else "project"
    if scope_norm not in {"project", "global", "both"}:
        scope_norm = "project"

    papers = _load_papers()
    if layered and scope_norm == "project":
        ref_ids = _project_ref_paper_ids()
        papers = [p for p in papers if p.get("id") in ref_ids]
    if source:
        papers = [p for p in papers if p.get("source") == source]
    papers = papers[: max(1, min(int(limit), 200))]

    ref_ids = _project_ref_paper_ids() if layered else set()
    out_papers = []
    for p in papers:
        item = {
            "id": p.get("id", ""),
            "title": p.get("title", ""),
            "year": p.get("year"),
            "source": p.get("source", ""),
            "has_full_text": bool(p.get("full_text_md")),
        }
        if layered:
            item["from_project"] = p.get("id", "") in ref_ids
        out_papers.append(item)
    return {"papers": out_papers, "count": len(out_papers)}


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
