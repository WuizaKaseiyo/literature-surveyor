"""verify_citations — Scan markdown for citations and verify each is real.

Recognizes 3 cite formats:
  [Author Year, arxiv:XXXX.XXXXX]
  [Author Year, doi:10.xxx/yyy]
  [Author Year, S2:abcd1234]

Verifies via:
  1. corpus_search local first (cheap, instant)
  2. arxiv API for arxiv: cites
  3. Semantic Scholar API for doi: and S2: cites

Returns structured pass/fail report.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

# Match: [Anything, arxiv:2401.12345] or [Anything, doi:10.x/y] or [Anything, S2:abc]
# Captures: full match, kind (arxiv/doi/S2), id
CITE_PATTERN = re.compile(
    r"\[([^\]]*?),\s*(arxiv|doi|S2):([^\]\s]+)\]",
    re.IGNORECASE,
)

ARXIV_ABS_URL = "https://export.arxiv.org/api/query"
S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
TIMEOUT_S = 15
CONTEXT_CHARS = 80


def _corpus_dir() -> Path:
    # A2 layered mode: entity store (papers.jsonl) lives in the global dir.
    g = os.getenv("LITSURVEY_GLOBAL_CORPUS_DIR")
    if g:
        path = Path(g).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        return Path(p).expanduser()
    if (Path.cwd() / "corpus").exists():
        return Path.cwd() / "corpus"
    return Path.home() / ".litsurvey_corpus"


def _load_corpus_ids() -> set[str]:
    """Set of all paper IDs in corpus for fast cross-check."""
    path = _corpus_dir() / "papers.jsonl"
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                if pid := p.get("id"):
                    out.add(pid)
                # Also record arxiv_id / doi cross-refs
                if ax := p.get("arxiv_id"):
                    out.add(ax)
                if doi := p.get("doi"):
                    out.add(doi)
            except json.JSONDecodeError:
                continue
    return out


@tool
def verify_citations(markdown: str) -> dict[str, Any]:
    """Scan markdown for citations and verify each one against real sources.

    Recognizes:
        [Author Year, arxiv:XXXX.XXXXX]
        [Author Year, doi:10.xxx/yyy]
        [Author Year, S2:abcd1234]

    Verification order per cite:
        1. Local corpus (instant)
        2. Live API (arxiv / Semantic Scholar)

    Args:
        markdown: The markdown text to scan.

    Returns:
        {
          "verified": [{"raw": "[...]", "kind": "arxiv", "id": "...",
                        "title": "...", "matched_corpus": true}, ...],
          "unverified": [{"raw": "[...]", "kind": "arxiv", "id": "...",
                          "reason": "...", "context_around": "..."}, ...],
          "verified_count": N,
          "unverified_count": M,
          "total_cites": N+M,
          "verification_method": "local + live API"
        }
    """
    if not markdown:
        return {
            "verified": [],
            "unverified": [],
            "verified_count": 0,
            "unverified_count": 0,
            "total_cites": 0,
            "verification_method": "no input",
        }

    corpus_ids = _load_corpus_ids()
    verified: list[dict] = []
    unverified: list[dict] = []

    matches = list(CITE_PATTERN.finditer(markdown))
    seen: set[tuple[str, str]] = set()  # dedupe (kind, id)

    for m in matches:
        raw = m.group(0)
        author_year = m.group(1).strip()
        kind = m.group(2).lower()
        cite_id = m.group(3).strip().rstrip(".,;:)")

        if (kind, cite_id) in seen:
            # Same cite appears multiple times in markdown — verify once
            continue
        seen.add((kind, cite_id))

        ctx = _context_around(markdown, m.start(), m.end())

        # Step 1: corpus check
        if cite_id in corpus_ids:
            verified.append(
                {
                    "raw": raw,
                    "kind": kind,
                    "id": cite_id,
                    "author_year": author_year,
                    "title": "(in corpus)",
                    "matched_corpus": True,
                }
            )
            continue

        # Step 2: live verification
        ok, info = _verify_remote(kind, cite_id)
        if ok:
            verified.append(
                {
                    "raw": raw,
                    "kind": kind,
                    "id": cite_id,
                    "author_year": author_year,
                    "title": info.get("title", ""),
                    "matched_corpus": False,
                }
            )
        else:
            unverified.append(
                {
                    "raw": raw,
                    "kind": kind,
                    "id": cite_id,
                    "author_year": author_year,
                    "reason": info.get("reason", "not found"),
                    "context_around": ctx,
                }
            )

    return {
        "verified": verified,
        "unverified": unverified,
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "total_cites": len(verified) + len(unverified),
        "verification_method": "local + live API",
    }


def _context_around(text: str, start: int, end: int) -> str:
    s = max(0, start - CONTEXT_CHARS)
    e = min(len(text), end + CONTEXT_CHARS)
    return text[s:e].replace("\n", " ")


def _verify_remote(kind: str, cite_id: str) -> tuple[bool, dict]:
    """Returns (ok, info) where info has title (if ok) or reason (if not)."""
    try:
        if kind == "arxiv":
            return _verify_arxiv(cite_id)
        if kind == "doi":
            return _verify_doi(cite_id)
        if kind == "s2":
            return _verify_s2(cite_id)
    except Exception as e:
        return False, {"reason": f"verification request failed: {e}"}
    return False, {"reason": f"unknown cite kind: {kind}"}


def _verify_arxiv(arxiv_id: str) -> tuple[bool, dict]:
    """Hit arxiv export API by id."""
    params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    url = f"{ARXIV_ABS_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            xml = resp.read().decode("utf-8")
    except Exception as e:
        return False, {"reason": f"arxiv lookup failed: {e}"}

    time.sleep(0.5)  # arxiv polite

    if "<entry>" not in xml:
        return False, {"reason": f"arxiv ID {arxiv_id} not found"}

    # crude title extract
    import re as _re

    m = _re.search(r"<title>(.*?)</title>", xml, _re.DOTALL)
    title = m.group(1).strip() if m else ""
    # arxiv returns query metadata as first <title>; the entry's <title> is second
    titles = _re.findall(r"<title>(.*?)</title>", xml, _re.DOTALL)
    if len(titles) >= 2:
        title = titles[1].strip()
    return True, {"title": " ".join(title.split())}


def _verify_doi(doi: str) -> tuple[bool, dict]:
    """Verify via crossref.org (free, no key)."""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    try:
        rsp = requests.get(url, timeout=TIMEOUT_S, headers={"User-Agent": "literature-surveyor/0.1"})
    except Exception as e:
        return False, {"reason": f"crossref request failed: {e}"}

    time.sleep(0.3)

    if rsp.status_code == 404:
        return False, {"reason": f"DOI {doi} not found in crossref"}
    if not rsp.ok:
        return False, {"reason": f"crossref returned {rsp.status_code}"}

    try:
        body = rsp.json()
        title = (body.get("message", {}).get("title") or [""])[0]
        return True, {"title": title}
    except Exception as e:
        return False, {"reason": f"crossref parse failed: {e}"}


def _verify_s2(s2_id: str) -> tuple[bool, dict]:
    """Verify via Semantic Scholar paper endpoint."""
    headers = {}
    if api_key := os.getenv("S2_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"{S2_PAPER_URL}/{s2_id}?fields=title"
    try:
        rsp = requests.get(url, headers=headers, timeout=TIMEOUT_S)
    except Exception as e:
        return False, {"reason": f"S2 request failed: {e}"}

    time.sleep(0.5 if api_key else 1.5)

    if rsp.status_code == 404:
        return False, {"reason": f"S2 ID {s2_id} not found"}
    if not rsp.ok:
        return False, {"reason": f"S2 returned {rsp.status_code}"}

    try:
        body = rsp.json()
        return True, {"title": body.get("title", "")}
    except Exception as e:
        return False, {"reason": f"S2 parse failed: {e}"}
