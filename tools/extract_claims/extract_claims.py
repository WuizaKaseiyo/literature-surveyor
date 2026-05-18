"""extract_claims — Use an LLM to extract structured claims from a paper in corpus.

Pulls paper full_text_md (or abstract fallback) from corpus, prompts a small/cheap
LLM to return a JSON array of claims, validates with Pydantic, persists to claims.jsonl.

Resolves LLM via the OMC `make_llm` API if available, falls back to OpenRouter
direct via openai client. Defaults to a cheap model (gpt-4o-mini class) since
this is structured extraction, not creative work.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

CLAIMS_FILENAME = "claims.jsonl"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
MAX_INPUT_CHARS = 30000  # truncate very long papers (v1 only)
V2_MAX_SECTIONS = 8       # per-paper cap on sections sent to LLM
V2_SECTION_CHAR_CAP = 8000
V2_MIN_SECTION_CHARS = 300


# ---------------------------------------------------------------------------
# Pydantic schema (kept local to this tool — see notes in TOOL.md)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    claim_text: str = Field(description="One-sentence factual statement from the paper")
    claim_type: str = Field(description="factual | methodological | negative_result | conjecture")
    evidence_span: str = Field(description="Section + table/figure reference, e.g. 'Section 4.2, Table 3'")
    evidence_quote: str = Field(default="", description="Short verbatim source quote supporting the claim")
    source_section: str = Field(default="", description="Section name or heading containing the evidence")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    applies_to: str = Field(
        default="", description="Scope: model size, dataset, domain — used by conflict detection"
    )


# ---------------------------------------------------------------------------
# Corpus access (mirrors corpus_store internals; kept local to be self-contained)
# ---------------------------------------------------------------------------


def _corpus_dir() -> Path:
    # A2 layered mode: papers.jsonl + claims.jsonl live in the global store,
    # so claims written here are shared across projects automatically.
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


def _load_paper(paper_id: str) -> dict | None:
    path = _corpus_dir() / "papers.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                if p.get("id") == paper_id:
                    return p
            except json.JSONDecodeError:
                continue
    return None


def _append_claims(claims: list[dict]) -> None:
    cd = _corpus_dir()
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / CLAIMS_FILENAME
    with path.open("a") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# LLM call — uses OMC make_llm if importable, else openai/openrouter direct
# ---------------------------------------------------------------------------


def _call_llm_with_retry(system: str, user: str, model: str, max_retries: int = 1) -> str:
    """Wrap _invoke_llm with a single retry on empty response.

    Only retries when the first call returned "" — that's the transient case
    (API blip, rate limit, momentary auth issue). Does not retry on malformed
    JSON — that's handled by _parse_claims_response. Backoff 0.3s between
    attempts, capped at max_retries (default 1).
    """
    raw = _invoke_llm(system, user, model)
    if raw:
        return raw
    for attempt in range(max_retries):
        time.sleep(0.3 * (attempt + 1))
        raw = _invoke_llm(system, user, model)
        if raw:
            return raw
    return ""


def _salvage_json_objects(raw: str) -> list[dict]:
    """Extract balanced {…} blocks from a string. String-aware: braces inside
    JSON strings (and their escape sequences) do not count toward depth.

    Used by _parse_claims_response as the partial-parse fallback when the
    LLM wraps its output with prose or truncates the last object.
    """
    out: list[dict] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, c in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            out.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return out


def _parse_claims_response(raw: str) -> tuple[list[dict], str]:
    """Best-effort parse of an LLM JSON response into a list of claim dicts.

    Pipeline (cheap → tolerant):
      1. Strip ```json``` fences.
      2. Strict json.loads → list, or {"claims": [...]}, or single claim dict.
      3. Salvage: extract balanced {…} blocks and parse each independently.
         Survives prose prefixes ("Here are the claims: [...]") and a
         truncated last object (max_tokens cutoff mid-array).

    Returns:
        (claims, parse_mode) — mode is one of
        empty / strict_array / strict_claims_key / strict_dict / salvaged_N / failed.
        Callers should surface salvaged_N / failed to talent-visible warnings.
    """
    raw = (raw or "").strip()
    if not raw:
        return [], "empty"
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)], "strict_array"
        if isinstance(data, dict):
            if isinstance(data.get("claims"), list):
                return [d for d in data["claims"] if isinstance(d, dict)], "strict_claims_key"
            if "claim_text" in data:
                return [data], "strict_dict"
    except json.JSONDecodeError:
        pass

    salvaged = _salvage_json_objects(raw)
    if salvaged:
        return salvaged, f"salvaged_{len(salvaged)}"
    return [], "failed"


def _call_llm_for_claims(
    paper_text: str, paper_meta: dict, model: str
) -> tuple[list[dict], str]:
    """v1 single-shot claim extraction.

    Returns (claims, parse_mode). parse_mode is forwarded to talent warnings
    when it's salvaged_* or failed so the caller knows the response needed
    rescue (or was unrescuable).
    """
    system = (
        "You are a precise academic claim extractor. Given a paper, return ONLY a JSON "
        "array of 5-15 claims. Each claim is a factual statement made by the paper "
        "(not your opinion). Each claim must have an evidence_span (section + table/figure). "
        "Return ONLY valid JSON, no commentary, no markdown fences."
    )
    user = (
        f"Paper title: {paper_meta.get('title','')}\n"
        f"Authors: {', '.join(paper_meta.get('authors',[])[:5])}\n"
        f"Year: {paper_meta.get('year','')}\n\n"
        f"Paper text (may be truncated):\n---\n{paper_text[:MAX_INPUT_CHARS]}\n---\n\n"
        "Extract 5-15 structured claims. Format strictly as a JSON array of objects with keys:\n"
        '  claim_text (string, one sentence),\n'
        '  claim_type (one of: factual, methodological, negative_result, conjecture),\n'
        '  evidence_span (string, e.g. "Section 4.2, Table 3"),\n'
        '  evidence_quote (string, one short quote from the source text that supports the claim),\n'
        '  source_section (string, section or heading where the evidence appears),\n'
        '  confidence (float 0-1, paper\'s own claimed confidence),\n'
        '  applies_to (string, scope qualifier like "models 7B-13B, English only").\n\n'
        "Return ONLY the JSON array, nothing else."
    )

    raw = _call_llm_with_retry(system, user, model)
    return _parse_claims_response(raw)


# ---------------------------------------------------------------------------
# v2 helpers — section split, quote round-trip verify, per-section LLM call
# ---------------------------------------------------------------------------


_HEAD_PAT = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
_SKIP_HEADING_PAT = re.compile(
    r'\b(references|bibliography|acknowledg|appendix|supplementary|'
    r'figure\s+\d|table\s+\d|author\s+contribution)\b',
    re.IGNORECASE,
)
_CLAIM_KEYWORD_PAT = re.compile(
    r'\b(method|methods|approach|model|models|architecture|algorithm|'
    r'experiment|experiments|result|results|finding|findings|evaluation|'
    r'ablation|ablations|analysis|discussion|conclusion|conclusions|'
    r'limitation|limitations)\b',
    re.IGNORECASE,
)


def _split_sections(text: str, max_level: int = 3) -> list[dict[str, Any]]:
    """Split a markdown document into sections by ATX headings (#…###).

    Returns [{heading, level, text}, …] in document order. Empty list if no
    qualifying headings were found.
    """
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    cur_heading = ""
    cur_level = 0
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_heading and cur_lines:
            body = "\n".join(cur_lines).strip()
            if body:
                sections.append({"heading": cur_heading, "level": cur_level, "text": body})

    for line in lines:
        m = _HEAD_PAT.match(line)
        if m and len(m.group(1)) <= max_level:
            flush()
            cur_heading = m.group(2).strip()
            cur_level = len(m.group(1))
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()
    return sections


def _section_relevant(sec: dict[str, Any]) -> bool:
    heading = sec.get("heading", "") or ""
    if _SKIP_HEADING_PAT.search(heading):
        return False
    if len(sec.get("text", "")) < V2_MIN_SECTION_CHARS:
        return False
    return True


def _rank_and_cap_sections(
    sections: list[dict[str, Any]], max_sections: int = V2_MAX_SECTIONS
) -> list[dict[str, Any]]:
    """Keep at most `max_sections`, prioritizing claim-bearing headings."""
    def score(sec: dict[str, Any]) -> int:
        s = 0
        if _CLAIM_KEYWORD_PAT.search(sec.get("heading", "") or ""):
            s += 100
        # Length contributes up to 50 points (caps at ~50K chars).
        s += min(len(sec.get("text", "")) // 1000, 50)
        return s

    return sorted(sections, key=score, reverse=True)[:max_sections]


def _normalize_for_match(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s%.-]", "", t)
    return t.strip()


def _quote_in_source(quote: str, source: str) -> bool:
    """True if a 60%-of-quote-length contiguous run of quote appears verbatim in source.

    Same algorithm as `tests/eval/compare_extraction.quote_verified` — kept local
    so this tool stays self-contained (OMC copies each tool dir to assets/).
    """
    if not quote or len(quote) < 10:
        return False
    nq = _normalize_for_match(quote)
    nt = _normalize_for_match(source)
    if not nq or not nt:
        return False
    if nq in nt:
        return True
    window = min(60, max(20, int(len(nq) * 0.6)))
    if window > len(nq):
        return False
    step = max(1, window // 4)
    for i in range(0, len(nq) - window + 1, step):
        if nq[i : i + window] in nt:
            return True
    return False


def _call_llm_for_claims_section(
    section_text: str, section_heading: str, paper_meta: dict, model: str
) -> tuple[list[dict], str]:
    """Per-section LLM call. Aims for 0-4 claims, returns ([], mode) if the
    section has none. Same shape as _call_llm_for_claims so per-section
    parse_mode bubbles up to talent warnings."""
    system = (
        "You extract structured claims from ONE section of an academic paper. "
        "Return ONLY a JSON array of 0-4 claims. Each evidence_quote MUST be a "
        "VERBATIM substring of the section text (no rewording, no paraphrase). "
        "Return [] if the section has no concrete claims."
    )
    user = (
        f"Paper: {paper_meta.get('title', '')}\n"
        f"Section: {section_heading}\n\n"
        f"Section text:\n---\n{section_text[:V2_SECTION_CHAR_CAP]}\n---\n\n"
        "Each claim object MUST have these keys:\n"
        '  claim_text (string, 1 sentence),\n'
        '  claim_type (one of: factual / methodological / negative_result / conjecture),\n'
        '  evidence_span (string, e.g. "Table 3" or "Eq. 4" — or the section heading if no specific anchor),\n'
        '  evidence_quote (VERBATIM excerpt from the section text — must be findable by substring search),\n'
        '  applies_to_dims (object — fill any of: model_size, dataset, domain, language, regime, metric — omit keys not stated in the section),\n'
        '  confidence (float 0-1 — paper\'s own claimed confidence).\n\n'
        "Return ONLY the JSON array, no commentary, no markdown fences."
    )

    raw = _call_llm_with_retry(system, user, model)
    return _parse_claims_response(raw)


def _extract_v2(paper: dict, model: str) -> tuple[list[dict], list[str]]:
    """Section-aware extraction. Returns (raw_claims_with_section_info, warnings)."""
    full_text = paper.get("full_text_md") or ""
    warnings: list[str] = []

    if full_text:
        sections = _split_sections(full_text, max_level=3)
        if not sections:
            warnings.append("no markdown headings detected — falling back to single-section mode")
            sections = [{"heading": "(unstructured)", "level": 1, "text": full_text[:60000]}]
    else:
        abstract = paper.get("abstract") or ""
        if len(abstract) < 200:
            return [], ["paper has no full_text_md and abstract too short for v2"]
        warnings.append("paper has no full_text_md — extracting from abstract only")
        sections = [{"heading": "Abstract", "level": 1, "text": abstract}]

    relevant = [s for s in sections if _section_relevant(s)]
    if not relevant:
        relevant = sections[:1]
    selected = _rank_and_cap_sections(relevant, max_sections=V2_MAX_SECTIONS)

    quote_source = full_text or paper.get("abstract", "")
    all_raw: list[dict] = []
    for sec in selected:
        raw_claims, parse_mode = _call_llm_for_claims_section(
            sec["text"], sec["heading"], paper, model
        )
        if parse_mode.startswith("salvaged_"):
            warnings.append(
                f"section '{sec['heading']}': LLM response malformed, "
                f"rescued via partial parse ({parse_mode})"
            )
        elif parse_mode == "failed":
            warnings.append(
                f"section '{sec['heading']}': LLM response unparseable, 0 claims this section"
            )
        elif parse_mode == "empty":
            warnings.append(
                f"section '{sec['heading']}': empty LLM response after retry"
            )
        for c in raw_claims:
            quote = c.get("evidence_quote", "")
            verified = _quote_in_source(quote, quote_source)
            c["evidence_quote_verified"] = verified
            c["section_path"] = [sec["heading"]]
            c["source_section"] = sec["heading"]
            if not verified and quote:
                warnings.append(
                    f"unverified quote in '{sec['heading']}': {quote[:60]!r}"
                )
            all_raw.append(c)
    return all_raw, warnings


def _invoke_llm(system: str, user: str, model: str) -> str:
    """Try multiple LLM clients in order of preference."""

    # Option 1: OMC's make_llm (if running inside OMC and importable)
    try:
        from onemancompany.agents.base import make_llm  # type: ignore

        llm = make_llm("00007")  # any researcher id; we just want the LLM client
        result = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return getattr(result, "content", str(result))
    except Exception:
        pass

    # Option 2: openai client (works for OpenAI direct + OpenRouter)
    try:
        from openai import OpenAI  # type: ignore

        api_key = (
            os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        )
        if not api_key:
            return ""
        base_url = (
            "https://openrouter.ai/api/v1"
            if os.getenv("OPENROUTER_API_KEY")
            else None
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        rsp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return rsp.choices[0].message.content or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# @tool
# ---------------------------------------------------------------------------


def _load_existing_claims_for(paper_id: str) -> list[dict]:
    """Return claims previously persisted for a paper, empty if none."""
    path = _corpus_dir() / CLAIMS_FILENAME
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                if c.get("paper_id") == paper_id:
                    out.append(c)
            except json.JSONDecodeError:
                continue
    return out


@tool
def extract_claims(
    paper_id: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
    version: str = "v1",
    force: bool = False,
) -> dict[str, Any]:
    """Extract 5-15 structured claims from a paper in the corpus and persist them.

    Args:
        paper_id: ID of paper already added via corpus_add_paper.
        model: LLM model to use. Default openai/gpt-4o-mini (cheap structured extraction).
        version: Extraction strategy version. "v1" (current) = single-shot on first
            30K chars of full text. "v2" = section-aware split + per-section extraction
            + evidence_quote ground-truth check + structured applies_to_dims.
            See OPTIMIZATION.md section C for v2 design.
        force: If False (default), short-circuit when claims for this paper_id
            already exist (saves LLM calls + avoids duplicate rows). In A2
            layered mode this is what enables cross-project claim reuse —
            project B importing a paper that project A already extracted gets
            the existing claims for free. Set True to re-extract anyway.

    Returns:
        {
          "paper_id": "...",
          "version": "v1" | "v2",
          "status": "extracted" | "reused" | "error",
          "claims_extracted": N,
          "claims": [{"id": "claim-...", "claim_text": "...", ...}, ...],
          "warnings": [...]
        }
    """
    if not paper_id:
        return {"error": "paper_id required"}

    if version not in {"v1", "v2"}:
        return {
            "error": f"unknown extraction version {version!r}, expected 'v1' or 'v2'",
            "paper_id": paper_id,
        }

    # A2 cross-project reuse short-circuit: if claims already exist for this
    # paper, return them without burning an LLM call (the main reason layered
    # corpus mode pays off).
    if not force:
        existing = _load_existing_claims_for(paper_id)
        if existing:
            return {
                "paper_id": paper_id,
                "version": existing[0].get("version", "unknown"),
                "status": "reused",
                "claims_extracted": len(existing),
                "claims": existing,
                "warnings": [
                    f"reused {len(existing)} previously extracted claims; "
                    "pass force=True to re-extract"
                ],
            }

    paper = _load_paper(paper_id)
    if not paper:
        return {"error": f"paper not found in corpus: {paper_id}"}

    if version == "v2":
        raw_claims, warnings = _extract_v2(paper, model)
        if not raw_claims:
            return {
                "error": "v2 extraction returned no claims",
                "paper_id": paper_id,
                "version": "v2",
                "claims_extracted": 0,
                "claims": [],
                "warnings": warnings,
            }
        validated: list[dict] = []
        for i, raw in enumerate(raw_claims):
            try:
                dims = raw.get("applies_to_dims") or {}
                # Backfill legacy free-form applies_to from structured dims.
                applies_to_legacy = ", ".join(
                    f"{k}={v}" for k, v in dims.items() if v
                )
                c = Claim(
                    claim_text=raw.get("claim_text", ""),
                    claim_type=raw.get("claim_type", "factual"),
                    evidence_span=raw.get("evidence_span", ""),
                    evidence_quote=raw.get("evidence_quote", ""),
                    source_section=raw.get("source_section", ""),
                    confidence=raw.get("confidence", 0.5),
                    applies_to=applies_to_legacy,
                )
                validated.append(
                    {
                        "id": f"{paper_id}#claim-{i+1}",
                        "paper_id": paper_id,
                        "claim_text": c.claim_text,
                        "claim_type": c.claim_type,
                        "evidence_span": c.evidence_span,
                        "evidence_quote": c.evidence_quote,
                        "evidence_quote_verified": bool(raw.get("evidence_quote_verified", False)),
                        "source_section": c.source_section,
                        "section_path": raw.get("section_path", []),
                        "confidence": c.confidence,
                        "applies_to": c.applies_to,
                        "applies_to_dims": dims,
                        "extracted_at": time.time(),
                        "model": model,
                        "version": "v2",
                    }
                )
            except ValidationError as e:
                warnings.append(f"claim {i} invalid: {e.errors()[:1]}")

        if validated:
            _append_claims(validated)
        return {
            "paper_id": paper_id,
            "version": "v2",
            "status": "extracted",
            "claims_extracted": len(validated),
            "claims": validated,
            "warnings": warnings,
        }

    text = paper.get("full_text_md") or paper.get("abstract") or ""
    if len(text) < 200:
        return {
            "error": f"paper text too short ({len(text)} chars) — need full_text_md or longer abstract",
            "paper_id": paper_id,
        }

    raw_claims, v1_parse_mode = _call_llm_for_claims(text, paper, model)
    if not raw_claims:
        return {
            "error": f"LLM returned no parseable claims ({v1_parse_mode})",
            "paper_id": paper_id,
            "version": "v1",
            "claims_extracted": 0,
            "claims": [],
        }

    validated: list[dict] = []
    warnings: list[str] = []
    if v1_parse_mode.startswith("salvaged_"):
        warnings.append(
            f"v1 LLM response malformed; rescued via partial parse ({v1_parse_mode})"
        )
    for i, raw in enumerate(raw_claims):
        try:
            c = Claim(**raw)
            validated.append(
                {
                    "id": f"{paper_id}#claim-{i+1}",
                    "paper_id": paper_id,
                    "claim_text": c.claim_text,
                    "claim_type": c.claim_type,
                    "evidence_span": c.evidence_span,
                    "evidence_quote": c.evidence_quote,
                    "source_section": c.source_section,
                    "confidence": c.confidence,
                    "applies_to": c.applies_to,
                    "extracted_at": time.time(),
                    "model": model,
                    "version": "v1",
                }
            )
        except ValidationError as e:
            warnings.append(f"claim {i} invalid: {e.errors()[:1]}")

    if validated:
        _append_claims(validated)

    return {
        "paper_id": paper_id,
        "version": "v1",
        "status": "extracted",
        "claims_extracted": len(validated),
        "claims": validated,
        "warnings": warnings,
    }
