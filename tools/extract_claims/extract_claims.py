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

# v3 — meta_pass + grounded per-section + global rerank + deterministic hard filter
V3_META_ABSTRACT_CHARS = 4000
V3_META_INTRO_CHARS = 2000
V3_META_CONCLUSION_CHARS = 1500
V3_MAX_SECTIONS = 8
V3_SECTION_CLAIM_BUDGET = 3       # was 4 in v2
V3_RERANK_KEEP_TOPK = 8            # hard cap per paper after rerank
V3_NOISE_SALIENCY = frozenset({
    "background", "methodology_footnote", "self_promotion", "cited_other_work",
})


# ---------------------------------------------------------------------------
# Pydantic schema (kept local to this tool — see notes in TOOL.md)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    claim_text: str = Field(description="One-sentence factual statement from the paper")
    claim_type: str = Field(description="factual | methodological | negative_result | conjecture")
    saliency_type: str = Field(
        default="empirical_finding",
        description=(
            "Saliency for survey synthesis. One of: empirical_finding, evaluation_result, "
            "method_proposed, limitation (substantive — surfaced to finding synthesis); "
            "background, methodology_footnote, self_promotion, cited_other_work (noise — "
            "kept in store but filtered from finding digest)."
        ),
    )
    evidence_span: str = Field(description="Section + table/figure reference, e.g. 'Section 4.2, Table 3'")
    evidence_quote: str = Field(default="", description="Short verbatim source quote supporting the claim")
    source_section: str = Field(default="", description="Section name or heading containing the evidence")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    applies_to: str = Field(
        default="", description="Scope: model size, dataset, domain — used by conflict detection"
    )
    contribution_idx: int = Field(
        default=-1,
        description=(
            "v3 only. 0-based index into meta_pass.contributions that this claim grounds. "
            "-1 = local content (limitation, method footnote) with no contribution anchor."
        ),
    )
    dedup_rank: int | None = Field(
        default=None,
        description=(
            "v3 only. 0-based rank after global rerank (lower = more important). "
            "None on pre-rerank claims; dropped claims are not persisted."
        ),
    )


# ---------------------------------------------------------------------------
# Corpus access (mirrors corpus_store internals; kept local to be self-contained)
# ---------------------------------------------------------------------------


def _corpus_dir() -> Path:
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


def _replace_claims_for(paper_id: str, new_claims: list[dict]) -> int:
    """Atomically swap a paper's claims for `new_claims`.

    Used on force=True re-extraction: a plain append leaves the prior rows
    behind, producing duplicate `#claim-N` ids and stale text downstream.
    Rewrites claims.jsonl via tmp + rename so readers never see a torn file.
    Returns the number of prior rows replaced.
    """
    cd = _corpus_dir()
    cd.mkdir(parents=True, exist_ok=True)
    path = cd / CLAIMS_FILENAME
    kept: list[dict] = []
    removed = 0
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if c.get("paper_id") == paper_id:
                    removed += 1
                else:
                    kept.append(c)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
        for c in new_claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return removed


# ---------------------------------------------------------------------------
# LLM call — uses OMC make_llm if importable, else openai/openrouter direct
# ---------------------------------------------------------------------------


def _call_llm_with_retry(system: str, user: str, model: str, max_retries: int = 1) -> str:
    """Retry _invoke_llm once on empty response (transient API blip).

    Malformed JSON is handled by _parse_claims_response, not retried here."""
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


_V3_RESULTS_PAT = re.compile(
    r"\b(result|results|finding|findings|evaluation|ablation|ablations|"
    r"experiment|experiments|limitation|limitations)\b",
    re.IGNORECASE,
)
_V3_RESTATER_PAT = re.compile(
    r"\b(discussion|conclusion|conclusions|summary)\b",
    re.IGNORECASE,
)


def _rank_and_cap_sections(
    sections: list[dict[str, Any]],
    max_sections: int = V2_MAX_SECTIONS,
    version: str = "v2",
) -> list[dict[str, Any]]:
    """Keep at most `max_sections`, prioritizing claim-bearing headings.

    v2 scoring: flat +100 for any claim keyword.
    v3 scoring: Results/Ablation/Experiment/Limitations +150 (where novel
    findings live); Method/Approach +100; Discussion/Conclusion demoted to
    +50 because they typically restate Results rather than introduce new
    claims — and v2's flat weighting caused those restated findings to be
    extracted a second time, inflating the per-paper count.
    """
    def score_v2(sec: dict[str, Any]) -> int:
        s = 0
        if _CLAIM_KEYWORD_PAT.search(sec.get("heading", "") or ""):
            s += 100
        s += min(len(sec.get("text", "")) // 1000, 50)
        return s

    def score_v3(sec: dict[str, Any]) -> int:
        heading = sec.get("heading", "") or ""
        s = 0
        if _V3_RESULTS_PAT.search(heading):
            s += 150
        elif _V3_RESTATER_PAT.search(heading):
            s += 50
        elif _CLAIM_KEYWORD_PAT.search(heading):
            s += 100
        s += min(len(sec.get("text", "")) // 1000, 50)
        return s

    scorer = score_v3 if version == "v3" else score_v2
    return sorted(sections, key=scorer, reverse=True)[:max_sections]


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
        "Return [] if the section has no concrete claims.\n\n"
        "CRITICAL: if the section heading is 'Related Work', 'Background', "
        "'Prior Work', 'Literature Review', or the section text paraphrases "
        "ANOTHER paper's contribution (sentences like 'Smith et al. proposed X', "
        "'Prior work in Y shows...', 'X et al. introduced...', 'RAFT enhances Y'), "
        "DO NOT treat those sentences as the current paper's claims. Either return "
        "an empty array, OR if you must extract, set saliency_type='cited_other_work' "
        "and make the evidence_quote the verbatim paraphrase. Never list another "
        "paper's contribution as this paper's empirical_finding or method_proposed."
    )
    user = (
        f"Paper: {paper_meta.get('title', '')}\n"
        f"Section: {section_heading}\n\n"
        f"Section text:\n---\n{section_text[:V2_SECTION_CHAR_CAP]}\n---\n\n"
        "Each claim object MUST have these keys:\n"
        '  claim_text (string, 1 sentence),\n'
        '  claim_type (one of: factual / methodological / negative_result / conjecture),\n'
        '  saliency_type (one of the 8 below — be strict, this decides whether the claim is shown to survey synthesis):\n'
        '    - empirical_finding: substantive empirical result with concrete details (numbers, comparisons, observed phenomena)\n'
        '    - evaluation_result: benchmark/metric scores, accuracy, win-rates with specific numbers\n'
        '    - method_proposed: describes a specific technique THIS paper introduces (not just labels it)\n'
        '    - limitation: author-acknowledged shortcoming or scope limit of THIS paper\n'
        '    - background: textbook context, generic motivation, field history, definitions ("RAG is...", "Since 2022, LLMs...")\n'
        '    - methodology_footnote: implementation detail, hyperparameter, scoring formula, annotation procedure note\n'
        '    - self_promotion: author marketing language ("our method addresses critical gaps", "we expect widespread adoption", "this approach is promising")\n'
        '    - cited_other_work: paraphrases ANOTHER paper\'s contribution ("Smith et al. proposed X", "RAFT enhances Y", "Prior work in Z shows...")\n'
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


# ---------------------------------------------------------------------------
# v3 — meta_pass + contribution-grounded per-section + global rerank
# ---------------------------------------------------------------------------


def _meta_pass(paper: dict, model: str) -> tuple[dict[str, Any], str]:
    """Single LLM call to extract this paper's self-stated contributions + scope.

    Used by v3 as an anchor for downstream per-section extraction: each claim
    must ground to one of these contributions (or be marked local).

    Returns:
        (meta, mode) where meta = {"contributions": [...], "scope": {...}}
        and mode ∈ {"ok", "fallback_no_text", "fallback_parse_failed",
                    "fallback_llm_empty"}. On any fallback, meta has empty
        lists/dicts and caller falls back to v2-style extraction.
    """
    abstract = (paper.get("abstract") or "")[:V3_META_ABSTRACT_CHARS]
    full_text = paper.get("full_text_md") or ""

    # Try to grab intro head and conclusion head deterministically.
    intro_text = ""
    conclusion_text = ""
    if full_text:
        sections = _split_sections(full_text, max_level=3)
        for s in sections:
            h = (s.get("heading") or "").lower()
            if not intro_text and re.search(r"\bintroduction\b|\bintro\b", h):
                intro_text = s["text"][:V3_META_INTRO_CHARS]
            if not conclusion_text and re.search(r"\bconclusion\b|\bsummary\b", h):
                conclusion_text = s["text"][:V3_META_CONCLUSION_CHARS]
            if intro_text and conclusion_text:
                break

    if not abstract and not intro_text:
        return {"contributions": [], "scope": {}}, "fallback_no_text"

    system = (
        "You read an academic paper's abstract, introduction head, and conclusion "
        "head, and return a JSON object describing what THIS paper claims to "
        "contribute and the scope it operates in. You are NOT extracting claims "
        "yet — only the paper's self-described contributions and the experimental "
        "scope.\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary."
    )
    user = (
        f"Paper: {paper.get('title', '')}\n"
        f"Year: {paper.get('year', '')}\n\n"
        f"Abstract:\n{abstract}\n\n"
        + (f"Introduction head:\n{intro_text}\n\n" if intro_text else "")
        + (f"Conclusion head:\n{conclusion_text}\n\n" if conclusion_text else "")
        + "Return JSON of exactly this shape:\n"
        '{\n'
        '  "contributions": [ "<short noun phrase, 1 line each, 2-5 items>", ... ],\n'
        '  "scope": {\n'
        '    "model_size":  "<e.g. 7B-13B, omit if not stated>",\n'
        '    "dataset":     "<e.g. MMLU, BBH, omit if not stated>",\n'
        '    "domain":      "<e.g. code, math, dialog>",\n'
        '    "language":    "<e.g. English, multilingual>",\n'
        '    "regime":      "<e.g. zero-shot, fine-tuning>",\n'
        '    "metric":      "<e.g. accuracy, win-rate>"\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- contributions must be the paper's own stated contributions (look for "
        '"we propose", "we contribute", "our main contribution"). Do NOT include '
        "prior-work descriptions.\n"
        "- omit scope keys that the paper does not state. Empty string is fine.\n"
        "- 2-5 contributions, each one short phrase, not a sentence."
    )

    raw = _call_llm_with_retry(system, user, model)
    if not raw:
        return {"contributions": [], "scope": {}}, "fallback_llm_empty"

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        salvaged = _salvage_json_objects(raw)
        data = salvaged[0] if salvaged else None

    if not isinstance(data, dict):
        return {"contributions": [], "scope": {}}, "fallback_parse_failed"

    contribs = data.get("contributions") or []
    if not isinstance(contribs, list):
        contribs = []
    contribs = [str(c).strip() for c in contribs if str(c).strip()][:5]

    scope = data.get("scope") or {}
    if not isinstance(scope, dict):
        scope = {}
    scope = {k: str(v).strip() for k, v in scope.items() if str(v or "").strip()}

    return {"contributions": contribs, "scope": scope}, "ok"


def _call_llm_for_claims_section_v3(
    section_text: str,
    section_heading: str,
    paper_meta: dict,
    contributions: list[str],
    scope_hint: dict[str, str],
    model: str,
) -> tuple[list[dict], str]:
    """Per-section claim extraction grounded to meta_pass contributions.

    Each claim must either (a) ground to one of `contributions` (claim.contribution_idx
    points to the 0-based index), or (b) be a local limitation/method footnote
    with contribution_idx=-1. Anti-examples in the prompt teach the LLM to
    skip background, cited_other_work, and self-promotion sentences.
    """
    contrib_block = (
        "\n".join(f"  [{i}] {c}" for i, c in enumerate(contributions))
        if contributions else "  (none provided — use contribution_idx=-1)"
    )
    scope_block = (
        "\n".join(f"  {k}: {v}" for k, v in scope_hint.items())
        if scope_hint else "  (none)"
    )

    system = (
        "You extract structured claims from ONE section of an academic paper. "
        "You are given THIS paper's self-stated contributions; every claim you "
        "extract must either ground to one of those contributions, OR be a "
        "limitation / methodology footnote local to this paper.\n\n"
        "Return ONLY a JSON array of 0-3 claims. Return [] if the section has "
        "no concrete claims that meet the bar below.\n\n"
        "QUALITY BAR (be strict — reject more than you keep):\n"
        "- INCLUDE: empirical findings with numbers/comparisons, evaluation "
        "results with specific metrics, the paper's own proposed method when "
        "named with concrete detail, author-acknowledged limitations.\n"
        "- EXCLUDE: textbook background ('RAG is...', 'Since 2022, LLMs...'), "
        "field history, generic motivation, marketing language ('our approach "
        "is promising', 'we expect widespread adoption'), paraphrased prior "
        "work ('Smith et al. proposed X', 'RAFT enhances Y'), and any claim "
        "that simply restates a contribution without new evidence.\n\n"
        "GOOD claim: 'LoRA at rank=8 matches full fine-tuning on GLUE within "
        "0.3 points across 5 tasks.' (concrete number, scope, comparison)\n"
        "BAD claim: 'Our method addresses key challenges in the field.' "
        "(self-promotion, no evidence)\n"
        "BAD claim: 'Prior work has shown that attention is important.' "
        "(cited_other_work)\n"
        "BAD claim: 'We propose a novel approach.' (restates contribution, "
        "no evidence)\n\n"
        "Each evidence_quote MUST be a VERBATIM substring of the section text."
    )
    user = (
        f"Paper: {paper_meta.get('title', '')}\n"
        f"Section: {section_heading}\n\n"
        f"This paper's contributions (claims should ground to one of these):\n"
        f"{contrib_block}\n\n"
        f"Inherited scope (carry into applies_to_dims unless the section overrides):\n"
        f"{scope_block}\n\n"
        f"Section text:\n---\n{section_text[:V2_SECTION_CHAR_CAP]}\n---\n\n"
        "Each claim object MUST have these keys:\n"
        '  claim_text (string, one sentence, must contain concrete detail),\n'
        '  claim_type (one of: factual / methodological / negative_result / conjecture),\n'
        '  saliency_type (one of: empirical_finding / evaluation_result / '
        'method_proposed / limitation — do NOT use background / methodology_footnote '
        '/ self_promotion / cited_other_work; if a candidate would fall in those, '
        'omit it instead),\n'
        '  contribution_idx (integer — 0-based index into the contributions list above, '
        'or -1 if this is a limitation or methodology note local to this paper),\n'
        '  evidence_span (string, e.g. "Table 3" or "Eq. 4"),\n'
        '  evidence_quote (VERBATIM substring of the section text — must be findable),\n'
        '  applies_to_dims (object — fill any of: model_size, dataset, domain, language, '
        'regime, metric; inherit from the scope above for any key the section does not '
        'override),\n'
        '  confidence (float 0-1, paper\'s own claimed confidence).\n\n'
        "Return ONLY the JSON array, no commentary, no markdown fences."
    )

    raw = _call_llm_with_retry(system, user, model)
    return _parse_claims_response(raw)


def _global_rerank(
    candidates: list[dict], paper_meta: dict, model: str, top_k: int = V3_RERANK_KEEP_TOPK
) -> tuple[list[str], dict[str, str], str]:
    """Cross-section dedup + importance rerank via a single LLM call.

    Returns:
        (keep_ids, drop_reasons, mode)
        - keep_ids: list of cand-N ids in priority order (best first), up to top_k
        - drop_reasons: { "cand-N": "duplicate of cand-M" | "background" | ... }
        - mode ∈ {"ok", "fallback_empty", "fallback_parse_failed"}
        On fallback the caller is expected to use deterministic _hard_filter
        and trim to top_k by first-seen order.
    """
    if not candidates:
        return [], {}, "ok"

    # Build a compact view for the LLM — strip evidence_quote to save tokens;
    # keep claim_text, saliency, section, contribution_idx.
    lines: list[str] = []
    for i, c in enumerate(candidates):
        cid = f"cand-{i}"
        sal = c.get("saliency_type", "?")
        sec = c.get("source_section", "?")
        cidx = c.get("contribution_idx", -1)
        txt = (c.get("claim_text") or "").replace("\n", " ").strip()[:240]
        lines.append(f"  {cid} | saliency={sal} | sec={sec} | contrib={cidx} | {txt}")
    cand_block = "\n".join(lines)

    system = (
        "You are a literature-survey editor selecting the most informative, "
        "non-redundant claims from a paper. You will be given a numbered list of "
        f"candidate claims; pick at most {top_k} to keep, ordered by importance, "
        "and explain why each dropped claim was dropped.\n\n"
        "Drop rules (in order):\n"
        "1. background / methodology_footnote / self_promotion / cited_other_work "
        "saliency — drop unconditionally.\n"
        "2. duplicates: two claims stating the same finding from different sections "
        "— keep the one with clearer numbers/scope, drop the other(s).\n"
        "3. vague claims without quantitative or scoped detail — drop unless "
        "saliency=limitation or method_proposed.\n"
        "4. if more than top_k remain, drop the least informative until at most "
        f"{top_k} are kept.\n\n"
        "Return ONLY valid JSON, no markdown fences."
    )
    user = (
        f"Paper: {paper_meta.get('title', '')}\n\n"
        f"Candidate claims ({len(candidates)}):\n{cand_block}\n\n"
        f"Return JSON of exactly this shape:\n"
        '{\n'
        '  "keep": ["cand-3", "cand-7", ...],  // ordered, most important first, '
        f'up to {top_k} entries\n'
        '  "drop_reasons": { "cand-2": "duplicate of cand-3", '
        '"cand-5": "background", ... }\n'
        "}"
    )

    raw = _call_llm_with_retry(system, user, model)
    if not raw:
        return [], {}, "fallback_empty"
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        salvaged = _salvage_json_objects(raw)
        data = salvaged[0] if salvaged else None

    if not isinstance(data, dict):
        return [], {}, "fallback_parse_failed"

    keep_raw = data.get("keep") or []
    keep_ids = [str(x) for x in keep_raw if isinstance(x, (str, int))][:top_k]
    # Re-stringify ints in case the LLM returned bare integers.
    keep_ids = [k if k.startswith("cand-") else f"cand-{k}" for k in keep_ids]

    drop_raw = data.get("drop_reasons") or {}
    drop_reasons: dict[str, str] = {}
    if isinstance(drop_raw, dict):
        for k, v in drop_raw.items():
            kk = str(k)
            if not kk.startswith("cand-"):
                kk = f"cand-{kk}"
            drop_reasons[kk] = str(v)[:160]

    valid = {f"cand-{i}" for i in range(len(candidates))}
    keep_ids = [k for k in keep_ids if k in valid]

    return keep_ids, drop_reasons, "ok"


def _hard_filter(candidate: dict) -> tuple[bool, str]:
    """Deterministic out-of-LLM gate. Returns (keep, drop_reason).

    Applied after rerank (to catch what rerank missed) AND as the only filter
    when rerank itself falls back.
    """
    sal = candidate.get("saliency_type") or "empirical_finding"
    if sal in V3_NOISE_SALIENCY:
        return False, f"noise_saliency:{sal}"

    if candidate.get("evidence_quote") and not candidate.get("evidence_quote_verified", False):
        return False, "unverified_quote"

    # Factual claim with no scope at all is too vague to be useful for synthesis.
    ctype = candidate.get("claim_type", "factual")
    dims = candidate.get("applies_to_dims") or {}
    if ctype == "factual" and sal not in ("limitation", "method_proposed"):
        dims_nonempty = isinstance(dims, dict) and any(v for v in dims.values())
        if not dims_nonempty:
            return False, "factual_no_scope"

    return True, ""


def _extract_v3(paper: dict, model: str) -> tuple[list[dict], list[str], dict[str, Any]]:
    """v3 extraction: meta_pass → grounded per-section → global rerank → hard filter.

    Returns:
        (kept_claims, warnings, meta) where
        - kept_claims: list of raw claim dicts (no DB id yet) ordered by dedup_rank,
          each already carrying section_path / source_section / contribution_idx /
          dedup_rank / evidence_quote_verified.
        - warnings: human-readable warning lines for the caller to surface.
        - meta: { "contributions": [...], "scope": {...},
                  "meta_mode": ..., "rerank_mode": ...,
                  "n_candidates": N, "n_dropped_rerank": M,
                  "n_dropped_hard": K }
    """
    warnings: list[str] = []
    meta_info: dict[str, Any] = {
        "contributions": [], "scope": {}, "meta_mode": "skipped",
        "rerank_mode": "skipped",
        "n_candidates": 0, "n_dropped_rerank": 0, "n_dropped_hard": 0,
    }

    # ---- Step 0: meta_pass --------------------------------------------------
    meta, meta_mode = _meta_pass(paper, model)
    meta_info["contributions"] = meta["contributions"]
    meta_info["scope"] = meta["scope"]
    meta_info["meta_mode"] = meta_mode
    if meta_mode != "ok":
        warnings.append(
            f"meta_pass fell back ({meta_mode}) — per-section claims will not have "
            "contribution grounding; v3 will still run but quality may drop"
        )

    # ---- Step 1: section split + v3-weighted ranking ------------------------
    full_text = paper.get("full_text_md") or ""
    if full_text:
        sections = _split_sections(full_text, max_level=3)
        if not sections:
            warnings.append("no markdown headings detected — falling back to single-section mode")
            sections = [{"heading": "(unstructured)", "level": 1, "text": full_text[:60000]}]
    else:
        abstract = paper.get("abstract") or ""
        if len(abstract) < 200:
            return [], ["paper has no full_text_md and abstract too short for v3"], meta_info
        warnings.append("paper has no full_text_md — extracting from abstract only")
        sections = [{"heading": "Abstract", "level": 1, "text": abstract}]

    relevant = [s for s in sections if _section_relevant(s)]
    if not relevant:
        relevant = sections[:1]
    selected = _rank_and_cap_sections(
        relevant, max_sections=V3_MAX_SECTIONS, version="v3"
    )

    # ---- Step 2: per-section grounded extraction ----------------------------
    quote_source = full_text or paper.get("abstract", "")
    candidates: list[dict] = []
    for sec in selected:
        raw_claims, parse_mode = _call_llm_for_claims_section_v3(
            sec["text"], sec["heading"], paper,
            meta["contributions"], meta["scope"], model,
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

        # Cap at V3_SECTION_CLAIM_BUDGET defensively even if LLM ignored the
        # prompt instruction.
        for c in raw_claims[:V3_SECTION_CLAIM_BUDGET]:
            quote = c.get("evidence_quote", "")
            verified = _quote_in_source(quote, quote_source)
            c["evidence_quote_verified"] = verified
            c["section_path"] = [sec["heading"]]
            c["source_section"] = sec["heading"]
            if not verified and quote:
                warnings.append(
                    f"unverified quote in '{sec['heading']}': {quote[:60]!r}"
                )
            # Inherit scope from meta if section omitted applies_to_dims keys.
            dims = c.get("applies_to_dims") or {}
            if isinstance(dims, dict):
                for k, v in meta["scope"].items():
                    if v and not dims.get(k):
                        dims[k] = v
                c["applies_to_dims"] = dims
            candidates.append(c)

    meta_info["n_candidates"] = len(candidates)
    if not candidates:
        return [], warnings + ["v3: no candidates produced by per-section pass"], meta_info

    # ---- Step 3: global rerank ----------------------------------------------
    keep_ids, drop_reasons, rerank_mode = _global_rerank(
        candidates, paper, model, top_k=V3_RERANK_KEEP_TOPK
    )
    meta_info["rerank_mode"] = rerank_mode

    if rerank_mode == "ok" and keep_ids:
        keep_index: dict[str, int] = {cid: rank for rank, cid in enumerate(keep_ids)}
        kept_pre_filter: list[dict] = []
        for i, c in enumerate(candidates):
            cid = f"cand-{i}"
            if cid in keep_index:
                c["dedup_rank"] = keep_index[cid]
                kept_pre_filter.append(c)
            else:
                reason = drop_reasons.get(cid, "not_selected_by_rerank")
                warnings.append(f"rerank dropped {cid}: {reason}")
        meta_info["n_dropped_rerank"] = len(candidates) - len(kept_pre_filter)
    else:
        # Fallback: keep order, trim to top_k deterministically; hard_filter
        # below will still do its job.
        warnings.append(
            f"rerank fell back ({rerank_mode}) — keeping first {V3_RERANK_KEEP_TOPK} "
            "candidates in document order and relying on hard_filter only"
        )
        kept_pre_filter = []
        for i, c in enumerate(candidates[:V3_RERANK_KEEP_TOPK]):
            c["dedup_rank"] = i
            kept_pre_filter.append(c)
        meta_info["n_dropped_rerank"] = max(0, len(candidates) - len(kept_pre_filter))

    # ---- Step 4: deterministic hard filter ----------------------------------
    kept: list[dict] = []
    for c in kept_pre_filter:
        keep, reason = _hard_filter(c)
        if keep:
            kept.append(c)
        else:
            warnings.append(f"hard_filter dropped: {reason} ({(c.get('claim_text') or '')[:60]!r})")
            meta_info["n_dropped_hard"] += 1

    # Re-pack dedup_rank to be dense 0..k-1 after hard_filter drops.
    kept.sort(key=lambda c: c.get("dedup_rank", 0))
    for i, c in enumerate(kept):
        c["dedup_rank"] = i

    return kept, warnings, meta_info


def _invoke_llm(system: str, user: str, model: str) -> str:
    """Try multiple LLM clients in order of preference."""

    # Option 1: OMC's make_llm (if running inside OMC and importable).
    # Employee id resolution matches sibling tools: prefer the calling
    # employee's id from env, fall back to a known researcher id only as
    # a last resort.
    try:
        from onemancompany.agents.base import make_llm  # type: ignore

        employee_id = os.getenv("OMC_EMPLOYEE_ID") or os.getenv("LITSURVEY_EMPLOYEE_ID") or ""
        llm = make_llm(employee_id) if employee_id else make_llm("00007")
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
    version: str = "v3",
    force: bool = False,
) -> dict[str, Any]:
    """Extract 5-15 structured claims from a paper in the corpus and persist them.

    Args:
        paper_id: ID of paper already added via corpus_add_paper.
        model: LLM model to use. Default openai/gpt-4o-mini (cheap structured extraction).
        version: Extraction strategy. Default **"v3"** = meta_pass (contributions +
            scope) → contribution-grounded per-section extraction → global rerank
            with top-K cap → deterministic hard filter (drops noise saliency,
            unverified quotes, scope-less factual claims). Aims for ~8 high-signal
            claims per paper vs v2's ~25.
            "v2" = section-aware split + per-section extraction + evidence_quote
            ground-truth check + structured applies_to_dims, no global dedup.
            "v1" is the legacy single-shot 30K-char-truncated path, retained
            for backward compat — emits a deprecation warning when used.
            See docs/internal/OPTIMIZATION.md section C for v2 design; v3 plan
            lives in the v3 PR description.
        force: If False (default), short-circuit when claims for this paper_id
            already exist (saves LLM calls + avoids duplicate rows). In A2
            layered mode this is what enables cross-project claim reuse —
            project B importing a paper that project A already extracted gets
            the existing claims for free. Set True to re-extract anyway.

    Returns:
        {
          "paper_id": "...",
          "version": "v1" | "v2" | "v3",
          "status": "extracted" | "reused" | "error",
          "claims_extracted": N,
          "claims": [{"id": "claim-...", "claim_text": "...", ...}, ...],
          "warnings": [...],
          "meta": {...}  # v3 only: contributions, scope, dedup counts
        }
    """
    if not paper_id:
        return {"error": "paper_id required"}

    if version not in {"v1", "v2", "v3"}:
        return {
            "error": f"unknown extraction version {version!r}, expected 'v1' / 'v2' / 'v3'",
            "paper_id": paper_id,
        }

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

    if version == "v3":
        raw_claims, warnings, meta_info = _extract_v3(paper, model)
        if not raw_claims:
            return {
                "error": "v3 extraction returned no claims",
                "paper_id": paper_id,
                "version": "v3",
                "claims_extracted": 0,
                "claims": [],
                "warnings": warnings,
                "meta": meta_info,
            }
        validated: list[dict] = []
        for i, raw in enumerate(raw_claims):
            try:
                dims = raw.get("applies_to_dims") or {}
                applies_to_legacy = ", ".join(
                    f"{k}={v}" for k, v in dims.items() if v
                )
                c = Claim(
                    claim_text=raw.get("claim_text", ""),
                    claim_type=raw.get("claim_type", "factual"),
                    saliency_type=raw.get("saliency_type", "empirical_finding"),
                    evidence_span=raw.get("evidence_span", ""),
                    evidence_quote=raw.get("evidence_quote", ""),
                    source_section=raw.get("source_section", ""),
                    confidence=raw.get("confidence", 0.5),
                    applies_to=applies_to_legacy,
                    contribution_idx=int(raw.get("contribution_idx", -1)),
                    dedup_rank=raw.get("dedup_rank"),
                )
                validated.append(
                    {
                        "id": f"{paper_id}#claim-{i+1}",
                        "paper_id": paper_id,
                        "claim_text": c.claim_text,
                        "claim_type": c.claim_type,
                        "saliency_type": c.saliency_type,
                        "evidence_span": c.evidence_span,
                        "evidence_quote": c.evidence_quote,
                        "evidence_quote_verified": bool(raw.get("evidence_quote_verified", False)),
                        "source_section": c.source_section,
                        "section_path": raw.get("section_path", []),
                        "confidence": c.confidence,
                        "applies_to": c.applies_to,
                        "applies_to_dims": dims,
                        "contribution_idx": c.contribution_idx,
                        "dedup_rank": c.dedup_rank,
                        "extracted_at": time.time(),
                        "model": model,
                        "version": "v3",
                    }
                )
            except ValidationError as e:
                warnings.append(f"claim {i} invalid: {e.errors()[:1]}")

        if validated:
            if force:
                _replace_claims_for(paper_id, validated)
            else:
                _append_claims(validated)
        return {
            "paper_id": paper_id,
            "version": "v3",
            "status": "extracted",
            "claims_extracted": len(validated),
            "claims": validated,
            "warnings": warnings,
            "meta": meta_info,
        }

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
                    saliency_type=raw.get("saliency_type", "empirical_finding"),
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
                        "saliency_type": c.saliency_type,
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
            if force:
                _replace_claims_for(paper_id, validated)
            else:
                _append_claims(validated)
        return {
            "paper_id": paper_id,
            "version": "v2",
            "status": "extracted",
            "claims_extracted": len(validated),
            "claims": validated,
            "warnings": warnings,
        }

    v1_deprecation = (
        "v1 is the legacy 30K-truncated single-shot path; default is now v2. "
        "Pass version='v2' explicitly for section-aware extraction + quote verification."
    )
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
    warnings: list[str] = [f"DEPRECATED: {v1_deprecation}"]
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
        if force:
            _replace_claims_for(paper_id, validated)
        else:
            _append_claims(validated)

    return {
        "paper_id": paper_id,
        "version": "v1",
        "status": "extracted",
        "claims_extracted": len(validated),
        "claims": validated,
        "warnings": warnings,
    }
