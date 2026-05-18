"""fact_check_rendered_survey — Verify final text claims against cited papers.

This is the second-stage quality gate after verify_citations:
  1. Parse the final Markdown into sentence-level claim/citation pairs.
  2. Resolve each citation to a paper in the local corpus.
  3. Check whether the cited paper supports the sentence it is attached to.

The checker is self-contained for OMC asset-tool loading. It uses a conservative
heuristic by default and can use an LLM judge when OMC/OpenAI-compatible clients
are available.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

CITE_PATTERN = re.compile(
    r"\[([^\]]*?),\s*(arxiv|doi|S2):([^\]\s]+)\]",
    re.IGNORECASE,
)
FENCED_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?%?|\d+(?:\.\d+)?[KMBTkmbt]?)")
# E2: number-with-unit pattern. Captures the digit string + the unit slot in one shot.
# Unit alternatives (in priority order):
#   \s*%                          → percent sign
#   \s+(?:percent|percentage|pct) → percent word form (must be preceded by space)
#   [KMBTkmbt]\b                  → scale suffix (Big-K/M/B/T) with word boundary
#   ε                             → bare number, no unit
_NUM_WITH_UNIT_PAT = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(-?\d+(?:\.\d+)?)"
    r"(\s*%|\s+(?:percent|percentage|pct)\b|[KMBTkmbt]\b|)",
    re.IGNORECASE,
)
_NUMBER_TOLERANCE = 0.05  # relative tolerance for numeric matching
MAX_SOURCE_CHARS = 50000
MAX_CONTEXT_CHARS = 6000
EXPANDED_CONTEXT_CHARS = 12000          # E5: borderline-retry budget
LOW_CONFIDENCE_THRESHOLD = 0.6          # E5: confidence floor below which we retry
EXPANDABLE_VERDICTS = {"unsupported", "partially_supported"}  # only retry these

VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "source_irrelevant",
    "source_not_in_corpus",
    "no_source_text",
    "judge_error",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
_STOP = frozenset(
    "a an the of in on at to for with from by and or but if then is are was were be been "
    "this that these those it its as we our you your they their he she them his her "
    "i me my so do does did have has had not no can may will would could should "
    "via using used use also more most less than which who what when where why how "
    "paper study studies result results method methods model models data dataset datasets "
    "show shows showed find finds found suggest suggests proposed propose et al".split()
)


def _corpus_dir() -> Path:
    # A2 layered mode: entity store (papers.jsonl, claims.jsonl) lives in the
    # global dir so cross-project anchoring works automatically.
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


def _load_papers() -> list[dict]:
    path = _corpus_dir() / "papers.jsonl"
    if not path.exists():
        return []
    papers: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def _load_claims_for_paper(paper_id: str) -> list[dict]:
    """Read pre-extracted claims for one paper. Empty list if no claims.jsonl."""
    if not paper_id:
        return []
    path = _corpus_dir() / "claims.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
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


def _match_claim_to_sentence(
    sentence: str, claims: list[dict]
) -> tuple[dict | None, float]:
    """Pick the claim most likely supporting `sentence`.

    Token-overlap scoring (Jaccard against claim_text + evidence_quote) with a
    minimum threshold to avoid spurious matches. Returns (best_claim, score) or
    (None, 0.0).

    Threshold rationale: require ≥ 3 shared content tokens AND Jaccard ≥ 0.12.
    Below this, the BM25-on-full-text path is more reliable than a forced anchor.
    """
    if not claims:
        return None, 0.0
    sent_tokens = set(_tokenize(sentence))
    if len(sent_tokens) < 2:
        return None, 0.0

    best: dict | None = None
    best_score = 0.0
    best_overlap = 0
    for c in claims:
        text = (c.get("claim_text") or "") + " " + (c.get("evidence_quote") or "")
        c_tokens = set(_tokenize(text))
        if not c_tokens:
            continue
        overlap = sent_tokens & c_tokens
        if len(overlap) < 3:
            continue
        score = len(overlap) / max(len(sent_tokens | c_tokens), 1)
        if score > best_score:
            best_score = score
            best_overlap = len(overlap)
            best = c

    if best is None or best_score < 0.12 or best_overlap < 3:
        return None, best_score
    return best, best_score


def _paper_lookup(papers: list[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for p in papers:
        candidates = [
            ("id", p.get("id")),
            ("arxiv", p.get("arxiv_id")),
            ("doi", p.get("doi")),
            ("s2", p.get("paper_id")),
            ("s2", p.get("s2_id")),
            ("openalex", p.get("openalex_id")),
        ]
        pid = p.get("id")
        if pid:
            candidates.extend([
                ("arxiv", pid),
                ("doi", pid),
                ("s2", pid),
            ])
        for kind, value in candidates:
            if value:
                lookup[(kind.lower(), _normalize_cite_id(str(value)))] = p
    return lookup


def _normalize_cite_id(cite_id: str) -> str:
    return cite_id.strip().rstrip(".,;:)").lower()


def _strip_code(markdown: str) -> str:
    markdown = FENCED_CODE_PATTERN.sub("", markdown or "")
    return INLINE_CODE_PATTERN.sub("", markdown)


def _clean_sentence(text: str) -> str:
    text = re.sub(r"^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s*)", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _without_cites(text: str) -> str:
    text = CITE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:;,.")


def _has_claim_text(text: str) -> bool:
    no_cite = _without_cites(text)
    return len(no_cite) >= 8 and bool(_TOKEN_RE.search(no_cite))


def _split_sentences(paragraph: str) -> list[str]:
    paragraph = re.sub(r"\s+", " ", paragraph.strip())
    if not paragraph:
        return []
    parts: list[str] = []
    start = 0
    bracket_depth = 0
    for i, ch in enumerate(paragraph):
        if ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth:
            bracket_depth -= 1
        elif ch in ".!?" and bracket_depth == 0:
            j = i + 1
            while j < len(paragraph) and paragraph[j].isspace():
                j += 1
            if j >= len(paragraph) or paragraph[j].isupper() or paragraph[j] in "\"'([" or paragraph[j].isdigit():
                parts.append(paragraph[start : i + 1])
                start = j
    if start < len(paragraph):
        parts.append(paragraph[start:])
    out: list[str] = []
    for part in parts:
        cleaned = _clean_sentence(part)
        if cleaned:
            out.append(cleaned)
    return out


def attributions_from_survey_json(survey_json: dict) -> list[dict[str, Any]]:
    """Build attributions directly from a LiteratureSurveySchema-shaped dict.

    Each finding becomes ONE attribution (no sentence splitting / backward
    attribution / footnote handling — those are markdown-parsing concerns).

    finding.cites[*].cite_text is parsed to extract (kind, id) so the same
    paper-lookup logic works for both entry points.

    finding.claim_ids are passed through as `preferred_claim_ids` so the
    judge can use the talent's pre-declared anchors instead of re-running
    token-overlap matching.
    """
    out: list[dict[str, Any]] = []
    cite_id_pat = re.compile(r"(arxiv|doi|s2):([^,\]\s]+)", re.IGNORECASE)
    for i, f in enumerate(survey_json.get("findings", [])):
        if not isinstance(f, dict):
            continue
        text = (f.get("text") or "").strip()
        if not text:
            continue
        cites_out: list[dict[str, Any]] = []
        for c in f.get("cites", []) or []:
            if not isinstance(c, dict):
                continue
            cite_text = c.get("cite_text", "") or ""
            paper_id = c.get("paper_id", "") or ""
            m = cite_id_pat.search(cite_text)
            if m:
                kind = m.group(1).lower()
                cid = m.group(2).strip().rstrip(".,;:)")
            elif paper_id:
                # Heuristic: if paper_id looks like arxiv (e.g., 2401.12345 or
                # cs.CL/0123456), assume arxiv. Otherwise default to arxiv too;
                # verify_citations / paper lookup will surface mismatches.
                kind = "arxiv"
                cid = paper_id
            else:
                continue
            cites_out.append({
                "raw": f"[{cite_text or paper_id}]",
                "kind": kind,
                "id": cid,
            })
        if not cites_out:
            continue
        out.append({
            "id": f"finding-{i + 1:04d}",
            "claim_text": text,
            "sentence_with_cites": text,
            "citations": cites_out,
            "preferred_claim_ids": list(f.get("claim_ids") or []),
        })
    return out


def parse_attributions(markdown: str) -> list[dict[str, Any]]:
    """Parse Markdown into sentence-level claim/citation pairs.

    Backward attribution: when a citation appears in a sentence, it is also
    attached to preceding uncited claim sentences in the same paragraph.
    """
    text = _strip_code(markdown)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    attributions: list[dict[str, Any]] = []
    attr_id = 1

    for para_index, paragraph in enumerate(paragraphs):
        sentences = _split_sentences(paragraph)
        pending: list[str] = []

        for sentence in sentences:
            matches = list(CITE_PATTERN.finditer(sentence))
            if not matches:
                if _has_claim_text(sentence):
                    pending.append(sentence)
                continue

            citations = [
                {
                    "raw": m.group(0),
                    "author_year": m.group(1).strip(),
                    "kind": m.group(2).lower(),
                    "id": m.group(3).strip().rstrip(".,;:)"),
                }
                for m in matches
            ]

            candidate_sentences = pending + ([sentence] if _has_claim_text(sentence) else [])
            for claim_sentence in candidate_sentences:
                claim = _without_cites(claim_sentence)
                if not claim:
                    continue
                attributions.append(
                    {
                        "id": f"attr-{attr_id:04d}",
                        "paragraph_index": para_index,
                        "claim_text": claim,
                        "sentence_with_cites": claim_sentence,
                        "citations": citations,
                    }
                )
                attr_id += 1
            pending = []

    return attributions


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP]


def _numbers(text: str) -> set[str]:
    """Legacy bag-of-numbers extractor. Retained for callers that want a coarse
    set of stringified numbers; the heuristic now uses
    _extract_numbers_with_unit + _numbers_match for unit-aware tolerance.
    """
    out = set()
    for n in NUMBER_PATTERN.findall(text or ""):
        out.add(n.lower())
        out.add(n.lower().rstrip("%"))
    return out


def _extract_numbers_with_unit(text: str) -> list[tuple[float, str]]:
    """Return [(value, unit)] for each number in `text`.

    Unit canonical form: "%" / "K" / "M" / "B" / "T" / "" (no unit).
    Word forms ("30 percent", "30 pct") normalize to "%". Sign is preserved
    when captured (limited: only when not glued to a preceding alphanumeric).

    Examples:
        "30%"           → [(30.0, "%")]
        "30 percent"    → [(30.0, "%")]
        "1.5B params"   → [(1.5, "B")]
        "improved by 13" → [(13.0, "")]
    """
    out: list[tuple[float, str]] = []
    for m in _NUM_WITH_UNIT_PAT.finditer(text or ""):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        u_raw = (m.group(2) or "").strip().lower()
        if u_raw == "%":
            unit = "%"
        elif u_raw in ("percent", "percentage", "pct"):
            unit = "%"
        elif u_raw in ("k", "m", "b", "t"):
            unit = u_raw.upper()
        else:
            unit = ""
        out.append((v, unit))
    return out


def _fmt_num(v: float, u: str) -> str:
    return f"{int(v) if v == int(v) else v}{u}"


def _numbers_match(
    claim_text: str, source_text: str, rel_tol: float = _NUMBER_TOLERANCE
) -> tuple[bool, list[str]]:
    """For each (value, unit) in claim, look for any source (v', u') with the
    SAME unit and `|v - v'| / max(|v|, |v'|) <= rel_tol`. Returns
    (all_matched, missing_pretty_strings).

    Same-unit constraint intentional: 30% ≢ 0.30 without context, even though
    they're numerically equivalent. Cross-unit equivalence is a downstream LLM
    judge's job, not the cheap heuristic's.
    """
    claim_nums = _extract_numbers_with_unit(claim_text)
    if not claim_nums:
        return True, []
    source_nums = _extract_numbers_with_unit(source_text)
    missing: list[str] = []
    for cv, cu in claim_nums:
        ok = False
        for sv, su in source_nums:
            if cu != su:
                continue
            denom = max(abs(cv), abs(sv))
            if denom == 0:
                ok = (cv == sv)
                if ok:
                    break
                continue
            if abs(cv - sv) / denom <= rel_tol:
                ok = True
                break
        if not ok:
            missing.append(_fmt_num(cv, cu))
    return len(missing) == 0, missing


def _source_text(paper: dict) -> str:
    return (
        paper.get("full_text_md")
        or paper.get("markdown")
        or paper.get("abstract")
        or ""
    )[:MAX_SOURCE_CHARS]


def _source_sentences(source: str) -> list[str]:
    source = re.sub(r"\s+", " ", source or "")
    if not source:
        return []
    # Also break on Markdown headings/bullets by normalizing them into periods.
    source = re.sub(r"\s+(#{1,6}\s+|[-*+]\s+)", ". ", source)
    return [s.strip() for s in SENTENCE_BOUNDARY_PATTERN.split(source) if s.strip()]


def _select_context(
    claim: str,
    source: str,
    max_chars: int = MAX_CONTEXT_CHARS,
    max_sentences: int = 12,
) -> tuple[str, list[str], float]:
    """Pick up to `max_sentences` best-overlapping source sentences capped at
    `max_chars`. Defaults preserve the original 6K/12-sentence baseline budget;
    E5 expansion bumps both."""
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return source[:max_chars], [], 0.0

    claim_counts = Counter(claim_tokens)
    claim_vocab = set(claim_counts)
    ranked: list[tuple[float, str]] = []

    for sent in _source_sentences(source):
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            continue
        sent_counts = Counter(sent_tokens)
        overlap = sum(min(claim_counts[t], sent_counts[t]) for t in claim_vocab)
        if overlap <= 0:
            continue
        density = overlap / max(len(sent_tokens), 1)
        coverage = len(set(sent_tokens) & claim_vocab) / max(len(claim_vocab), 1)
        score = overlap + density + coverage
        ranked.append((score, sent))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected: list[str] = []
    total_len = 0
    for _, sent in ranked[:max_sentences]:
        if total_len + len(sent) > max_chars:
            break
        selected.append(sent)
        total_len += len(sent) + 1

    if not selected:
        return source[:max_chars], [], 0.0

    context = " ".join(selected)
    context_vocab = set(_tokenize(context))
    coverage = len(context_vocab & claim_vocab) / max(len(claim_vocab), 1)
    return context, selected[:3], coverage


def _heuristic_judge(claim: str, source: str) -> dict[str, Any]:
    if not source.strip():
        return {
            "verdict": "no_source_text",
            "confidence": 0.0,
            "evidence_quote": "",
            "explanation": "The cited paper has no full text or abstract in the corpus.",
            "judge": "heuristic",
        }

    context, top_sentences, coverage = _select_context(claim, source)
    if not top_sentences:
        return {
            "verdict": "source_irrelevant",
            "confidence": 0.25,
            "evidence_quote": "",
            "explanation": "No meaningful token overlap was found between the claim and cited source.",
            "judge": "heuristic",
        }

    matched, missing_numbers = _numbers_match(claim, context)
    if not matched:
        return {
            "verdict": "unsupported",
            "confidence": 0.55,
            "evidence_quote": top_sentences[0],
            "explanation": (
                "The cited source is topically related, but the selected evidence "
                f"does not contain the claim's numeric value(s) within "
                f"{int(_NUMBER_TOLERANCE * 100)}% tolerance: "
                f"{', '.join(missing_numbers[:5])}."
            ),
            "judge": "heuristic",
        }

    if coverage >= 0.45:
        verdict = "supported"
        confidence = 0.65
        explanation = "The claim has strong lexical overlap with retrieved evidence from the cited source."
    elif coverage >= 0.22:
        verdict = "partially_supported"
        confidence = 0.5
        explanation = "The cited source appears related, but the evidence only partially matches the claim."
    else:
        verdict = "unsupported"
        confidence = 0.45
        explanation = "The cited source is weakly related and does not clearly support the specific claim."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_quote": top_sentences[0],
        "explanation": explanation,
        "judge": "heuristic",
    }


_CONFIDENCE_WORDS = {
    "very high": 0.95, "high": 0.85, "medium-high": 0.75,
    "medium": 0.6, "moderate": 0.6,
    "medium-low": 0.45, "low": 0.3, "very low": 0.15,
}


def _parse_confidence(value: Any) -> float:
    """Accept float, int, numeric string, or word ('high'/'medium'/'low').

    Why this matters for E5: any sloppy LLM confidence string used to default
    to 0.5, which is below LOW_CONFIDENCE_THRESHOLD (0.6) → spurious E5
    retries cost money. We try harder to recover a number from messy strings
    ("approximately 0.7", "confidence ~85%") before falling back.
    """
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[s]
        # Fast path: whole-string numeric ("0.85", "85%")
        try:
            return max(0.0, min(1.0, float(s.rstrip("%")) / (100.0 if "%" in s else 1.0)))
        except ValueError:
            pass
        # Slow path: pull the first numeric substring out of prose
        # ("approximately 0.7", "confidence ~85%", "high (≈0.9)")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                v = float(m.group())
                if "%" in s or v > 1.0:
                    v = v / 100.0
                return max(0.0, min(1.0, v))
            except ValueError:
                pass
        return 0.5
    return 0.5


def _parse_llm_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _invoke_llm(system: str, user: str, model: str) -> str:
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

    try:
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            return ""
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
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


def _llm_judge(claim: str, context: str, model: str) -> dict[str, Any] | None:
    system = (
        "You are a strict source attribution fact checker. Judge only whether the "
        "provided source excerpt supports the claim. Return only JSON."
    )
    user = (
        "Claim:\n"
        f"{claim}\n\n"
        "Source excerpt:\n"
        f"{context[:MAX_CONTEXT_CHARS]}\n\n"
        "Rubric:\n"
        "- supported: the excerpt directly supports all concrete facts in the claim.\n"
        "- partially_supported: the excerpt supports the topic or part of the claim, but not every concrete detail.\n"
        "- unsupported: the excerpt is relevant but does not support the specific claim, or evidence is absent/uncertain.\n"
        "- contradicted: the excerpt contradicts the claim.\n"
        "- source_irrelevant: the excerpt is not about the same topic.\n\n"
        "Return JSON with keys: verdict, confidence, evidence_quote, explanation. "
        "Use one verdict from the rubric."
    )
    raw = _invoke_llm(system, user, model)
    data = _parse_llm_json(raw)
    if not data:
        return None

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return None
    return {
        "verdict": verdict,
        "confidence": _parse_confidence(data.get("confidence")),
        "evidence_quote": str(data.get("evidence_quote", "") or "")[:1200],
        "explanation": str(data.get("explanation", "") or "")[:1200],
        "judge": f"llm:{model}",
    }


def _build_anchor_block(matched_claim: dict | None) -> str:
    if not matched_claim or not matched_claim.get("evidence_quote"):
        return ""
    lines = [
        f"Extracted claim from paper: {matched_claim.get('claim_text', '')}",
        f"Verbatim evidence quote: {matched_claim.get('evidence_quote', '')}",
    ]
    if matched_claim.get("source_section"):
        lines.append(f"Section: {matched_claim.get('source_section', '')}")
    return "\n".join(lines)


def _check_claim_against_paper(
    claim: str,
    paper: dict | None,
    judge_model: str,
    use_llm: bool,
    use_extracted_claims: bool = True,
    expand_on_low_confidence: bool = True,
    preferred_claim_id: str = "",
) -> dict[str, Any]:
    """When called from the survey_json fast path, `preferred_claim_id` is the
    claim id the talent declared in Finding.claim_ids; we use it as the anchor
    directly, skipping token-overlap matching. Empty string = fall back to the
    matcher (markdown path or no preferred anchor)."""
    if paper is None:
        return {
            "verdict": "source_not_in_corpus",
            "confidence": 0.0,
            "evidence_quote": "",
            "explanation": "The citation could not be resolved to a paper in the local corpus.",
            "judge": "resolver",
            "matched_claim_id": "",
            "matched_claim_quote": "",
            "expanded_retry": "",
        }

    source = _source_text(paper)
    context, top_sentences, _ = _select_context(claim, source)

    matched_claim: dict | None = None
    if use_extracted_claims:
        claims = _load_claims_for_paper(paper.get("id", ""))
        if preferred_claim_id:
            for c in claims:
                if c.get("id") == preferred_claim_id:
                    matched_claim = c
                    break
        if matched_claim is None:
            matched_claim, _match_score = _match_claim_to_sentence(claim, claims)

    anchor_block = _build_anchor_block(matched_claim)
    if anchor_block:
        # Use the pre-extracted claim as the primary anchor for the LLM judge.
        # Old BM25 context is still appended for additional grounding.
        judge_context = anchor_block + "\n\n---\n\nAdditional source context:\n" + context[:3000]
    else:
        judge_context = context

    result: dict[str, Any]
    if use_llm and judge_context.strip():
        judged = _llm_judge(claim, judge_context, judge_model)
        if judged:
            result = judged
        else:
            result = _heuristic_judge(claim, source)
    else:
        result = _heuristic_judge(claim, source)

    # E5: borderline retry with expanded BM25 window. Only kicks in when the
    # LLM judge ran AND verdict is one of the "uncertain rejection" ones AND
    # confidence is below the threshold. Take the STRICTER verdict (more
    # concerning) to stay conservative — talent should still investigate.
    expanded_retry_status = ""
    if (
        use_llm
        and expand_on_low_confidence
        and str(result.get("judge", "")).startswith("llm:")
        and result.get("verdict") in EXPANDABLE_VERDICTS
        and float(result.get("confidence", 1.0)) < LOW_CONFIDENCE_THRESHOLD
    ):
        expanded_context, _, _ = _select_context(
            claim, source, max_chars=EXPANDED_CONTEXT_CHARS, max_sentences=24
        )
        if anchor_block:
            retry_context = (
                anchor_block
                + "\n\n---\n\nAdditional source context:\n"
                + expanded_context
            )
        else:
            retry_context = expanded_context
        retry_judged = _llm_judge(claim, retry_context, judge_model)
        if retry_judged:
            # Use STRICTNESS rank — escalating unsupported → contradicted is
            # the whole point of expanded-context retry; rollup rank would
            # incorrectly block that escalation (see C1 review note).
            orig_rank = _STRICTNESS_RANK.get(result.get("verdict", ""), 0)
            new_rank = _STRICTNESS_RANK.get(retry_judged.get("verdict", ""), 0)
            if new_rank < orig_rank:
                result = retry_judged
                expanded_retry_status = "took_stricter"
            else:
                expanded_retry_status = "kept_original"
        else:
            expanded_retry_status = "retry_failed"

    if not result.get("evidence_quote") and top_sentences:
        result["evidence_quote"] = top_sentences[0]

    if matched_claim:
        result["matched_claim_id"] = matched_claim.get("id", "")
        result["matched_claim_quote"] = matched_claim.get("evidence_quote", "")
        # Prefer the matched claim's verbatim quote as the surface evidence.
        if matched_claim.get("evidence_quote"):
            result["evidence_quote"] = matched_claim["evidence_quote"]
    else:
        result.setdefault("matched_claim_id", "")
        result.setdefault("matched_claim_quote", "")
    result["expanded_retry"] = expanded_retry_status
    return result


# Two distinct verdict orderings — they encode different judgements.
#
# _ROLLUP_RANK (max wins): used to aggregate multi-cite items into a single
# per-attribution verdict. The intent is "most actionable for the talent" —
# supported > partial > contradicted > unsupported. We rank contradicted ABOVE
# unsupported here because, when no positive cite exists, "the paper says the
# opposite" is more informative than "the paper doesn't say". The talent's
# action differs (rewrite vs find better cite), and the more pointed verdict
# should win the rollup display.
_ROLLUP_RANK = {
    "supported": 6,
    "partially_supported": 5,
    "contradicted": 4,
    "unsupported": 3,
    "source_irrelevant": 2,
    "source_not_in_corpus": 1,
    "no_source_text": 0,
    "judge_error": -1,
}

# _STRICTNESS_RANK (lower wins for "stricter"): used by E5 retry. The retry
# replaces the original only when the new verdict is STRICTER (more concerning
# about the claim's truthhood). Direction matters here: a retry that escalates
# unsupported → contradicted must be taken (extra context revealed an opposing
# fact); a retry that relaxes contradicted → supported must NOT be taken (we
# never relax a confident rejection).
#
# Strictness lattice (lowest = strictest):
#   contradicted ≺ no_source_text ≺ source_not_in_corpus ≺ source_irrelevant
#   ≺ unsupported ≺ partially_supported ≺ supported
# judge_error is sentinel — treated as neutral.
_STRICTNESS_RANK = {
    "supported": 6,
    "partially_supported": 5,
    "unsupported": 4,
    "source_irrelevant": 3,
    "source_not_in_corpus": 2,
    "no_source_text": 1,
    "judge_error": 0,
    "contradicted": -1,
}
# Legacy alias retained so any external caller importing _VERDICT_RANK (none
# in this branch, but defensive) keeps working with the rollup semantics.
_VERDICT_RANK = _ROLLUP_RANK
_NON_BLOCKING = {"supported", "partially_supported"}


def _aggregate_attributions(items: list[dict]) -> list[dict]:
    """Group per-cite items by attribution_id; pick the most-supportive verdict
    per group as the attribution's final verdict.

    Items missing attribution_id (shouldn't happen, but defensive) are treated
    as singleton groups keyed by their index.
    """
    groups: dict[str, list[dict]] = {}
    for i, it in enumerate(items):
        key = it.get("attribution_id") or f"_anon_{i}"
        groups.setdefault(key, []).append(it)

    per_attr: list[dict] = []
    for attr_id, sub in groups.items():
        best = max(sub, key=lambda x: _ROLLUP_RANK.get(x.get("verdict", "judge_error"), -1))
        per_attr.append({
            "attribution_id": attr_id,
            "claim_text": best.get("claim_text", ""),
            "final_verdict": best.get("verdict", "judge_error"),
            "final_matched_claim_id": best.get("matched_claim_id", ""),
            "sub_results": [
                {
                    "citation_raw": s.get("citation_raw", ""),
                    "citation_id": s.get("citation_id", ""),
                    "paper_id": s.get("paper_id", ""),
                    "verdict": s.get("verdict", ""),
                    "confidence": s.get("confidence", 0.0),
                    "judge": s.get("judge", ""),
                }
                for s in sub
            ],
        })
    return per_attr


def _summary(items: list[dict], per_attribution: list[dict]) -> dict[str, Any]:
    """Aggregate counts. Per-cite item counts are reported for backward compat;
    the canonical `blocking_count` is now per-attribution (the gate talent should
    actually act on)."""
    counts = Counter(item.get("verdict", "judge_error") for item in items)
    attr_counts = Counter(a.get("final_verdict", "judge_error") for a in per_attribution)
    blocking_attributions = sum(
        n for v, n in attr_counts.items() if v not in _NON_BLOCKING
    )
    return {
        "supported_count": counts.get("supported", 0),
        "partial_count": counts.get("partially_supported", 0),
        "unsupported_count": counts.get("unsupported", 0),
        "contradicted_count": counts.get("contradicted", 0),
        "source_irrelevant_count": counts.get("source_irrelevant", 0),
        "source_not_in_corpus_count": counts.get("source_not_in_corpus", 0),
        "no_source_text_count": counts.get("no_source_text", 0),
        "judge_error_count": counts.get("judge_error", 0),
        # Backward-compat: legacy `blocking_count` was the per-cite count.
        # New canonical: blocking attributions = attributions whose ROLLED-UP
        # verdict is non-supportive. Talent should check `blocking_count` (now
        # the attribution one); `blocking_cite_count` is the legacy view.
        "blocking_count": blocking_attributions,
        "blocking_cite_count": (
            counts.get("unsupported", 0)
            + counts.get("contradicted", 0)
            + counts.get("source_irrelevant", 0)
            + counts.get("source_not_in_corpus", 0)
            + counts.get("no_source_text", 0)
            + counts.get("judge_error", 0)
        ),
        "total_checked": len(items),
        "total_attributions": len(per_attribution),
    }


@tool
def fact_check_rendered_survey(
    markdown: str = "",
    use_llm: bool = False,
    judge_model: str = "openai/gpt-4o-mini",
    max_checks: int = 200,
    use_extracted_claims: bool = True,
    expand_on_low_confidence: bool = True,
    survey_json: dict | None = None,
) -> dict[str, Any]:
    """Fact-check final survey claims against their cited corpus papers.

    Two entry points (use one):
      - `markdown`: legacy. Sentence-level parsing of the rendered Stage 2
        markdown. Handles backward attribution and code-block skipping.
      - `survey_json`: E6 fast path. A `LiteratureSurveySchema`-shaped dict.
        Each `findings[i]` becomes one attribution; `claim_ids[*]` are honored
        as pre-declared anchors (skips token-overlap matching against
        claims.jsonl). More deterministic, no parser edge cases.

    Args:
        markdown: Final rendered survey Markdown. Ignored when survey_json is supplied.
        use_llm: If True, use an available OMC/OpenAI-compatible LLM judge;
            otherwise use a conservative deterministic heuristic.
        judge_model: LLM judge model for use_llm=True.
        max_checks: Hard cap on attribution-citation pairs to evaluate.
        use_extracted_claims: If True (default), when claims.jsonl is populated,
            anchor the judge to the matched pre-extracted claim's evidence_quote
            instead of (only) BM25-selecting context from raw paper text. Set
            False to compare against the BM25-only baseline.
        expand_on_low_confidence: If True (default) and the LLM judge returned
            "unsupported" or "partially_supported" with confidence < 0.6, retry
            once with an expanded BM25 window (12K chars, 24 sentences). If the
            retry verdict is stricter (more concerning), it replaces the first;
            otherwise the first is kept. Each item exposes `expanded_retry`
            ∈ {"", "took_stricter", "kept_original", "retry_failed"}.
        survey_json: E6 fast path input. Mutually preferred over markdown when both supplied.

    Returns:
        {
          "items": [...],
          "supported_count": N,
          "unsupported_count": M,
          "blocking_count": K,
          "parsed_attributions": A,
          "total_checked": C,
          "input_mode": "survey_json" | "markdown" | "empty"
        }
    """
    if survey_json:
        attributions = attributions_from_survey_json(survey_json)
        input_mode = "survey_json"
    elif markdown:
        attributions = parse_attributions(markdown)
        input_mode = "markdown"
    else:
        empty = _summary([], [])
        return {
            **empty,
            "items": [],
            "per_attribution": [],
            "parsed_attributions": 0,
            "input_mode": "empty",
            "verification_method": "final attribution fact check",
        }

    papers = _load_papers()
    lookup = _paper_lookup(papers)

    items: list[dict[str, Any]] = []
    for attr in attributions:
        preferred_ids = attr.get("preferred_claim_ids") or []
        for cite in attr["citations"]:
            if len(items) >= max(1, min(int(max_checks), 1000)):
                break
            kind = cite["kind"].lower()
            cite_id = _normalize_cite_id(cite["id"])
            paper = lookup.get((kind, cite_id))
            # E6: pick the preferred_claim_id that targets THIS cite's paper.
            preferred_cid = ""
            if paper and preferred_ids:
                pid = paper.get("id", "")
                for cid in preferred_ids:
                    if isinstance(cid, str) and cid.startswith(f"{pid}#"):
                        preferred_cid = cid
                        break
            result = _check_claim_against_paper(
                attr["claim_text"],
                paper,
                judge_model,
                use_llm,
                use_extracted_claims=use_extracted_claims,
                expand_on_low_confidence=expand_on_low_confidence,
                preferred_claim_id=preferred_cid,
            )
            items.append(
                {
                    "attribution_id": attr["id"],
                    "claim_text": attr["claim_text"],
                    "citation_raw": cite["raw"],
                    "citation_kind": kind,
                    "citation_id": cite["id"],
                    "paper_id": paper.get("id", "") if paper else "",
                    "paper_title": paper.get("title", "") if paper else "",
                    "verdict": result["verdict"],
                    "confidence": result.get("confidence", 0.0),
                    "evidence_quote": result.get("evidence_quote", ""),
                    "explanation": result.get("explanation", ""),
                    "judge": result.get("judge", "unknown"),
                    "matched_claim_id": result.get("matched_claim_id", ""),
                    "matched_claim_quote": result.get("matched_claim_quote", ""),
                    "expanded_retry": result.get("expanded_retry", ""),
                }
            )
        if len(items) >= max(1, min(int(max_checks), 1000)):
            break

    per_attribution = _aggregate_attributions(items)
    summary = _summary(items, per_attribution)
    return {
        **summary,
        "items": items,
        "per_attribution": per_attribution,
        "parsed_attributions": len(attributions),
        "corpus_size": len(papers),
        "checked_at": time.time(),
        "input_mode": input_mode,
        "verification_method": f"final attribution fact check ({input_mode})",
    }
