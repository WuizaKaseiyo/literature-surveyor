"""detect_conflicts — Cross-paper claim contradiction detector.

Scans `<corpus>/claims.jsonl` for pairs of claims that:
  (a) come from different papers,
  (b) describe overlapping experimental settings (via `applies_to_dims`),
  (c) actually contradict each other (LLM judge).

This is the 6th use of extracted claims and the natural payoff of v2's
structured `applies_to_dims` — the dim dict is the join key that filters
~125 claims × 125 = 15K naive pairs down to a few dozen candidates worth
showing to an LLM judge.

Output schema matches schemas/literature_survey_schema.Conflict so the result
can be inlined directly into `stage2.json`.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

CLAIMS_FILENAME = "claims.jsonl"
DEFAULT_MODEL = "openai/gpt-4o-mini"
LEVELS = {"direct", "methodological", "scope", "temporal", "none"}

# Hard caps to keep cost predictable.
MAX_CANDIDATE_PAIRS = 200
MAX_LLM_JUDGED = 50

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
_STOP = frozenset(
    "a an the of in on at to for with from by and or but if then is are was were be been "
    "this that these those it its as we our you your they their he she them his her "
    "i me my so do does did have has had not no can may will would could should "
    "via using used use also more most less than which who what when where why how".split()
)


# ---------------------------------------------------------------------------
# Corpus + claim access
# ---------------------------------------------------------------------------


def _corpus_dir() -> Path:
    # A2 layered mode: claims live in global. Mirror the resolution used by
    # other tools so detect_conflicts sees the same data.
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


def _load_claims() -> list[dict]:
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
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_year_map() -> dict[str, Any]:
    """Build paper_id → year map for temporal-level judgement.

    detect_conflicts used to look year up via `claim.get('paper', {}).get('year')`,
    but claim records don't carry a nested paper dict — that path was dead.
    This function reads papers.jsonl once per detect_conflicts invocation.
    Cheap (one JSONL scan) compared to LLM-judge cost.
    """
    path = _corpus_dir() / "papers.jsonl"
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
                pid = p.get("id")
                if pid:
                    out[pid] = p.get("year", "")
            except json.JSONDecodeError:
                continue
    return out


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP}


# ---------------------------------------------------------------------------
# Pair filtering — pre-LLM cheap heuristics
# ---------------------------------------------------------------------------


def _normalize_value(v: Any) -> str:
    return re.sub(r"\s+", "", str(v or "").lower())


def _share_setting(dims_a: dict | None, dims_b: dict | None) -> dict | None:
    """Return a dict of shared {key: (val_a, val_b)} for keys both claims fill
    with overlapping values, or None if no setting overlap.

    Overlap rule: bidirectional substring after lowercasing + whitespace strip.
    "7B" matches "7B-13B"; "MMLU" matches "MMLU and BBH"; "TruthfulQA" doesn't
    match "MMLU".
    """
    dims_a = dims_a or {}
    dims_b = dims_b or {}
    if not isinstance(dims_a, dict) or not isinstance(dims_b, dict):
        return None
    shared_keys = set(dims_a.keys()) & set(dims_b.keys())
    if not shared_keys:
        return None
    out: dict = {}
    for k in shared_keys:
        na, nb = _normalize_value(dims_a[k]), _normalize_value(dims_b[k])
        if not na or not nb:
            continue
        if na in nb or nb in na:
            out[k] = (dims_a[k], dims_b[k])
    return out or None


def _topic_overlap(text_a: str, text_b: str) -> float:
    """Token Jaccard on claim_text. Cheap pre-filter so the LLM judge isn't
    asked to compare claims with no shared vocabulary at all."""
    ta = _tokenize(text_a)
    tb = _tokenize(text_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def _candidate_pairs(claims: list[dict], min_jaccard: float) -> list[dict]:
    """Generate (a, b) pairs ready to be LLM-judged.

    Filtering chain (cheapest → most expensive):
      1. Different paper_id
      2. Neither is pure conjecture (per conflict-detection SKILL.md)
      3. applies_to_dims share at least one key with overlapping value
      4. claim_text token Jaccard >= min_jaccard
    """
    out: list[dict] = []
    for a, b in combinations(claims, 2):
        if a.get("paper_id") == b.get("paper_id"):
            continue
        # Skip pairs where both are conjecture — those are speculation, not
        # establishable as contradicting facts.
        if a.get("claim_type") == "conjecture" and b.get("claim_type") == "conjecture":
            continue
        shared = _share_setting(a.get("applies_to_dims"), b.get("applies_to_dims"))
        if not shared:
            continue
        topic = _topic_overlap(a.get("claim_text", ""), b.get("claim_text", ""))
        if topic < min_jaccard:
            continue
        out.append({"a": a, "b": b, "shared_setting": shared, "topic_overlap": topic})
        if len(out) >= MAX_CANDIDATE_PAIRS:
            break
    # Rank by topic overlap so the LLM judge sees the most-likely-real pairs first
    out.sort(key=lambda x: x["topic_overlap"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# LLM judge (per pair)
# ---------------------------------------------------------------------------


def _invoke_llm(system: str, user: str, model: str) -> str:
    """Mirror the OMC + OpenAI/OpenRouter fallback chain used by sibling tools."""
    # Option 1: OMC in-process
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


def _parse_json(raw: str) -> dict | None:
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


def _llm_judge_pair(
    claim_a: dict,
    claim_b: dict,
    shared_setting: dict,
    model: str,
    year_map: dict[str, Any] | None = None,
) -> dict | None:
    """LLM judge per candidate pair.

    `year_map` is paper_id → year, used by the temporal-level rule. Claims
    don't carry a nested `paper` dict, so we look year up externally; without
    it the temporal level is unreachable.
    """
    system = (
        "You judge whether two claims from different papers contradict each "
        "other under a shared experimental setting. Return only JSON. "
        "Pay special attention to the verbatim evidence_quote excerpts — "
        "they're the ground truth for what each paper actually says; ignore "
        "paraphrasing differences in claim_text alone."
    )
    year_map = year_map or {}
    a_year = year_map.get(claim_a.get("paper_id", ""), "")
    b_year = year_map.get(claim_b.get("paper_id", ""), "")

    def _block(label: str, c: dict, year: Any) -> str:
        lines = [
            f"{label} (paper {c.get('paper_id', '?')}, year {year or '?'}):",
            f"  claim_text: {c.get('claim_text', '')}",
        ]
        if c.get("evidence_quote"):
            lines.append(f"  evidence_quote: {c.get('evidence_quote', '')}")
        lines.append(
            f"  applies_to_dims: {json.dumps(c.get('applies_to_dims', {}))}"
        )
        return "\n".join(lines)

    user = (
        f"{_block('Claim A', claim_a, a_year)}\n\n"
        f"{_block('Claim B', claim_b, b_year)}\n\n"
        f"Shared setting (computed): {json.dumps(shared_setting)}\n\n"
        "Levels:\n"
        "- direct: same setup + same metric, opposite numbers or opposite qualitative result\n"
        "- methodological: same problem, different methods, different conclusions\n"
        "- scope: claims agree on overlap but differ on extent (e.g. A: works at 7B; B: doesn't at 70B)\n"
        "- temporal: later paper overturns earlier (note in addition to level above)\n"
        "- none: not contradicting (agree, independent, or not comparable)\n\n"
        "Return JSON with keys: verdict ('contradicted' or 'not_contradicted'), "
        "level (one of direct/methodological/scope/temporal/none), "
        "shared_setting (one-line description), "
        "description (one-line natural-language explanation), "
        "confidence (float 0-1)."
    )
    raw = _invoke_llm(system, user, model)
    data = _parse_json(raw)
    if not data:
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    level = str(data.get("level", "")).strip().lower()
    if level not in LEVELS:
        level = "none"
    try:
        conf = float(data.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "verdict": verdict,
        "level": level,
        "shared_setting": str(data.get("shared_setting", ""))[:300],
        "description": str(data.get("description", ""))[:500],
        "confidence": max(0.0, min(1.0, conf)),
    }


# ---------------------------------------------------------------------------
# @tool
# ---------------------------------------------------------------------------


@tool
def detect_conflicts(
    paper_ids: list[str] | None = None,
    use_llm: bool = True,
    model: str = DEFAULT_MODEL,
    level_filter: list[str] | None = None,
    min_topic_jaccard: float = 0.08,
    max_pairs: int = MAX_LLM_JUDGED,
) -> dict[str, Any]:
    """Find cross-paper claim contradictions in the current corpus.

    Algorithm:
      1. Load all claims from claims.jsonl (layered: from global pool)
      2. Filter pairs: different paper, not both conjecture, share setting
         via applies_to_dims, claim_text token Jaccard >= min_topic_jaccard
      3. Rank candidates by topic overlap, take top max_pairs
      4. LLM judge each (or stop here when use_llm=False — useful for
         smoke-testing the pair-filter logic without spending tokens)

    Args:
        paper_ids: Optional restriction to claims from these papers.
        use_llm: If False, return pair candidates only (no verdict).
        model: LLM model for the judge.
        level_filter: Optional post-hoc filter on verdict level
            (e.g. ["direct", "methodological"]).
        min_topic_jaccard: Pre-LLM token-overlap floor on claim_text. Default
            0.08 keeps reasonably-similar claims and drops random ones.
        max_pairs: Hard cap on LLM calls. Default 50 → ~$0.02-0.05 with DS-V3.

    Returns:
        {
          "conflicts": [
            {"id": "conflict-001", "level": "direct",
             "claim_a_paper_id": "...", "claim_a_text": "...", "claim_a_id": "...",
             "claim_b_paper_id": "...", "claim_b_text": "...", "claim_b_id": "...",
             "shared_setting": "...", "description": "...", "confidence": 0.8},
            ...
          ],
          "candidates_screened": N,
          "candidates_judged": M,
          "judge": "llm:<model>" or "candidates_only",
          "checked_at": <epoch>,
        }
    """
    all_claims = _load_claims()
    if paper_ids:
        keep = set(paper_ids)
        all_claims = [c for c in all_claims if c.get("paper_id") in keep]

    if len(all_claims) < 2:
        return {
            "conflicts": [],
            "candidates_screened": 0,
            "candidates_judged": 0,
            "judge": "no_claims",
            "checked_at": time.time(),
        }

    pairs = _candidate_pairs(all_claims, min_jaccard=float(min_topic_jaccard))
    cap = max(1, min(int(max_pairs), 200))
    to_judge = pairs[:cap]

    if not use_llm:
        return {
            "conflicts": [],
            "candidates": [
                {
                    "claim_a_id": p["a"].get("id", ""),
                    "claim_a_paper_id": p["a"].get("paper_id", ""),
                    "claim_a_text": p["a"].get("claim_text", ""),
                    "claim_b_id": p["b"].get("id", ""),
                    "claim_b_paper_id": p["b"].get("paper_id", ""),
                    "claim_b_text": p["b"].get("claim_text", ""),
                    "shared_setting": p["shared_setting"],
                    "topic_overlap": round(p["topic_overlap"], 3),
                }
                for p in to_judge
            ],
            "candidates_screened": len(pairs),
            "candidates_judged": 0,
            "judge": "candidates_only",
            "checked_at": time.time(),
        }

    conflicts: list[dict] = []
    cid = 1
    level_set = (
        {l.lower() for l in level_filter} if level_filter else None
    )
    # Load year map once per invocation so temporal-level rule isn't dead code.
    year_map = _load_year_map()
    for p in to_judge:
        verdict = _llm_judge_pair(
            p["a"], p["b"], p["shared_setting"], model, year_map=year_map
        )
        if verdict is None:
            continue
        if verdict["verdict"] != "contradicted":
            continue
        if level_set and verdict["level"] not in level_set:
            continue
        conflicts.append(
            {
                "id": f"conflict-{cid:03d}",
                "level": verdict["level"],
                "claim_a_id": p["a"].get("id", ""),
                "claim_a_paper_id": p["a"].get("paper_id", ""),
                "claim_a_text": p["a"].get("claim_text", ""),
                "claim_b_id": p["b"].get("id", ""),
                "claim_b_paper_id": p["b"].get("paper_id", ""),
                "claim_b_text": p["b"].get("claim_text", ""),
                "shared_setting": verdict["shared_setting"]
                or ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in p["shared_setting"].items()),
                "description": verdict["description"],
                "confidence": verdict["confidence"],
                "topic_overlap": round(p["topic_overlap"], 3),
            }
        )
        cid += 1

    return {
        "conflicts": conflicts,
        "candidates_screened": len(pairs),
        "candidates_judged": len(to_judge),
        "judge": f"llm:{model}",
        "checked_at": time.time(),
    }
