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
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

CLAIMS_FILENAME = "claims.jsonl"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
MAX_INPUT_CHARS = 30000  # truncate very long papers


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


def _call_llm_for_claims(paper_text: str, paper_meta: dict, model: str) -> list[dict]:
    """Returns list of dicts (raw, pre-validation). Empty list on hard failure."""

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

    raw = _invoke_llm(system, user, model)
    if not raw:
        return []

    # Strip ```json fences if model added them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "claims" in data:
            return data["claims"]
    except json.JSONDecodeError:
        pass
    return []


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


@tool
def extract_claims(
    paper_id: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
) -> dict[str, Any]:
    """Extract 5-15 structured claims from a paper in the corpus and persist them.

    Args:
        paper_id: ID of paper already added via corpus_add_paper.
        model: LLM model to use. Default openai/gpt-4o-mini (cheap structured extraction).

    Returns:
        {
          "paper_id": "...",
          "claims_extracted": N,
          "claims": [{"id": "claim-...", "claim_text": "...", ...}, ...],
          "warnings": [...]
        }
    """
    if not paper_id:
        return {"error": "paper_id required"}

    paper = _load_paper(paper_id)
    if not paper:
        return {"error": f"paper not found in corpus: {paper_id}"}

    text = paper.get("full_text_md") or paper.get("abstract") or ""
    if len(text) < 200:
        return {
            "error": f"paper text too short ({len(text)} chars) — need full_text_md or longer abstract",
            "paper_id": paper_id,
        }

    raw_claims = _call_llm_for_claims(text, paper, model)
    if not raw_claims:
        return {
            "error": "LLM returned no parseable claims",
            "paper_id": paper_id,
            "claims_extracted": 0,
            "claims": [],
        }

    validated: list[dict] = []
    warnings: list[str] = []
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
                }
            )
        except ValidationError as e:
            warnings.append(f"claim {i} invalid: {e.errors()[:1]}")

    if validated:
        _append_claims(validated)

    return {
        "paper_id": paper_id,
        "claims_extracted": len(validated),
        "claims": validated,
        "warnings": warnings,
    }
