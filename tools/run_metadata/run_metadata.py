"""run_metadata — Per-run provenance: hashes, model identifiers, stage timings.

Three @tool functions form a small lifecycle:

    run_start(research_question, ...)   → writes initial run.json
    run_stage_done(stage, elapsed_s, ...) → appends a stage entry (many calls)
    run_finalize(output_paths=...)      → stamps completed_at, snapshots corpus

The final `run.json` lets you:
  * diff two runs of the same talent to see what changed (prompt hash, model)
  * audit which prompt version produced a given output
  * see per-stage wall-clock breakdown (where did time go?)

Workspace resolution (where `run.json` lands), in priority order:
  1. $LITSURVEY_RUN_DIR
  2. parent dir of $LITSURVEY_CORPUS_DIR
  3. current working directory (if it already has a `corpus/` subdir)
  4. ~/.litsurvey_corpus/

All three tools degrade gracefully: if `run.json` doesn't exist yet, later
tools return an `error` field rather than raising. The talent's persona
instructs the LLM to call them in order.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------


def _workspace_dir() -> Path:
    """Where run.json lives. Resolves via the cascade above."""
    p = os.getenv("LITSURVEY_RUN_DIR")
    if p:
        return Path(p).expanduser()
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        return Path(p).expanduser().parent
    if (Path.cwd() / "corpus").exists():
        return Path.cwd()
    return Path.home() / ".litsurvey_corpus"


def _run_json_path() -> Path:
    return _workspace_dir() / "run.json"


def _corpus_papers_path() -> Path:
    p = os.getenv("LITSURVEY_CORPUS_DIR")
    if p:
        return Path(p).expanduser() / "papers.jsonl"
    if (Path.cwd() / "corpus").exists():
        return Path.cwd() / "corpus" / "papers.jsonl"
    return Path.home() / ".litsurvey_corpus" / "papers.jsonl"


def _talent_root() -> Path | None:
    """Find the dir containing `prompts/` and `skills/` (for hashing).

    Honors $LITSURVEY_TALENT_ROOT, else walks up from this file's location.
    Returns None when neither prompts/ nor skills/ are found — hashing
    degrades to an empty dict in that case.
    """
    env = os.getenv("LITSURVEY_TALENT_ROOT")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve().parent
    for _ in range(5):
        if (here / "prompts").is_dir() and (here / "skills").is_dir():
            return here
        here = here.parent
    return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_prompt_files() -> dict[str, str]:
    """SHA-256 of every prompt/SKILL.md the talent uses. Empty dict if not found."""
    root = _talent_root()
    if root is None:
        return {}
    out: dict[str, str] = {}
    prompts_dir = root / "prompts"
    if prompts_dir.is_dir():
        for md in sorted(prompts_dir.glob("*.md")):
            out[f"prompts/{md.stem}"] = _sha256_text(md.read_text(encoding="utf-8"))
    skills_dir = root / "skills"
    if skills_dir.is_dir():
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            sf = skill_dir / "SKILL.md"
            if sf.exists():
                out[f"skills/{skill_dir.name}"] = _sha256_text(
                    sf.read_text(encoding="utf-8")
                )
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# run.json read/write
# ---------------------------------------------------------------------------


def _load_run_json() -> dict[str, Any]:
    p = _run_json_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_run_json(data: dict[str, Any]) -> None:
    p = _run_json_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def run_start(
    research_question: str,
    main_model: str = "",
    extract_model: str = "",
) -> dict[str, Any]:
    """Begin provenance tracking. Call this BEFORE any retrieval/extraction work.

    Writes a partial `run.json` containing:
      - a fresh run_id (UUID hex)
      - SHA-256 of the research question (canonicalised: stripped)
      - SHA-256 of every prompt and SKILL.md file the talent uses
      - model identifiers (passed in, or read from env vars)
      - started_at timestamp
      - empty stages[] and output_paths[] to be filled by later calls

    Args:
        research_question: The Stage 2 input research question. Hashed for diff.
        main_model: e.g. "claude-sonnet-4-5". If empty, reads $LITSURVEY_MAIN_MODEL.
        extract_model: e.g. "gpt-4o-mini". If empty, reads $LITSURVEY_EXTRACT_MODEL.

    Returns:
        {
          "run_id": "<uuid hex>",
          "run_json_path": "/path/to/run.json",
          "started_at": "ISO8601 UTC",
          "prompt_hash_count": N
        }
    """
    run_id = uuid.uuid4().hex
    prompt_hashes = _hash_prompt_files()
    data: dict[str, Any] = {
        "schema_version": "1",
        "run_id": run_id,
        "started_at": _now_iso(),
        "completed_at": None,
        "research_question_sha256": _sha256_text(research_question.strip()),
        "research_question_preview": research_question.strip()[:200],
        "prompt_hashes": prompt_hashes,
        "models": {
            "main": main_model or os.getenv("LITSURVEY_MAIN_MODEL", "unknown"),
            "extract": extract_model or os.getenv("LITSURVEY_EXTRACT_MODEL", "unknown"),
        },
        "stages": [],
        "corpus_size_final": None,
        "output_paths": [],
        "notes": [],
    }
    _write_run_json(data)
    return {
        "run_id": run_id,
        "run_json_path": str(_run_json_path()),
        "started_at": data["started_at"],
        "prompt_hash_count": len(prompt_hashes),
    }


@tool
def run_stage_done(
    stage: str,
    elapsed_s: float,
    notes: str = "",
) -> dict[str, Any]:
    """Record completion of one workflow stage with wall-clock duration.

    Call after each major stage (search / pdf_extract / extract_claims /
    conflict / verify / etc.) so the final run.json contains a timing
    breakdown. Multiple calls for the same stage name are allowed (e.g.
    if you call extract_claims in a loop, log each round separately).

    Args:
        stage: Stage name. Free-form, but stick to a small vocabulary
            (search, filter, pdf_extract, claim_extract, conflict_detect,
            verify, render).
        elapsed_s: Wall-clock seconds the stage took.
        notes: Optional free-text note (e.g. "3 retries on rate limit",
            "skipped — corpus already had 28 papers").

    Returns:
        {"stages_recorded": N, "total_elapsed_s": float}
    """
    data = _load_run_json()
    if not data:
        return {
            "error": "no run.json present — call run_start first",
            "stages_recorded": 0,
            "total_elapsed_s": 0.0,
        }

    entry: dict[str, Any] = {
        "name": stage,
        "elapsed_s": round(float(elapsed_s), 3),
        "recorded_at": _now_iso(),
    }
    if notes:
        entry["notes"] = notes

    data.setdefault("stages", []).append(entry)
    _write_run_json(data)

    total = sum(s["elapsed_s"] for s in data["stages"])
    return {
        "stages_recorded": len(data["stages"]),
        "total_elapsed_s": round(total, 3),
    }


@tool
def run_finalize(output_paths: list[str] | None = None) -> dict[str, Any]:
    """Finalize run.json: stamp completed_at, snapshot corpus, record outputs.

    Call AFTER all stages are done and stage2.json + markdown are written.
    Idempotent (calling twice just overwrites completed_at; stages remain).

    Args:
        output_paths: List of paths the talent produced (e.g. ["stage2.json",
            "stage2_literature_surveyor.md"]). Stored verbatim.

    Returns:
        {
          "run_id": "...",
          "completed_at": "ISO8601",
          "total_elapsed_s": float,
          "stages_recorded": N,
          "corpus_size_final": int | None,
          "run_json_path": "..."
        }
    """
    data = _load_run_json()
    if not data:
        return {"error": "no run.json present — call run_start first"}

    data["completed_at"] = _now_iso()
    if output_paths is not None:
        data["output_paths"] = [str(p) for p in output_paths]

    papers_path = _corpus_papers_path()
    if papers_path.exists():
        try:
            with papers_path.open(encoding="utf-8") as f:
                data["corpus_size_final"] = sum(1 for line in f if line.strip())
        except OSError:
            pass

    _write_run_json(data)

    total = sum(s["elapsed_s"] for s in data.get("stages", []))
    return {
        "run_id": data.get("run_id"),
        "completed_at": data["completed_at"],
        "total_elapsed_s": round(total, 3),
        "stages_recorded": len(data.get("stages", [])),
        "corpus_size_final": data.get("corpus_size_final"),
        "run_json_path": str(_run_json_path()),
    }
