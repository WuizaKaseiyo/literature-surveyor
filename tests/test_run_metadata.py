"""Tests for run_metadata — pure I/O + hashing, no LLM call."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tools.run_metadata.run_metadata import (
    _hash_prompt_files,
    _run_json_path,
    _sha256_text,
    _talent_root,
    run_finalize,
    run_stage_done,
    run_start,
)


# ---------------------------------------------------------------------------
# Workspace fixture — share `isolated_corpus` from conftest. Workspace becomes
# parent of the corpus dir, so run.json lands one level up from corpus/.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_workspace(isolated_corpus, monkeypatch):
    """Set up clean workspace + corpus dirs, and a minimal talent tree to hash."""
    workspace = isolated_corpus.parent
    # Stand up a tiny prompts/ + skills/ tree the hash walker can find.
    (workspace / "prompts").mkdir()
    (workspace / "prompts" / "talent_persona.md").write_text("persona v1", encoding="utf-8")
    (workspace / "skills" / "systematic-review").mkdir(parents=True)
    (workspace / "skills" / "systematic-review" / "SKILL.md").write_text(
        "skill v1", encoding="utf-8"
    )
    monkeypatch.setenv("LITSURVEY_TALENT_ROOT", str(workspace))
    return workspace


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_sha256_text_deterministic():
    assert _sha256_text("hello") == _sha256_text("hello")
    assert _sha256_text("hello") != _sha256_text("world")


def test_hash_prompt_files_picks_up_md_files(isolated_workspace):
    hashes = _hash_prompt_files()
    assert "prompts/talent_persona" in hashes
    assert "skills/systematic-review" in hashes
    # Both are 64-char hex (sha-256).
    assert all(len(h) == 64 for h in hashes.values())


def test_hash_prompt_files_changes_when_content_changes(isolated_workspace):
    h1 = _hash_prompt_files()
    (isolated_workspace / "prompts" / "talent_persona.md").write_text(
        "persona v2 EDITED", encoding="utf-8"
    )
    h2 = _hash_prompt_files()
    assert h1["prompts/talent_persona"] != h2["prompts/talent_persona"]
    # Other hashes unchanged.
    assert h1["skills/systematic-review"] == h2["skills/systematic-review"]


# ---------------------------------------------------------------------------
# run_start
# ---------------------------------------------------------------------------


def test_run_start_creates_run_json(isolated_workspace):
    result = run_start.invoke({
        "research_question": "What are RLHF methods for small LMs?",
        "main_model": "claude-sonnet-4-5",
        "extract_model": "gpt-4o-mini",
    })

    assert "run_id" in result
    assert len(result["run_id"]) == 32  # uuid hex
    assert result["prompt_hash_count"] >= 2  # at least the two we stood up

    path = Path(result["run_json_path"])
    assert path.exists()
    assert path == _run_json_path()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert data["models"]["main"] == "claude-sonnet-4-5"
    assert data["models"]["extract"] == "gpt-4o-mini"
    assert data["stages"] == []
    assert data["completed_at"] is None
    assert data["research_question_preview"].startswith("What are RLHF")
    assert len(data["research_question_sha256"]) == 64


def test_run_start_falls_back_to_env_for_models(isolated_workspace, monkeypatch):
    monkeypatch.setenv("LITSURVEY_MAIN_MODEL", "env-main-model")
    monkeypatch.setenv("LITSURVEY_EXTRACT_MODEL", "env-extract-model")

    run_start.invoke({"research_question": "x" * 30})
    data = json.loads(_run_json_path().read_text(encoding="utf-8"))
    assert data["models"]["main"] == "env-main-model"
    assert data["models"]["extract"] == "env-extract-model"


# ---------------------------------------------------------------------------
# run_stage_done
# ---------------------------------------------------------------------------


def test_run_stage_done_appends_and_sums(isolated_workspace):
    run_start.invoke({"research_question": "anything long enough"})
    r1 = run_stage_done.invoke({"stage": "search", "elapsed_s": 45.2})
    r2 = run_stage_done.invoke({"stage": "extract_claims", "elapsed_s": 312.5})

    assert r1["stages_recorded"] == 1
    assert r2["stages_recorded"] == 2
    assert r2["total_elapsed_s"] == pytest.approx(357.7)

    data = json.loads(_run_json_path().read_text(encoding="utf-8"))
    assert [s["name"] for s in data["stages"]] == ["search", "extract_claims"]
    assert [s["elapsed_s"] for s in data["stages"]] == [45.2, 312.5]


def test_run_stage_done_accepts_optional_notes(isolated_workspace):
    run_start.invoke({"research_question": "anything long enough"})
    run_stage_done.invoke({"stage": "search", "elapsed_s": 10.0, "notes": "3 retries"})

    data = json.loads(_run_json_path().read_text(encoding="utf-8"))
    assert data["stages"][0]["notes"] == "3 retries"


def test_run_stage_done_without_run_start_returns_error(isolated_workspace):
    # Note: no run_start called.
    result = run_stage_done.invoke({"stage": "search", "elapsed_s": 10.0})
    assert "error" in result
    assert result["stages_recorded"] == 0


# ---------------------------------------------------------------------------
# run_finalize
# ---------------------------------------------------------------------------


def test_run_finalize_stamps_completed_at(isolated_workspace):
    run_start.invoke({"research_question": "anything long enough"})
    run_stage_done.invoke({"stage": "search", "elapsed_s": 10.0})
    time.sleep(0.01)
    result = run_finalize.invoke({
        "output_paths": ["stage2.json", "stage2_literature_surveyor.md"]
    })

    assert result["completed_at"] is not None
    assert result["total_elapsed_s"] == pytest.approx(10.0)
    assert result["stages_recorded"] == 1

    data = json.loads(_run_json_path().read_text(encoding="utf-8"))
    assert data["completed_at"] is not None
    assert data["output_paths"] == ["stage2.json", "stage2_literature_surveyor.md"]


def test_run_finalize_snapshots_corpus_size(isolated_workspace, fake_paper):
    from tools.corpus_store.corpus_store import corpus_add_paper

    run_start.invoke({"research_question": "anything long enough"})
    for i in range(3):
        p = dict(fake_paper)
        p["id"] = f"2401.{1000+i}"
        corpus_add_paper.invoke({"paper": p})

    result = run_finalize.invoke({})
    assert result["corpus_size_final"] == 3


def test_run_finalize_without_run_start_returns_error(isolated_workspace):
    result = run_finalize.invoke({})
    assert "error" in result


# ---------------------------------------------------------------------------
# End-to-end mini lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_produces_complete_run_json(isolated_workspace):
    """Exercise the documented 3-call sequence and confirm run.json shape."""
    rs = run_start.invoke({
        "research_question": "Methods and bottlenecks for RLHF on small LMs.",
        "main_model": "claude-sonnet-4-5",
        "extract_model": "gpt-4o-mini",
    })

    for stage, t in [
        ("search", 45.2),
        ("filter", 8.1),
        ("pdf_extract", 22.8),
        ("claim_extract", 312.5),
        ("conflict_detect", 28.1),
        ("verify", 12.0),
        ("render", 4.2),
    ]:
        run_stage_done.invoke({"stage": stage, "elapsed_s": t})

    fin = run_finalize.invoke({
        "output_paths": ["stage2.json", "stage2_literature_surveyor.md"]
    })

    assert fin["run_id"] == rs["run_id"]
    assert fin["stages_recorded"] == 7
    assert fin["total_elapsed_s"] == pytest.approx(432.9)

    data = json.loads(_run_json_path().read_text(encoding="utf-8"))
    # Stage order preserved.
    assert [s["name"] for s in data["stages"]] == [
        "search", "filter", "pdf_extract", "claim_extract",
        "conflict_detect", "verify", "render",
    ]
    # Provenance fields present.
    assert data["models"]["main"] == "claude-sonnet-4-5"
    assert data["models"]["extract"] == "gpt-4o-mini"
    assert "prompts/talent_persona" in data["prompt_hashes"]


# ---------------------------------------------------------------------------
# Talent-root resolution fallback
# ---------------------------------------------------------------------------


def test_hash_prompt_files_returns_empty_when_no_talent_root(monkeypatch, tmp_path):
    """If LITSURVEY_TALENT_ROOT points to a dir with no prompts/skills, return {}."""
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv("LITSURVEY_TALENT_ROOT", str(bare))
    assert _hash_prompt_files() == {}
