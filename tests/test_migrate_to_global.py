"""Tests for the legacy → global corpus migration script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.corpus_store.migrate_to_global import migrate


def _seed_legacy_corpus(corpus_dir: Path) -> None:
    papers = [
        {
            "id": "2401.10001",
            "title": "Legacy paper 1",
            "abstract": "abstract 1",
            "source": "arxiv",
            "source_query": "old query 1",
            "added_at": 1700000000.0,
        },
        {
            "id": "2401.10002",
            "title": "Legacy paper 2",
            "abstract": "abstract 2",
            "source": "arxiv",
            "source_query": "old query 2",
            "added_at": 1700000010.0,
        },
    ]
    claims = [
        {
            "id": "2401.10001#claim-1",
            "paper_id": "2401.10001",
            "claim_text": "Legacy claim 1.",
            "claim_type": "factual",
            "evidence_quote": "Legacy claim 1.",
            "evidence_span": "S1",
            "source_section": "S1",
            "confidence": 0.7,
            "applies_to": "",
            "version": "v1",
        },
    ]
    with (corpus_dir / "papers.jsonl").open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    with (corpus_dir / "claims.jsonl").open("w") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")


def test_migrate_promotes_and_writes_refs(tmp_path):
    legacy_dir = tmp_path / "legacy_project_corpus"
    legacy_dir.mkdir()
    global_dir = tmp_path / "global_corpus"

    _seed_legacy_corpus(legacy_dir)

    summary = migrate(source=legacy_dir, global_dir=global_dir, dry_run=False)
    assert summary["papers_promoted"] == 2
    assert summary["claims_promoted"] == 1

    # Global got the entities
    global_papers = [
        json.loads(l) for l in (global_dir / "papers.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(global_papers) == 2
    global_claims = [
        json.loads(l) for l in (global_dir / "claims.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(global_claims) == 1

    # Project got refs.jsonl
    refs = [
        json.loads(l) for l in (legacy_dir / "refs.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(refs) == 2
    assert {r["paper_id"] for r in refs} == {"2401.10001", "2401.10002"}
    assert all(r["migrated_from"].endswith("papers.jsonl") for r in refs)

    # project_meta.json got created
    meta = json.loads((legacy_dir / "project_meta.json").read_text())
    assert meta["migrated_from_legacy"] is True
    assert "project_id" in meta

    # Legacy entity files renamed (not deleted)
    assert not (legacy_dir / "papers.jsonl").exists()
    assert not (legacy_dir / "claims.jsonl").exists()
    bak = list(legacy_dir.glob("papers.jsonl.bak.*"))
    assert len(bak) == 1


def test_migrate_dry_run_writes_nothing(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    global_dir = tmp_path / "global"

    _seed_legacy_corpus(legacy_dir)
    summary = migrate(source=legacy_dir, global_dir=global_dir, dry_run=True)
    assert summary["papers_promoted"] == 2
    assert summary["dry_run"] is True

    # No writes happened
    assert not (global_dir / "papers.jsonl").exists()
    assert not (legacy_dir / "refs.jsonl").exists()
    assert (legacy_dir / "papers.jsonl").exists()  # original untouched


def test_migrate_idempotent_skips_existing_global(tmp_path):
    """Re-running after a previous migration shouldn't re-promote."""
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    global_dir = tmp_path / "global"

    _seed_legacy_corpus(legacy_dir)
    migrate(source=legacy_dir, global_dir=global_dir)

    # Re-seed legacy (simulate another stale project) and re-run
    _seed_legacy_corpus(legacy_dir)
    second = migrate(source=legacy_dir, global_dir=global_dir)
    assert second["papers_promoted"] == 0, "global already has these papers"
    assert second["papers_already_in_global"] == 2
    # But refs.jsonl is overwritten with current legacy state
    refs = [
        json.loads(l) for l in (legacy_dir / "refs.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(refs) == 2


def test_migrate_no_source(tmp_path):
    legacy = tmp_path / "does_not_exist"
    out = migrate(source=legacy, global_dir=tmp_path / "global")
    assert out.get("error") == "source_missing"


def test_migrated_corpus_works_in_layered_mode(tmp_path, monkeypatch):
    """After migration, setting both envs should let corpus_store see the
    migrated paper as a project ref + global entity."""
    from tools.corpus_store.corpus_store import corpus_status, corpus_list_papers

    legacy_dir = tmp_path / "legacy_project"
    legacy_dir.mkdir()
    global_dir = tmp_path / "global"
    _seed_legacy_corpus(legacy_dir)

    migrate(source=legacy_dir, global_dir=global_dir)

    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(legacy_dir))
    monkeypatch.setenv("LITSURVEY_GLOBAL_CORPUS_DIR", str(global_dir))

    status = corpus_status.invoke({})
    assert status["mode"] == "layered"
    assert status["corpus_size"] == 2, "project's 2 refs should appear"
    assert status["global_corpus_size"] == 2

    listed = corpus_list_papers.invoke({"scope": "project"})
    ids = {p["id"] for p in listed["papers"]}
    assert ids == {"2401.10001", "2401.10002"}
