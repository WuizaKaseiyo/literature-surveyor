"""Tests for A2 layered corpus mode (cross-project sharing).

Layered mode is opt-in via LITSURVEY_GLOBAL_CORPUS_DIR. When set:
- Paper entities live in `<global>/papers.jsonl` (deduped by id)
- Each project records its references in `<project>/refs.jsonl`
- Searching/listing/status operate at the project scope by default but can
  cross-cut into global via the `scope` parameter

These tests cover the contract; tests using the legacy `isolated_corpus`
fixture confirm the default mode is unchanged (see `test_corpus_store.py`).
"""

from __future__ import annotations

import json

import pytest

from tools.corpus_store.corpus_store import (
    corpus_add_paper,
    corpus_get_paper,
    corpus_list_papers,
    corpus_search,
    corpus_status,
)


def _paper(pid: str, title: str, **kw) -> dict:
    base = {
        "id": pid,
        "title": title,
        "authors": ["A. Author"],
        "year": 2024,
        "venue": "arxiv",
        "abstract": "We study attention pruning in small language models.",
        "source": "arxiv",
        "source_query": kw.pop("source_query", "small models attention"),
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Add behavior
# ---------------------------------------------------------------------------


def test_add_writes_global_entity_and_project_ref(layered_corpus):
    project_dir, global_dir = layered_corpus
    r = corpus_add_paper.invoke({"paper": _paper("2401.00001", "Test paper 1")})
    assert r["status"] == "added"
    assert r["global_status"] == "added"
    assert r["project_status"] == "added_ref"
    assert r["corpus_size"] == 1            # project view
    assert r["global_corpus_size"] == 1

    # Files exist where expected
    assert (global_dir / "papers.jsonl").exists()
    assert (project_dir / "refs.jsonl").exists()
    assert (project_dir / "project_meta.json").exists()

    # Global has the full paper
    papers_in_global = [
        json.loads(l) for l in (global_dir / "papers.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(papers_in_global) == 1
    assert papers_in_global[0]["title"] == "Test paper 1"

    # Project has only a ref (no full paper duplicated)
    refs = [
        json.loads(l) for l in (project_dir / "refs.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(refs) == 1
    assert refs[0]["paper_id"] == "2401.00001"
    assert refs[0]["source_query"] == "small models attention"


def test_add_idempotent_global_but_appends_ref_for_distinct_query(layered_corpus):
    """Same paper found by two different queries → 1 global entity, 2 ref rows."""
    project_dir, global_dir = layered_corpus
    corpus_add_paper.invoke({
        "paper": _paper("2401.00001", "P1", source_query="query A"),
    })
    r2 = corpus_add_paper.invoke({
        "paper": _paper("2401.00001", "P1", source_query="query B"),
    })
    assert r2["global_status"] == "skipped_duplicate"
    assert r2["project_status"] == "added_ref"

    papers_in_global = [
        json.loads(l) for l in (global_dir / "papers.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(papers_in_global) == 1, "global should not duplicate by id"
    refs = [
        json.loads(l) for l in (project_dir / "refs.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(refs) == 2, "audit trail should preserve both query rows"
    assert {refs[0]["source_query"], refs[1]["source_query"]} == {"query A", "query B"}


# ---------------------------------------------------------------------------
# Cross-project sharing
# ---------------------------------------------------------------------------


def test_two_projects_share_global_pool(two_layered_projects, monkeypatch):
    """Adding the same paper from two projects: 1 global, 1 ref per project."""
    project_a, project_b, global_dir = two_layered_projects

    # Project A adds
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper("2401.00001", "Shared paper")})

    # Project B adds same paper
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    r = corpus_add_paper.invoke({"paper": _paper("2401.00001", "Shared paper")})
    assert r["global_status"] == "skipped_duplicate"
    assert r["project_status"] == "added_ref"

    # Global has 1 paper
    papers_in_global = [
        json.loads(l) for l in (global_dir / "papers.jsonl").read_text().splitlines() if l.strip()
    ]
    assert len(papers_in_global) == 1

    # Each project has its own ref
    refs_a = [json.loads(l) for l in (project_a / "refs.jsonl").read_text().splitlines() if l.strip()]
    refs_b = [json.loads(l) for l in (project_b / "refs.jsonl").read_text().splitlines() if l.strip()]
    assert len(refs_a) == 1
    assert len(refs_b) == 1
    assert refs_a[0]["paper_id"] == refs_b[0]["paper_id"] == "2401.00001"


def test_project_only_sees_its_refs_via_status(two_layered_projects, monkeypatch):
    """Each project's corpus_status returns its own ref count, with shared global size."""
    project_a, project_b, _ = two_layered_projects

    # A adds 2 papers, B adds 1 different paper
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper("p1", "P one")})
    corpus_add_paper.invoke({"paper": _paper("p2", "P two")})

    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    corpus_add_paper.invoke({"paper": _paper("p3", "P three")})

    status_b = corpus_status.invoke({})
    assert status_b["mode"] == "layered"
    assert status_b["corpus_size"] == 1, "project B should see only its 1 ref"
    assert status_b["global_corpus_size"] == 3, "global should see all 3"

    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    status_a = corpus_status.invoke({})
    assert status_a["corpus_size"] == 2
    assert status_a["global_corpus_size"] == 3


# ---------------------------------------------------------------------------
# Search scope behavior
# ---------------------------------------------------------------------------


def test_search_scope_project_filters_to_refs(two_layered_projects, monkeypatch):
    """scope=project should only return papers this project has ref'd."""
    project_a, project_b, _ = two_layered_projects

    # A has p1, B has p2 — both globally indexed
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper("p1", "Attention in small LMs")})

    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    corpus_add_paper.invoke({"paper": _paper("p2", "Attention in large LMs")})

    # From project B, scope=project should only see p2
    res_proj = corpus_search.invoke({"query": "attention", "scope": "project"})
    ids = {r["paper"]["id"] for r in res_proj["results"]}
    assert ids == {"p2"}, f"expected only project B's p2, got {ids}"
    assert res_proj["scope"] == "project"

    # scope=global should see both
    res_global = corpus_search.invoke({"query": "attention", "scope": "global"})
    ids_global = {r["paper"]["id"] for r in res_global["results"]}
    assert ids_global == {"p1", "p2"}

    # scope=both (default) returns all with from_project tag
    res_both = corpus_search.invoke({"query": "attention"})
    for r in res_both["results"]:
        assert "from_project" in r
        assert r["from_project"] == (r["paper"]["id"] == "p2")


def test_list_papers_scope_project_default(two_layered_projects, monkeypatch):
    project_a, project_b, _ = two_layered_projects

    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper("p1", "P1")})
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    corpus_add_paper.invoke({"paper": _paper("p2", "P2")})

    # B's default list = only B's papers
    listing = corpus_list_papers.invoke({})
    ids = {p["id"] for p in listing["papers"]}
    assert ids == {"p2"}

    # global scope = both
    listing_g = corpus_list_papers.invoke({"scope": "global"})
    ids_g = {p["id"] for p in listing_g["papers"]}
    assert ids_g == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Entity-fetching tools transparently read from global
# ---------------------------------------------------------------------------


def test_corpus_get_paper_reads_global_regardless_of_project(two_layered_projects, monkeypatch):
    """get_paper should resolve by id no matter which project is active."""
    project_a, project_b, _ = two_layered_projects

    # A adds p1
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper("p1", "Project A paper")})

    # Switch to B (which has not ref'd p1) — get_paper still works
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    out = corpus_get_paper.invoke({"paper_id": "p1"})
    assert out["title"] == "Project A paper"


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def test_mode_label_in_status_layered(layered_corpus):
    out = corpus_status.invoke({})
    assert out["mode"] == "layered"
    assert "global_dir" in out
    assert "project_dir" in out


def test_mode_label_absent_in_legacy(isolated_corpus):
    """Legacy mode does not include the 'mode' key (preserving old return shape)."""
    out = corpus_status.invoke({})
    assert "mode" not in out
    assert "global_corpus_size" not in out


# ---------------------------------------------------------------------------
# Cross-project claim reuse (the real cost-savings of A2)
# ---------------------------------------------------------------------------


def test_extract_claims_reuses_existing(layered_corpus):
    """If claims for paper_id already exist in global, extract_claims
    short-circuits and returns them — no LLM call needed."""
    from tools.extract_claims.extract_claims import extract_claims

    project_dir, global_dir = layered_corpus
    pid = "2401.99001"
    corpus_add_paper.invoke({"paper": _paper(pid, "Reused-claim paper")})

    # Pre-seed claims.jsonl in global as if a prior project extracted them
    claims_path = global_dir / "claims.jsonl"
    pre_existing = [
        {
            "id": f"{pid}#claim-1",
            "paper_id": pid,
            "claim_text": "Pre-extracted claim 1.",
            "claim_type": "factual",
            "evidence_span": "Section 3",
            "evidence_quote": "Pre-extracted claim 1.",
            "source_section": "Section 3",
            "confidence": 0.8,
            "applies_to": "x",
            "version": "v2",
        },
        {
            "id": f"{pid}#claim-2",
            "paper_id": pid,
            "claim_text": "Pre-extracted claim 2.",
            "claim_type": "factual",
            "evidence_span": "Section 4",
            "evidence_quote": "Pre-extracted claim 2.",
            "source_section": "Section 4",
            "confidence": 0.7,
            "applies_to": "x",
            "version": "v2",
        },
    ]
    with claims_path.open("w") as f:
        for c in pre_existing:
            f.write(json.dumps(c) + "\n")

    # Call extract_claims with NO API key set — short-circuit should return cached
    out = extract_claims.invoke({"paper_id": pid, "version": "v2"})
    assert out["status"] == "reused"
    assert out["claims_extracted"] == 2
    assert out["claims"][0]["claim_text"] == "Pre-extracted claim 1."


def test_extract_claims_force_bypasses_cache(layered_corpus):
    """force=True must re-attempt extraction even when cached claims exist.
    Without an API key the re-extraction returns an error — that's the proof
    the short-circuit was bypassed."""
    from tools.extract_claims.extract_claims import extract_claims

    project_dir, global_dir = layered_corpus
    pid = "2401.99002"
    corpus_add_paper.invoke({"paper": _paper(pid, "Forced re-extract paper")})

    (global_dir / "claims.jsonl").write_text(
        json.dumps({
            "id": f"{pid}#claim-1",
            "paper_id": pid,
            "claim_text": "cached",
            "claim_type": "factual",
            "evidence_span": "S1",
            "evidence_quote": "cached",
            "source_section": "S1",
            "confidence": 0.5,
            "applies_to": "",
            "version": "v1",
        }) + "\n"
    )

    # Defensive: ensure no LLM key leaks into this test
    import os
    saved = {
        k: os.environ.pop(k, None)
        for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY")
    }
    try:
        out = extract_claims.invoke({"paper_id": pid, "version": "v1", "force": True})
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

    # The cached path was bypassed → re-extraction tried → LLM unavailable → error
    assert out.get("status") != "reused"
    assert out.get("error"), out


def test_claim_store_sees_cross_project_claims(two_layered_projects, monkeypatch):
    """A claim extracted from project A's perspective is searchable from
    project B's perspective — exactly because layered mode puts claims in
    the shared global store."""
    from tools.claim_store.claim_store import claim_list_by_paper

    project_a, project_b, global_dir = two_layered_projects
    pid = "2401.99003"

    # Project A: add paper + write a claim to global claims.jsonl
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    corpus_add_paper.invoke({"paper": _paper(pid, "Cross-project claim paper")})
    claim = {
        "id": f"{pid}#claim-1",
        "paper_id": pid,
        "claim_text": "Sharing across projects works.",
        "claim_type": "factual",
        "evidence_span": "S1",
        "evidence_quote": "Sharing across projects works.",
        "source_section": "S1",
        "confidence": 0.9,
        "applies_to": "",
        "version": "v2",
    }
    (global_dir / "claims.jsonl").write_text(json.dumps(claim) + "\n")

    # Project B: switch and query the same claim
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_b))
    out = claim_list_by_paper.invoke({"paper_id": pid})
    assert out["count"] == 1
    assert out["claims"][0]["claim_text"] == "Sharing across projects works."
