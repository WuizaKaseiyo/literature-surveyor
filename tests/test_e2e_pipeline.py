"""End-to-end pipeline test (offline, no LLM, no network).

Exercises the full data plumbing from a populated corpus to a validated
rendered survey, mirroring what `systematic-review/SKILL.md` step 6 → step 9
prescribes. Each step's output feeds the next:

    corpus_status
        → claim_search (find backing claims for a topic)
        → build Finding with claim_ids from the search
        → serialize LiteratureSurveySchema → JSON
        → render markdown using G段 evidence-footnote convention
        → verify_citations (resolves all cites to local corpus → 0 unverified)
        → fact_check_rendered_survey (matched_claim_id ⊆ Finding.claim_ids → 0 blocking)

Uses the existing `tests/fixtures/factcheck_fixture/corpus/` (5 papers, ~70
real DS-V3-extracted claims) copied to a fresh tmp dir per test so the fixture
stays read-only. No external services or LLMs invoked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from schemas.literature_survey_schema import (
    CitationRef,
    CorpusSummary,
    Finding,
    LiteratureSurveySchema,
    SearchStrategy,
)
from tools.claim_store.claim_store import claim_search, claim_status
from tools.corpus_store.corpus_store import corpus_status
from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
    fact_check_rendered_survey,
)
from tools.verify_citations.verify_citations import verify_citations

FIXTURE_CORPUS = (
    Path(__file__).parent / "fixtures" / "factcheck_fixture" / "corpus"
)


@pytest.fixture
def e2e_corpus(tmp_path, monkeypatch):
    """Copy the factcheck_fixture corpus to an isolated tmp dir + point env there."""
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(FIXTURE_CORPUS, corpus_dir)
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(corpus_dir))
    return corpus_dir


def test_pipeline_corpus_has_expected_inputs(e2e_corpus):
    """Sanity check: the fixture corpus we're about to drive the pipeline with is intact."""
    cs = corpus_status.invoke({})
    assert cs["corpus_size"] == 5, "expected 5 papers in fixture"
    assert cs["claims_count"] > 50, "expected ~70 claims in fixture"
    cls = claim_status.invoke({})
    assert cls["claims_count"] == cs["claims_count"]
    assert cls["papers_with_claims"] == 5


def test_pipeline_end_to_end_offline(e2e_corpus):
    """Drive the pipeline from corpus → finding → render → verify → fact_check.

    Asserts that:
      1. claim_search returns backing claims for a known topic
      2. LiteratureSurveySchema accepts a Finding with claim_ids from that search
      3. verify_citations resolves the cited paper in the corpus (no API)
      4. fact_check finds matched_claim_id ⊆ Finding.claim_ids
      5. blocking_count == 0
    """
    # --- Step 7.5 (from SKILL.md): find backing claims for a candidate finding ---
    finding_text = "DPO matches PPO-based RLHF in summarization quality."
    target_paper = "2305.18290"

    backing = claim_search.invoke({
        "query": "DPO PPO summarization quality response",
        "paper_ids": [target_paper],
        "top_k": 3,
    })
    assert backing["results"], "claim_search should find at least 1 backing claim"
    backing_claim_ids = [r["claim"]["id"] for r in backing["results"]]
    top_claim = backing["results"][0]["claim"]

    # --- Step 8: build Finding + schema, serialize JSON ---
    finding = Finding(
        text=finding_text,
        cites=[CitationRef(
            paper_id=target_paper,
            cite_text=f"Rafailov et al. 2023, arxiv:{target_paper}",
        )],
        claim_ids=backing_claim_ids[:2],
        confidence=0.85,
    )

    survey = LiteratureSurveySchema(
        research_question="Effectiveness of preference optimization in alignment.",
        search_strategy=SearchStrategy(
            queries=["DPO PPO preference optimization"],
            sources=["arxiv"],
        ),
        corpus_summary=CorpusSummary(total_papers=5),
        taxonomy=[],
        findings=[finding],
    )
    # Schema must serialize without errors
    survey_json = survey.model_dump_json()
    assert json.loads(survey_json)["findings"][0]["claim_ids"] == backing_claim_ids[:2]

    # --- Step 8b: render markdown per G段 convention ---
    rendered = (
        f"# Stage 2: Literature Survey\n\n"
        f"## Findings\n\n"
        f"{finding.text[:-1]} [Rafailov et al. 2023, arxiv:{target_paper}].\n"
        f"\n"
        f"> evidence: \"{top_claim['evidence_quote']}\"\n"
        f"> — {top_claim.get('source_section', 'Section')} "
        f"(arxiv:{target_paper}#{top_claim['id'].split('#')[-1]})\n"
    )

    # --- Step 9a: verify_citations (corpus-only, no network) ---
    cite_res = verify_citations.invoke({"markdown": rendered})
    assert cite_res["unverified_count"] == 0, (
        f"verify_citations failed: {cite_res['unverified']}"
    )
    assert cite_res["verified_count"] == 1
    assert cite_res["verified"][0]["id"] == target_paper

    # --- Step 9b: fact_check (no LLM) ---
    fc_res = fact_check_rendered_survey.invoke({
        "markdown": rendered,
        "use_llm": False,
    })
    # Footnote must not be double-counted
    assert fc_res["total_checked"] == 1, fc_res["items"]
    item = fc_res["items"][0]

    # The matched_claim_id from fact_check must be one of the Finding.claim_ids
    # (the G段 cross-check that step 9 in SKILL.md prescribes)
    assert item["matched_claim_id"], "fact_check should have anchored to an extracted claim"
    assert item["matched_claim_id"] in finding.claim_ids, (
        f"cross-check failed: fact_check anchored {item['matched_claim_id']!r} "
        f"but Finding.claim_ids = {finding.claim_ids}"
    )

    # No blocking verdicts on a well-formed survey
    assert fc_res["blocking_count"] == 0, (
        f"blocking_count={fc_res['blocking_count']} on well-formed survey: {item}"
    )


@pytest.fixture
def e2e_corpus_layered(tmp_path, monkeypatch):
    """Same fixture as e2e_corpus but in A2 layered mode.

    Splits the factcheck_fixture's papers + claims into a global dir and
    creates an empty project dir + refs covering all 5 papers, so the project
    sees the same corpus through the layered abstraction.
    """
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()

    # Copy entities to global
    shutil.copyfile(FIXTURE_CORPUS / "papers.jsonl", global_dir / "papers.jsonl")
    shutil.copyfile(FIXTURE_CORPUS / "claims.jsonl", global_dir / "claims.jsonl")

    # Build refs in project for every paper id
    import json as _json
    paper_ids = []
    with (global_dir / "papers.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                paper_ids.append(_json.loads(line)["id"])
    with (project_dir / "refs.jsonl").open("w") as f:
        for pid in paper_ids:
            f.write(_json.dumps({
                "paper_id": pid,
                "source_query": "fixture",
                "found_via": "arxiv",
                "added_at": 0,
                "kept": True,
            }) + "\n")

    monkeypatch.setenv("LITSURVEY_GLOBAL_CORPUS_DIR", str(global_dir))
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_dir))
    return project_dir, global_dir


def test_pipeline_end_to_end_layered_mode(e2e_corpus_layered):
    """The same e2e pipeline, but in A2 layered mode. Validates that the
    cross-project data plumbing (global entities, project refs) is wired
    correctly across all 7 tools."""
    project_dir, global_dir = e2e_corpus_layered

    # Sanity: status reports layered mode + correct counts
    cs = corpus_status.invoke({})
    assert cs["mode"] == "layered"
    assert cs["corpus_size"] == 5
    assert cs["global_corpus_size"] == 5

    # claim_search reads from global
    finding_text = "DPO matches PPO-based RLHF in summarization quality."
    target_paper = "2305.18290"
    backing = claim_search.invoke({
        "query": "DPO PPO summarization quality response",
        "paper_ids": [target_paper],
        "top_k": 3,
    })
    assert backing["results"], "claim_search should find backing claims via global"
    backing_claim_ids = [r["claim"]["id"] for r in backing["results"]]
    top_claim = backing["results"][0]["claim"]

    finding = Finding(
        text=finding_text,
        cites=[CitationRef(
            paper_id=target_paper,
            cite_text=f"Rafailov et al. 2023, arxiv:{target_paper}",
        )],
        claim_ids=backing_claim_ids[:2],
        confidence=0.85,
    )
    survey = LiteratureSurveySchema(
        research_question="Effectiveness of preference optimization.",
        search_strategy=SearchStrategy(queries=["DPO"], sources=["arxiv"]),
        corpus_summary=CorpusSummary(total_papers=5),
        taxonomy=[],
        findings=[finding],
    )

    rendered = (
        f"# Stage 2 (layered-mode e2e)\n\n"
        f"{finding.text[:-1]} [Rafailov et al. 2023, arxiv:{target_paper}].\n\n"
        f"> evidence: \"{top_claim['evidence_quote']}\"\n"
        f"> — {top_claim.get('source_section', '?')} "
        f"(arxiv:{target_paper}#{top_claim['id'].split('#')[-1]})\n"
    )

    cite_res = verify_citations.invoke({"markdown": rendered})
    assert cite_res["unverified_count"] == 0, cite_res["unverified"]

    fc_res = fact_check_rendered_survey.invoke({"markdown": rendered, "use_llm": False})
    assert fc_res["total_checked"] == 1, fc_res["items"]
    item = fc_res["items"][0]
    assert item["matched_claim_id"] in finding.claim_ids, (
        f"layered-mode cross-check failed: matched={item['matched_claim_id']!r} "
        f"declared={finding.claim_ids}"
    )
    assert fc_res["blocking_count"] == 0


def test_pipeline_orphan_finding_caught_at_step_7_5(e2e_corpus):
    """A finding about a topic the corpus doesn't cover returns no usable backing.

    Documents the step 7.5 guard-rail: the *system* always returns BM25 results
    (even an unrelated query can hit one common stop-token-survivor and score
    >0), but `matched_terms` is the signal the talent should check. Few or
    weak matched_terms = not a real anchor → skip the finding.
    """
    # Query about a topic absent from the fixture (DPO paper doesn't cover quantum physics)
    out = claim_search.invoke({
        "query": "quantum entanglement bell inequality bohr nucleus",
        "paper_ids": ["2305.18290"],
        "top_k": 3,
    })
    # Either nothing comes back, or at most 1 incidental token overlap
    for r in out["results"]:
        assert len(r["matched_terms"]) <= 1, (
            f"unexpected multi-term match on unrelated query: {r['matched_terms']}"
        )

    # Compare to a related query — should produce much richer matched_terms
    related = claim_search.invoke({
        "query": "DPO PPO preference optimization reward model",
        "paper_ids": ["2305.18290"],
        "top_k": 3,
    })
    assert related["results"]
    assert len(related["results"][0]["matched_terms"]) >= 2, (
        "related query should match multiple terms"
    )
