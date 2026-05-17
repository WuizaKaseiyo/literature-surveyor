"""Tests for the LiteratureSurveySchema Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.literature_survey_schema import (
    AttributionAuditItem,
    AttributionAuditSummary,
    CitationRef,
    Conflict,
    CorpusSummary,
    Finding,
    Gap,
    LiteratureSurveySchema,
    OpenQuestion,
    SearchStrategy,
    TaxonomyNode,
)


def _minimal_survey() -> dict:
    return {
        "research_question": "Does X cause Y?",
        "search_strategy": {
            "queries": ["x y"],
            "sources": ["arxiv"],
            "time_window": "2020-2024",
            "inclusion_criteria": ["peer-reviewed"],
            "exclusion_criteria": ["position papers"],
        },
        "corpus_summary": {
            "total_papers": 5,
            "year_distribution": {"2023": 3, "2024": 2},
            "venue_distribution": {"NeurIPS": 2, "ICLR": 3},
            "source_distribution": {"arxiv": 5},
            "with_full_text": 4,
        },
        "taxonomy": [
            {"name": "Method A", "description": "...", "paper_ids": ["a"], "children": []}
        ],
        "findings": [
            {
                "text": "X causes Y in some setting",
                "cites": [{"paper_id": "a", "cite_text": "Smith 2024, arxiv:2401.1"}],
                "confidence": 0.8,
            }
        ],
    }


def test_minimal_valid():
    survey = LiteratureSurveySchema(**_minimal_survey())
    assert survey.research_question.startswith("Does")
    assert survey.corpus_summary.total_papers == 5


def test_missing_required_fails():
    bad = _minimal_survey()
    del bad["research_question"]
    with pytest.raises(ValidationError):
        LiteratureSurveySchema(**bad)


def test_finding_confidence_bounds():
    bad_finding = {
        "text": "x",
        "cites": [{"paper_id": "a", "cite_text": "..."}],
        "confidence": 1.5,  # out of [0,1]
    }
    with pytest.raises(ValidationError):
        Finding(**bad_finding)


def test_finding_claim_ids_default_empty():
    f = Finding(
        text="A bare finding without backing claim_ids.",
        cites=[{"paper_id": "a", "cite_text": "..."}],
    )
    assert f.claim_ids == []


def test_finding_claim_ids_accepts_list():
    f = Finding(
        text="A finding with two backing claims.",
        cites=[{"paper_id": "2305.18290", "cite_text": "Rafailov et al. 2023"}],
        claim_ids=["2305.18290#claim-3", "2305.18290#claim-7"],
    )
    assert f.claim_ids == ["2305.18290#claim-3", "2305.18290#claim-7"]


def test_conflict_level_enum():
    valid = Conflict(
        id="c1",
        level="direct",
        claim_a_paper_id="a",
        claim_a_text="...",
        claim_b_paper_id="b",
        claim_b_text="...",
        shared_setting="x",
        description="...",
    )
    assert valid.level == "direct"

    with pytest.raises(ValidationError):
        Conflict(
            id="c1",
            level="weird-thing",
            claim_a_paper_id="a",
            claim_a_text="...",
            claim_b_paper_id="b",
            claim_b_text="...",
            shared_setting="x",
            description="...",
        )


def test_gap_actionability_enum():
    g = Gap(
        id="g1",
        title="X gap",
        description="...",
        evidence_paper_ids=["a"],
        actionability="high",
        estimated_difficulty="medium",
    )
    assert g.actionability == "high"


def test_taxonomy_recursion():
    deep = TaxonomyNode(
        name="root",
        description="r",
        paper_ids=[],
        children=[
            TaxonomyNode(name="child", description="c", paper_ids=["x"], children=[])
        ],
    )
    assert deep.children[0].name == "child"


def test_serialize_roundtrip():
    survey = LiteratureSurveySchema(**_minimal_survey())
    json_str = survey.model_dump_json()
    parsed = LiteratureSurveySchema.model_validate_json(json_str)
    assert parsed.research_question == survey.research_question


def test_attribution_audit_schema():
    item = AttributionAuditItem(
        claim_text="X improves Y.",
        citation_raw="[Smith et al. 2024, arxiv:2401.12345]",
        citation_kind="arxiv",
        citation_id="2401.12345",
        verdict="supported",
        confidence=0.8,
    )
    audit = AttributionAuditSummary(supported_count=1, total_checked=1, items=[item])
    assert audit.items[0].verdict == "supported"

    bad = item.model_dump()
    bad["verdict"] = "looks_fine"
    with pytest.raises(ValidationError):
        AttributionAuditItem(**bad)
