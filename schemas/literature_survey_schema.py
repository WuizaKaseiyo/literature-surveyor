"""Pydantic schema for Stage 2 Literature Survey output.

The talent is expected to:
  1. Build an instance of LiteratureSurveySchema
  2. Write `stage2.json` (model.model_dump_json(indent=2))
  3. Render `stage2_literature_surveyor.md` from the schema (human-readable)
  4. Run verify_citations and fact_check_rendered_survey on the markdown

Downstream Stage 3 (idea_generator) consumes `stage2.json` directly — no markdown parsing.

This file is a REFERENCE — talent tools have their own local copies of any
fields they validate (each @tool .py file is self-contained because OMC's
install_talent_functions copies tool dirs into company/assets/tools/<name>/
and Python import paths don't carry over).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchStrategy(BaseModel):
    queries: list[str] = Field(description="The search queries actually run")
    sources: list[str] = Field(description="Which sources were queried")
    time_window: str = Field(default="", description="e.g. '2020-2025' or 'last 5 years'")
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class CorpusSummary(BaseModel):
    total_papers: int
    year_distribution: dict[str, int] = Field(default_factory=dict)
    venue_distribution: dict[str, int] = Field(default_factory=dict)
    source_distribution: dict[str, int] = Field(default_factory=dict)
    with_full_text: int = 0


class TaxonomyNode(BaseModel):
    name: str
    description: str
    paper_ids: list[str] = Field(description="IDs of papers in this category")
    children: list["TaxonomyNode"] = Field(default_factory=list)


TaxonomyNode.model_rebuild()


class MethodCluster(BaseModel):
    name: str
    description: str
    paper_ids: list[str]
    representative_paper_id: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class CitationRef(BaseModel):
    paper_id: str
    cite_text: str = Field(description="As it should appear in markdown, e.g. 'Smith et al. 2024, arxiv:2401.12345'")


class Finding(BaseModel):
    text: str = Field(description="The finding statement")
    cites: list[CitationRef]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # PR-4 (G段): which extracted claims back this finding. Populated by talent
    # via claim_search during step 7.5 of systematic-review. Used downstream by
    # fact_check_rendered_survey to anchor judgement, and by markdown rendering
    # to surface verbatim evidence footnotes under the finding.
    claim_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of extracted claims that back this finding (format "
            "'<paper_id>#claim-N'). Empty list = unverified / orphan finding."
        ),
    )


class Conflict(BaseModel):
    id: str
    level: Literal["direct", "methodological", "scope", "temporal"]
    claim_a_paper_id: str
    claim_a_text: str
    claim_b_paper_id: str
    claim_b_text: str
    shared_setting: str
    description: str
    stage3_hint: str = ""


class OpenQuestion(BaseModel):
    id: str
    question: str
    raised_by_paper_ids: list[str]
    quotes: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    id: str
    title: str
    description: str
    evidence_paper_ids: list[str]
    actionability: Literal["high", "medium", "low"]
    estimated_difficulty: Literal["low", "medium", "high"]


class AttributionAuditItem(BaseModel):
    claim_text: str
    citation_raw: str
    citation_kind: Literal["arxiv", "doi", "s2"]
    citation_id: str
    paper_id: str = ""
    paper_title: str = ""
    verdict: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "source_irrelevant",
        "source_not_in_corpus",
        "no_source_text",
        "judge_error",
    ]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quote: str = ""
    explanation: str = ""
    judge: str = ""
    # PR-3 (E1): when fact_check found a pre-extracted claim that anchored the
    # judgement, these fields record which one. Empty when no anchor was used
    # (claims.jsonl missing, or no claim matched the sentence well enough).
    matched_claim_id: str = ""
    matched_claim_quote: str = ""


class AttributionAuditSummary(BaseModel):
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    source_irrelevant_count: int = 0
    source_not_in_corpus_count: int = 0
    no_source_text_count: int = 0
    judge_error_count: int = 0
    blocking_count: int = 0
    total_checked: int = 0
    items: list[AttributionAuditItem] = Field(default_factory=list)


class LiteratureSurveySchema(BaseModel):
    """Top-level schema for Stage 2 output."""

    research_question: str
    search_strategy: SearchStrategy
    corpus_summary: CorpusSummary
    taxonomy: list[TaxonomyNode]
    methods_landscape: list[MethodCluster] = Field(default_factory=list)
    findings: list[Finding]
    conflicts: list[Conflict] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    suggested_directions: list[str] = Field(default_factory=list)

    # Quality gates
    citation_audit: dict = Field(
        default_factory=dict,
        description="Output of verify_citations and fact_check_rendered_survey.",
    )
    attribution_audit: AttributionAuditSummary = Field(
        default_factory=AttributionAuditSummary,
        description="Sentence-level support audit for final rendered markdown.",
    )
    corpus_insufficient_on_topic: list[str] = Field(
        default_factory=list,
        description="Topics where the corpus came up empty — be honest, don't invent",
    )

    # Metadata
    talent_version: str = "0.1.0"
    generated_at: float = 0.0
