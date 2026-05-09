"""Tests for self_assess — pure heuristic, no LLM call."""

from __future__ import annotations

from tools.corpus_store.corpus_store import corpus_add_paper
from tools.self_assess.self_assess import self_assess


def test_empty_corpus_insufficient(isolated_corpus):
    res = self_assess.invoke({"research_question": "anything"})
    assert res["verdict"] == "insufficient"
    assert res["suggested_next_action"] == "search_more"


def test_below_min_insufficient(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    res = self_assess.invoke({"research_question": "x", "min_papers": 10})
    assert res["verdict"] == "insufficient"


def test_in_band_sufficient(isolated_corpus, fake_paper):
    # Pad to 12 papers across 3 years
    for i in range(12):
        p = dict(fake_paper)
        p["id"] = f"2401.{1000 + i}"
        p["year"] = 2022 + (i % 3)
        p["source"] = ["arxiv", "openalex", "semantic_scholar"][i % 3]
        corpus_add_paper.invoke({"paper": p})

    res = self_assess.invoke({"research_question": "x", "min_papers": 10, "max_papers": 30})
    assert res["verdict"] == "sufficient"
    assert res["suggested_next_action"] == "extract_claims"
    assert res["metrics"]["paper_count"] == 12
    assert res["metrics"]["year_spread"] == 3
    assert res["metrics"]["source_diversity"] == 3


def test_over_max_saturated(isolated_corpus, fake_paper):
    for i in range(35):
        p = dict(fake_paper)
        p["id"] = f"2401.{2000 + i}"
        corpus_add_paper.invoke({"paper": p})

    res = self_assess.invoke({"research_question": "x", "min_papers": 10, "max_papers": 30})
    assert res["verdict"] == "saturated"
    assert res["suggested_next_action"] == "extract_claims"
