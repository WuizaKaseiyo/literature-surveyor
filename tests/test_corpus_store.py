"""Tests for corpus_store — exercises add/search/status/list/get end-to-end."""

from __future__ import annotations

from tools.corpus_store.corpus_store import (
    corpus_add_paper,
    corpus_get_paper,
    corpus_list_papers,
    corpus_search,
    corpus_status,
)


def test_add_then_status(isolated_corpus, fake_paper):
    result = corpus_add_paper.invoke({"paper": fake_paper})
    assert result["status"] == "added"
    assert result["corpus_size"] == 1

    status = corpus_status.invoke({})
    assert status["corpus_size"] == 1
    assert status["by_source"] == {"arxiv": 1}
    assert status["by_year"] == {"2024": 1}


def test_add_idempotent(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    second = corpus_add_paper.invoke({"paper": fake_paper})
    assert second["status"] == "skipped_duplicate"
    assert second["corpus_size"] == 1


def test_add_rejects_missing_required(isolated_corpus):
    no_id = corpus_add_paper.invoke({"paper": {"title": "x"}})
    assert no_id["status"] == "rejected"

    no_title = corpus_add_paper.invoke({"paper": {"id": "x"}})
    assert no_title["status"] == "rejected"


def test_search_returns_relevant(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    corpus_add_paper.invoke({
        "paper": {
            "id": "2305.55555",
            "title": "Diffusion Models for Image Generation",
            "abstract": "We present a new diffusion model architecture for images.",
            "year": 2023,
            "source": "arxiv",
        }
    })

    res = corpus_search.invoke({"query": "attention small language models", "top_k": 5})
    assert res["total_corpus_size"] == 2
    assert len(res["results"]) >= 1
    # Top hit should be the relevant paper
    assert res["results"][0]["paper"]["id"] == "2401.12345"
    assert "attention" in res["results"][0]["matched_terms"]


def test_search_no_match_returns_empty(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    res = corpus_search.invoke({"query": "quantum chromodynamics", "top_k": 5})
    assert res["results"] == []


def test_list_filtered_by_source(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    corpus_add_paper.invoke({
        "paper": {"id": "ss-1", "title": "From SS", "source": "semantic_scholar"}
    })

    arxiv_only = corpus_list_papers.invoke({"limit": 10, "source": "arxiv"})
    assert arxiv_only["count"] == 1
    assert arxiv_only["papers"][0]["source"] == "arxiv"


def test_get_paper_roundtrip(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    got = corpus_get_paper.invoke({"paper_id": "2401.12345"})
    assert got["title"] == fake_paper["title"]
    assert got["authors"] == fake_paper["authors"]


def test_get_paper_not_found(isolated_corpus):
    got = corpus_get_paper.invoke({"paper_id": "nonexistent"})
    assert got["error"] == "not_found"
