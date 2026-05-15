"""Tests for final rendered-survey attribution fact checking."""

from __future__ import annotations

from tools.corpus_store.corpus_store import corpus_add_paper
from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
    fact_check_rendered_survey,
    parse_attributions,
)


def test_parse_backward_attribution():
    md = (
        "The paper studies attention in small language models. "
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )

    attrs = parse_attributions(md)

    assert len(attrs) == 2
    assert attrs[0]["claim_text"] == "The paper studies attention in small language models"
    assert attrs[1]["claim_text"].startswith("Pruning attention heads")
    assert all(a["citations"][0]["id"] == "2401.12345" for a in attrs)


def test_parser_ignores_code_blocks():
    md = """```text
Fake claim [Nobody 2024, arxiv:9999.99999]
```

Real claim [Smith et al. 2024, arxiv:2401.12345].
"""
    attrs = parse_attributions(md)
    assert len(attrs) == 1
    assert attrs[0]["claim_text"] == "Real claim"


def test_fact_check_supported_by_corpus(isolated_corpus, fake_paper):
    paper = dict(fake_paper)
    paper["abstract"] = (
        "We study attention patterns in language models below 7B parameters. "
        "Pruning attention heads leads to 30% latency reduction with negligible MMLU loss."
    )
    corpus_add_paper.invoke({"paper": paper})

    md = (
        "Pruning attention heads reduces latency by 30% in language models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    assert res["total_checked"] == 1
    assert res["supported_count"] == 1
    assert res["blocking_count"] == 0
    assert res["items"][0]["paper_id"] == "2401.12345"


def test_fact_check_catches_numeric_mismatch(isolated_corpus, fake_paper):
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})

    md = (
        "Pruning attention heads reduces latency by 50% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    assert res["unsupported_count"] == 1
    assert res["blocking_count"] == 1
    assert "50" in res["items"][0]["explanation"]


def test_fact_check_requires_cited_paper_in_corpus(isolated_corpus):
    md = "A real-looking claim [Smith et al. 2024, arxiv:2401.12345]."
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    assert res["source_not_in_corpus_count"] == 1
    assert res["blocking_count"] == 1

