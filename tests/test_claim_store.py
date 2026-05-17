"""Tests for claim_store — verifies the five read-side @tools on a synthetic
claims.jsonl. extract_claims (which actually populates claims.jsonl) is not
exercised here; we write claims directly so tests stay LLM-free and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.claim_store.claim_store import (
    claim_find_evidence,
    claim_get,
    claim_list_by_paper,
    claim_search,
    claim_status,
)


def _write_claims(corpus_dir: Path, claims: list[dict]) -> None:
    path = corpus_dir / "claims.jsonl"
    with path.open("w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


@pytest.fixture
def sample_claims(isolated_corpus) -> list[dict]:
    claims = [
        {
            "id": "2305.18290#claim-1",
            "paper_id": "2305.18290",
            "claim_text": "Direct Preference Optimization achieves comparable performance to PPO without a reward model.",
            "claim_type": "factual",
            "evidence_span": "Section 6, Table 1",
            "evidence_quote": "DPO matches PPO performance on summarization at temperature 0.0",
            "source_section": "Section 6",
            "confidence": 0.85,
            "applies_to": "GPT-2 large, summarization",
        },
        {
            "id": "2305.18290#claim-2",
            "paper_id": "2305.18290",
            "claim_text": "DPO is more stable to train than RLHF using PPO.",
            "claim_type": "methodological",
            "evidence_span": "Section 5",
            "evidence_quote": "DPO training is more stable than the corresponding PPO baseline",
            "source_section": "Section 5",
            "confidence": 0.7,
            "applies_to": "language models",
        },
        {
            "id": "2305.11206#claim-1",
            "paper_id": "2305.11206",
            "claim_text": "LIMA fine-tuned on 1000 examples achieves strong instruction following.",
            "claim_type": "factual",
            "evidence_span": "Section 4, Table 2",
            "evidence_quote": "LIMA outperforms RLHF-tuned baselines on 43% of cases",
            "source_section": "Section 4",
            "confidence": 0.8,
            "applies_to": "LLaMa 65B",
        },
    ]
    _write_claims(isolated_corpus, claims)
    return claims


def test_status_empty(isolated_corpus):
    out = claim_status.invoke({})
    assert out["claims_count"] == 0
    assert out["papers_with_claims"] == 0
    assert out["by_type"] == {}


def test_status_after_write(isolated_corpus, sample_claims):
    out = claim_status.invoke({})
    assert out["claims_count"] == 3
    assert out["papers_with_claims"] == 2
    assert out["by_type"] == {"factual": 2, "methodological": 1}
    assert out["with_evidence_quote"] == 3


def test_search_finds_relevant(isolated_corpus, sample_claims):
    out = claim_search.invoke({"query": "DPO PPO performance"})
    assert out["total_claims"] == 3
    assert len(out["results"]) >= 1
    top = out["results"][0]
    assert "2305.18290" in top["claim"]["id"]


def test_search_paper_filter(isolated_corpus, sample_claims):
    out = claim_search.invoke({"query": "training stability", "paper_ids": ["2305.11206"]})
    # 2305.11206 doesn't talk about stability — should be empty or non-2305.18290
    for r in out["results"]:
        assert r["claim"]["paper_id"] == "2305.11206"


def test_search_type_filter(isolated_corpus, sample_claims):
    out = claim_search.invoke({"query": "DPO", "claim_type": "methodological"})
    for r in out["results"]:
        assert r["claim"]["claim_type"] == "methodological"


def test_search_empty_query(isolated_corpus, sample_claims):
    out = claim_search.invoke({"query": ""})
    assert out.get("error") == "empty query"


def test_list_by_paper(isolated_corpus, sample_claims):
    out = claim_list_by_paper.invoke({"paper_id": "2305.18290"})
    assert out["count"] == 2
    assert all(c["paper_id"] == "2305.18290" for c in out["claims"])


def test_list_by_paper_unknown(isolated_corpus, sample_claims):
    out = claim_list_by_paper.invoke({"paper_id": "9999.99999"})
    assert out["count"] == 0
    assert out["claims"] == []


def test_get_found(isolated_corpus, sample_claims):
    out = claim_get.invoke({"claim_id": "2305.18290#claim-2"})
    assert out["claim_text"].startswith("DPO is more stable")


def test_get_not_found(isolated_corpus, sample_claims):
    out = claim_get.invoke({"claim_id": "doesnt-exist"})
    assert out.get("error") == "not_found"


def test_find_evidence_anchors_to_paper(isolated_corpus, sample_claims):
    """The key use case: rendered sentence → matched claim in same paper."""
    out = claim_find_evidence.invoke({
        "claim_text": "DPO achieves the same summarization quality as PPO",
        "paper_id": "2305.18290",
    })
    assert out["paper_id"] == "2305.18290"
    assert len(out["matches"]) >= 1
    top = out["matches"][0]
    assert top["claim_id"] == "2305.18290#claim-1"
    assert "DPO matches PPO" in top["evidence_quote"]


def test_find_evidence_constrained_to_paper(isolated_corpus, sample_claims):
    """Even if a different paper would match better, only this paper's claims return."""
    out = claim_find_evidence.invoke({
        "claim_text": "DPO is more stable than PPO",
        "paper_id": "2305.11206",  # LIMA, not DPO
    })
    for m in out["matches"]:
        assert m["claim_id"].startswith("2305.11206#")


def test_index_rebuilt_on_stale(isolated_corpus, sample_claims):
    """First search builds the index; appending a new claim triggers a rebuild."""
    out1 = claim_search.invoke({"query": "RLHF"})
    initial_total = out1["total_claims"]

    # Append a new claim with content not previously indexed.
    new = {
        "id": "9999.99#claim-1",
        "paper_id": "9999.99",
        "claim_text": "Novel reinforcement signal annihilates reward hacking.",
        "claim_type": "factual",
        "evidence_span": "S1",
        "evidence_quote": "annihilates reward hacking",
        "source_section": "S1",
        "confidence": 0.5,
        "applies_to": "synthetic",
    }
    import time as _t
    _t.sleep(0.01)  # ensure mtime difference
    with (isolated_corpus / "claims.jsonl").open("a") as f:
        f.write(json.dumps(new) + "\n")

    out2 = claim_search.invoke({"query": "annihilates reward hacking"})
    assert out2["total_claims"] == initial_total + 1
    assert any("9999.99" in r["claim"]["id"] for r in out2["results"])
