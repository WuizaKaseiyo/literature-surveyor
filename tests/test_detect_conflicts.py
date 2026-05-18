"""Tests for detect_conflicts — exercises pair-filtering logic (no LLM)."""

from __future__ import annotations

import json

import pytest

from tools.detect_conflicts.detect_conflicts import (
    _candidate_pairs,
    _share_setting,
    _topic_overlap,
    detect_conflicts,
)


def _write_claims(corpus_dir, claims):
    path = corpus_dir / "claims.jsonl"
    with path.open("w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _c(cid, pid, text, claim_type="factual", dims=None):
    return {
        "id": cid,
        "paper_id": pid,
        "claim_text": text,
        "claim_type": claim_type,
        "evidence_span": "",
        "evidence_quote": "",
        "source_section": "",
        "confidence": 0.7,
        "applies_to": "",
        "applies_to_dims": dims or {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_share_setting_token_overlap():
    """Token-set intersection (M6 fix): "7B" shares a token with "7B-13B"."""
    out = _share_setting(
        {"model_size": "7B-13B", "dataset": "MMLU"},
        {"model_size": "7B", "dataset": "BBH"},
    )
    assert out is not None
    assert "model_size" in out
    assert "dataset" not in out


def test_share_setting_rejects_substring_false_positive():
    """M6 regression guard: "L1" must NOT match "L17B" (substring would have);
    token sets give {"l1"} vs {"l17b"} which don't intersect."""
    out = _share_setting(
        {"model_size": "L1"},
        {"model_size": "L17B"},
    )
    assert out is None, "L1 must not match L17B — distinct tokens"


def test_share_setting_rejects_digit_run_false_positive():
    """M6 regression guard: "7B" must NOT match "L17B" — different tokens
    even though digit substring "7" appears in both."""
    out = _share_setting(
        {"model_size": "7B"},
        {"model_size": "L17B"},
    )
    assert out is None


def test_share_setting_multi_value_handled():
    """Comma-separated dim values tokenize into multiple atoms."""
    out = _share_setting(
        {"model_size": "7B, 13B, 70B"},
        {"model_size": "70B"},
    )
    assert out is not None and "model_size" in out


def test_share_setting_no_keys_overlap():
    out = _share_setting(
        {"model_size": "7B"},
        {"dataset": "MMLU"},
    )
    assert out is None


def test_share_setting_empty_inputs():
    assert _share_setting({}, {}) is None
    assert _share_setting(None, {"x": "y"}) is None


def test_share_setting_normalizes_whitespace():
    out = _share_setting(
        {"dataset": "TruthfulQA"},
        {"dataset": "  truthful qa"},
    )
    # Different whitespace + casing — bidirectional substring after strip
    assert out is not None
    assert "dataset" in out


def test_topic_overlap_jaccard():
    a = "DPO matches PPO on summarization"
    b = "DPO is worse than PPO on summarization"
    overlap = _topic_overlap(a, b)
    assert overlap > 0.3


def test_topic_overlap_disjoint():
    a = "quantum entanglement bell inequality"
    b = "DPO RLHF preference reward model"
    assert _topic_overlap(a, b) == 0.0


# ---------------------------------------------------------------------------
# Candidate filter
# ---------------------------------------------------------------------------


def test_candidate_skips_same_paper():
    claims = [
        _c("p1#c1", "p1", "claim a about 7B", dims={"model_size": "7B"}),
        _c("p1#c2", "p1", "claim b about 7B", dims={"model_size": "7B"}),
    ]
    pairs = _candidate_pairs(claims, min_jaccard=0.0)
    assert pairs == []


def test_candidate_skips_pure_conjecture_pair():
    claims = [
        _c("p1#c1", "p1", "speculative claim 7B", "conjecture", {"model_size": "7B"}),
        _c("p2#c1", "p2", "another speculative 7B", "conjecture", {"model_size": "7B"}),
    ]
    pairs = _candidate_pairs(claims, min_jaccard=0.0)
    assert pairs == []


def test_candidate_skips_no_shared_setting():
    claims = [
        _c("p1#c1", "p1", "same words but no dims", dims={"model_size": "7B"}),
        _c("p2#c1", "p2", "same words but no dims", dims={"dataset": "MMLU"}),
    ]
    pairs = _candidate_pairs(claims, min_jaccard=0.0)
    assert pairs == []


def test_candidate_finds_overlapping_setting_pair():
    claims = [
        _c("p1#c1", "p1", "PPO RLHF improves MMLU score in 7B models", dims={"model_size": "7B", "dataset": "MMLU"}),
        _c("p2#c1", "p2", "PPO RLHF degrades MMLU performance in 7B reasoning", dims={"model_size": "7B-13B", "dataset": "MMLU"}),
    ]
    pairs = _candidate_pairs(claims, min_jaccard=0.05)
    assert len(pairs) == 1
    assert pairs[0]["a"]["paper_id"] == "p1"
    assert pairs[0]["b"]["paper_id"] == "p2"
    assert "model_size" in pairs[0]["shared_setting"]
    assert "dataset" in pairs[0]["shared_setting"]


def test_candidate_pairs_sort_before_cap_keeps_high_score_late():
    """M9 regression guard: a high-overlap pair appearing LATE in iteration
    order must not be dropped by the cap. Old code did `if len >= CAP: break`
    inside the loop; correct code sorts then caps."""
    from tools.detect_conflicts.detect_conflicts import MAX_CANDIDATE_PAIRS
    # Generate MAX_CANDIDATE_PAIRS + 5 weak candidate pairs first, then one
    # very high-overlap pair last. The latter must survive the cap.
    weak_claims = []
    for i in range(MAX_CANDIDATE_PAIRS + 5):
        weak_claims.append(_c(
            f"weak-p{i}#c1", f"weak-p{i}",
            f"Token unique{i} thing happens here with attention models",
            dims={"model_size": "7B"},
        ))
    # Two STRONG-overlap claims at the tail
    strong_a = _c(
        "strong-a#c1", "strong-a",
        "DPO matches PPO on summarization quality across runs reliably consistently",
        dims={"model_size": "7B"},
    )
    strong_b = _c(
        "strong-b#c1", "strong-b",
        "DPO matches PPO on summarization quality across runs reliably consistently",
        dims={"model_size": "7B"},
    )
    claims = weak_claims + [strong_a, strong_b]

    pairs = _candidate_pairs(claims, min_jaccard=0.0)
    assert len(pairs) <= MAX_CANDIDATE_PAIRS, "must respect the cap"
    # The strong pair MUST be present despite being added last in the loop
    strong_present = any(
        {pair["a"]["paper_id"], pair["b"]["paper_id"]} == {"strong-a", "strong-b"}
        for pair in pairs
    )
    assert strong_present, "high-overlap tail pair dropped by early break (M9 regression)"


def test_candidate_topic_threshold_drops_unrelated():
    claims = [
        _c("p1#c1", "p1", "PPO RLHF improves MMLU", dims={"model_size": "7B"}),
        _c("p2#c1", "p2", "Tokenizer choice affects throughput", dims={"model_size": "7B"}),
    ]
    pairs = _candidate_pairs(claims, min_jaccard=0.3)
    assert pairs == []


# ---------------------------------------------------------------------------
# Top-level @tool — no LLM mode
# ---------------------------------------------------------------------------


def test_detect_conflicts_no_llm_returns_candidates(isolated_corpus):
    claims = [
        _c("p1#c1", "p1", "PPO RLHF on Llama-7B improves MMLU by 5 points",
           dims={"model_size": "7B", "method": "PPO"}),
        _c("p2#c1", "p2", "PPO RLHF on Llama-7B drops MMLU by 3 points",
           dims={"model_size": "7B", "method": "PPO"}),
        _c("p3#c1", "p3", "Quantization halves memory footprint",
           dims={"technique": "INT8"}),
    ]
    _write_claims(isolated_corpus, claims)

    out = detect_conflicts.invoke({"use_llm": False, "min_topic_jaccard": 0.05})
    assert out["judge"] == "candidates_only"
    assert out["candidates_screened"] >= 1
    assert out["conflicts"] == []
    # The 7B-PPO pair is a candidate; the quantization claim has no setting overlap
    cand_paper_pairs = {
        (c["claim_a_paper_id"], c["claim_b_paper_id"])
        for c in out["candidates"]
    }
    assert ("p1", "p2") in cand_paper_pairs or ("p2", "p1") in cand_paper_pairs
    assert all("p3" not in pair for pair in cand_paper_pairs)


def test_detect_conflicts_paper_ids_filter(isolated_corpus):
    claims = [
        _c("p1#c1", "p1", "X works at 7B", dims={"model_size": "7B"}),
        _c("p2#c1", "p2", "X works at 7B", dims={"model_size": "7B"}),
        _c("p3#c1", "p3", "X fails at 7B", dims={"model_size": "7B"}),
    ]
    _write_claims(isolated_corpus, claims)

    out = detect_conflicts.invoke({
        "use_llm": False,
        "paper_ids": ["p1", "p2"],
        "min_topic_jaccard": 0.0,
    })
    # Should only see candidates between p1 and p2 — p3 filtered out
    for c in out["candidates"]:
        assert c["claim_a_paper_id"] in {"p1", "p2"}
        assert c["claim_b_paper_id"] in {"p1", "p2"}


def test_detect_conflicts_empty_corpus(isolated_corpus):
    out = detect_conflicts.invoke({"use_llm": False})
    assert out["conflicts"] == []
    assert out["candidates_screened"] == 0
