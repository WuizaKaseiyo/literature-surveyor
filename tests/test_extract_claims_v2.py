"""Tests for v2 helpers in extract_claims — LLM-free."""

from __future__ import annotations

import json
from unittest.mock import patch

from tools.extract_claims.extract_claims import (
    _call_llm_with_retry,
    _parse_claims_response,
    _quote_in_source,
    _rank_and_cap_sections,
    _salvage_json_objects,
    _section_relevant,
    _split_sections,
)


SAMPLE_MD = (
    "# 1 Introduction\n\n"
    + (
        "This paper studies attention patterns in small language models. "
        "We pretrain on 2T tokens. We contribute a new pruning recipe that "
        "preserves benchmark scores while reducing latency substantially. "
        "Our experiments span seven model sizes from 0.5B to 13B parameters. "
        "The paper is organized into method, experiments, results, and discussion. "
    ) * 3
    + "\n\n## 1.1 Contributions\n\n"
    + (
        "We make three contributions to the field. First, a pruning algorithm. "
        "Second, a training stability technique. Third, an empirical evaluation. "
    ) * 3
    + "\n\n# 2 Method\n\n"
    + (
        "We use PPO-style RLHF on Llama-7B with a 50k preference dataset. "
        "The reward model is trained with the standard pairwise loss. "
        "Pruning is applied per-head with magnitude criterion at fine-tuning. "
    ) * 4
    + "\n\n## 2.1 Loss\n\n"
    + (
        "Our loss combines KL divergence with reward maximization. "
        "The temperature parameter controls exploration during training. "
    ) * 4
    + "\n\n# 3 Experiments\n\n"
    + (
        "We evaluate on MMLU and BBH benchmarks across seven model scales. "
        "Our 7B model improves MMLU from 45.3 to 51.2 percentage points. "
        "Latency is measured on a single A100 80GB across multiple seeds. "
    ) * 4
    + "\n\n# 4 Results\n\n"
    + (
        "Pruning attention heads reduces latency by 30% with negligible MMLU "
        "loss in 1B-7B models. Larger 13B models show small regressions. "
        "Stability metrics are reported in Table 3 across runs. "
    ) * 4
    + "\n\n# 5 References\n\n"
    + "[1] Smith et al. 2024.\n[2] Brown et al. 2023.\n[3] Wang 2022.\n"
    + "\n# 6 Appendix A: Training Details\n\n"
    + "Additional hyperparameters: lr=2e-5, batch=64.\n"
)


def test_split_sections_by_atx_headings():
    secs = _split_sections(SAMPLE_MD)
    headings = [s["heading"] for s in secs]
    assert "1 Introduction" in headings
    assert "1.1 Contributions" in headings
    assert "2 Method" in headings
    assert "3 Experiments" in headings
    assert "5 References" in headings
    assert "6 Appendix A: Training Details" in headings


def test_split_sections_respects_max_level():
    secs = _split_sections(SAMPLE_MD, max_level=1)
    headings = [s["heading"] for s in secs]
    # H1 only — H2 "1.1 Contributions" rolls into its parent
    assert "1.1 Contributions" not in headings
    assert "1 Introduction" in headings


def test_section_relevant_skips_references_and_appendix():
    secs = _split_sections(SAMPLE_MD)
    by_heading = {s["heading"]: s for s in secs}
    assert _section_relevant(by_heading["3 Experiments"])
    # References and appendix should be filtered out
    refs = by_heading["5 References"]
    appx = by_heading["6 Appendix A: Training Details"]
    assert _section_relevant(refs) is False
    assert _section_relevant(appx) is False


def test_section_relevant_skips_too_short():
    short_sec = {"heading": "Tiny", "text": "x"}
    assert _section_relevant(short_sec) is False


def test_rank_and_cap_prioritizes_claim_keywords():
    secs = [
        {"heading": "Acknowledgments", "text": "a" * 5000},
        {"heading": "1 Method",        "text": "b" * 1000},
        {"heading": "2 Results",       "text": "c" * 1000},
        {"heading": "Background",      "text": "d" * 2000},
    ]
    out = _rank_and_cap_sections(secs, max_sections=2)
    out_headings = [s["heading"] for s in out]
    # The two claim-keyword sections must beat the longer non-keyword ones.
    assert "1 Method" in out_headings
    assert "2 Results" in out_headings


def test_quote_verified_exact_substring():
    src = "Pruning attention heads reduces latency by 30% in 1B-7B models."
    assert _quote_in_source("reduces latency by 30%", src) is True


def test_quote_verified_minor_whitespace_change():
    src = "Pruning attention heads  reduces  latency by 30%   in   1B-7B models."
    assert _quote_in_source("Pruning attention heads reduces latency by 30%", src) is True


def test_quote_verified_rejects_fabricated():
    src = "Pruning attention heads reduces latency by 30% in 1B-7B models."
    # Completely different wording, no shared phrase
    assert _quote_in_source(
        "Carbon footprint of pretraining was offset entirely by sustainability initiatives.",
        src,
    ) is False


def test_quote_verified_rejects_too_short():
    src = "Anything."
    assert _quote_in_source("short", src) is False


# ---------------------------------------------------------------------------
# C6 — robust parsing + retry
# ---------------------------------------------------------------------------


def test_parse_strict_array():
    raw = json.dumps([{"claim_text": "a"}, {"claim_text": "b"}])
    claims, mode = _parse_claims_response(raw)
    assert mode == "strict_array"
    assert len(claims) == 2


def test_parse_strict_claims_key():
    raw = json.dumps({"claims": [{"claim_text": "a"}]})
    claims, mode = _parse_claims_response(raw)
    assert mode == "strict_claims_key"
    assert len(claims) == 1


def test_parse_strict_single_dict():
    raw = json.dumps({"claim_text": "lonely"})
    claims, mode = _parse_claims_response(raw)
    assert mode == "strict_dict"
    assert claims == [{"claim_text": "lonely"}]


def test_parse_strips_json_fence():
    raw = '```json\n[{"claim_text": "fenced"}]\n```'
    claims, mode = _parse_claims_response(raw)
    assert mode == "strict_array"
    assert claims[0]["claim_text"] == "fenced"


def test_parse_empty_returns_empty_mode():
    claims, mode = _parse_claims_response("")
    assert claims == []
    assert mode == "empty"

    claims, mode = _parse_claims_response("   \n\n  ")
    assert claims == []
    assert mode == "empty"


def test_parse_salvage_prose_prefix():
    """LLM wrote prose before the array — strict fails, salvage picks objects."""
    raw = (
        "Here are the claims I extracted from the paper:\n\n"
        '{"claim_text": "first claim"}\n'
        '{"claim_text": "second claim"}\n'
        "Hope these are useful!"
    )
    claims, mode = _parse_claims_response(raw)
    assert mode == "salvaged_2"
    assert {c["claim_text"] for c in claims} == {"first claim", "second claim"}


def test_parse_salvage_truncated_last_object():
    """LLM hit max_tokens mid-array — partial parse keeps the complete ones."""
    raw = (
        '[{"claim_text": "complete one"}, '
        '{"claim_text": "another complete"}, '
        '{"claim_text": "incomp'   # truncated, unbalanced quote and brace
    )
    claims, mode = _parse_claims_response(raw)
    assert mode == "salvaged_2"
    assert {c["claim_text"] for c in claims} == {"complete one", "another complete"}


def test_parse_salvage_handles_brace_in_string():
    """Brace inside a JSON string must not split the object boundary."""
    raw = (
        "preamble: "
        '{"claim_text": "has {brace} in text"} '
        '{"claim_text": "next"}'
    )
    claims, mode = _parse_claims_response(raw)
    assert mode == "salvaged_2"
    assert claims[0]["claim_text"] == "has {brace} in text"


def test_parse_failed_returns_failed_mode():
    claims, mode = _parse_claims_response("no JSON at all, just words")
    assert claims == []
    assert mode == "failed"


def test_salvage_skips_non_dict():
    """Salvage finds {} blocks but should only return dicts."""
    raw = '[1, 2, 3] {"claim_text": "real"} [4, 5]'
    found = _salvage_json_objects(raw)
    assert found == [{"claim_text": "real"}]


def test_retry_invokes_twice_on_empty_response():
    """First call returns "" → retry once → success."""
    calls = []

    def fake(system, user, model):
        calls.append((system, user, model))
        return "" if len(calls) == 1 else "ok-second"

    with patch("tools.extract_claims.extract_claims._invoke_llm", side_effect=fake), \
         patch("tools.extract_claims.extract_claims.time.sleep"):
        result = _call_llm_with_retry("sys", "usr", "model-x")

    assert len(calls) == 2
    assert result == "ok-second"


def test_retry_skipped_when_first_call_succeeds():
    calls = []

    def fake(system, user, model):
        calls.append(1)
        return "first-ok"

    with patch("tools.extract_claims.extract_claims._invoke_llm", side_effect=fake), \
         patch("tools.extract_claims.extract_claims.time.sleep"):
        result = _call_llm_with_retry("s", "u", "m")

    assert len(calls) == 1
    assert result == "first-ok"


def test_retry_gives_up_after_max():
    """If retries also return "", final result is also "" — no infinite loop."""
    calls = []

    def fake(system, user, model):
        calls.append(1)
        return ""

    with patch("tools.extract_claims.extract_claims._invoke_llm", side_effect=fake), \
         patch("tools.extract_claims.extract_claims.time.sleep"):
        result = _call_llm_with_retry("s", "u", "m", max_retries=2)

    assert result == ""
    assert len(calls) == 3  # 1 initial + 2 retries
