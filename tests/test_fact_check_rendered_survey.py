"""Tests for final rendered-survey attribution fact checking."""

from __future__ import annotations

import json

from tools.corpus_store.corpus_store import corpus_add_paper
from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
    fact_check_rendered_survey,
    parse_attributions,
)


def _write_claims(corpus_dir, claims):
    path = corpus_dir / "claims.jsonl"
    with path.open("w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


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


def test_parser_ignores_evidence_footnote():
    """G段 evidence footnote format must not be parsed as a second attribution.

    The convention from systematic-review/SKILL.md is:
        Some finding [Author, arxiv:ID].

        > evidence: "verbatim quote"
        > — Section N (arxiv:ID#claim-K)

    The blockquote uses parens around the claim ref, so it should:
      (a) not match the [author, arxiv:id] cite regex
      (b) be dropped because no cite anchors the paragraph
    """
    md = (
        "DPO matches PPO on summarization "
        "[Rafailov et al. 2023, arxiv:2305.18290].\n"
        "\n"
        "> evidence: \"matches or improves response quality in summarization\"\n"
        "> — Section 6, Table 1 (arxiv:2305.18290#claim-3)\n"
    )
    attrs = parse_attributions(md)
    # exactly 1 attribution: the finding sentence
    assert len(attrs) == 1
    assert attrs[0]["citations"][0]["id"] == "2305.18290"
    # the #claim-3 ref inside parens must NOT be parsed as a cite
    for a in attrs:
        for c in a["citations"]:
            assert "claim" not in c["id"]


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


# ---------------------------------------------------------------------------
# E2 — unit-aware number matching in heuristic judge
# ---------------------------------------------------------------------------


def test_e2_tolerance_accepts_near_value(isolated_corpus, fake_paper):
    """Claim "31% reduction" should NOT be flagged when source says "30%"
    — within the 5% relative tolerance band."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})
    md = (
        "Pruning attention heads reduces latency by 31% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})
    item = res["items"][0]
    assert item["verdict"] != "unsupported", (
        f"31% should be within tolerance of 30%; got {item['verdict']}"
    )


def test_e2_recognizes_percent_word_form(isolated_corpus, fake_paper):
    """`30 percent` in source should be canonicalized as 30% and match claim's
    `30%` — heuristic shouldn't be defeated by word vs symbol."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning reduces latency by 30 percent in language models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})
    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})
    item = res["items"][0]
    assert item["verdict"] != "unsupported"


def test_e2_unit_mismatch_still_caught(isolated_corpus, fake_paper):
    """Different units mean different things — 1M params ≠ 1B params even
    though the digit "1" is the same."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "We trained 1M-parameter models on synthetic data with attention heads."
    )
    corpus_add_paper.invoke({"paper": paper})
    md = (
        "We trained 1B-parameter models on synthetic data with attention heads "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})
    item = res["items"][0]
    # Should not be supported — 1B vs 1M is a 1000× difference, definitely outside tolerance
    assert item["verdict"] in {"unsupported", "partially_supported"}, item


def test_e2_outside_tolerance_still_unsupported(isolated_corpus, fake_paper):
    """Sanity-preserve: 50% claim against 30% source is still wrong."""
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
    assert "50" in res["items"][0]["explanation"]


def test_e2_extract_helper_unit_table():
    """Direct test of _extract_numbers_with_unit canonicalization."""
    from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
        _extract_numbers_with_unit,
    )
    cases = {
        "30%": [(30.0, "%")],
        "30 percent reduction": [(30.0, "%")],
        "30 pct": [(30.0, "%")],
        "1.5B params": [(1.5, "B")],
        "7B and 13B": [(7.0, "B"), (13.0, "B")],
        "the rate was 30": [(30.0, "")],
        "no numbers here": [],
    }
    for text, expected in cases.items():
        got = _extract_numbers_with_unit(text)
        assert got == expected, f"{text!r} → {got}, want {expected}"


def test_fact_check_requires_cited_paper_in_corpus(isolated_corpus):
    md = "A real-looking claim [Smith et al. 2024, arxiv:2401.12345]."
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    assert res["source_not_in_corpus_count"] == 1
    assert res["blocking_count"] == 1


# ---------------------------------------------------------------------------
# E1 — claim-anchored path (PR-3)
# ---------------------------------------------------------------------------


def test_anchors_to_matched_claim_when_available(isolated_corpus, fake_paper):
    """When claims.jsonl has a strong match, the result records matched_claim_id
    and surfaces the claim's verbatim evidence_quote (not a BM25 top sentence)."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Section 4.2 establishes that pruning attention heads in small models "
        "leads to noticeable speedups. Specific numbers in Table 3."
    )
    corpus_add_paper.invoke({"paper": paper})
    _write_claims(isolated_corpus, [
        {
            "id": "2401.12345#claim-1",
            "paper_id": "2401.12345",
            "claim_text": "Pruning attention heads reduces latency by 30% in models below 7B.",
            "claim_type": "factual",
            "evidence_span": "Section 4.2, Table 3",
            "evidence_quote": "Pruning attention heads leads to 30% latency reduction in 1B-7B models.",
            "source_section": "Section 4.2",
            "confidence": 0.85,
            "applies_to": "models 1B-7B",
        },
    ])

    md = (
        "Pruning attention heads reduces latency by 30% in language models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    assert res["total_checked"] == 1
    item = res["items"][0]
    assert item["matched_claim_id"] == "2401.12345#claim-1"
    assert "30% latency reduction" in item["matched_claim_quote"]
    # The surface evidence_quote is the matched claim's verbatim quote.
    assert item["evidence_quote"] == item["matched_claim_quote"]


def test_falls_back_to_bm25_when_no_claim_matches(isolated_corpus, fake_paper):
    """Claims exist but none overlap the sentence — anchor is skipped, BM25 path
    runs, matched_claim_id stays empty."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})
    _write_claims(isolated_corpus, [
        {
            "id": "2401.12345#claim-1",
            "paper_id": "2401.12345",
            "claim_text": "Carbon emissions during pretraining were offset by Meta sustainability programs.",
            "claim_type": "factual",
            "evidence_span": "Section 2.2.1",
            "evidence_quote": "Total emissions were fully offset.",
            "source_section": "Section 2.2.1",
            "confidence": 0.7,
            "applies_to": "Llama 2 pretraining",
        },
    ])

    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    item = res["items"][0]
    assert item["matched_claim_id"] == ""
    assert item["matched_claim_quote"] == ""
    # Still passes via BM25 + heuristic since abstract supports the claim.
    assert res["supported_count"] == 1


def test_no_claims_file_no_regression(isolated_corpus, fake_paper):
    """claims.jsonl absent: old behavior preserved, matched_* fields are empty."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})

    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    item = res["items"][0]
    assert item["matched_claim_id"] == ""
    assert item["matched_claim_quote"] == ""
    assert res["supported_count"] == 1


def test_e2e_finding_claim_ids_match_factcheck_matched_claim_id(isolated_corpus, fake_paper):
    """G段 闭环：Finding 声明的 claim_ids 应该和 fact_check 实际找到的 matched_claim_id 一致。

    这是 step 9 的交叉校验流程的 happy path 测试 —— 验证 SKILL.md 里 step 7.5
    (写 finding 前 claim_search) 和 step 9 (fact_check 后核对 matched_claim_id)
    数据闭环。
    """
    # 1. corpus 里放一篇 paper + 一条 backing claim
    paper = dict(fake_paper)
    paper["abstract"] = (
        "DPO matches or improves response quality vs PPO-based RLHF on "
        "summarization and single-turn dialogue tasks."
    )
    corpus_add_paper.invoke({"paper": paper})
    backing_claim = {
        "id": "2401.12345#claim-3",
        "paper_id": "2401.12345",
        "claim_text": "DPO matches or improves response quality vs PPO-based RLHF on summarization.",
        "claim_type": "factual",
        "evidence_span": "Section 6, Table 1",
        "evidence_quote": (
            "matches or improves response quality in summarization "
            "and single-turn dialogue with significantly less compute"
        ),
        "source_section": "Section 6",
        "confidence": 0.85,
        "applies_to": "GPT-2 large, summarization",
    }
    _write_claims(isolated_corpus, [backing_claim])

    # 2. talent 在 stage2.json 里写的 Finding（含 claim_ids）
    from schemas.literature_survey_schema import CitationRef, Finding
    finding = Finding(
        text="DPO matches PPO-based RLHF on summarization quality.",
        cites=[CitationRef(paper_id="2401.12345", cite_text="Smith et al. 2024, arxiv:2401.12345")],
        claim_ids=["2401.12345#claim-3"],
    )
    assert finding.claim_ids == ["2401.12345#claim-3"]

    # 3. talent 按 G2 约定渲染 markdown（finding + evidence footnote）
    rendered = (
        f"{finding.text[:-1]} [Smith et al. 2024, arxiv:2401.12345].\n\n"
        f"> evidence: \"{backing_claim['evidence_quote']}\"\n"
        f"> — {backing_claim['source_section']} (arxiv:{backing_claim['paper_id']}#claim-3)\n"
    )

    # 4. fact_check 跑
    res = fact_check_rendered_survey.invoke({"markdown": rendered, "use_llm": False})

    # 5. 只 parse 出 1 个 attribution（footnote 没被算第二条）
    assert res["total_checked"] == 1, res["items"]

    # 6. matched_claim_id 真的命中了 Finding 声明的那条
    item = res["items"][0]
    assert item["matched_claim_id"] == "2401.12345#claim-3"
    assert item["matched_claim_id"] in finding.claim_ids, (
        f"cross-check failed: fact_check matched {item['matched_claim_id']} "
        f"but Finding.claim_ids = {finding.claim_ids}"
    )


# ---------------------------------------------------------------------------
# E3 — multi-cite attribution aggregation (PR-?, post-A2)
# ---------------------------------------------------------------------------


def test_multi_cite_supported_wins_over_missing(isolated_corpus, fake_paper):
    """Sentence with two cites — A in corpus and supports, B not in corpus.
    Per-cite items: 1 supported + 1 source_not_in_corpus. Per-attribution
    rollup: supported (non-blocking). blocking_count should be 0."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})

    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345][Wang et al. 2024, arxiv:9999.99999]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})

    # Two per-cite items, but exactly one rolled-up attribution
    assert res["total_checked"] == 2
    assert res["total_attributions"] == 1
    assert res["blocking_count"] == 0, "supported cite should non-block the attribution"
    assert res["blocking_cite_count"] >= 1, "the missing cite still counts in legacy per-cite view"

    per = res["per_attribution"][0]
    assert per["final_verdict"] == "supported"
    assert len(per["sub_results"]) == 2
    verdicts = {s["verdict"] for s in per["sub_results"]}
    assert "supported" in verdicts
    assert "source_not_in_corpus" in verdicts


def test_multi_cite_all_bad_stays_blocking(isolated_corpus):
    """Sentence with two cites, both unresolved → attribution still blocking."""
    md = (
        "Some claim [Anonymous 2024, arxiv:1111.11111]"
        "[Anonymous 2024, arxiv:2222.22222]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})
    assert res["total_attributions"] == 1
    assert res["blocking_count"] == 1
    per = res["per_attribution"][0]
    assert per["final_verdict"] == "source_not_in_corpus"


def test_single_cite_backward_compat(isolated_corpus, fake_paper):
    """Single-cite case: total_attributions equals total_checked, behavior
    identical to pre-E3."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})

    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({"markdown": md, "use_llm": False})
    assert res["total_checked"] == res["total_attributions"] == 1
    assert res["blocking_count"] == 0
    assert res["blocking_count"] == res["blocking_cite_count"], (
        "no multi-cite: attribution and per-cite blocking should agree"
    )


def test_use_extracted_claims_false_skips_anchor(isolated_corpus, fake_paper):
    """Explicitly disabling extracted claims keeps the BM25-only baseline path."""
    paper = dict(fake_paper)
    paper["abstract"] = (
        "Pruning attention heads leads to 30% latency reduction in models below 7B."
    )
    corpus_add_paper.invoke({"paper": paper})
    _write_claims(isolated_corpus, [
        {
            "id": "2401.12345#claim-1",
            "paper_id": "2401.12345",
            "claim_text": "Pruning attention heads reduces latency by 30% in models below 7B.",
            "claim_type": "factual",
            "evidence_span": "Section 4.2",
            "evidence_quote": "30% latency reduction in pruned attention heads.",
            "source_section": "Section 4.2",
            "confidence": 0.85,
            "applies_to": "1B-7B",
        },
    ])

    md = (
        "Pruning attention heads reduces latency by 30% in models below 7B "
        "[Smith et al. 2024, arxiv:2401.12345]."
    )
    res = fact_check_rendered_survey.invoke({
        "markdown": md,
        "use_llm": False,
        "use_extracted_claims": False,
    })

    item = res["items"][0]
    assert item["matched_claim_id"] == ""
    assert item["matched_claim_quote"] == ""

