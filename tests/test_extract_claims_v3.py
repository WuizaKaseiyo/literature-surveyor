"""Regression tests for extract_claims v3 — LLM-free (mocked _invoke_llm).

Covers two production bugs found reviewing the v3 PR:
  1. `contribution_idx=int(...)` crashed on null / "" / "n/a" / "1.5" LLM output
     (TypeError/ValueError escaped the `except ValidationError` and tore down
     the whole tool call).
  2. The `factual_no_scope` hard-filter rule could collapse a paper to ZERO
     claims when no applies_to_dims were filled (meta_pass fallback, or the
     LLM never emitted scope) — a recall regression vs v2.
"""

from __future__ import annotations

import json

import pytest

import tools.extract_claims.extract_claims as ec_mod
from tools.extract_claims.extract_claims import (
    _coerce_int,
    _hard_filter,
    _is_scopeless_factual,
    extract_claims,
)

PAPER_ID = "2401.55555"
SOURCE_MD = (
    "# Introduction\nWe propose a retrieval method for open-domain QA.\n\n"
    "# Results\nOur method reaches 88.0 accuracy on MMLU, beating the baseline. "
    "This verbatim sentence supports the claim about accuracy on MMLU here.\n\n"
    "# Conclusion\nWe summarized the gains observed across datasets.\n"
)
QUOTE = "Our method reaches 88.0 accuracy on MMLU, beating the baseline."


def _seed_paper(corpus_dir):
    paper = {
        "id": PAPER_ID,
        "title": "A Retrieval Method",
        "authors": ["Alice Smith"],
        "year": 2024,
        "abstract": "We propose a retrieval method for QA. " * 20,
        "full_text_md": SOURCE_MD,
        "source": "arxiv",
    }
    (corpus_dir / "papers.jsonl").write_text(json.dumps(paper) + "\n")


def _claim(contribution_idx, dims, saliency="empirical_finding", ctype="factual"):
    return {
        "claim_text": "The method reaches 88.0 accuracy on MMLU.",
        "claim_type": ctype,
        "saliency_type": saliency,
        "contribution_idx": contribution_idx,
        "evidence_span": "Results",
        "evidence_quote": QUOTE,
        "applies_to_dims": dims,
        "confidence": 0.8,
    }


def _dispatcher(section_claims, *, meta_ok=True, rerank_keep=("cand-0",)):
    """Build an _invoke_llm side_effect routing by system-prompt content."""

    def fake(system, user, model):
        s = system.lower()
        if "self-described contributions" in s:  # meta_pass
            if not meta_ok:
                return ""  # forces meta_pass fallback
            return json.dumps(
                {"contributions": ["a retrieval method"], "scope": {"dataset": "MMLU"}}
            )
        if "literature-survey editor" in s:  # global rerank
            return json.dumps({"keep": list(rerank_keep), "drop_reasons": {}})
        return json.dumps(section_claims)  # per-section extraction

    return fake


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, -1), ("", -1), ("n/a", -1), ("1.5", -1), ("3", 3), (2, 2), (2.9, 2), (-1, -1)],
)
def test_coerce_int_never_raises(value, expected):
    assert _coerce_int(value, default=-1) == expected


def test_hard_filter_no_longer_drops_scopeless_factual():
    """_hard_filter is now UNCONDITIONAL-only; scope-less factual passes it
    (given a verified quote — the unconditional rules are noise + unverified)."""
    c = _claim(-1, {})
    c["evidence_quote_verified"] = True
    keep, _ = _hard_filter(c)
    assert keep is True


def test_hard_filter_still_drops_noise_and_unverified():
    keep_noise, reason_noise = _hard_filter(_claim(-1, {"dataset": "MMLU"}, saliency="background"))
    assert keep_noise is False and reason_noise.startswith("noise_saliency")

    bad_quote = _claim(-1, {"dataset": "MMLU"})
    bad_quote["evidence_quote_verified"] = False  # has quote, not verified
    keep_q, reason_q = _hard_filter(bad_quote)
    assert keep_q is False and reason_q == "unverified_quote"


def test_is_scopeless_factual():
    assert _is_scopeless_factual(_claim(-1, {})) is True
    assert _is_scopeless_factual(_claim(-1, {"dataset": "MMLU"})) is False
    assert _is_scopeless_factual(_claim(-1, {}, saliency="limitation")) is False
    assert _is_scopeless_factual(_claim(-1, {}, ctype="methodological")) is False


# ---------------------------------------------------------------------------
# Bug #1: contribution_idx coercion no longer crashes extract_claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_idx", [None, "", "n/a", "1.5"])
def test_contribution_idx_malformed_does_not_crash(isolated_corpus, monkeypatch, bad_idx):
    _seed_paper(isolated_corpus)
    fake = _dispatcher([_claim(bad_idx, {"dataset": "MMLU"})])
    monkeypatch.setattr(ec_mod, "_invoke_llm", fake)
    monkeypatch.setattr(ec_mod.time, "sleep", lambda *a, **k: None)

    out = extract_claims.invoke({"paper_id": PAPER_ID, "version": "v3", "force": True})

    assert out["status"] == "extracted"
    assert out["claims_extracted"] == 1
    assert out["claims"][0]["contribution_idx"] == -1  # coerced safely


# ---------------------------------------------------------------------------
# Bug #2: scope-less factual claims no longer collapse a paper to zero
# ---------------------------------------------------------------------------


def test_scopeless_factual_kept_when_only_claims(isolated_corpus, monkeypatch):
    """meta_pass fell back + factual claim has no dims → must NOT return 0."""
    _seed_paper(isolated_corpus)
    fake = _dispatcher([_claim(-1, {})], meta_ok=False)
    monkeypatch.setattr(ec_mod, "_invoke_llm", fake)
    monkeypatch.setattr(ec_mod.time, "sleep", lambda *a, **k: None)

    out = extract_claims.invoke({"paper_id": PAPER_ID, "version": "v3", "force": True})

    assert out["status"] == "extracted"
    assert out["claims_extracted"] >= 1  # floor: kept rather than dropped to 0


def test_scopeless_factual_dropped_when_scoped_present(isolated_corpus, monkeypatch):
    """When a scoped claim co-exists, the scope-less one is still pruned.

    meta_ok=False so meta-pass scope isn't backfilled into the scope-less claim
    (otherwise every claim inherits paper-level scope and none stay scope-less)."""
    _seed_paper(isolated_corpus)
    section = [
        _claim(0, {"dataset": "MMLU"}),  # scoped — kept
        _claim(-1, {}),                  # scope-less — pruned when scoped exists
    ]
    fake = _dispatcher(section, meta_ok=False, rerank_keep=("cand-0", "cand-1"))
    monkeypatch.setattr(ec_mod, "_invoke_llm", fake)
    monkeypatch.setattr(ec_mod.time, "sleep", lambda *a, **k: None)

    out = extract_claims.invoke({"paper_id": PAPER_ID, "version": "v3", "force": True})

    assert out["status"] == "extracted"
    assert out["claims_extracted"] == 1
    assert out["claims"][0]["applies_to_dims"] == {"dataset": "MMLU"}
