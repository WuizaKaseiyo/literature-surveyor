"""Tests for verify_citations — uses corpus hits + mocks live APIs."""

from __future__ import annotations

from unittest.mock import patch

from tools.corpus_store.corpus_store import corpus_add_paper
from tools.verify_citations.verify_citations import verify_citations


def test_no_cites_in_text(isolated_corpus):
    res = verify_citations.invoke({"markdown": "Just plain text, no citations."})
    assert res["total_cites"] == 0
    assert res["verified_count"] == 0
    assert res["unverified_count"] == 0


def test_cite_resolved_by_corpus(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})

    md = "We rely on prior work [Smith et al. 2024, arxiv:2401.12345] for our setup."
    res = verify_citations.invoke({"markdown": md})

    assert res["verified_count"] == 1
    assert res["unverified_count"] == 0
    assert res["verified"][0]["matched_corpus"] is True
    assert res["verified"][0]["id"] == "2401.12345"


def test_unknown_arxiv_unverified(isolated_corpus):
    md = "Citing a fake paper [Imaginary Et Al 2099, arxiv:2099.99999]."

    # Mock arxiv API to return empty (no <entry>)
    fake_xml = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    class FakeResp:
        def read(self):
            return fake_xml

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    with patch(
        "tools.verify_citations.verify_citations.urllib.request.urlopen",
        return_value=FakeResp(),
    ):
        res = verify_citations.invoke({"markdown": md})

    assert res["unverified_count"] == 1
    assert res["unverified"][0]["id"] == "2099.99999"
    assert "not found" in res["unverified"][0]["reason"]


def test_dedupe_repeated_cites(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    md = (
        "Previously [Smith et al. 2024, arxiv:2401.12345] and elsewhere "
        "[Smith et al. 2024, arxiv:2401.12345] again."
    )
    res = verify_citations.invoke({"markdown": md})
    assert res["total_cites"] == 1  # deduped
    assert res["verified_count"] == 1


def test_ignores_non_compliant_formats(isolated_corpus, fake_paper):
    corpus_add_paper.invoke({"paper": fake_paper})
    md = (
        "Numeric cite [1] and APA-style (Smith, 2024) and bare [Smith et al] "
        "should all be ignored. Only [Smith et al. 2024, arxiv:2401.12345] counts."
    )
    res = verify_citations.invoke({"markdown": md})
    assert res["total_cites"] == 1
