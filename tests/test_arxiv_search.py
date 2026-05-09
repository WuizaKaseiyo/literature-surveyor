"""Tests for arxiv_search — uses fixture XML (no network)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from tools.arxiv_search.arxiv_search import arxiv_search

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-01T12:00:00Z</updated>
    <published>2024-01-25T09:00:00Z</published>
    <title>Attention Is Probably All You Need Again</title>
    <summary>We revisit attention. Results show
    things work fine.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Lee</name></author>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL"/>
    <arxiv:category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.99999</id>
    <published>2023-02-10T09:00:00Z</published>
    <title>Older Paper</title>
    <summary>From 2023.</summary>
    <author><name>Older Author</name></author>
  </entry>
</feed>
"""


def test_empty_query_rejected():
    result = arxiv_search.invoke({"query": "  ", "max_results": 5})
    assert result["error"] == "empty query"
    assert result["total"] == 0


def test_basic_parse():
    fake_resp = MagicMock()
    fake_resp.read.return_value = SAMPLE_XML
    fake_resp.__enter__ = lambda s: fake_resp
    fake_resp.__exit__ = lambda *a: None

    with patch("tools.arxiv_search.arxiv_search.urllib.request.urlopen", return_value=fake_resp):
        result = arxiv_search.invoke({"query": "attention", "max_results": 10})

    assert result["total"] == 2
    first = result["results"][0]
    assert first["arxiv_id"] == "2401.12345"  # version stripped
    assert first["title"] == "Attention Is Probably All You Need Again"
    assert first["authors"] == ["Alice Smith", "Bob Lee"]
    assert first["pdf_url"].startswith("http")
    assert "cs.CL" in first["categories"]


def test_since_filter_drops_old():
    fake_resp = MagicMock()
    fake_resp.read.return_value = SAMPLE_XML
    fake_resp.__enter__ = lambda s: fake_resp
    fake_resp.__exit__ = lambda *a: None

    with patch("tools.arxiv_search.arxiv_search.urllib.request.urlopen", return_value=fake_resp):
        result = arxiv_search.invoke({"query": "attention", "max_results": 10, "since": "2024-01-01"})

    # 2023 paper filtered out
    assert result["total"] == 1
    assert result["results"][0]["arxiv_id"] == "2401.12345"


def test_network_failure_returns_error():
    with patch(
        "tools.arxiv_search.arxiv_search.urllib.request.urlopen",
        side_effect=ConnectionError("network down"),
    ):
        result = arxiv_search.invoke({"query": "x"})

    assert "error" in result
    assert result["total"] == 0
    assert result["results"] == []
