"""Shared pytest fixtures.

Make sibling `tools/` and `schemas/` importable in tests, and isolate corpus
storage to a tmp dir per test (so tests don't pollute each other or your home).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to sys.path so tests can `from tools.foo.foo import foo`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def isolated_corpus(tmp_path, monkeypatch):
    """Point LITSURVEY_CORPUS_DIR to a fresh tmp dir for the duration of one test."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(corpus_dir))
    return corpus_dir


@pytest.fixture
def fake_paper() -> dict:
    return {
        "id": "2401.12345",
        "title": "Attention Patterns in Small Language Models",
        "authors": ["Alice Smith", "Bob Lee"],
        "year": 2024,
        "venue": "NeurIPS",
        "abstract": (
            "We study attention patterns in language models below 7B parameters. "
            "We find that pruning attention heads leads to 30% latency reduction "
            "with negligible MMLU score loss in models 1B-7B."
        ),
        "source": "arxiv",
        "source_query": "attention small language models",
    }
