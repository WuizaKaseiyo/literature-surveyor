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
    """Point LITSURVEY_CORPUS_DIR to a fresh tmp dir for the duration of one test.

    Does NOT set LITSURVEY_GLOBAL_CORPUS_DIR, so tools run in legacy single-tenant
    mode. Use `layered_corpus` for A2 layered-mode tests.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(corpus_dir))
    # Defensive: make sure no leftover global env from previous test pollutes.
    monkeypatch.delenv("LITSURVEY_GLOBAL_CORPUS_DIR", raising=False)
    return corpus_dir


@pytest.fixture
def layered_corpus(tmp_path, monkeypatch):
    """A2 layered-mode fixture — separate project + global dirs.

    Returns (project_dir, global_dir). Both env vars set, so tools route
    entities to global and refs to project.
    """
    project_dir = tmp_path / "project_corpus"
    global_dir = tmp_path / "global_corpus"
    project_dir.mkdir()
    global_dir.mkdir()
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_dir))
    monkeypatch.setenv("LITSURVEY_GLOBAL_CORPUS_DIR", str(global_dir))
    return project_dir, global_dir


@pytest.fixture
def two_layered_projects(tmp_path, monkeypatch):
    """Two project dirs sharing one global pool — for cross-project tests.

    Returns (project_a, project_b, global_dir). Switch active project with
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_x)).
    """
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    global_dir = tmp_path / "global"
    for d in (project_a, project_b, global_dir):
        d.mkdir()
    monkeypatch.setenv("LITSURVEY_GLOBAL_CORPUS_DIR", str(global_dir))
    monkeypatch.setenv("LITSURVEY_CORPUS_DIR", str(project_a))
    return project_a, project_b, global_dir


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
