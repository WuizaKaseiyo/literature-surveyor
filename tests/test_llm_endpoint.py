"""Regression tests for LLM endpoint resolution + structured error surfacing.

Issue: https://github.com/WuizaKaseiyo/literature-surveyor/issues/5

Two things must hold across extract_claims / detect_conflicts /
fact_check_rendered_survey:

  1. When `OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL` (or any of the other
     configurable base-url envs) are BOTH set, the OpenAI SDK client must
     receive the env base URL — NOT the hardcoded https://openrouter.ai/api/v1.
     LiteLLM / one-api gateways reuse the `OPENROUTER_API_KEY` env name; the
     pre-fix code branched on the env *name* and sent those requests to the
     wrong host, where they 401'd and the silent error path swallowed it.

  2. When the LLM call fails (auth / endpoint / model-not-found / connection
     error), the failure surfaces as a structured diagnostic in the tool's
     `warnings` (extract_claims) or `llm_warnings` (detect_conflicts,
     fact_check_rendered_survey). Pre-fix behaviour was to silently return
     `""` and let the caller report "0 claims" / "all unverifiable", which
     made deployment problems undebuggable.

No real HTTP or LLM calls — every test injects a fake OpenAI factory.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

import tools.detect_conflicts.detect_conflicts as dc_mod
import tools.extract_claims.extract_claims as ec_mod
import tools.fact_check_rendered_survey.fact_check_rendered_survey as fc_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every LLM-related env var so each test starts from a clean slate."""
    for name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENAI_BASE_URL",
        "DEFAULT_API_BASE_URL",
        "MEMENTO_MINI_BASE_URL",
        "OMC_EMPLOYEE_ID",
        "LITSURVEY_EMPLOYEE_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def _force_openai_path(monkeypatch: pytest.MonkeyPatch, *modules) -> None:
    """Stub `onemancompany.agents.base.make_llm` so option-1 always fails →
    every test deterministically exercises the openai-SDK code path that owns
    the bug (the base_url resolution + error swallowing)."""
    import sys

    fake_pkg = SimpleNamespace(
        agents=SimpleNamespace(
            base=SimpleNamespace(
                make_llm=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no OMC")),
            )
        )
    )
    # Make `from onemancompany.agents.base import make_llm` raise inside
    # each tool's _invoke_llm so we always fall through to the openai SDK path.
    sys.modules["onemancompany"] = fake_pkg
    sys.modules["onemancompany.agents"] = fake_pkg.agents
    sys.modules["onemancompany.agents.base"] = fake_pkg.agents.base
    for m in modules:
        # Drain any leftover error state from a previous test run.
        m._drain_llm_errors()


class _FakeOpenAIFactory:
    """Drop-in for `openai.OpenAI`. Captures init kwargs so a test can assert
    on the resolved `base_url`, and lets us script the chat.completions.create
    response (a payload string or an Exception to raise)."""

    def __init__(self, response: str | Exception):
        self.response = response
        self.init_calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.init_calls.append(kwargs)
        client = MagicMock()
        if isinstance(self.response, Exception):
            client.chat.completions.create.side_effect = self.response
        else:
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=self.response))
                ]
            )
        return client


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch, response: str | Exception
) -> _FakeOpenAIFactory:
    """Patch the `openai.OpenAI` symbol that each tool will import locally."""
    factory = _FakeOpenAIFactory(response)
    fake_openai_mod = SimpleNamespace(
        OpenAI=factory,
        # Exception classes used by _classify_llm_error — empty bases so
        # nothing in the real `openai` package is required at test time.
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        PermissionDeniedError=type("PermissionDeniedError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        BadRequestError=type("BadRequestError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
    )
    import sys

    sys.modules["openai"] = fake_openai_mod
    return factory


# ---------------------------------------------------------------------------
# 1. Endpoint resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected_base_url",
    [
        # Custom proxy via OPENROUTER_BASE_URL is respected even though the key
        # is named OPENROUTER_API_KEY (the bug from issue #5).
        (
            {
                "OPENROUTER_API_KEY": "sk-proxy",
                "OPENROUTER_BASE_URL": "https://litellm.example.invalid/v1",
            },
            "https://litellm.example.invalid/v1",
        ),
        # OPENAI_BASE_URL is also honoured when present.
        (
            {
                "OPENROUTER_API_KEY": "sk-proxy",
                "OPENAI_BASE_URL": "https://openai-proxy.example.invalid/v1",
            },
            "https://openai-proxy.example.invalid/v1",
        ),
        # DEFAULT_API_BASE_URL is a recognised fallback.
        (
            {
                "OPENROUTER_API_KEY": "sk-proxy",
                "DEFAULT_API_BASE_URL": "https://default.example.invalid/v1",
            },
            "https://default.example.invalid/v1",
        ),
        # MEMENTO_MINI_BASE_URL (lowest-priority deployment env) honoured too.
        (
            {
                "OPENROUTER_API_KEY": "sk-proxy",
                "MEMENTO_MINI_BASE_URL": "https://mini.example.invalid/v1",
            },
            "https://mini.example.invalid/v1",
        ),
        # Native OpenRouter: only OPENROUTER_API_KEY, no proxy env →
        # historical fallback URL is preserved.
        (
            {"OPENROUTER_API_KEY": "sk-or-direct"},
            "https://openrouter.ai/api/v1",
        ),
        # Pure OpenAI direct key → let the SDK pick its default (None).
        (
            {"OPENAI_API_KEY": "sk-oai-direct"},
            None,
        ),
    ],
)
@pytest.mark.parametrize(
    "module",
    [ec_mod, dc_mod, fc_mod],
    ids=["extract_claims", "detect_conflicts", "fact_check"],
)
def test_resolve_llm_endpoint_priority(monkeypatch, env, expected_base_url, module):
    """All three tools must resolve (api_key, base_url) with the same priority
    ladder. Hardcoding https://openrouter.ai/api/v1 from the env-var *name*
    breaks LiteLLM/one-api proxy deployments — issue #5."""
    _clear_llm_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    _, base_url = module._resolve_llm_endpoint()
    assert base_url == expected_base_url


@pytest.mark.parametrize(
    "module",
    [ec_mod, dc_mod, fc_mod],
    ids=["extract_claims", "detect_conflicts", "fact_check"],
)
def test_openai_client_receives_env_base_url(monkeypatch, module):
    """End-to-end: when OPENROUTER_API_KEY + OPENROUTER_BASE_URL are both set,
    the actual OpenAI client init must receive the env base URL — not the
    hardcoded openrouter.ai (the smoking-gun assertion for issue #5)."""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-proxy")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://litellm.example.invalid/v1")

    _force_openai_path(monkeypatch, module)
    factory = _install_fake_openai(monkeypatch, response="ok")

    out = module._invoke_llm("sys", "user", "openai/gpt-4o-mini")

    assert out == "ok"
    assert len(factory.init_calls) == 1
    assert factory.init_calls[0]["base_url"] == "https://litellm.example.invalid/v1"
    assert factory.init_calls[0]["base_url"] != "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# 2. Structured error surfacing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [ec_mod, dc_mod, fc_mod],
    ids=["extract_claims", "detect_conflicts", "fact_check"],
)
def test_invoke_llm_classifies_auth_error(monkeypatch, module):
    """A 401 from the SDK must be classified, never silently swallowed."""
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-bad")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://gateway.example.invalid/v1")

    _force_openai_path(monkeypatch, module)
    import sys

    auth_err_cls = type("AuthenticationError", (Exception,), {})
    sys.modules["openai"] = SimpleNamespace(
        OpenAI=_FakeOpenAIFactory(auth_err_cls("401 missing auth header")),
        AuthenticationError=auth_err_cls,
        PermissionDeniedError=type("PermissionDeniedError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        BadRequestError=type("BadRequestError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
    )

    out = module._invoke_llm("sys", "user", "openai/gpt-4o-mini")
    assert out == ""

    errs = module._drain_llm_errors()
    assert len(errs) >= 1
    err = errs[0]
    assert err["error_kind"] == "auth_error"
    assert err["base_url_host"] == "gateway.example.invalid"
    assert err["model"] == "openai/gpt-4o-mini"
    # Secrets safety: the api key must not leak into the error payload.
    assert "sk-bad" not in json.dumps(err)


@pytest.mark.parametrize(
    "module",
    [ec_mod, dc_mod, fc_mod],
    ids=["extract_claims", "detect_conflicts", "fact_check"],
)
def test_invoke_llm_missing_api_key_surfaces(monkeypatch, module):
    """No api key set → tools used to return "" silently. Now they must
    emit a `missing_api_key` diagnostic that the @tool will lift to warnings."""
    _clear_llm_env(monkeypatch)
    _force_openai_path(monkeypatch, module)

    out = module._invoke_llm("sys", "user", "openai/gpt-4o-mini")
    assert out == ""

    errs = module._drain_llm_errors()
    assert any(e.get("error_kind") == "missing_api_key" for e in errs)


# ---------------------------------------------------------------------------
# 3. Top-level @tool integration — extract_claims surfaces LLM error in warnings
# ---------------------------------------------------------------------------


def _seed_paper_for_extract(corpus_dir, paper_id="2401.99999"):
    paper = {
        "id": paper_id,
        "title": "Demo paper",
        "authors": ["Alice"],
        "year": 2024,
        "abstract": "We propose a thing. " * 30,
        "full_text_md": "# Intro\nWe propose a thing.\n\n# Results\nIt works.\n",
        "source": "arxiv",
    }
    (corpus_dir / "papers.jsonl").write_text(json.dumps(paper) + "\n")
    return paper_id


def test_extract_claims_surfaces_auth_error_in_warnings(
    monkeypatch, isolated_corpus
):
    """Before the fix this scenario returned `claims_extracted=0` with a
    generic 'no claims' error and no hint that the real cause was a 401.
    After the fix the warnings list must name the error_kind + host."""
    paper_id = _seed_paper_for_extract(isolated_corpus)

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-bad")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://gateway.example.invalid/v1")

    _force_openai_path(monkeypatch, ec_mod)
    monkeypatch.setattr(ec_mod.time, "sleep", lambda *a, **k: None)

    import sys

    auth_err_cls = type("AuthenticationError", (Exception,), {})
    sys.modules["openai"] = SimpleNamespace(
        OpenAI=_FakeOpenAIFactory(auth_err_cls("401 missing auth header")),
        AuthenticationError=auth_err_cls,
        PermissionDeniedError=type("PermissionDeniedError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        NotFoundError=type("NotFoundError", (Exception,), {}),
        BadRequestError=type("BadRequestError", (Exception,), {}),
        APIConnectionError=type("APIConnectionError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
    )

    out = ec_mod.extract_claims.invoke(
        {"paper_id": paper_id, "version": "v2", "force": True}
    )

    assert out.get("claims_extracted", 0) == 0
    warnings = out.get("warnings", [])
    combined = "\n".join(warnings)
    assert "llm_error" in combined
    assert "auth_error" in combined
    assert "gateway.example.invalid" in combined
    assert "sk-bad" not in combined  # secrets-safe surfacing
