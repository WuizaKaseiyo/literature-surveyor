"""pdf_extract — Extract a PDF to structured markdown.

Uses pymupdf4llm which preserves headings, tables, and math much better than
plain text extraction.

Falls back gracefully:
- If pymupdf4llm not installed → use pypdf for plain text + warning
- If both unavailable → return error dict
"""

from __future__ import annotations

import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DOWNLOAD_TIMEOUT_S = 60
MAX_PDF_SIZE_MB = 50


@tool
def pdf_extract(pdf_path_or_url: str) -> dict[str, Any]:
    """Extract a PDF as structured markdown (preserves headings, tables, math).

    Accepts a local file path OR an HTTP(S) URL. URLs are downloaded to a temp
    file. Uses pymupdf4llm for high-fidelity extraction; falls back to pypdf for
    plain text if pymupdf4llm is unavailable.

    Args:
        pdf_path_or_url: Local path (e.g. "/tmp/paper.pdf") or URL
            (e.g. "https://arxiv.org/pdf/2401.12345.pdf").

    Returns:
        {
          "markdown": "...",       # full text as markdown (or plain text fallback)
          "char_count": 45000,
          "source": "pymupdf4llm" | "pypdf" | "error",
          "format": "markdown" | "plain",
          "pdf_path": "...",       # local path used (downloaded if URL)
          "warnings": [...]
        }

    On failure: {"error": "...", "markdown": "", "char_count": 0, "source": "error"}
    """
    if not pdf_path_or_url or not pdf_path_or_url.strip():
        return _err("empty path/url")

    warnings: list[str] = []
    local_path = pdf_path_or_url

    # Resolve URL → local file
    if pdf_path_or_url.startswith(("http://", "https://")):
        try:
            local_path = _download_pdf(pdf_path_or_url, warnings)
        except Exception as e:
            return _err(f"download failed: {e}")

    p = Path(local_path)
    if not p.exists():
        return _err(f"file not found: {local_path}")
    if p.stat().st_size == 0:
        return _err(f"file is empty: {local_path}")
    if p.stat().st_size > MAX_PDF_SIZE_MB * 1024 * 1024:
        return _err(f"file too large (>{MAX_PDF_SIZE_MB}MB): {p.stat().st_size} bytes")

    # Try pymupdf4llm first
    try:
        import pymupdf4llm  # type: ignore

        md = pymupdf4llm.to_markdown(str(p))
        return {
            "markdown": md,
            "char_count": len(md),
            "source": "pymupdf4llm",
            "format": "markdown",
            "pdf_path": str(p),
            "warnings": warnings,
        }
    except ImportError:
        warnings.append("pymupdf4llm not installed; falling back to pypdf (plain text)")
    except Exception as e:
        warnings.append(f"pymupdf4llm failed: {e}; falling back to pypdf")

    # Fallback: pypdf plain text
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(p))
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception as e:
                warnings.append(f"pypdf page extract failed: {e}")
        text = "\n\n".join(text_parts)
        return {
            "markdown": text,
            "char_count": len(text),
            "source": "pypdf",
            "format": "plain",
            "pdf_path": str(p),
            "warnings": warnings,
        }
    except ImportError:
        return _err("neither pymupdf4llm nor pypdf installed")
    except Exception as e:
        return _err(f"pypdf extraction failed: {e}")


def _download_pdf(url: str, warnings: list[str]) -> str:
    """Download a PDF to a deterministic temp path (cached if already there)."""
    safe_name = urllib.parse.quote(url, safe="")[:200]
    tmp_path = Path(tempfile.gettempdir()) / f"litsurv_pdf_{safe_name}.pdf"

    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        warnings.append(f"using cached download: {tmp_path}")
        return str(tmp_path)

    req = urllib.request.Request(
        url, headers={"User-Agent": "literature-surveyor/0.1 (+research)"}
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        data = resp.read()
    if len(data) > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise ValueError(f"download exceeds {MAX_PDF_SIZE_MB}MB")

    tmp_path.write_bytes(data)
    return str(tmp_path)


def _err(msg: str) -> dict[str, Any]:
    return {
        "error": msg,
        "markdown": "",
        "char_count": 0,
        "source": "error",
        "format": "",
        "warnings": [],
    }
