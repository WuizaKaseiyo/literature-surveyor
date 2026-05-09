---
name: pdf_extract
description: Extract a PDF (URL or local path) to structured markdown. Preserves headings, tables, math.
---

# pdf_extract

Convert a PDF to markdown for downstream LLM consumption.

## When to use

- After `arxiv_search` / `semantic_scholar_search` — fetch the PDF for full-text
- When CEO uploads reference PDFs — extract before storing in corpus
- Any time you need text from a PDF

## When NOT to use

- For abstract-only needs — search tools already return abstracts; don't waste a download
- For scanned (image-based) PDFs — neither pymupdf4llm nor pypdf does OCR; will return mostly empty text

## How it works

1. If input is a URL, downloads to `/tmp/litsurv_pdf_<hash>.pdf` (cached on repeat)
2. Tries `pymupdf4llm.to_markdown()` first — preserves structure
3. Falls back to `pypdf` plain text if pymupdf4llm fails
4. Returns markdown + char_count + source identifier

## Limits

- Max PDF size: 50 MB (download or local)
- Download timeout: 60s
- Cached downloads in `/tmp` survive across calls within a session

## Quirks

- arxiv PDFs sometimes redirect — User-Agent is set to identify as research tool
- pymupdf4llm output occasionally has odd `**bold**` from PDF formatting noise; usually harmless for LLM
- For scanned PDFs the result will be near-empty; check `char_count < 500` and report `pdf_unparseable: true`
