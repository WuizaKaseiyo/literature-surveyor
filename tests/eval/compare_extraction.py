"""Compare extract_claims versions on a fixed paper fixture.

Run extract_claims with each requested version on every fixture paper, compute
quality metrics on the resulting claims, and write a markdown report under
tests/eval/reports/.

Usage:
    python tests/eval/compare_extraction.py
    python tests/eval/compare_extraction.py --versions v1 v2
    python tests/eval/compare_extraction.py --fixtures path/to/papers.jsonl

Requires OPENAI_API_KEY or OPENROUTER_API_KEY in env (extract_claims uses LLM).
Each (version, paper) pair runs in an isolated tmp corpus dir so claim files
from v1 and v2 do not mix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURES = REPO_ROOT / "tests/fixtures/golden_papers/papers.jsonl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "tests/eval/reports"

SECTION_BUCKETS: list[tuple[str, re.Pattern[str]]] = [
    ("intro",      re.compile(r"\b(introduction|intro)\b", re.I)),
    ("related",    re.compile(r"\b(related work|background|prior work)\b", re.I)),
    ("method",     re.compile(r"\b(method|approach|model|architecture|algorithm)\b", re.I)),
    ("experiment", re.compile(r"\b(experiment|setup|implementation|training)\b", re.I)),
    ("result",     re.compile(r"\b(result|finding|evaluation|ablation|analysis)\b", re.I)),
    ("discussion", re.compile(r"\b(discussion|limitation|future work)\b", re.I)),
    ("conclusion", re.compile(r"\b(conclusion|summary)\b", re.I)),
]


def _bucket_section(section: str) -> str:
    s = section or ""
    for name, pat in SECTION_BUCKETS:
        if pat.search(s):
            return name
    return "other"


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text.strip()


def quote_verified(quote: str, full_text: str) -> bool:
    """True if a substantial contiguous segment of `quote` appears in `full_text`.

    Set-based n-gram overlap gives false positives on long source texts because
    common English 4-grams blanket the source. We require at least one long
    contiguous substring of the quote (≥ 60% of the quote length, capped to 60
    chars) to match verbatim in the normalized source.
    """
    if not quote or len(quote) < 10:
        return False
    nq, nt = _normalize(quote), _normalize(full_text)
    if not nq or not nt:
        return False
    if nq in nt:
        return True
    window = min(60, max(20, int(len(nq) * 0.6)))
    if window > len(nq):
        return False
    step = max(1, window // 4)
    for i in range(0, len(nq) - window + 1, step):
        if nq[i : i + window] in nt:
            return True
    return False


def applies_to_filled(claim: dict[str, Any]) -> bool:
    dims = claim.get("applies_to_dims")
    if isinstance(dims, dict) and any(v for v in dims.values()):
        return True
    s = claim.get("applies_to", "")
    return bool(s and len(str(s).strip()) >= 3)


def run_one(paper: dict, version: str, model: str) -> dict[str, Any]:
    """Run extract_claims for one paper inside an isolated tmp corpus."""
    with tempfile.TemporaryDirectory() as td:
        os.environ["LITSURVEY_CORPUS_DIR"] = td
        # Re-import inside fresh env so the module picks up the new corpus dir.
        # Tools resolve corpus dir lazily via env, so a single import is fine.
        from tools.corpus_store.corpus_store import corpus_add_paper
        from tools.extract_claims.extract_claims import extract_claims

        corpus_add_paper.invoke({"paper": paper})
        t0 = time.time()
        out = extract_claims.invoke(
            {"paper_id": paper["id"], "model": model, "version": version}
        )
        elapsed = time.time() - t0
    return {
        "paper_id": paper["id"],
        "version": version,
        "claims": out.get("claims", []) or [],
        "error": out.get("error"),
        "warnings": out.get("warnings", []),
        "elapsed_s": elapsed,
    }


def compute_metrics(paper: dict, claims: list[dict]) -> dict[str, Any]:
    full_text = paper.get("full_text_md") or paper.get("abstract") or ""
    n = len(claims)
    sections = Counter(_bucket_section(c.get("source_section", "")) for c in claims)
    verified = sum(1 for c in claims if quote_verified(c.get("evidence_quote", ""), full_text))
    applied = sum(1 for c in claims if applies_to_filled(c))
    types = Counter(c.get("claim_type", "unknown") for c in claims)
    output_chars = sum(
        len(c.get("claim_text", "")) + len(c.get("evidence_quote", "")) for c in claims
    )
    return {
        "n_claims": n,
        "paper_chars": len(full_text),
        "section_dist": dict(sections),
        "type_dist": dict(types),
        "evidence_verified_rate": (verified / n) if n else 0.0,
        "applies_to_filled_rate": (applied / n) if n else 0.0,
        "approx_output_chars": output_chars,
    }


def render_report(rows: list[dict], versions: list[str], out_path: Path) -> None:
    by_paper: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_paper.setdefault(row["paper_id"], {})[row["version"]] = row

    lines: list[str] = [
        f"# extract_claims comparison — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"versions: {', '.join(versions)}",
        f"papers: {len(by_paper)}",
        "",
    ]

    lines.append("## Per-paper")
    lines.append("")
    for pid in sorted(by_paper):
        by_v = by_paper[pid]
        # Find any row with metrics to get paper_chars; fall back to "?" if all failed.
        valid = [r for r in by_v.values() if r.get("metrics")]
        paper_chars = valid[0]["metrics"]["paper_chars"] if valid else "?"
        lines.append(f"### `{pid}` ({paper_chars} chars)")
        lines.append("")
        lines.append("| version | n_claims | verified_quote | applies_to | elapsed | section_dist |")
        lines.append("|---|---|---|---|---|---|")
        for v in versions:
            row = by_v.get(v)
            if not row:
                lines.append(f"| {v} | — | — | — | — | (no data) |")
                continue
            if row.get("error") or not row.get("metrics"):
                err = row.get("error", "no metrics")
                lines.append(f"| {v} | error | — | — | — | {err} |")
                continue
            m = row["metrics"]
            lines.append(
                f"| {v} | {m['n_claims']} | {m['evidence_verified_rate']:.0%} | "
                f"{m['applies_to_filled_rate']:.0%} | {row['elapsed_s']:.1f}s | {m['section_dist']} |"
            )
        lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| version | papers | avg_claims | avg_verified | avg_applies_to | total_elapsed |")
    lines.append("|---|---|---|---|---|---|")
    for v in versions:
        v_rows = [r for r in rows if r["version"] == v and not r.get("error")]
        if not v_rows:
            lines.append(f"| {v} | 0 | — | — | — | — |")
            continue
        ns = [r["metrics"]["n_claims"] for r in v_rows]
        vr = [r["metrics"]["evidence_verified_rate"] for r in v_rows]
        ar = [r["metrics"]["applies_to_filled_rate"] for r in v_rows]
        el = sum(r["elapsed_s"] for r in v_rows)
        lines.append(
            f"| {v} | {len(v_rows)} | {sum(ns) / len(ns):.1f} | "
            f"{sum(vr) / len(vr):.0%} | {sum(ar) / len(ar):.0%} | {el:.1f}s |"
        )
    lines.append("")

    lines.append("## Claim-count vs paper-length scatter")
    lines.append("")
    for v in versions:
        v_rows = [r for r in rows if r["version"] == v and not r.get("error")]
        if not v_rows:
            continue
        lines.append(f"- **{v}**: " + ", ".join(
            f"({r['metrics']['paper_chars']}ch, {r['metrics']['n_claims']})" for r in v_rows
        ))
    lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    ap.add_argument("--versions", nargs="+", default=["v1"])
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument(
        "--no-save-samples",
        action="store_true",
        help="skip writing raw claim JSONs (default: save)",
    )
    args = ap.parse_args()

    if not args.fixtures.exists():
        print(
            f"No fixtures at {args.fixtures}\n"
            "Add at least 1 paper jsonl line — see tests/fixtures/golden_papers/README.md",
            file=sys.stderr,
        )
        return 1

    papers = [
        json.loads(line) for line in args.fixtures.read_text().splitlines() if line.strip()
    ]
    if not papers:
        print("Fixtures file is empty.", file=sys.stderr)
        return 1

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print(
            "No LLM API key in env — extract_claims will return errors.",
            file=sys.stderr,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    samples_dir = args.out_dir / f"extraction_{timestamp}_samples"
    if not args.no_save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for version in args.versions:
        for paper in papers:
            print(f"[{version}] {paper['id']} …", file=sys.stderr)
            run = run_one(paper, version, args.model)
            if run.get("error"):
                rows.append({**run, "metrics": None})
                print(f"  -> error: {run['error']}", file=sys.stderr)
                continue
            metrics = compute_metrics(paper, run["claims"])
            rows.append({**run, "metrics": metrics})
            print(
                f"  -> {metrics['n_claims']} claims, "
                f"verified {metrics['evidence_verified_rate']:.0%}, "
                f"{run['elapsed_s']:.1f}s",
                file=sys.stderr,
            )

            if not args.no_save_samples:
                sample = {
                    "paper_id": paper["id"],
                    "paper_title": paper.get("title", ""),
                    "paper_chars": metrics["paper_chars"],
                    "version": version,
                    "model": args.model,
                    "elapsed_s": run["elapsed_s"],
                    "metrics": metrics,
                    "warnings": run.get("warnings", []),
                    "claims": run["claims"],
                }
                sample_path = samples_dir / f"{paper['id']}__{version}.json"
                sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2))

    out_path = args.out_dir / f"extraction_{timestamp}.md"
    render_report(rows, args.versions, out_path)
    if not args.no_save_samples:
        print(f"wrote {out_path} (samples in {samples_dir.name}/)")
    else:
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
