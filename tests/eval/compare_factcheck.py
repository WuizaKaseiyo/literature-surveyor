"""Compare fact_check_rendered_survey with vs without the extracted-claim anchor.

For the same rendered Stage 2 markdown + corpus, run the @tool twice:
  1. use_extracted_claims=True   (PR-3 path: anchor judge to matched claim's evidence_quote)
  2. use_extracted_claims=False  (BM25-only baseline: select context from raw paper text)

Compare verdict distributions, anchor hit rate, judge elapsed time, and the
per-item agreement between the two paths.

Usage:
    python tests/eval/compare_factcheck.py
    python tests/eval/compare_factcheck.py --use-llm                # judge via LLM (needs API key)
    python tests/eval/compare_factcheck.py --use-llm --model openai/gpt-4o-mini

Default uses the deterministic heuristic judge — fast and free but coarser.
LLM judge gives a more realistic comparison of the two modes; recommended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/factcheck_fixture"
DEFAULT_OUT_DIR = REPO_ROOT / "tests/eval/reports"


def run_one(markdown: str, use_extracted_claims: bool, use_llm: bool, model: str) -> dict:
    from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
        fact_check_rendered_survey,
    )

    t0 = time.time()
    out = fact_check_rendered_survey.invoke({
        "markdown": markdown,
        "use_llm": use_llm,
        "judge_model": model,
        "use_extracted_claims": use_extracted_claims,
    })
    out["__elapsed_s"] = time.time() - t0
    return out


def verdict_counts(items: list[dict]) -> dict[str, int]:
    return dict(Counter(i.get("verdict", "?") for i in items))


def anchor_hit_rate(items: list[dict]) -> float:
    if not items:
        return 0.0
    return sum(1 for i in items if i.get("matched_claim_id")) / len(items)


def diff_verdicts(items_a: list[dict], items_b: list[dict]) -> list[dict]:
    """Pair items by attribution_id and report differences."""
    a_by = {i["attribution_id"] + "__" + i["citation_id"]: i for i in items_a}
    b_by = {i["attribution_id"] + "__" + i["citation_id"]: i for i in items_b}
    diffs = []
    for key in sorted(set(a_by) | set(b_by)):
        a, b = a_by.get(key), b_by.get(key)
        va = a.get("verdict") if a else None
        vb = b.get("verdict") if b else None
        if va != vb:
            diffs.append({
                "attribution_id": key,
                "claim_text": (a or b).get("claim_text", "")[:120],
                "citation_id": (a or b).get("citation_id", ""),
                "anchored_verdict": va,
                "bm25_verdict": vb,
                "matched_claim_id": (a or {}).get("matched_claim_id", ""),
            })
    return diffs


def render_report(
    anchored: dict,
    bm25_only: dict,
    out_path: Path,
    fixture: Path,
    use_llm: bool,
    model: str,
) -> None:
    a_items = anchored.get("items", [])
    b_items = bm25_only.get("items", [])
    a_verdicts = verdict_counts(a_items)
    b_verdicts = verdict_counts(b_items)
    diffs = diff_verdicts(a_items, b_items)
    all_verdicts = sorted(set(a_verdicts) | set(b_verdicts))

    lines: list[str] = [
        f"# fact_check_rendered_survey comparison — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- fixture: `{fixture.name}`",
        f"- attributions parsed: {anchored.get('parsed_attributions', 0)}",
        f"- items checked: {anchored.get('total_checked', 0)}",
        f"- judge: {'LLM (' + model + ')' if use_llm else 'heuristic'}",
        "",
        "## Verdict distribution",
        "",
        "| verdict | anchored (PR-3) | BM25-only baseline |",
        "|---|---|---|",
    ]
    for v in all_verdicts:
        lines.append(f"| {v} | {a_verdicts.get(v, 0)} | {b_verdicts.get(v, 0)} |")
    lines.append("")

    lines.extend([
        "## Aggregate",
        "",
        "| metric | anchored | baseline |",
        "|---|---|---|",
        f"| blocking_count | {anchored.get('blocking_count', 0)} | {bm25_only.get('blocking_count', 0)} |",
        f"| supported_count | {anchored.get('supported_count', 0)} | {bm25_only.get('supported_count', 0)} |",
        f"| partial_count | {anchored.get('partial_count', 0)} | {bm25_only.get('partial_count', 0)} |",
        f"| unsupported_count | {anchored.get('unsupported_count', 0)} | {bm25_only.get('unsupported_count', 0)} |",
        f"| contradicted_count | {anchored.get('contradicted_count', 0)} | {bm25_only.get('contradicted_count', 0)} |",
        f"| anchor hit rate | {anchor_hit_rate(a_items):.0%} | n/a |",
        f"| elapsed | {anchored.get('__elapsed_s', 0):.1f}s | {bm25_only.get('__elapsed_s', 0):.1f}s |",
        "",
    ])

    if diffs:
        lines.extend([
            f"## Per-item differences ({len(diffs)})",
            "",
            "| attribution | citation | claim | anchored | baseline | matched_claim_id |",
            "|---|---|---|---|---|---|",
        ])
        for d in diffs:
            lines.append(
                f"| {d['attribution_id']} | {d['citation_id']} | "
                f"{d['claim_text']} | "
                f"`{d['anchored_verdict']}` | `{d['bm25_verdict']}` | "
                f"{d['matched_claim_id'] or '—'} |"
            )
    else:
        lines.append("## Per-item differences\n\nNone — both modes agree on every item.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--use-llm", action="store_true", help="enable LLM judge")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--no-save-samples", action="store_true")
    args = ap.parse_args()

    md_path = args.fixture / "stage2_rendered.md"
    corpus_path = args.fixture / "corpus"
    if not md_path.exists() or not corpus_path.exists():
        print(f"Missing fixture under {args.fixture}", file=sys.stderr)
        return 1

    markdown = md_path.read_text()
    os.environ["LITSURVEY_CORPUS_DIR"] = str(corpus_path)

    if args.use_llm and not (
        os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    ):
        print("--use-llm passed but no API key in env.", file=sys.stderr)
        return 1

    print("[anchored] use_extracted_claims=True …", file=sys.stderr)
    anchored = run_one(markdown, True, args.use_llm, args.model)
    print(
        f"  -> {anchored.get('total_checked')} items, "
        f"anchor hit {anchor_hit_rate(anchored.get('items', [])):.0%}, "
        f"{anchored['__elapsed_s']:.1f}s",
        file=sys.stderr,
    )

    print("[baseline] use_extracted_claims=False …", file=sys.stderr)
    bm25_only = run_one(markdown, False, args.use_llm, args.model)
    print(
        f"  -> {bm25_only.get('total_checked')} items, {bm25_only['__elapsed_s']:.1f}s",
        file=sys.stderr,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"factcheck_{timestamp}.md"
    render_report(anchored, bm25_only, out_path, args.fixture, args.use_llm, args.model)

    if not args.no_save_samples:
        samples_dir = args.out_dir / f"factcheck_{timestamp}_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        (samples_dir / "anchored.json").write_text(
            json.dumps(anchored, ensure_ascii=False, indent=2)
        )
        (samples_dir / "bm25_only.json").write_text(
            json.dumps(bm25_only, ensure_ascii=False, indent=2)
        )
        print(f"wrote {out_path} (samples in {samples_dir.name}/)")
    else:
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
