"""A2 layered-mode cross-project reuse demo.

Runs the full Stage 2 pipeline twice on the same set of papers, with two
DIFFERENT project dirs sharing ONE global corpus pool:

  Project A — cold start. Pays the full extract_claims LLM bill.
  Project B — same papers via the same global pool. Every extract_claims call
              should short-circuit to `status="reused"` with 0 LLM tokens.

Both runs go all the way through claim_search → render → verify_citations →
fact_check_rendered_survey to confirm the *full* pipeline (not just extraction)
sees the cached claims.

Output: tests/e2e/smoke_runs/<timestamp>_layered_demo/
  ├── global/              # shared corpus pool
  ├── project_a/           # cold-start project's refs
  ├── project_b/           # warm-start project's refs
  ├── project_a_run.json   # full pipeline output for A
  ├── project_b_run.json   # full pipeline output for B
  └── summary.md           # side-by-side cost / time comparison

Usage:
    OPENAI_API_KEY=sk-... python tests/e2e/run_layered_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_PAPERS = REPO_ROOT / "tests/fixtures/golden_papers/papers.jsonl"
SMOKE_RUNS_DIR = REPO_ROOT / "tests/e2e/smoke_runs"

FINDINGS_RECIPE = {
    "2305.11206": (
        "LIMA fine-tuned on 1000 curated examples is preferred or equivalent "
        "to GPT-4 in 43% of human comparisons.",
        "LIMA GPT-4 human preference comparison",
        "Zhou et al. 2023, arxiv:2305.11206",
    ),
    "2305.18290": (
        "DPO matches or improves response quality versus PPO-based RLHF on "
        "summarization and single-turn dialogue.",
        "DPO PPO summarization dialogue response quality",
        "Rafailov et al. 2023, arxiv:2305.18290",
    ),
}


def _load_two_smallest_papers() -> list[dict]:
    papers = []
    with GOLDEN_PAPERS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p["id"] in FINDINGS_RECIPE:
                papers.append(p)
    papers.sort(key=lambda p: p.get("char_count", 0))
    return papers[:2]


def run_project(
    project_dir: Path, global_dir: Path, papers: list[dict], model: str, label: str
) -> dict:
    """Run the full pipeline for one project against the shared global pool."""
    project_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LITSURVEY_GLOBAL_CORPUS_DIR"] = str(global_dir)
    os.environ["LITSURVEY_CORPUS_DIR"] = str(project_dir)

    from tools.claim_store.claim_store import claim_search
    from tools.corpus_store.corpus_store import corpus_add_paper, corpus_status
    from tools.extract_claims.extract_claims import extract_claims
    from tools.fact_check_rendered_survey.fact_check_rendered_survey import (
        fact_check_rendered_survey,
    )
    from tools.verify_citations.verify_citations import verify_citations
    from schemas.literature_survey_schema import (
        CitationRef,
        CorpusSummary,
        Finding,
        LiteratureSurveySchema,
        SearchStrategy,
    )

    result: dict = {"label": label, "papers": [p["id"] for p in papers]}
    print(f"\n==== {label} ====", file=sys.stderr)

    # 1) corpus_status snapshot
    pre_status = corpus_status.invoke({})
    print(
        f"  pre-status: project={pre_status.get('corpus_size', 0)}, "
        f"global={pre_status.get('global_corpus_size', '?')}",
        file=sys.stderr,
    )

    # 2) add papers + extract claims
    t0 = time.time()
    extract_summary: list[dict] = []
    for p in papers:
        corpus_add_paper.invoke({"paper": p})
        t_ext = time.time()
        out = extract_claims.invoke({
            "paper_id": p["id"],
            "model": model,
            "version": "v2",
        })
        elapsed = time.time() - t_ext
        extract_summary.append({
            "paper_id": p["id"],
            "status": out.get("status", "?"),
            "claims_extracted": out.get("claims_extracted", 0),
            "elapsed_s": elapsed,
            "error": out.get("error", ""),
        })
        print(
            f"  {p['id']}: status={out.get('status', '?')} "
            f"claims={out.get('claims_extracted', 0)} {elapsed:.1f}s",
            file=sys.stderr,
        )
    extract_total = time.time() - t0
    result["extract"] = {"per_paper": extract_summary, "total_elapsed_s": extract_total}

    # 3) build findings via claim_search
    all_claims_by_id: dict[str, dict] = {}
    for p in papers:
        for c in claim_search.invoke({
            "query": "x",  # ignored — we just want to enumerate
            "paper_ids": [p["id"]],
            "top_k": 50,
        }).get("results", []):
            all_claims_by_id[c["claim"]["id"]] = c["claim"]

    findings_built = []
    for p in papers:
        text, query, cite_text = FINDINGS_RECIPE[p["id"]]
        backing = claim_search.invoke({
            "query": query,
            "paper_ids": [p["id"]],
            "top_k": 3,
        })
        if not backing["results"]:
            continue
        findings_built.append(Finding(
            text=text,
            cites=[CitationRef(paper_id=p["id"], cite_text=cite_text)],
            claim_ids=[r["claim"]["id"] for r in backing["results"][:2]],
            confidence=0.8,
        ))

    # 4) build survey + render markdown
    survey = LiteratureSurveySchema(
        research_question="Effectiveness of post-training alignment recipes.",
        search_strategy=SearchStrategy(queries=["RLHF DPO"], sources=["arxiv"]),
        corpus_summary=CorpusSummary(total_papers=len(papers)),
        taxonomy=[],
        findings=findings_built,
    )

    md_lines = ["# Stage 2 (layered demo run)", "", "## Findings", ""]
    for f in findings_built:
        md_lines.append(
            f"{f.text[:-1]} [{f.cites[0].cite_text}]."
        )
        if f.claim_ids:
            top_claim = all_claims_by_id.get(f.claim_ids[0])
            if top_claim:
                md_lines.append("")
                md_lines.append(
                    f'> evidence: "{top_claim.get("evidence_quote", "")[:300]}"'
                )
                md_lines.append(
                    f"> — {top_claim.get('source_section') or '?'} "
                    f"(arxiv:{f.cites[0].paper_id}#{f.claim_ids[0].split('#')[-1]})"
                )
        md_lines.append("")
    markdown = "\n".join(md_lines)

    # 5) verify_citations
    t_v = time.time()
    cite_audit = verify_citations.invoke({"markdown": markdown})
    cite_elapsed = time.time() - t_v

    # 6) fact_check w/ LLM
    t_f = time.time()
    fc_audit = fact_check_rendered_survey.invoke({
        "markdown": markdown,
        "use_llm": True,
        "judge_model": model,
        "use_extracted_claims": True,
    })
    fc_elapsed = time.time() - t_f

    # 7) cross-check
    declared = {cid for f in findings_built for cid in f.claim_ids}
    matched = {it.get("matched_claim_id", "") for it in fc_audit["items"]}
    matched -= {""}

    result["verify"] = {
        "verified_count": cite_audit["verified_count"],
        "unverified_count": cite_audit["unverified_count"],
        "elapsed_s": cite_elapsed,
    }
    result["factcheck"] = {
        "supported_count": fc_audit.get("supported_count", 0),
        "blocking_count": fc_audit.get("blocking_count", 0),
        "elapsed_s": fc_elapsed,
        "matched_claim_ids": sorted(matched),
        "declared_claim_ids": sorted(declared),
        "anchor_consistent": matched.issubset(declared),
    }
    result["status_after"] = corpus_status.invoke({})
    result["markdown_chars"] = len(markdown)

    print(
        f"  fact_check: supported={fc_audit.get('supported_count', 0)} "
        f"blocking={fc_audit.get('blocking_count', 0)} {fc_elapsed:.1f}s",
        file=sys.stderr,
    )
    return result


def main() -> int:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print("No LLM API key in env.", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_RUNS_DIR / f"{timestamp}_layered_demo"
    global_dir = run_dir / "global"
    project_a = run_dir / "project_a"
    project_b = run_dir / "project_b"
    for d in (run_dir, global_dir, project_a, project_b):
        d.mkdir(parents=True, exist_ok=True)

    papers = _load_two_smallest_papers()
    if not papers:
        print("no fixture papers", file=sys.stderr)
        return 1
    print(f"using papers: {[p['id'] for p in papers]}", file=sys.stderr)

    model = "openai/gpt-4o-mini"

    # Phase 1: project A (cold start — pays the LLM bill)
    a = run_project(project_a, global_dir, papers, model, "Project A (cold start)")
    (run_dir / "project_a_run.json").write_text(json.dumps(a, default=str, indent=2))

    # Phase 2: project B (warm — should reuse global claims)
    b = run_project(project_b, global_dir, papers, model, "Project B (warm — shared global)")
    (run_dir / "project_b_run.json").write_text(json.dumps(b, default=str, indent=2))

    # Side-by-side summary
    a_extract_total = a["extract"]["total_elapsed_s"]
    b_extract_total = b["extract"]["total_elapsed_s"]
    a_reused = sum(1 for x in a["extract"]["per_paper"] if x["status"] == "reused")
    b_reused = sum(1 for x in b["extract"]["per_paper"] if x["status"] == "reused")
    a_claims = sum(x["claims_extracted"] for x in a["extract"]["per_paper"])
    b_claims = sum(x["claims_extracted"] for x in b["extract"]["per_paper"])

    saved_pct = (
        (1 - b_extract_total / a_extract_total) * 100 if a_extract_total > 0 else 0
    )

    summary = [
        f"# A2 layered-mode reuse demo — {timestamp}",
        "",
        f"papers: {', '.join(p['id'] for p in papers)}",
        f"model: {model}",
        f"global pool: `{global_dir}`",
        "",
        "## Phase comparison",
        "",
        "| metric | Project A (cold) | Project B (warm) | savings |",
        "|---|---|---|---|",
        f"| extract elapsed | {a_extract_total:.1f}s | {b_extract_total:.1f}s | {saved_pct:.0f}% |",
        f"| papers reused | {a_reused}/{len(papers)} | {b_reused}/{len(papers)} | — |",
        f"| claims surfaced | {a_claims} | {b_claims} | — |",
        f"| verify elapsed | {a['verify']['elapsed_s']:.2f}s | {b['verify']['elapsed_s']:.2f}s | — |",
        f"| fact_check elapsed | {a['factcheck']['elapsed_s']:.1f}s | {b['factcheck']['elapsed_s']:.1f}s | — |",
        f"| fact_check supported | {a['factcheck']['supported_count']} | {b['factcheck']['supported_count']} | — |",
        f"| fact_check blocking | {a['factcheck']['blocking_count']} | {b['factcheck']['blocking_count']} | — |",
        "",
        "## Per-paper extract status",
        "",
        "| paper | Project A status | A elapsed | Project B status | B elapsed |",
        "|---|---|---|---|---|",
    ]
    for ap, bp in zip(a["extract"]["per_paper"], b["extract"]["per_paper"]):
        summary.append(
            f"| {ap['paper_id']} | {ap['status']} | {ap['elapsed_s']:.1f}s | "
            f"{bp['status']} | {bp['elapsed_s']:.2f}s |"
        )
    summary.append("")

    summary.append("## Global pool after both runs")
    summary.append("")
    end_status = b["status_after"]
    summary.append(f"- mode: {end_status.get('mode', 'legacy')}")
    summary.append(f"- global papers: {end_status.get('global_corpus_size', '?')}")
    summary.append(f"- global claims: {end_status.get('global_claims_count', '?')}")
    summary.append(f"- project A refs: {sum(1 for _ in (project_a / 'refs.jsonl').open() if _.strip())}")
    summary.append(f"- project B refs: {sum(1 for _ in (project_b / 'refs.jsonl').open() if _.strip())}")
    summary.append("")

    if (
        b_reused == len(papers)
        and a_reused == 0
        and b_extract_total < a_extract_total / 5
        and b["factcheck"]["anchor_consistent"]
        and b["factcheck"]["blocking_count"] == 0
    ):
        summary.append("STATUS: ✅ A2 cross-project reuse works end-to-end")
    else:
        summary.append("STATUS: ⚠ unexpected — see audits")
    summary.append("")

    (run_dir / "summary.md").write_text("\n".join(summary))
    print(f"\nwrote {run_dir}/")
    print("\n" + "\n".join(summary[-12:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
