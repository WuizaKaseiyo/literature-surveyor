"""End-to-end smoke run: full pipeline with real LLM calls.

NOT a pytest test — manually invoked. Costs ~$0.02-0.10 per run depending on
how many papers you exercise. Run after any change that touches the LLM-driven
parts (extract_claims, fact_check_rendered_survey) to catch regressions the
offline e2e test (`tests/test_e2e_pipeline.py`) can't see.

Pipeline executed:
  1. Pick N papers from tests/fixtures/golden_papers/papers.jsonl
  2. corpus_add_paper each into a fresh tmp corpus
  3. extract_claims version="v2" on each (real LLM)
  4. For each paper, claim_search to surface backing claims for 1 finding
  5. Build LiteratureSurveySchema with the findings
  6. Render stage2_literature_surveyor.md following G段 convention
  7. verify_citations (corpus-only, no API)
  8. fact_check_rendered_survey with use_llm=True
  9. Save everything under tests/e2e/smoke_runs/<timestamp>/

Usage:
    OPENAI_API_KEY=sk-... python tests/e2e/run_smoke_pipeline.py
    OPENAI_API_KEY=sk-... python tests/e2e/run_smoke_pipeline.py --papers 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_PAPERS = REPO_ROOT / "tests/fixtures/golden_papers/papers.jsonl"
SMOKE_RUNS_DIR = REPO_ROOT / "tests/e2e/smoke_runs"

# Recipes: paper id -> (finding_text, claim_search_query, cite_text)
FINDINGS_RECIPE = {
    "2305.11206": (
        "LIMA fine-tuned on 1000 curated examples is preferred or equivalent "
        "to GPT-4 in 43% of human comparisons.",
        "LIMA GPT-4 human preference comparison 43",
        "Zhou et al. 2023, arxiv:2305.11206",
    ),
    "2305.18290": (
        "DPO matches or improves response quality versus PPO-based RLHF on "
        "summarization and single-turn dialogue.",
        "DPO PPO summarization single-turn dialogue response quality",
        "Rafailov et al. 2023, arxiv:2305.18290",
    ),
    "2307.09288": (
        "Llama 2 70B improves MMLU and BBH results by approximately 5 and 8 "
        "points compared to Llama 1 65B.",
        "Llama 2 70B MMLU BBH improvement Llama 1",
        "Touvron et al. 2023, arxiv:2307.09288",
    ),
    "2305.15717": (
        "Imitation models trained on outputs from stronger LLMs close little "
        "to no capability gap on targeted automatic evaluations.",
        "imitation models proprietary LLMs capability gap automatic evaluation",
        "Gudibande et al. 2023, arxiv:2305.15717",
    ),
    "2303.12712": (
        "The authors argue that early GPT-4 exhibits more general intelligence "
        "than previous AI models across mathematics, coding, and vision.",
        "GPT-4 general intelligence mathematics coding vision evaluation",
        "Bubeck et al. 2023, arxiv:2303.12712",
    ),
}


def _load_golden_papers(limit: int) -> list[dict]:
    papers = []
    with GOLDEN_PAPERS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    # Pick the smaller papers first so the run is fast.
    papers.sort(key=lambda p: p.get("char_count", 0))
    return papers[:limit]


def _render_markdown(survey_json: dict, claim_lookup: dict[str, dict]) -> str:
    lines = [
        "# Stage 2: Literature Survey (smoke run)",
        "",
        f"## Research question",
        "",
        survey_json["research_question"],
        "",
        "## Findings",
        "",
    ]
    for finding in survey_json["findings"]:
        cite_text = finding["cites"][0]["cite_text"]
        para = f"{finding['text'][:-1]} [{cite_text}]."
        lines.append(para)
        if finding.get("claim_ids"):
            top_cid = finding["claim_ids"][0]
            top_claim = claim_lookup.get(top_cid)
            if top_claim:
                quote = top_claim.get("evidence_quote", "")[:300]
                section = top_claim.get("source_section") or "(no section)"
                lines.append("")
                lines.append(f'> evidence: "{quote}"')
                lines.append(f"> — {section} (arxiv:{top_cid})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=2, help="how many papers to exercise")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument(
        "--extract-version", default="v2", choices=["v1", "v2"],
        help="which extract_claims version to run",
    )
    args = ap.parse_args()

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        print("No LLM API key in env.", file=sys.stderr)
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = run_dir / "corpus"
    corpus_dir.mkdir(parents=True)
    os.environ["LITSURVEY_CORPUS_DIR"] = str(corpus_dir)

    print(f"smoke run -> {run_dir}", file=sys.stderr)

    # Import here so env vars are set first.
    from tools.claim_store.claim_store import claim_search, claim_status
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

    # --- Step 1-2: populate corpus from golden_papers ---
    papers = _load_golden_papers(args.papers)
    if not papers:
        print("no golden papers found", file=sys.stderr)
        return 1
    print(f"\n[1] populating corpus with {len(papers)} papers ...", file=sys.stderr)
    for p in papers:
        r = corpus_add_paper.invoke({"paper": p})
        print(f"  + {p['id']} ({p.get('char_count', 0)} chars): {r['status']}", file=sys.stderr)

    cs = corpus_status.invoke({})
    print(f"  corpus_size={cs['corpus_size']}", file=sys.stderr)

    # --- Step 3: extract claims (real LLM) ---
    print(f"\n[2] extract_claims version={args.extract_version} ...", file=sys.stderr)
    t0 = time.time()
    extract_outs: dict[str, dict] = {}
    for p in papers:
        out = extract_claims.invoke({
            "paper_id": p["id"],
            "model": args.model,
            "version": args.extract_version,
        })
        extract_outs[p["id"]] = out
        n = out.get("claims_extracted", 0)
        wn = len(out.get("warnings", []))
        err = out.get("error", "")
        print(f"  {p['id']}: {n} claims ({wn} warnings) {err}", file=sys.stderr)
    elapsed_extract = time.time() - t0

    cls = claim_status.invoke({})
    print(f"  total claims in corpus: {cls['claims_count']}", file=sys.stderr)

    # --- Step 4: per paper, find backing claims for a candidate finding ---
    print(f"\n[3] claim_search for backing claims ...", file=sys.stderr)
    findings_built: list[Finding] = []
    all_claims_by_id: dict[str, dict] = {}
    for cls_out in [extract_outs[p["id"]] for p in papers]:
        for c in cls_out.get("claims", []):
            all_claims_by_id[c["id"]] = c

    for p in papers:
        recipe = FINDINGS_RECIPE.get(p["id"])
        if not recipe:
            continue
        text, query, cite_text = recipe
        search_out = claim_search.invoke({
            "query": query,
            "paper_ids": [p["id"]],
            "top_k": 3,
        })
        backing_ids = [r["claim"]["id"] for r in search_out["results"]]
        if not backing_ids:
            print(f"  WARN no backing claims for {p['id']} — skipping finding", file=sys.stderr)
            continue
        f = Finding(
            text=text,
            cites=[CitationRef(paper_id=p["id"], cite_text=cite_text)],
            claim_ids=backing_ids[:2],
            confidence=0.8,
        )
        findings_built.append(f)
        print(
            f"  {p['id']}: finding backed by {len(backing_ids)} claims "
            f"(top score {search_out['results'][0]['score']})",
            file=sys.stderr,
        )

    if not findings_built:
        print("no findings produced", file=sys.stderr)
        return 1

    # --- Step 5: build survey schema ---
    survey = LiteratureSurveySchema(
        research_question="Effectiveness of post-training alignment recipes across LLMs.",
        search_strategy=SearchStrategy(
            queries=["RLHF DPO instruction tuning alignment"],
            sources=["arxiv"],
            time_window="2023",
        ),
        corpus_summary=CorpusSummary(total_papers=len(papers)),
        taxonomy=[],
        findings=findings_built,
    )
    (run_dir / "stage2.json").write_text(survey.model_dump_json(indent=2))

    # --- Step 6: render markdown ---
    survey_dict = survey.model_dump()
    markdown = _render_markdown(survey_dict, all_claims_by_id)
    md_path = run_dir / "stage2_literature_surveyor.md"
    md_path.write_text(markdown)

    # --- Step 7: verify_citations (no network — corpus-only resolves first) ---
    print(f"\n[4] verify_citations ...", file=sys.stderr)
    cite_audit = verify_citations.invoke({"markdown": markdown})
    print(
        f"  verified={cite_audit['verified_count']} unverified={cite_audit['unverified_count']}",
        file=sys.stderr,
    )
    (run_dir / "citation_audit.json").write_text(json.dumps(cite_audit, indent=2))

    # --- Step 8: fact_check_rendered_survey (LLM in loop) ---
    print(f"\n[5] fact_check_rendered_survey use_llm=True ...", file=sys.stderr)
    t0 = time.time()
    fc_audit = fact_check_rendered_survey.invoke({
        "markdown": markdown,
        "use_llm": True,
        "judge_model": args.model,
        "use_extracted_claims": True,
    })
    elapsed_fc = time.time() - t0
    (run_dir / "factcheck_audit.json").write_text(json.dumps(fc_audit, indent=2))

    # --- Cross-check matched_claim_id ⊆ Finding.claim_ids ---
    declared_claim_ids = {cid for f in findings_built for cid in f.claim_ids}
    matched_ids = [it.get("matched_claim_id", "") for it in fc_audit["items"]]
    matched_set = {m for m in matched_ids if m}
    in_declared = matched_set & declared_claim_ids
    out_of_declared = matched_set - declared_claim_ids

    summary_lines = [
        f"# Smoke run summary — {timestamp}",
        "",
        f"- papers: {len(papers)} ({', '.join(p['id'] for p in papers)})",
        f"- extract version: {args.extract_version}, model: {args.model}",
        f"- claims extracted: {cls['claims_count']}",
        f"- claims by paper: " + ", ".join(
            f"{p['id']}={extract_outs[p['id']].get('claims_extracted', 0)}" for p in papers
        ),
        f"- extract elapsed: {elapsed_extract:.1f}s",
        "",
        "## Findings",
        f"- built: {len(findings_built)}",
        f"- declared claim_ids: {sorted(declared_claim_ids)}",
        "",
        "## Citation audit",
        f"- verified: {cite_audit['verified_count']}",
        f"- unverified: {cite_audit['unverified_count']}",
        "",
        "## Fact check audit (LLM judge)",
        f"- elapsed: {elapsed_fc:.1f}s",
        f"- total_checked: {fc_audit.get('total_checked', 0)}",
        f"- supported: {fc_audit.get('supported_count', 0)}",
        f"- partial: {fc_audit.get('partial_count', 0)}",
        f"- unsupported: {fc_audit.get('unsupported_count', 0)}",
        f"- contradicted: {fc_audit.get('contradicted_count', 0)}",
        f"- blocking: {fc_audit.get('blocking_count', 0)}",
        "",
        "## Cross-check: matched_claim_id vs Finding.claim_ids",
        f"- matched_claim_ids from fact_check: {sorted(matched_set)}",
        f"- ⊆ Finding.claim_ids ({len(in_declared)}/{len(matched_set)}): {sorted(in_declared)}",
        f"- not in Finding.claim_ids: {sorted(out_of_declared) or '(none — good)'}",
        "",
    ]
    if fc_audit.get("blocking_count", 0) == 0 and not out_of_declared:
        summary_lines.append("STATUS: ✅ pipeline produced a clean survey end to end")
    else:
        summary_lines.append("STATUS: ⚠ needs investigation — see audits")

    (run_dir / "summary.md").write_text("\n".join(summary_lines))

    print(f"\nwrote {run_dir}/")
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")
    print("\n" + "\n".join(summary_lines[-6:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
