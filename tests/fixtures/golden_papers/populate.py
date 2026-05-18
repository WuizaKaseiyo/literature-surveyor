"""Populate the golden paper fixtures from arxiv.

Five papers chosen to cover the profile in README.md (short alignment /
method-heavy / long tech report / negative result / long analysis).

Metadata is hardcoded (arxiv export API is rate-limited from many IPs and the
fixture should be stable regardless). PDF text is fetched fresh via pdf_extract
on each populate call. Re-running is idempotent — papers already present with
non-empty full_text_md are skipped unless --force.

Usage:
    python tests/fixtures/golden_papers/populate.py
    python tests/fixtures/golden_papers/populate.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

PAPERS_JSONL = Path(__file__).parent / "papers.jsonl"

PICKS: list[dict] = [
    {
        "id": "2305.11206",
        "title": "LIMA: Less Is More for Alignment",
        "authors": [
            "Chunting Zhou", "Pengfei Liu", "Puxin Xu", "Srini Iyer", "Jiao Sun",
            "Yuning Mao", "Xuezhe Ma", "Avia Efrat", "Ping Yu", "Lili Yu",
            "Susan Zhang", "Gargi Ghosh", "Mike Lewis", "Luke Zettlemoyer", "Omer Levy",
        ],
        "year": 2023,
        "venue": "NeurIPS",
        "abstract": (
            "Large language models are trained in two stages: (1) unsupervised pretraining "
            "from raw text, to learn general-purpose representations, and (2) large scale "
            "instruction tuning and reinforcement learning, to better align to end tasks "
            "and user preferences. We measure the relative importance of these two stages "
            "by training LIMA, a 65B parameter LLaMa language model fine-tuned with the "
            "standard supervised loss on only 1,000 carefully curated prompts and responses..."
        ),
        "fixture_role": "short alignment paper",
    },
    {
        "id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": [
            "Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon",
            "Christopher D. Manning", "Chelsea Finn",
        ],
        "year": 2023,
        "venue": "NeurIPS",
        "abstract": (
            "While large-scale unsupervised language models (LMs) learn broad world "
            "knowledge and some reasoning skills, achieving precise control of their "
            "behavior is difficult due to the completely unsupervised nature of their "
            "training. Existing methods for gaining such steerability collect human "
            "labels of the relative quality of model generations and fine-tune the "
            "unsupervised LM to align with these preferences, often with reinforcement "
            "learning from human feedback (RLHF)..."
        ),
        "fixture_role": "method-heavy",
    },
    {
        "id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": [
            "Hugo Touvron", "Louis Martin", "Kevin Stone", "Peter Albert", "Amjad Almahairi",
            "Yasmine Babaei", "et al.",
        ],
        "year": 2023,
        "venue": "arxiv",
        "abstract": (
            "In this work, we develop and release Llama 2, a collection of pretrained and "
            "fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 "
            "billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized "
            "for dialogue use cases..."
        ),
        "fixture_role": "long tech report",
    },
    {
        "id": "2305.15717",
        "title": "The False Promise of Imitating Proprietary LLMs",
        "authors": [
            "Arnav Gudibande", "Eric Wallace", "Charlie Snell", "Xinyang Geng",
            "Hao Liu", "Pieter Abbeel", "Sergey Levine", "Dawn Song",
        ],
        "year": 2023,
        "venue": "arxiv",
        "abstract": (
            "An emerging method to cheaply improve a weaker language model is to finetune "
            "it on outputs from a stronger model, such as a proprietary system like ChatGPT. "
            "This approach looks to cheaply imitate the proprietary model's capabilities... "
            "we critically analyze this approach. We first finetune a series of LMs that "
            "imitate ChatGPT using varying base model sizes (1.5B--13B), data sources, and "
            "imitation data amounts (0.3M--150M tokens). We then evaluate the models using "
            "crowd raters and canonical NLP benchmarks. Initially, we were surprised by the "
            "output quality of our imitation models -- they appear far better at following "
            "instructions, and crowd workers rate their outputs as competitive with ChatGPT. "
            "However, when conducting more targeted automatic evaluations, we find that "
            "imitation models close little to none of the gap..."
        ),
        "fixture_role": "negative result",
    },
    {
        "id": "2303.12712",
        "title": "Sparks of Artificial General Intelligence: Early experiments with GPT-4",
        "authors": [
            "Sebastien Bubeck", "Varun Chandrasekaran", "Ronen Eldan", "Johannes Gehrke",
            "Eric Horvitz", "Ece Kamar", "Peter Lee", "Yin Tat Lee", "Yuanzhi Li",
            "Scott Lundberg", "Harsha Nori", "Hamid Palangi", "Marco Tulio Ribeiro",
            "Yi Zhang",
        ],
        "year": 2023,
        "venue": "arxiv",
        "abstract": (
            "Artificial intelligence (AI) researchers have been developing and refining "
            "large language models (LLMs) that exhibit remarkable capabilities across a "
            "variety of domains and tasks... We contend that (this early version of) "
            "GPT-4 is part of a new cohort of LLMs that exhibit more general intelligence "
            "than previous AI models..."
        ),
        "fixture_role": "long evaluation/analysis",
    },
]


def _normalize_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip().lower())


def _load_existing() -> dict[str, dict]:
    if not PAPERS_JSONL.exists():
        return {}
    out: dict[str, dict] = {}
    for line in PAPERS_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
            out[_normalize_id(p["id"])] = p
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _save_all(papers: dict[str, dict]) -> None:
    with PAPERS_JSONL.open("w") as f:
        for p in papers.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _extract(pdf_url: str) -> dict:
    from tools.pdf_extract.pdf_extract import pdf_extract
    return pdf_extract.invoke({"pdf_path_or_url": pdf_url})


def populate(force: bool = False) -> int:
    existing = _load_existing()
    print(f"existing fixtures: {len(existing)}")

    fetched = 0
    failed: list[str] = []
    for pick in PICKS:
        nid = _normalize_id(pick["id"])
        if nid in existing and not force and existing[nid].get("full_text_md"):
            print(f"  skip {pick['id']} ({pick['fixture_role']}) — already present")
            continue

        print(f"\n[{pick['id']}] {pick['fixture_role']}")
        pdf_url = f"https://arxiv.org/pdf/{pick['id']}.pdf"
        print(f"  fetching: {pdf_url}")
        ex = _extract(pdf_url)
        if ex.get("source") == "error":
            print(f"  ! pdf_extract failed: {ex.get('error')}")
            failed.append(pick["id"])
            continue

        paper = {
            "id": nid,
            "title": pick["title"],
            "authors": pick["authors"],
            "year": pick["year"],
            "venue": pick["venue"],
            "abstract": pick["abstract"],
            "full_text_md": ex.get("markdown", ""),
            "source": "arxiv",
            "source_query": "fixture",
            "fixture_role": pick["fixture_role"],
            "pdf_source": ex.get("source"),
            "char_count": ex.get("char_count", 0),
            "added_at": time.time(),
        }
        existing[nid] = paper
        fetched += 1
        print(f"  -> {ex.get('char_count')} chars via {ex.get('source')}")
        time.sleep(2)

    _save_all(existing)
    print(f"\nwrote {PAPERS_JSONL}: {len(existing)} papers ({fetched} new)")
    if failed:
        print(f"failed: {failed}")
    return 0 if not failed else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    args = ap.parse_args()
    return populate(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
