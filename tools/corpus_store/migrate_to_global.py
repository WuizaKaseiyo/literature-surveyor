"""Migrate a legacy single-tenant corpus into the new layered global store.

Reads `<source>/papers.jsonl` and `<source>/claims.jsonl` (if present), copies
deduplicated entries into the global store, builds a `<source>/refs.jsonl`
that links each paper to the project, and renames the source files to
`.bak.<timestamp>` so the old layout doesn't shadow the new one.

Idempotent: re-running picks up only entries not already in global. Use
`--dry-run` to see what would happen without writing anything.

Usage:
    python -m tools.corpus_store.migrate_to_global --source /path/to/project/corpus
    python -m tools.corpus_store.migrate_to_global --source /path/to/project/corpus --global ~/my-global-pool
    python -m tools.corpus_store.migrate_to_global --source /path/to/project/corpus --dry-run

Set LITSURVEY_GLOBAL_CORPUS_DIR in your shell afterwards to opt every project
into layered mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _existing_ids(jsonl_path: Path, key: str = "id") -> set[str]:
    return {r.get(key) for r in _read_jsonl(jsonl_path) if r.get(key)}


def migrate(
    source: Path,
    global_dir: Path,
    dry_run: bool = False,
    keep_backup: bool = True,
) -> dict:
    source = source.expanduser().resolve()
    global_dir = global_dir.expanduser().resolve()
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return {"error": "source_missing"}

    global_dir.mkdir(parents=True, exist_ok=True)

    src_papers = source / "papers.jsonl"
    src_claims = source / "claims.jsonl"
    dst_papers = global_dir / "papers.jsonl"
    dst_claims = global_dir / "claims.jsonl"
    dst_refs = source / "refs.jsonl"
    project_meta = source / "project_meta.json"

    legacy_papers = _read_jsonl(src_papers)
    legacy_claims = _read_jsonl(src_claims)
    if not legacy_papers and not legacy_claims:
        print(f"nothing to migrate from {source}", file=sys.stderr)
        return {"papers_promoted": 0, "claims_promoted": 0, "refs_written": 0}

    existing_global_paper_ids = _existing_ids(dst_papers, "id")
    existing_global_claim_ids = _existing_ids(dst_claims, "id")

    papers_to_promote = [p for p in legacy_papers if p.get("id") not in existing_global_paper_ids]
    claims_to_promote = [c for c in legacy_claims if c.get("id") not in existing_global_claim_ids]
    refs_to_write = [
        {
            "paper_id": p["id"],
            "source_query": p.get("source_query", ""),
            "found_via": p.get("source", ""),
            "added_at": p.get("added_at", time.time()),
            "kept": True,
            "migrated_from": str(src_papers),
        }
        for p in legacy_papers
    ]

    summary = {
        "source": str(source),
        "global": str(global_dir),
        "papers_in_source": len(legacy_papers),
        "claims_in_source": len(legacy_claims),
        "papers_promoted": len(papers_to_promote),
        "papers_already_in_global": len(legacy_papers) - len(papers_to_promote),
        "claims_promoted": len(claims_to_promote),
        "claims_already_in_global": len(legacy_claims) - len(claims_to_promote),
        "refs_to_write": len(refs_to_write),
        "dry_run": dry_run,
    }

    if dry_run:
        print(json.dumps(summary, indent=2))
        return summary

    # Write to global (append-only)
    if papers_to_promote:
        with dst_papers.open("a") as f:
            for p in papers_to_promote:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if claims_to_promote:
        with dst_claims.open("a") as f:
            for c in claims_to_promote:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Project refs (overwrite — refs.jsonl is per-project derived data)
    if refs_to_write:
        with dst_refs.open("w") as f:
            for r in refs_to_write:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # project_meta.json
    if not project_meta.exists():
        import hashlib
        pid = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        project_meta.write_text(json.dumps({
            "project_id": pid,
            "project_path": str(source),
            "created_at": time.time(),
            "migrated_from_legacy": True,
            "retired_paper_ids": [],
        }, ensure_ascii=False, indent=2))

    # Rename legacy single-tenant entity files so they don't shadow global.
    # In layered mode tools resolve papers.jsonl via global; renaming the
    # source copy ensures any tool that still uses the old path doesn't get
    # stale data.
    ts = time.strftime("%Y%m%d_%H%M%S")
    if keep_backup:
        if src_papers.exists():
            src_papers.rename(source / f"papers.jsonl.bak.{ts}")
        if src_claims.exists():
            src_claims.rename(source / f"claims.jsonl.bak.{ts}")
        summary["backups_kept"] = True
    else:
        src_papers.unlink(missing_ok=True)
        src_claims.unlink(missing_ok=True)

    print(json.dumps(summary, indent=2))
    print(
        "\nset LITSURVEY_GLOBAL_CORPUS_DIR={} in your shell to activate layered mode."
        .format(global_dir),
        file=sys.stderr,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="legacy corpus dir")
    ap.add_argument(
        "--global", dest="global_dir", type=Path,
        default=Path.home() / ".litsurvey_corpus_global",
        help="target global corpus dir (default ~/.litsurvey_corpus_global)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-backup", action="store_true",
        help="delete legacy papers.jsonl/claims.jsonl instead of renaming",
    )
    args = ap.parse_args()

    out = migrate(
        source=args.source,
        global_dir=args.global_dir,
        dry_run=args.dry_run,
        keep_backup=not args.no_backup,
    )
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
