#!/usr/bin/env python3
"""Filter reviewed MOVi-E candidate pairs into an evaluation parquet dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Candidate pair JSONL")
    parser.add_argument("allowlist", type=Path, help="One approved source_qid per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _allowlist(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        source_id = line.split("#", 1)[0].strip()
        if source_id:
            ids.add(source_id)
    if not ids:
        raise ValueError("The allowlist contains no source IDs")
    return ids


def main() -> None:
    args = parse_args()
    approved = _allowlist(args.allowlist)
    rows = _read_jsonl(args.input)
    selected = [row for row in rows if row.get("source_qid") in approved]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_source[str(row.get("source_qid"))].append(row)
    missing = approved - set(by_source)
    incomplete = [source_id for source_id, pair in by_source.items() if {row.get("diagnostic_answer_format") for row in pair} != {"direction", "object"}]
    if missing or incomplete:
        details = []
        if missing:
            details.append(f"{len(missing)} unknown ID(s), e.g. {sorted(missing)[0]!r}")
        if incomplete:
            details.append(f"{len(incomplete)} incomplete pair(s), e.g. {incomplete[0]!r}")
        raise ValueError("Invalid allowlist: " + "; ".join(details))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"movi_e_{args.split}_direction_object.jsonl"
    parquet_path = args.output_dir / f"movi_e_{args.split}_direction_object.parquet"
    sample_path = args.output_dir / "sample_pairs.jsonl"
    info_path = args.output_dir / "dataset_info.json"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in selected[:10]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    pq.write_table(pa.Table.from_pylist(selected), parquet_path)
    info = {
        "num_records": len(selected),
        "num_matched_pairs": len(by_source),
        "candidate_jsonl": str(args.input.resolve()),
        "allowlist": str(args.allowlist.resolve()),
        "allowlist_sha256": hashlib.sha256(args.allowlist.read_bytes()).hexdigest(),
        "jsonl_path": str(jsonl_path.resolve()),
        "parquet_path": str(parquet_path.resolve()),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
