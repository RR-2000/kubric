#!/usr/bin/env python3
"""Run build → visual review rendering → optional allowlist finalization for MOVi-E."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/home/ramanathan/data/movi_e_export"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("/home/ramanathan/data/movi_e_direction_object_candidates"))
    parser.add_argument("--review-dir", type=Path, default=Path("/home/ramanathan/data/movi_e_direction_object_review"))
    parser.add_argument("--final-dir", type=Path, default=Path("/home/ramanathan/data/movi_e_better_sample"))
    parser.add_argument("--approved-ids", type=Path, help="Allowlist created after visual review")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sequences", type=int)
    return parser.parse_args()


def _run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    candidate_jsonl = args.candidate_dir / f"movi_e_{args.split}_direction_object.jsonl"
    build = [
        sys.executable, str(SCRIPT_DIR / "build_movi_e_direction_object_dataset.py"),
        "--input-dir", str(args.input_dir), "--output-dir", str(args.candidate_dir),
        "--split", args.split, "--seed", str(args.seed),
    ]
    if args.max_sequences is not None:
        build.extend(["--max-sequences", str(args.max_sequences)])
    _run(*build)
    _run(sys.executable, str(SCRIPT_DIR / "render_movi_e_direction_object_review.py"), str(candidate_jsonl), "--output-dir", str(args.review_dir))
    if args.approved_ids is None:
        print("Review the contact sheets, save one approved source_qid per line, then rerun with --approved-ids PATH.")
        return
    _run(sys.executable, str(SCRIPT_DIR / "finalize_movi_e_direction_object_dataset.py"), str(candidate_jsonl), str(args.approved_ids), "--output-dir", str(args.final_dir), "--split", args.split)


if __name__ == "__main__":
    main()
