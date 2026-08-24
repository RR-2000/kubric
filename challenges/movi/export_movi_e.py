#!/usr/bin/env python3
"""Download MOVi-E from TFDS and export local frames, modalities, and metadata."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

try:
    from .export_movi_a import (
        build_example_metadata,
        save_json,
        save_modality_frames,
        save_rgb_frames,
    )
except ImportError:
    from export_movi_a import (
        build_example_metadata,
        save_json,
        save_modality_frames,
        save_rgb_frames,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="movi_e/256x256")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data-dir", default="gs://kubric-public/tfds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ramanathan/data/movi_e_export"),
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-modalities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save segmentation, depth, flow, normals, and object coordinates.",
    )
    return parser.parse_args()


def decode_video_name(value: object, fallback: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_video_name(value.item(), fallback)
    rendered = str(value).strip()
    return rendered or fallback


def export_is_complete(example_dir: Path, num_frames: int) -> bool:
    frames_dir = example_dir / "frames"
    return (
        (example_dir / "metadata.json").is_file()
        and frames_dir.is_dir()
        and sum(1 for _ in frames_dir.glob("frame_*.png")) == num_frames
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = tfds.as_numpy(
        tfds.load(
            args.dataset,
            split=args.split,
            data_dir=args.data_dir,
            shuffle_files=False,
        )
    )

    exported = 0
    skipped_complete = 0
    visited = 0
    stop_index = (
        args.start_index + args.max_examples
        if args.max_examples is not None
        else None
    )
    selected_examples = itertools.islice(
        enumerate(dataset), args.start_index, stop_index
    )
    for example_idx, example in selected_examples:
        visited += 1

        fallback = f"example_{example_idx:06d}"
        video_name = decode_video_name(example["metadata"].get("video_name"), fallback)
        example_dir = args.output_dir / video_name
        num_frames = int(example["metadata"]["num_frames"])

        if example_dir.exists() and not args.overwrite:
            if export_is_complete(example_dir, num_frames):
                skipped_complete += 1
                print(f"[SKIP] Complete export already exists: {example_dir}")
                continue
            print(f"[RESUME] Rebuilding incomplete export: {example_dir}")

        example_dir.mkdir(parents=True, exist_ok=True)
        save_rgb_frames(np.asarray(example["video"]), example_dir / "frames")
        if args.save_modalities:
            save_modality_frames(example, example_dir)

        # Write metadata last. Its presence marks an export as complete.
        save_json(
            example_dir / "metadata.json",
            build_example_metadata(args.dataset, args.split, example_idx, example),
        )
        exported += 1
        print(f"[OK] Exported {video_name} -> {example_dir}")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "tfds_data_dir": args.data_dir,
        "output_dir": str(args.output_dir.resolve()),
        "save_modalities": args.save_modalities,
        "start_index": args.start_index,
        "max_examples": args.max_examples,
        "visited_examples": visited,
        "exported_examples": exported,
        "skipped_complete_examples": skipped_complete,
    }
    (args.output_dir / "export_info.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
