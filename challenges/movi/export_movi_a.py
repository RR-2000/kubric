#!/usr/bin/env python3
"""Download MOVi-A from TFDS and export frames plus metadata to disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load MOVi-A from TFDS and save RGB frames plus all structured "
            "metadata for each example."
        )
    )
    parser.add_argument(
        "--dataset",
        default="movi_a/256x256",
        help="TFDS dataset name/config to export.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="TFDS split to export.",
    )
    parser.add_argument(
        "--data-dir",
        default="gs://kubric-public/tfds",
        help="TFDS data_dir containing the MOVi datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where exported examples will be written.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of examples to export.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip examples before this zero-based index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported example directories.",
    )
    parser.add_argument(
        "--save-modalities",
        action="store_true",
        help=(
            "Also save non-RGB per-frame modalities such as segmentations, depth, "
            "flow, normals, and object coordinates."
        ),
    )
    return parser.parse_args()


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return to_serializable(value.numpy())
    if all(hasattr(value, attr) for attr in ("ymin", "xmin", "ymax", "xmax")):
        return {
            "ymin": float(value.ymin),
            "xmin": float(value.xmin),
            "ymax": float(value.ymax),
            "xmax": float(value.xmax),
        }
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def save_rgb_frames(video: np.ndarray, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(video):
        iio.imwrite(frames_dir / f"frame_{frame_idx:05d}.png", frame)


def save_modality_frames(example: dict[str, Any], example_dir: Path) -> None:
    modality_names = [
        "segmentations",
        "depth",
        "forward_flow",
        "backward_flow",
        "normal",
        "object_coordinates",
    ]
    for modality_name in modality_names:
        if modality_name not in example:
            continue

        modality = np.asarray(example[modality_name])
        modality_dir = example_dir / modality_name
        modality_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, frame in enumerate(modality):
            suffix = ".npy"
            if modality_name == "segmentations":
                output_path = modality_dir / f"{modality_name}_{frame_idx:05d}.png"
                if frame.ndim == 3 and frame.shape[-1] == 1:
                    frame = frame[..., 0]
                iio.imwrite(output_path, frame)
                continue

            output_path = modality_dir / f"{modality_name}_{frame_idx:05d}{suffix}"
            np.save(output_path, frame)


def build_example_metadata(
    dataset_name: str,
    split: str,
    example_idx: int,
    example: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "split": split,
        "example_index": example_idx,
        "metadata": example["metadata"],
        "camera": example["camera"],
        "instances": example["instances"],
        "events": example["events"],
    }


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
    for example_idx, example in enumerate(dataset):
        if example_idx < args.start_index:
            continue
        if args.max_examples is not None and exported >= args.max_examples:
            break

        video_name = example["metadata"].get("video_name", f"example_{example_idx:06d}")
        example_dir = args.output_dir / str(video_name)

        if example_dir.exists():
            if not args.overwrite:
                print(f"[SKIP] {example_dir} already exists")
                continue
        example_dir.mkdir(parents=True, exist_ok=True)

        save_rgb_frames(np.asarray(example["video"]), example_dir / "frames")
        save_json(
            example_dir / "metadata.json",
            build_example_metadata(args.dataset, args.split, example_idx, example),
        )

        if args.save_modalities:
            save_modality_frames(example, example_dir)

        exported += 1
        print(f"[OK] Exported {video_name} -> {example_dir}")

    print(f"[DONE] Exported {exported} examples to {args.output_dir}")


if __name__ == "__main__":
    main()

""" 
python /home/ramanathan/VLM/kubric/challenges/movi/export_movi_a.py \
  --output-dir /home/ramanathan/data/movi_a_export \
  --split validation \
  --save-modalities
"""
