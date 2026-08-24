#!/usr/bin/env python3
"""Build a matched direction-versus-object diagnostic from local MOVi-E exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DIRECTIONS = ("left", "right", "front", "behind")
OPTION_LETTERS = tuple("ABCD")
CATEGORY_LABELS = (
    "Action Figures",
    "Bag",
    "Board Games",
    "Bottles and Cans and Cups",
    "Camera",
    "Car Seat",
    "Consumer Goods",
    "Hat",
    "Headphones",
    "Keyboard",
    "Legos",
    "Media Cases",
    "Mouse",
    "None",
    "Shoe",
    "Stuffed Toys",
    "Toys",
)
RELATION_PHRASES = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
}
SKIP_REASONS = (
    "missing_metadata",
    "missing_frame",
    "malformed_annotation",
    "fewer_than_4_visible_objects",
    "duplicate_object_names",
    "unclear_object_label",
    "relation_below_threshold",
    "multiple_correct_object_options",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/ramanathan/data/movi_e_export"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ramanathan/data/movi_e_better_sample"),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-frame-start", type=int, default=20)
    parser.add_argument("--max-frames-per-sequence", type=int, default=4)
    parser.add_argument("--max-pairs-per-sequence", type=int, default=24)
    parser.add_argument("--min-visible-pixels", type=int, default=300)
    parser.add_argument("--min-bbox-area", type=float, default=0.01)
    parser.add_argument("--min-relation-separation", type=float, default=0.35)
    parser.add_argument("--min-relation-margin", type=float, default=0.10)
    parser.add_argument("--max-sequences", type=int)
    return parser.parse_args()


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value).strip()


def _humanize_asset_id(value: object) -> str:
    text = _decode_text(value)
    text = re.sub(r"[_\-/]+", " ", text)
    return " ".join(text.split())


def _display_name(value: object) -> str | None:
    """Remove model codes and opaque asset suffixes from a MOVi object name.

    The export's ``asset_id`` is not a display label: it often contains a SKU
    (for example ``S17MOCHA``) or a random asset suffix.  Keep descriptive
    words, discard digit-containing tokens, and reject labels that collapse to
    a single opaque or generic word.  This is a deterministic first-pass
    filter; visual review remains necessary to certify semantic recognizability.
    """
    tokens = _humanize_asset_id(value).split()
    while tokens and re.fullmatch(r"[A-Za-z0-9]+", tokens[-1]) and (
        any(char.isdigit() for char in tokens[-1])
        or (len(tokens[-1]) >= 8 and sum(char.isupper() for char in tokens[-1]) >= 3)
    ):
        tokens.pop()
    tokens = [token for token in tokens if not any(char.isdigit() for char in token)]
    label = " ".join(tokens).strip()
    words = re.findall(r"[A-Za-z]+", label)
    if len(words) < 2 or len(label) > 80:
        return None
    return label


def _category_name(value: object) -> str:
    if isinstance(value, str) and not value.strip().isdigit():
        return value.strip()
    try:
        index = int(value)
    except (TypeError, ValueError):
        return "unknown category"
    return CATEGORY_LABELS[index] if 0 <= index < len(CATEGORY_LABELS) else "unknown category"


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def _normalize(values: list[float]) -> list[float]:
    length = _norm(values)
    return [value / length for value in values] if length > 1e-8 else [0.0, 0.0, 0.0]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def compute_relation(
    anchor_position: list[float],
    target_position: list[float],
    camera_position: list[float],
) -> dict[str, Any] | None:
    """Classify target position in the anchor-local frame facing the camera."""
    forward = _subtract(camera_position, anchor_position)
    forward[2] = 0.0
    if _norm(forward) < 1e-8:
        return None
    forward = _normalize(forward)
    right_axis = _normalize(_cross([0.0, 0.0, 1.0], forward))
    relative = _subtract(target_position, anchor_position)
    right_coordinate = _dot(relative, right_axis)
    front_coordinate = _dot(relative, forward)
    scores = {
        "left": right_coordinate,
        "right": -right_coordinate,
        "front": front_coordinate,
        "behind": -front_coordinate,
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "relation": ranked[0][0],
        "score": float(ranked[0][1]),
        "margin": float(ranked[0][1] - ranked[1][1]),
        "axis_scores": {key: float(value) for key, value in scores.items()},
        "relative_local_coordinates": {
            "right": float(right_coordinate),
            "front": float(front_coordinate),
            "up": float(relative[2]),
        },
    }


def _bbox_for_frame(instances: dict[str, Any], object_idx: int, frame_idx: int):
    frames = [int(value) for value in instances["bbox_frames"][object_idx]]
    try:
        offset = frames.index(frame_idx)
    except ValueError:
        return None
    bboxes = instances["bboxes"][object_idx]
    return bboxes[offset] if offset < len(bboxes) else None


def _bbox_area(bbox: list[float]) -> float:
    ymin, xmin, ymax, xmax = bbox
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


def _bbox_pixels(bbox: list[float], width: int, height: int) -> list[int]:
    ymin, xmin, ymax, xmax = bbox
    return [
        int(round(xmin * width)),
        int(round(ymin * height)),
        int(round(xmax * width)),
        int(round(ymax * height)),
    ]


def build_visible_objects(
    data: dict[str, Any],
    frame_idx: int,
    min_visible_pixels: int,
    min_bbox_area: float,
) -> list[dict[str, Any]]:
    instances = data["instances"]
    width = int(data["metadata"]["width"])
    height = int(data["metadata"]["height"])
    objects = []
    for object_idx in range(len(instances["positions"])):
        visibility = int(instances["visibility"][object_idx][frame_idx])
        bbox = _bbox_for_frame(instances, object_idx, frame_idx)
        if visibility < min_visible_pixels or bbox is None or _bbox_area(bbox) < min_bbox_area:
            continue
        bbox = [float(value) for value in bbox]
        asset_id = _decode_text(instances["asset_id"][object_idx])
        display_name = _display_name(asset_id)
        if display_name is None:
            continue
        category = _category_name(instances["category"][object_idx])
        objects.append(
            {
                "object_idx": object_idx,
                "segmentation_id": object_idx + 1,
                "name": display_name,
                "raw_asset_id": asset_id,
                "asset_id": asset_id,
                "category": category,
                "scale": float(instances["scale"][object_idx]),
                "is_dynamic": bool(instances["is_dynamic"][object_idx]),
                "visibility_pixels": visibility,
                "bbox_2d_norm": bbox,
                "bbox_2d_xyxy_pixels": _bbox_pixels(bbox, width, height),
                "image_position_2d": [
                    float(value)
                    for value in instances["image_positions"][object_idx][frame_idx]
                ],
                "position_3d": [
                    float(value)
                    for value in instances["positions"][object_idx][frame_idx]
                ],
                "bbox_3d": instances["bboxes_3d"][object_idx][frame_idx],
            }
        )
    return objects


def _stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _render_prompt(question: str, options: list[str]) -> str:
    return question + "\n" + "\n".join(
        f"{letter}. {option}" for letter, option in zip(OPTION_LETTERS, options)
    )


def _make_record(
    *,
    common: dict[str, Any],
    variant: str,
    answer_format: str,
    question: str,
    options: list[str],
    answer_text: str,
) -> dict[str, Any]:
    answer_idx = options.index(answer_text)
    gold_letter = OPTION_LETTERS[answer_idx]
    return {
        **common,
        "qid": f"{common['source_relation_id']}::{variant}",
        "index": f"{common['source_relation_id']}::{variant}",
        "diagnostic_variant": variant,
        "diagnostic_answer_format": answer_format,
        "question": question,
        "question_prompt": _render_prompt(question, options),
        "options": options,
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3],
        "answer": gold_letter,
        "answer_text": answer_text,
        "diagnostic_target": answer_text,
        "answer_idx": answer_idx,
        "gold_option_letter": gold_letter,
        "num_options": 4,
    }


def build_pair(
    *,
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    anchor_idx: int,
    target_idx: int,
    camera_position: list[float],
    min_separation: float,
    min_margin: float,
    seed: int,
) -> tuple[list[dict[str, Any]], str | None]:
    anchor = objects[anchor_idx]
    target = objects[target_idx]
    relation_info = compute_relation(
        anchor["position_3d"], target["position_3d"], camera_position
    )
    if (
        relation_info is None
        or relation_info["score"] < min_separation
        or relation_info["margin"] < min_margin
    ):
        return [], "relation_below_threshold"

    source_id = (
        f"movi_e:{sequence_name}:{frame_idx}:"
        f"{anchor['object_idx']}:{target['object_idx']}"
    )
    rng = _stable_rng(seed, source_id)
    distractor_indices = [
        index for index in range(len(objects)) if index not in {anchor_idx, target_idx}
    ]
    if len(distractor_indices) < 3:
        return [], "fewer_than_4_visible_objects"
    chosen_indices = [target_idx, *rng.sample(distractor_indices, 3)]
    rng.shuffle(chosen_indices)

    correct_indices = []
    for option_idx in chosen_indices:
        option_relation = compute_relation(
            anchor["position_3d"], objects[option_idx]["position_3d"], camera_position
        )
        if option_relation and option_relation["relation"] == relation_info["relation"]:
            correct_indices.append(option_idx)
    if len(correct_indices) != 1 or correct_indices[0] != target_idx:
        return [], "multiple_correct_object_options"

    object_options = [objects[index]["name"] for index in chosen_indices]
    relation = relation_info["relation"]
    common = {
        "image": str(frame_path.resolve()),
        "image_path": str(frame_path.resolve()),
        "img_path": str(frame_path.resolve()),
        "sequence_name": sequence_name,
        "frame_index": frame_idx,
        "source_qid": source_id,
        "source_relation_id": source_id,
        "diagnostic_anchor": anchor["name"],
        "diagnostic_anchor_object_id": anchor["object_idx"],
        "diagnostic_target_object": target["name"],
        "diagnostic_target_object_id": target["object_idx"],
        "diagnostic_relation": relation,
        "candidate_pool": object_options,
        "candidate_object_ids": [objects[index]["object_idx"] for index in chosen_indices],
        "diagnostic_correct_option_object_ids": [target["object_idx"]],
        "diagnostic_correct_object_options": [target["name"]],
        "num_correct_object_options": 1,
        "relation_source": "movi_e_positions_3d",
        "source_task_family": "object_centric_relative_position",
        "dataset_family": "movi_e",
        "object_label_source": "cleaned_asset_id",
        "visible_objects": objects,
        "referenced_object_ids": [anchor["object_idx"]] + [
            objects[index]["object_idx"] for index in chosen_indices
        ],
        "task_metadata": {
            "anchor_object": anchor["name"],
            "target_object": target["name"],
            "relation": relation,
            "separation": relation_info["score"],
            "relation_margin": relation_info["margin"],
            "minimum_separation": min_separation,
            "minimum_relation_margin": min_margin,
            "axis_scores": relation_info["axis_scores"],
            "relative_local_coordinates": relation_info[
                "relative_local_coordinates"
            ],
            "camera_position": camera_position,
        },
    }
    direction_question = (
        f"Imagine standing at the {anchor['name']} and facing the camera. "
        f"Where is the {target['name']} relative to the {anchor['name']}? "
        "Answer with the option letter only."
    )
    object_question = (
        f"Imagine standing at the {anchor['name']} and facing the camera. "
        f"Which object is {RELATION_PHRASES[relation]} the {anchor['name']}? "
        "Answer with the option letter only."
    )
    return [
        _make_record(
            common=common,
            variant="native",
            answer_format="direction",
            question=direction_question,
            options=list(DIRECTIONS),
            answer_text=relation,
        ),
        _make_record(
            common=common,
            variant="inverse",
            answer_format="object",
            question=object_question,
            options=object_options,
            answer_text=target["name"],
        ),
    ], None


def build_sequence_records(
    sequence_dir: Path, args: argparse.Namespace, stats: Counter
) -> list[dict[str, Any]]:
    metadata_path = sequence_dir / "metadata.json"
    if not metadata_path.is_file():
        stats["missing_metadata"] += 1
        return []
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if data.get("split") != args.split:
            return []
        num_frames = int(data["metadata"]["num_frames"])
        sequence_name = _decode_text(data["metadata"]["video_name"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        stats["malformed_annotation"] += 1
        return []

    rng = _stable_rng(args.seed, sequence_name)
    frame_indices = list(range(max(0, args.sample_frame_start), num_frames))
    rng.shuffle(frame_indices)
    frame_indices = sorted(frame_indices[: args.max_frames_per_sequence])
    records = []
    accepted_pairs = 0

    for frame_position, frame_idx in enumerate(frame_indices):
        remaining_pair_budget = args.max_pairs_per_sequence - accepted_pairs
        remaining_frames = len(frame_indices) - frame_position
        frame_pair_quota = math.ceil(remaining_pair_budget / remaining_frames)
        frame_accepted_pairs = 0
        frame_path = sequence_dir / "frames" / f"frame_{frame_idx:05d}.png"
        if not frame_path.is_file():
            stats["missing_frame"] += 1
            continue
        try:
            objects = build_visible_objects(
                data,
                frame_idx,
                args.min_visible_pixels,
                args.min_bbox_area,
            )
            camera_position = [
                float(value) for value in data["camera"]["positions"][frame_idx]
            ]
        except (KeyError, IndexError, TypeError, ValueError):
            stats["malformed_annotation"] += 1
            continue
        if len(objects) < 4:
            stats["unclear_object_label"] += 1
            continue
        names = [obj["name"].casefold() for obj in objects]
        if len(names) != len(set(names)):
            stats["duplicate_object_names"] += 1
            continue

        ordered_pairs = [
            (anchor_idx, target_idx)
            for anchor_idx in range(len(objects))
            for target_idx in range(len(objects))
            if anchor_idx != target_idx
        ]
        rng.shuffle(ordered_pairs)
        for anchor_idx, target_idx in ordered_pairs:
            pair, skip_reason = build_pair(
                sequence_name=sequence_name,
                frame_idx=frame_idx,
                frame_path=frame_path,
                objects=objects,
                anchor_idx=anchor_idx,
                target_idx=target_idx,
                camera_position=camera_position,
                min_separation=args.min_relation_separation,
                min_margin=args.min_relation_margin,
                seed=args.seed,
            )
            if skip_reason:
                stats[skip_reason] += 1
                continue
            records.extend(pair)
            accepted_pairs += 1
            frame_accepted_pairs += 1
            if accepted_pairs >= args.max_pairs_per_sequence:
                return records
            if frame_accepted_pairs >= frame_pair_quota:
                break
    return records


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    errors = []
    grouped = defaultdict(list)
    for row_number, record in enumerate(records, 1):
        grouped[record.get("source_relation_id")].append(record)
        options = record.get("options", [])
        if len(options) != 4 or record.get("num_options") != 4:
            errors.append(f"row {row_number}: expected exactly four options")
        if not Path(str(record.get("image_path", ""))).is_file():
            errors.append(f"row {row_number}: missing image")
        if record.get("diagnostic_answer_format") == "object":
            folded = [str(option).casefold() for option in options]
            if len(folded) != len(set(folded)):
                errors.append(f"row {row_number}: duplicate object options")
            if any(
                any(char.isdigit() for char in str(option))
                or len(re.findall(r"[A-Za-z]+", str(option))) < 2
                for option in options
            ):
                errors.append(f"row {row_number}: unclear object display label")
            if record.get("num_correct_object_options") != 1:
                errors.append(f"row {row_number}: multiple correct object options")
            if record.get("diagnostic_correct_object_options") != [
                record.get("answer_text")
            ]:
                errors.append(f"row {row_number}: incorrect object-answer audit data")

    matched_pairs = 0
    shared_fields = (
        "image_path",
        "sequence_name",
        "frame_index",
        "diagnostic_anchor_object_id",
        "diagnostic_target_object_id",
        "diagnostic_relation",
        "candidate_pool",
    )
    for source_id, pair in grouped.items():
        by_format = {record.get("diagnostic_answer_format"): record for record in pair}
        if len(pair) != 2 or set(by_format) != {"direction", "object"}:
            errors.append(f"source {source_id}: incomplete pair")
            continue
        if any(
            by_format["direction"].get(field) != by_format["object"].get(field)
            for field in shared_fields
        ):
            errors.append(f"source {source_id}: mismatched pair metadata")
            continue
        matched_pairs += 1

    return {
        "valid": not errors,
        "errors": errors[:100],
        "num_errors": len(errors),
        "num_records": len(records),
        "num_source_groups": len(grouped),
        "num_matched_pairs": matched_pairs,
        "num_unique_images": len({row["image_path"] for row in records}),
        "num_sequences_with_records": len(
            {row["sequence_name"] for row in records}
        ),
        "answer_format_distribution": dict(
            sorted(Counter(row["diagnostic_answer_format"] for row in records).items())
        ),
        "relation_distribution": dict(
            sorted(Counter(row["diagnostic_relation"] for row in records).items())
        ),
        "gold_option_letter_distribution": dict(
            sorted(Counter(row["gold_option_letter"] for row in records).items())
        ),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if args.max_sequences is not None:
        sequence_dirs = sequence_dirs[: args.max_sequences]

    stats = Counter({reason: 0 for reason in SKIP_REASONS})
    records = []
    for sequence_number, sequence_dir in enumerate(sequence_dirs, 1):
        sequence_records = build_sequence_records(sequence_dir, args, stats)
        records.extend(sequence_records)
        print(
            f"[{sequence_number}/{len(sequence_dirs)}] {sequence_dir.name}: "
            f"{len(sequence_records) // 2} pairs"
        )

    audit = audit_records(records)
    audit["skipped_counts"] = {reason: stats[reason] for reason in SKIP_REASONS}
    audit["seed"] = args.seed
    audit["split"] = args.split
    audit["num_sequences"] = len(sequence_dirs)
    audit["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    jsonl_path = args.output_dir / f"movi_e_{args.split}_direction_object.jsonl"
    parquet_path = args.output_dir / f"movi_e_{args.split}_direction_object.parquet"
    sample_path = args.output_dir / "sample_pairs.jsonl"
    audit_path = args.output_dir / "audit_report.json"
    info_path = args.output_dir / "dataset_info.json"
    _write_jsonl(jsonl_path, records)
    _write_jsonl(sample_path, records[:10])
    pq.write_table(pa.Table.from_pylist(records), parquet_path)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    info = {
        **audit,
        "jsonl_path": str(jsonl_path.resolve()),
        "parquet_path": str(parquet_path.resolve()),
        "sample_path": str(sample_path.resolve()),
        "audit_path": str(audit_path.resolve()),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    if not audit["valid"]:
        raise SystemExit("Dataset audit failed")


if __name__ == "__main__":
    main()
