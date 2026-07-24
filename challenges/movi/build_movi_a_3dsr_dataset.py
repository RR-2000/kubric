#!/usr/bin/env python3
"""Build 3DSR-style question-answer samples from exported MOVi-A sequences."""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SHAPE_LABELS = ["cube", "cylinder", "sphere"]
SIZE_LABELS = ["small", "large"]
COLOR_LABELS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
MATERIAL_LABELS = ["metal", "rubber"]

OBJECT_RELATIONS = {
    "left": ("to the left of", "left"),
    "right": ("to the right of", "right"),
    "front": ("in front of", "front"),
    "behind": ("behind", "behind"),
}
YES_NO_OPTIONS = ["yes", "no"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan MOVi-A exports and build a JSONL + Parquet dataset of "
            "3DSR-style spatial reasoning tasks."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/ramanathan/data/movi_a_export"),
        help="Directory containing exported MOVi-A sequence folders.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/home/ramanathan/data/movi_a_3dsr_new/movi_a_3dsr.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("/home/ramanathan/data/movi_a_3dsr_new/movi_a_3dsr.parquet"),
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible frame sampling.",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=12,
        help="Maximum candidate frames to sample from each sequence.",
    )
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=24,
        help="Stop after writing this many task samples for one sequence.",
    )
    parser.add_argument(
        "--max-object-centric-samples-per-sequence",
        type=int,
        default=16,
        help="Maximum number of 2-choice object-centric samples to export per sequence.",
    )
    parser.add_argument(
        "--max-object-centric-multi-samples-per-sequence",
        type=int,
        default=16,
        help="Maximum number of multi-object object-centric samples to export per sequence.",
    )
    parser.add_argument(
        "--max-object-centric-direction-samples-per-sequence",
        type=int,
        default=16,
        help="Maximum number of binary object-centric direction samples to export per sequence.",
    )
    parser.add_argument(
        "--max-camera-pose-samples-per-sequence",
        type=int,
        default=12,
        help="Maximum number of camera-pose samples to export per sequence.",
    )
    parser.add_argument(
        "--min-visible-pixels",
        type=int,
        default=300,
        help="Minimum visible pixels for an object to be considered usable.",
    )
    parser.add_argument(
        "--min-bbox-area",
        type=float,
        default=0.01,
        help="Minimum normalized bbox area for an object to be considered usable.",
    )
    parser.add_argument(
        "--min-image-separation",
        type=float,
        default=0.08,
        help="Minimum separation in normalized image coordinates for image-view tasks.",
    )
    parser.add_argument(
        "--min-height-separation",
        type=float,
        default=0.25,
        help="Minimum world-space z separation for higher/lower tasks.",
    )
    parser.add_argument(
        "--min-camera-depth-separation",
        type=float,
        default=0.75,
        help="Minimum distance-to-camera separation for closer/farther tasks.",
    )
    parser.add_argument(
        "--min-object-centric-separation",
        type=float,
        default=0.35,
        help="Minimum local-axis separation for object-centric tasks.",
    )
    parser.add_argument(
        "--min-option-margin",
        type=float,
        default=0.5,
        help="Minimum margin between the best option and the runner-up for multi-option tasks.",
    )
    parser.add_argument(
        "--similar-color-threshold",
        type=float,
        default=0.25,
        help="Skip frames with same-shape objects whose RGB distance is below this threshold.",
    )
    
    parser.add_argument(
        "--sample-frame-start",
        type=int,
        default=0,
        help="Frame index to start sampling from.",
    )
    return parser.parse_args()


def norm3(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def subtract3(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def normalize3(values: list[float]) -> list[float]:
    length = norm3(values)
    if length < 1e-8:
        return [0.0, 0.0, 0.0]
    return [v / length for v in values]


def dot3(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def color_distance(color_a: list[float], color_b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(color_a, color_b)))


def decode_label(index: int, labels: list[str]) -> str:
    return labels[index]


def decode_video_name(raw_name: str) -> str:
    if raw_name.startswith("b'") and raw_name.endswith("'"):
        try:
            return raw_name[2:-1].encode("utf-8").decode("unicode_escape")
        except Exception:
            return raw_name
    return raw_name


def bbox_area(bbox: list[float]) -> float:
    ymin, xmin, ymax, xmax = bbox
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    ymin, xmin, ymax, xmax = bbox
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def bbox_to_xyxy_pixels(bbox: list[float], width: int, height: int) -> list[int]:
    ymin, xmin, ymax, xmax = bbox
    return [
        int(round(xmin * width)),
        int(round(ymin * height)),
        int(round(xmax * width)),
        int(round(ymax * height)),
    ]


def get_bbox_for_frame(instances: dict[str, Any], object_idx: int, frame_idx: int) -> list[float] | None:
    bbox_frames = instances["bbox_frames"][object_idx]
    try:
        bbox_offset = bbox_frames.index(frame_idx)
    except ValueError:
        return None
    bboxes = instances["bboxes"][object_idx]
    if bbox_offset >= len(bboxes):
        return None
    return bboxes[bbox_offset]


def build_object_record(
    instances: dict[str, Any],
    object_idx: int,
    frame_idx: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    bbox = get_bbox_for_frame(instances, object_idx, frame_idx)
    if bbox is None:
        raise ValueError(f"Missing bbox for object {object_idx} frame {frame_idx}")
    center_x, center_y = bbox_center(bbox)
    descriptor = " ".join(
        [
            decode_label(instances["size_label"][object_idx], SIZE_LABELS),
            decode_label(instances["color_label"][object_idx], COLOR_LABELS),
            decode_label(instances["material_label"][object_idx], MATERIAL_LABELS),
            decode_label(instances["shape_label"][object_idx], SHAPE_LABELS),
        ]
    )
    return {
        "object_idx": object_idx,
        "segmentation_id": object_idx + 1,
        "name": descriptor,
        "shape": decode_label(instances["shape_label"][object_idx], SHAPE_LABELS),
        "size": decode_label(instances["size_label"][object_idx], SIZE_LABELS),
        "material": decode_label(instances["material_label"][object_idx], MATERIAL_LABELS),
        "color_name": decode_label(instances["color_label"][object_idx], COLOR_LABELS),
        "color_rgb": [float(v) for v in instances["color"][object_idx]],
        "visibility_pixels": int(instances["visibility"][object_idx][frame_idx]),
        "bbox_2d_norm": [float(v) for v in bbox],
        "bbox_2d_xyxy_pixels": bbox_to_xyxy_pixels(bbox, width, height),
        "bbox_center_norm": [float(center_x), float(center_y)],
        "image_position_2d": [float(v) for v in instances["image_positions"][object_idx][frame_idx]],
        "position_3d": [float(v) for v in instances["positions"][object_idx][frame_idx]],
        "bbox_3d": instances["bboxes_3d"][object_idx][frame_idx],
    }


def is_frame_unambiguous(
    objects: list[dict[str, Any]],
    similar_color_threshold: float,
) -> bool:
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if obj_a["shape"] != obj_b["shape"]:
                continue
            if color_distance(obj_a["color_rgb"], obj_b["color_rgb"]) < similar_color_threshold:
                return False
    return True


def choose_frame_indices(num_frames: int, max_frames: int, sample_frame_start: int, rng: random.Random) -> list[int]:
    sample_frame_start = max(0, min(sample_frame_start, num_frames - 1))
    
    frame_indices = list(range(sample_frame_start, num_frames))
    rng.shuffle(frame_indices)
    return sorted(frame_indices[: min(num_frames - sample_frame_start, max_frames)])


def make_qid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def distance3(a: list[float], b: list[float]) -> float:
    return norm3(subtract3(a, b))


def classify_difficulty(separation: float, min_separation: float) -> str:
    if min_separation <= 0:
        if separation >= 1.0:
            return "easy"
        if separation >= 0.5:
            return "hard"
        return "very_hard"

    ratio = separation / min_separation
    if ratio >= 3.0:
        return "easy"
    if ratio >= 1.75:
        return "hard"
    return "very_hard"


def compute_anchor_frame(
    anchor_position: list[float],
    camera_position: list[float],
) -> tuple[list[float], list[float]] | None:
    up = [0.0, 0.0, 1.0]
    forward = subtract3(camera_position, anchor_position)
    forward[2] = 0.0
    if norm3(forward) < 1e-8:
        return None
    forward = normalize3(forward)
    right = normalize3(cross3(up, forward))
    return forward, right


def compute_object_centric_relation(
    anchor_position: list[float],
    other_position: list[float],
    camera_position: list[float],
) -> dict[str, Any] | None:
    anchor_frame = compute_anchor_frame(anchor_position, camera_position)
    if anchor_frame is None:
        return None
    forward, right = anchor_frame
    relative = subtract3(other_position, anchor_position)
    rel_right = dot3(relative, right)
    rel_front = dot3(relative, forward)
    axis_scores = {
        "left": rel_right,
        "right": -rel_right,
        "front": rel_front,
        "behind": -rel_front,
    }
    # print(axis_scores)
    # exit()
    candidates = list(axis_scores.items())
    relation, score = max(candidates, key=lambda item: item[1])
    _, canonical_relation = OBJECT_RELATIONS[relation]
    return {
        "relation": canonical_relation,
        "score": score,
        "axis_scores": axis_scores,
        "relative_local_coordinates": {
            "right": rel_right,
            "front": rel_front,
            "up": relative[2],
        },
    }


def compute_camera_world_vector(
    anchor_position: list[float],
    camera_position: list[float],
) -> dict[str, Any]:
    vector = subtract3(camera_position, anchor_position)
    distance = norm3(vector)
    horizontal_scores = {
        "left": vector[0],
        "right": -vector[0],
        "front": vector[1],
        "behind": -vector[1],
    }
    relation, score = max(horizontal_scores.items(), key=lambda item: item[1])
    _, canonical_relation = OBJECT_RELATIONS[relation]
    return {
        "vector_world_aligned": {
            "right": float(vector[0]),
            "up": float(vector[2]),
            "front": float(vector[1]),
        },
        "distance": float(distance),
        "horizontal_scores": horizontal_scores,
        "dominant_horizontal_relation": canonical_relation,
        "dominant_horizontal_score": float(score),
    }


def make_base_record(
    *,
    qid: str,
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    task_family: str,
    question: str,
    answer: str,
    options: list[str],
    objects: list[dict[str, Any]],
    referenced_object_indices: list[int],
    task_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "index": qid,
        "sequence_name": sequence_name,
        "frame_index": frame_idx,
        "image": str(frame_path),
        "img_path": str(frame_path),
        "task_family": task_family,
        "question": question,
        "A": options[0] if len(options) > 0 else None,
        "B": options[1] if len(options) > 1 else None,
        "C": options[2] if len(options) > 2 else None,
        "D": options[3] if len(options) > 3 else None,
        "answer": answer,
        "visible_objects": objects,
        "referenced_visible_indices": referenced_object_indices,
        "referenced_object_ids": [objects[idx]["object_idx"] for idx in referenced_object_indices],
        "bbox_items": [objects[idx]["name"] for idx in referenced_object_indices],
        "difficulty": task_metadata.get("difficulty"),
        "task_metadata": task_metadata,
    }


def build_image_view_task(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    pair: tuple[int, int],
    min_sep: float,
) -> dict[str, Any] | None:
    obj_a = objects[pair[0]]
    obj_b = objects[pair[1]]
    ax, ay = obj_a["bbox_center_norm"]
    bx, by = obj_b["bbox_center_norm"]

    # Position of A compared to B in the image plane (2D)
    candidates = [
        ("left", bx - ax),
        ("right", ax - bx),
        ("above", by - ay),
        ("below", ay - by),
    ]
    relation, score = max(candidates, key=lambda item: item[1])
    if score < min_sep:
        return None

    # If flip, change direction
    flip_map = {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
    }
    # Select object to be the answer
    answer = "A" if random.random() < 0.5 else "B"
    if answer == "B":
        relation = flip_map[relation]
    adjective = {
        "left": "more to the left",
        "right": "more to the right",
        "above": "higher in the image",
        "below": "lower in the image",
    }[relation]
    question = (
        f"As seen from the camera, which object is {adjective}: "
        f"{obj_a['name']} or {obj_b['name']}?"
    )
    difficulty = classify_difficulty(score, min_sep)
    return make_base_record(
        qid=make_qid("movi_image"),
        sequence_name=sequence_name,
        frame_idx=frame_idx,
        frame_path=frame_path,
        task_family="camera_relative_position",
        question=question,
        answer=answer,
        options=[obj_a["name"], obj_b["name"]],
        objects=objects,
        referenced_object_indices=[pair[0], pair[1]],
        task_metadata={
            "relation": relation,
            "difficulty": difficulty,
            "separation": score,
            "minimum_separation": min_sep,
            "center_delta": [bx - ax, by - ay],
            "position_type": "image_plane",
        },
    )


def build_height_task(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    pair: tuple[int, int],
    min_sep: float,
) -> dict[str, Any] | None:
    obj_a = objects[pair[0]]
    obj_b = objects[pair[1]]
    za = obj_a["position_3d"][2]
    zb = obj_b["position_3d"][2]
    if abs(za - zb) < min_sep:
        return None

    ask_higher = ((frame_idx + pair[0] + pair[1]) % 2 == 0)
    if za > zb:
        higher = obj_a
        lower = obj_b
        higher_answer = "A"
    else:
        higher = obj_b
        lower = obj_a
        higher_answer = "B"

    if ask_higher:
        question = (
            "Consider the real-world 3D locations of the objects. "
            f"Which object is higher: {obj_a['name']} or {obj_b['name']}?"
        )
        answer = higher_answer
        target = "higher"
        correct_object = higher["name"]
    else:
        question = (
            "Consider the real-world 3D locations of the objects. "
            f"Which object is lower: {obj_a['name']} or {obj_b['name']}?"
        )
        answer = "A" if higher_answer == "B" else "B"
        target = "lower"
        correct_object = lower["name"]

    difficulty = classify_difficulty(abs(za - zb), min_sep)

    return make_base_record(
        qid=make_qid("movi_height"),
        sequence_name=sequence_name,
        frame_idx=frame_idx,
        frame_path=frame_path,
        task_family="height_relative_3d",
        question=question,
        answer=answer,
        options=[obj_a["name"], obj_b["name"]],
        objects=objects,
        referenced_object_indices=[pair[0], pair[1]],
        task_metadata={
            "relation": target,
            "difficulty": difficulty,
            "separation": abs(za - zb),
            "minimum_separation": min_sep,
            "object_a_name": obj_a["name"],
            "object_a_z": za,
            "object_b_name": obj_b["name"],
            "object_b_z": zb,
            "correct_object": correct_object,
        },
    )


def build_camera_distance_task(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    pair: tuple[int, int],
    camera_position: list[float],
    min_sep: float,
) -> dict[str, Any] | None:
    obj_a = objects[pair[0]]
    obj_b = objects[pair[1]]
    dist_a = distance3(obj_a["position_3d"], camera_position)
    dist_b = distance3(obj_b["position_3d"], camera_position)
    if abs(dist_a - dist_b) < min_sep:
        return None

    ask_closer = ((frame_idx + pair[0] + pair[1]) % 2 == 1)
    if dist_a < dist_b:
        closer = obj_a
        farther = obj_b
        closer_answer = "A"
    else:
        closer = obj_b
        farther = obj_a
        closer_answer = "B"

    if ask_closer:
        question = (
            f"Relative to the camera, which object is closer: "
            f"{obj_a['name']} or {obj_b['name']}?"
        )
        answer = closer_answer
        relation = "closer"
        correct_object = closer["name"]
    else:
        question = (
            f"Relative to the camera, which object is farther: "
            f"{obj_a['name']} or {obj_b['name']}?"
        )
        answer = "A" if closer_answer == "B" else "B"
        relation = "farther"
        correct_object = farther["name"]

    difficulty = classify_difficulty(abs(dist_a - dist_b), min_sep)

    return make_base_record(
        qid=make_qid("movi_depth"),
        sequence_name=sequence_name,
        frame_idx=frame_idx,
        frame_path=frame_path,
        task_family="camera_distance",
        question=question,
        answer=answer,
        options=[obj_a["name"], obj_b["name"]],
        objects=objects,
        referenced_object_indices=[pair[0], pair[1]],
        task_metadata={
            "relation": relation,
            "difficulty": difficulty,
            "separation": abs(dist_a - dist_b),
            "minimum_separation": min_sep,
            "object_a_name": obj_a["name"],
            "object_a_camera_distance": dist_a,
            "object_b_name": obj_b["name"],
            "object_b_camera_distance": dist_b,
            "correct_object": correct_object,
            "camera_position": camera_position,
        },
    )


def build_object_centric_task(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    anchor_idx: int,
    other_idx: int,
    camera_position: list[float],
    min_sep: float,
) -> dict[str, Any] | None:
    anchor = objects[anchor_idx]
    other = objects[other_idx]
    relation_info = compute_object_centric_relation(
        anchor["position_3d"],
        other["position_3d"],
        camera_position,
    )
    if relation_info is None or relation_info["score"] < min_sep:
        return None

    question = (
        f"Imagine standing at the {anchor['name']} and facing the camera. "
        f"Where is the {other['name']} relative to the {anchor['name']}?"
    )
    
    options = ["left", "right", "front", "behind"]
    answer = {label: chr(ord('A') + idx) for idx, label in enumerate(options)}[relation_info["relation"]]
    
    difficulty = classify_difficulty(relation_info["score"], min_sep)
    return make_base_record(
        qid=make_qid("movi_object"),
        sequence_name=sequence_name,
        frame_idx=frame_idx,
        frame_path=frame_path,
        task_family="object_centric_relative_position",
        question=question,
        answer=answer,
        options=options,
        objects=objects,
        referenced_object_indices=[anchor_idx, other_idx],
        task_metadata={
            "anchor_object": anchor["name"],
            "target_object": other["name"],
            "relation": relation_info["relation"],
            "difficulty": difficulty,
            "separation": relation_info["score"],
            "minimum_separation": min_sep,
            "relative_local_coordinates": relation_info["relative_local_coordinates"],
            "camera_position": camera_position,
        },
    )


def build_object_centric_direction_tasks(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    anchor_idx: int,
    other_idx: int,
    camera_position: list[float],
    min_sep: float,
) -> list[dict[str, Any]]:
    anchor = objects[anchor_idx]
    other = objects[other_idx]
    relation_info = compute_object_centric_relation(
        anchor["position_3d"],
        other["position_3d"],
        camera_position,
    )
    if relation_info is None or relation_info["score"] < min_sep:
        return []

    truth_relation = relation_info["relation"]
    axis_scores = relation_info["axis_scores"]
    incorrect_relations = [label for label in OBJECT_RELATIONS if label != truth_relation]
    hardest_negative = max(incorrect_relations, key=lambda label: axis_scores[label])
    relation_candidates = [truth_relation, hardest_negative]
    tasks: list[dict[str, Any]] = []

    for queried_relation in relation_candidates:
        is_true = queried_relation == truth_relation
        score = axis_scores[queried_relation]
        margin = relation_info["score"] - max(
            axis_scores[label] for label in OBJECT_RELATIONS if label != queried_relation
        )
        question = (
            f"Imagine standing at the {anchor['name']} and facing the camera. "
            f"Is the {other['name']} {OBJECT_RELATIONS[queried_relation][0]} the {anchor['name']}?"
        )
        difficulty = classify_difficulty(abs(score), min_sep)
        tasks.append(
            make_base_record(
                qid=make_qid("movi_object_dir"),
                sequence_name=sequence_name,
                frame_idx=frame_idx,
                frame_path=frame_path,
                task_family="object_centric_direction_binary",
                question=question,
                answer="A" if is_true else "B",
                options=YES_NO_OPTIONS,
                objects=objects,
                referenced_object_indices=[anchor_idx, other_idx],
                task_metadata={
                    "anchor_object": anchor["name"],
                    "target_object": other["name"],
                    "queried_relation": queried_relation,
                    "truth_relation": truth_relation,
                    "is_true": is_true,
                    "difficulty": difficulty,
                    "separation": abs(score),
                    "minimum_separation": min_sep,
                    "axis_margin": margin,
                    "relative_local_coordinates": relation_info["relative_local_coordinates"],
                    "axis_scores": axis_scores,
                    "camera_position": camera_position,
                },
            )
        )
    return tasks


def build_camera_pose_task(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    anchor_idx: int,
    camera_position: list[float],
    min_sep: float,
) -> dict[str, Any] | None:
    anchor = objects[anchor_idx]
    camera_info = compute_camera_world_vector(anchor["position_3d"], camera_position)
    if camera_info["dominant_horizontal_score"] < min_sep:
        return None

    relation = camera_info["dominant_horizontal_relation"]
    question = (
        f"Using world-aligned directions where +X is right and +Y is front, "
        f"which horizontal direction best describes the camera position relative to the {anchor['name']}?"
    )
    options = ["left", "right", "front", "behind"]
    answer = chr(ord("A") + options.index(relation))
    difficulty = classify_difficulty(camera_info["dominant_horizontal_score"], min_sep)
    return make_base_record(
        qid=make_qid("movi_camera_pose"),
        sequence_name=sequence_name,
        frame_idx=frame_idx,
        frame_path=frame_path,
        task_family="object_centric_camera_pose",
        question=question,
        answer=answer,
        options=options,
        objects=objects,
        referenced_object_indices=[anchor_idx],
        task_metadata={
            "anchor_object": anchor["name"],
            "relation": relation,
            "difficulty": difficulty,
            "separation": camera_info["dominant_horizontal_score"],
            "minimum_separation": min_sep,
            "camera_vector_world_aligned": camera_info["vector_world_aligned"],
            "camera_distance": camera_info["distance"],
            "horizontal_scores": camera_info["horizontal_scores"],
            "camera_position": camera_position,
        },
    )


def build_object_centric_multi_tasks(
    sequence_name: str,
    frame_idx: int,
    frame_path: Path,
    objects: list[dict[str, Any]],
    anchor_idx: int,
    candidate_indices: list[int],
    camera_position: list[float],
    min_sep: float,
    min_option_margin: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    anchor = objects[anchor_idx]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "left": [],
        "right": [],
        "front": [],
        "behind": [],
    }
    for other_idx in candidate_indices:
        if other_idx == anchor_idx:
            continue
        relation_info = compute_object_centric_relation(
            anchor["position_3d"],
            objects[other_idx]["position_3d"],
            camera_position,
        )
        if relation_info is None or relation_info["score"] < min_sep:
            continue
        grouped[relation_info["relation"]].append((other_idx, relation_info))

    viable_relations = [label for label, items in grouped.items() if items]
    if not viable_relations:
        return []

    relation_order = sorted(
        viable_relations,
        key=lambda label: max(item[1]["axis_scores"][label] for item in grouped[label]),
        reverse=True,
    )
    tasks: list[dict[str, Any]] = []
    for relation in relation_order:
        ranked_for_relation = sorted(
            grouped[relation],
            key=lambda item: item[1]["axis_scores"][relation],
            reverse=True,
        )
        correct_idx, correct_info = ranked_for_relation[0]
        competing_scores = [
            info["axis_scores"][relation]
            for _, items in grouped.items()
            for idx, info in items
            if idx != correct_idx
        ]
        option_margin = None
        if competing_scores:
            option_margin = correct_info["axis_scores"][relation] - max(competing_scores)
            if option_margin < min_option_margin:
                continue

        distractor_pool = [idx for idx in candidate_indices if idx not in {anchor_idx, correct_idx}]
        if len(distractor_pool) < 2:
            continue

        rng.shuffle(distractor_pool)
        option_indices = [correct_idx] + distractor_pool[: min(3, len(distractor_pool))]
        rng.shuffle(option_indices)
        answer = chr(ord("A") + option_indices.index(correct_idx))
        options = [objects[idx]["name"] for idx in option_indices]

        question = (
            f"Imagine standing at the {anchor['name']} and facing the camera. "
            f"Which object is farthest {OBJECT_RELATIONS[relation][0]} the {anchor['name']}?"
        )
        margin_for_difficulty = option_margin if option_margin is not None else correct_info["score"]
        difficulty = classify_difficulty(margin_for_difficulty, min_option_margin)
        tasks.append(
            make_base_record(
                qid=make_qid("movi_object_multi"),
                sequence_name=sequence_name,
                frame_idx=frame_idx,
                frame_path=frame_path,
                task_family="object_centric_relative_position_multi",
                question=question,
                answer=answer,
                options=options,
                objects=objects,
                referenced_object_indices=[anchor_idx] + option_indices,
                task_metadata={
                    "anchor_object": anchor["name"],
                    "relation": relation,
                    "correct_object": objects[correct_idx]["name"],
                    "candidate_objects": [objects[idx]["name"] for idx in option_indices],
                    "difficulty": difficulty,
                    "separation": margin_for_difficulty,
                    "minimum_separation": min_option_margin,
                    "option_margin": option_margin,
                    "correct_relative_local_coordinates": correct_info["relative_local_coordinates"],
                    "camera_position": camera_position,
                },
            )
        )
    return tasks


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def build_records_for_sequence(
    sequence_dir: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    metadata_path = sequence_dir / "metadata.json"
    frames_dir = sequence_dir / "frames"
    if not metadata_path.exists() or not frames_dir.exists():
        return []

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    width = int(data["metadata"]["width"])
    height = int(data["metadata"]["height"])
    num_frames = int(data["metadata"]["num_frames"])
    instances = data["instances"]
    sequence_name = decode_video_name(str(data["metadata"]["video_name"]))

    records: list[dict[str, Any]] = []
    object_centric_count = 0
    object_centric_multi_count = 0
    object_centric_direction_count = 0
    camera_pose_count = 0
    frame_indices = choose_frame_indices(num_frames, args.max_frames_per_sequence, args.sample_frame_start, rng)

    for frame_idx in frame_indices:
        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        if not frame_path.exists():
            continue

        objects: list[dict[str, Any]] = []
        for object_idx in range(len(instances["positions"])):
            visible_pixels = int(instances["visibility"][object_idx][frame_idx])
            bbox = get_bbox_for_frame(instances, object_idx, frame_idx)
            if visible_pixels < args.min_visible_pixels:
                continue
            if bbox is None:
                continue
            if bbox_area(bbox) < args.min_bbox_area:
                continue
            objects.append(build_object_record(instances, object_idx, frame_idx, width, height))

        if len(objects) < 2:
            continue
        if not is_frame_unambiguous(objects, args.similar_color_threshold):
            continue

        object_idx_to_visible_idx = {obj["object_idx"]: idx for idx, obj in enumerate(objects)}
        visible_original_indices = [obj["object_idx"] for obj in objects]
        camera_position = [float(v) for v in data["camera"]["positions"][frame_idx]]
        visible_indices = list(range(len(objects)))
        anchor_candidates = visible_indices[:]
        rng.shuffle(anchor_candidates)

        for anchor_idx in anchor_candidates:
            if camera_pose_count >= args.max_camera_pose_samples_per_sequence:
                break
            camera_pose_task = build_camera_pose_task(
                sequence_name,
                frame_idx,
                frame_path,
                objects,
                anchor_idx,
                camera_position,
                args.min_object_centric_separation,
            )
            if camera_pose_task is not None:
                records.append(camera_pose_task)
                camera_pose_count += 1

        if len(objects) >= 3:
            for anchor_idx in anchor_candidates:
                if object_centric_multi_count >= args.max_object_centric_multi_samples_per_sequence:
                    break
                multi_tasks = build_object_centric_multi_tasks(
                    sequence_name,
                    frame_idx,
                    frame_path,
                    objects,
                    anchor_idx,
                    visible_indices,
                    camera_position,
                    args.min_object_centric_separation,
                    args.min_option_margin,
                    rng,
                )
                for multi_task in multi_tasks:
                    if object_centric_multi_count >= args.max_object_centric_multi_samples_per_sequence:
                        break
                    records.append(multi_task)
                    object_centric_multi_count += 1
            if len(records) >= args.max_samples_per_sequence:
                return records[: args.max_samples_per_sequence]

        for i in range(len(visible_original_indices)):
            for j in range(i + 1, len(visible_original_indices)):
                pair = (
                    object_idx_to_visible_idx[visible_original_indices[i]],
                    object_idx_to_visible_idx[visible_original_indices[j]],
                )

                image_task = build_image_view_task(
                    sequence_name,
                    frame_idx,
                    frame_path,
                    objects,
                    pair,
                    args.min_image_separation,
                )
                if image_task is not None:
                    records.append(image_task)

                height_task = build_height_task(
                    sequence_name,
                    frame_idx,
                    frame_path,
                    objects,
                    pair,
                    args.min_height_separation,
                )
                if height_task is not None:
                    records.append(height_task)

                distance_task = build_camera_distance_task(
                    sequence_name,
                    frame_idx,
                    frame_path,
                    objects,
                    pair,
                    camera_position,
                    args.min_camera_depth_separation,
                )
                if distance_task is not None:
                    records.append(distance_task)

                for anchor_idx, other_idx in (pair, (pair[1], pair[0])):
                    if object_centric_count >= args.max_object_centric_samples_per_sequence:
                        break
                    object_task = build_object_centric_task(
                        sequence_name,
                        frame_idx,
                        frame_path,
                        objects,
                        anchor_idx,
                        other_idx,
                        camera_position,
                        args.min_object_centric_separation,
                    )
                    if object_task is not None:
                        records.append(object_task)
                        object_centric_count += 1

                    if object_centric_direction_count < args.max_object_centric_direction_samples_per_sequence:
                        direction_tasks = build_object_centric_direction_tasks(
                            sequence_name,
                            frame_idx,
                            frame_path,
                            objects,
                            anchor_idx,
                            other_idx,
                            camera_position,
                            args.min_object_centric_separation,
                        )
                        for direction_task in direction_tasks:
                            if object_centric_direction_count >= args.max_object_centric_direction_samples_per_sequence:
                                break
                            records.append(direction_task)
                            object_centric_direction_count += 1

                if len(records) >= args.max_samples_per_sequence:
                    return records[: args.max_samples_per_sequence]

    return records[: args.max_samples_per_sequence]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    sequence_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())

    all_records: list[dict[str, Any]] = []
    for sequence_dir in sequence_dirs:
        all_records.extend(build_records_for_sequence(sequence_dir, args, rng))

    write_jsonl(args.output_jsonl, all_records)
    write_parquet(args.output_parquet, all_records)

    summary = {
        "num_sequences": len(sequence_dirs),
        "num_records": len(all_records),
        "output_jsonl": str(args.output_jsonl),
        "output_parquet": str(args.output_parquet),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
