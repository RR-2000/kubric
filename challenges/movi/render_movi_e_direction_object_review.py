#!/usr/bin/env python3
"""Render MOVi-E direction/object pairs as reviewable contact sheets.

The script reads the candidate JSONL emitted by
``build_movi_e_direction_object_dataset.py``.  It renders one panel per
matched source pair, overlays the anchor and displayed object choices, and
writes a manifest mapping every panel to its ``source_qid``.  Reviewers can
copy approved source IDs from the manifest into a plain-text allowlist.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Candidate pair JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-sheet", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=384)
    return parser.parse_args()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object on line {line_number} of {path}")
                yield value


def _matched_pairs(rows: Iterator[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        source_id = str(row.get("source_qid", ""))
        answer_format = str(row.get("diagnostic_answer_format", ""))
        if source_id and answer_format in {"direction", "object"}:
            grouped[source_id][answer_format] = row
    incomplete = [source_id for source_id, pair in grouped.items() if set(pair) != {"direction", "object"}]
    if incomplete:
        raise ValueError(f"Found {len(incomplete)} incomplete source pair(s), e.g. {incomplete[0]!r}")
    return [(source_id, pair["direction"], pair["object"]) for source_id, pair in sorted(grouped.items())]


def _box_for_object(row: dict[str, Any], object_id: int) -> list[int] | None:
    for item in row.get("visible_objects", []):
        if item.get("object_idx") == object_id:
            box = item.get("bbox_2d_xyxy_pixels")
            if isinstance(box, list) and len(box) == 4:
                return [int(value) for value in box]
    return None


def _draw_box(draw: ImageDraw.ImageDraw, box: list[int] | None, scale: float, color: str, label: str) -> None:
    if box is None:
        return
    scaled = [int(value * scale) for value in box]
    draw.rectangle(scaled, outline=color, width=3)
    draw.text((scaled[0] + 3, max(0, scaled[1] - 14)), label, fill=color, stroke_width=1, stroke_fill="black")


def _panel(source_id: str, direction: dict[str, Any], object_row: dict[str, Any], image_width: int) -> Image.Image:
    image_path = Path(str(direction.get("image_path") or direction.get("img_path") or direction.get("image")))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    scale = image_width / image.width
    image = image.resize((image_width, round(image.height * scale)))
    raw_text_lines = [
        source_id,
        f"Anchor: {direction.get('diagnostic_anchor')} | Target: {direction.get('diagnostic_target_object')}",
        f"Relation: {direction.get('diagnostic_relation')}",
        "Object options: " + " | ".join(
            f"{letter}. {value}" for letter, value in zip("ABCD", object_row.get("options", []))
        ),
        "Approve only if every named object is identifiable in the image.",
    ]
    text_lines = [
        wrapped
        for line in raw_text_lines
        for wrapped in textwrap.wrap(line, width=max(36, image_width // 7))
    ]
    font = ImageFont.load_default()
    text_height = 16 * len(text_lines) + 10
    panel = Image.new("RGB", (image.width, image.height + text_height), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    anchor_id = direction.get("diagnostic_anchor_object_id")
    if isinstance(anchor_id, int):
        _draw_box(draw, _box_for_object(direction, anchor_id), scale, "yellow", "anchor")
    for letter, object_id in zip("ABCD", object_row.get("candidate_object_ids", [])):
        if isinstance(object_id, int):
            _draw_box(draw, _box_for_object(direction, object_id), scale, "cyan", letter)
    for line_number, line in enumerate(text_lines):
        draw.text((4, image.height + 4 + line_number * 16), line, fill="black", font=font)
    return panel


def main() -> None:
    args = parse_args()
    if args.pairs_per_sheet < 1:
        raise ValueError("--pairs-per-sheet must be positive")
    pairs = _matched_pairs(_read_jsonl(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "review_manifest.jsonl"
    manifest = []
    for sheet_index in range(math.ceil(len(pairs) / args.pairs_per_sheet)):
        chunk = pairs[sheet_index * args.pairs_per_sheet : (sheet_index + 1) * args.pairs_per_sheet]
        panels = [_panel(*pair, args.image_width) for pair in chunk]
        width = max(panel.width for panel in panels)
        height = sum(panel.height for panel in panels)
        sheet = Image.new("RGB", (width, height), "white")
        y_offset = 0
        for panel_index, (panel, (source_id, direction, object_row)) in enumerate(zip(panels, chunk)):
            sheet.paste(panel, (0, y_offset))
            manifest.append({
                "source_qid": source_id,
                "sheet": f"review_{sheet_index:04d}.png",
                "panel_index": panel_index,
                "direction_qid": direction.get("qid"),
                "object_qid": object_row.get("qid"),
            })
            y_offset += panel.height
        sheet.save(args.output_dir / f"review_{sheet_index:04d}.png")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item) + "\n")
    print(f"Rendered {len(pairs)} pairs into {args.output_dir}")
    print(f"Review manifest: {manifest_path}")


if __name__ == "__main__":
    main()
