from pathlib import Path

from challenges.movi.build_movi_e_direction_object_dataset import build_pair


def _object(object_idx, name, position):
    return {
        "object_idx": object_idx,
        "name": name,
        "position_3d": position,
    }


def test_build_pair_emits_matched_direction_and_object_rows():
    objects = [
        _object(0, "anchor", [0.0, 0.0, 0.0]),
        _object(1, "left object", [-3.0, 0.0, 0.0]),
        _object(2, "right object", [3.0, 0.0, 0.0]),
        _object(3, "front object", [0.0, 3.0, 0.0]),
        _object(4, "behind object", [0.0, -3.0, 0.0]),
    ]
    pair, skip_reason = build_pair(
        sequence_name="example",
        frame_idx=20,
        frame_path=Path("/tmp/frame.png"),
        objects=objects,
        anchor_idx=0,
        target_idx=1,
        camera_position=[0.0, 10.0, 5.0],
        min_separation=0.35,
        min_margin=0.1,
        seed=0,
    )
    assert skip_reason is None
    assert len(pair) == 2
    assert {row["diagnostic_answer_format"] for row in pair} == {
        "direction",
        "object",
    }
    assert pair[0]["source_relation_id"] == pair[1]["source_relation_id"]
    assert all(row["diagnostic_relation"] == "left" for row in pair)
    assert all(row["num_correct_object_options"] == 1 for row in pair)


def test_build_pair_drops_multiple_correct_displayed_objects():
    objects = [
        _object(0, "anchor", [0.0, 0.0, 0.0]),
        _object(1, "left object one", [-3.0, 0.0, 0.0]),
        _object(2, "left object two", [-2.0, 0.0, 0.0]),
        _object(3, "right object", [3.0, 0.0, 0.0]),
        _object(4, "front object", [0.0, 3.0, 0.0]),
    ]
    pair, skip_reason = build_pair(
        sequence_name="example",
        frame_idx=20,
        frame_path=Path("/tmp/frame.png"),
        objects=objects,
        anchor_idx=0,
        target_idx=1,
        camera_position=[0.0, 10.0, 5.0],
        min_separation=0.35,
        min_margin=0.1,
        seed=0,
    )
    assert pair == []
    assert skip_reason == "multiple_correct_object_options"
