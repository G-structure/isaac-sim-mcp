"""Contract tests for pure continuous-collision evidence."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac.sim.mcp_extension"
    / "isaac_sim_mcp_extension"
    / "continuous_collision.py"
)
SPEC = importlib.util.spec_from_file_location("continuous_collision_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
continuous_collision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuous_collision
SPEC.loader.exec_module(continuous_collision)


def _pose(
    x: float,
    *,
    orientation_wxyz: list[float] | None = None,
) -> object:
    return continuous_collision.RigidBodyPose.parse(
        [x, 0.0, 0.0],
        orientation_wxyz or [1.0, 0.0, 0.0, 0.0],
    )


def _classify(**overrides):
    values = {
        "sensor_path": "/World/left_finger",
        "filter_path": "/World/cube",
        "previous_sensor": _pose(0.0),
        "current_sensor": _pose(0.1),
        "previous_filter": _pose(1.0),
        "current_filter": _pose(1.0),
        "previous_endpoint_contact": False,
        "current_endpoint_contact": False,
        "sweep_hits": [],
        "sweep_query_available": True,
        "sweep_saturated": False,
        "max_sweep_hits": 8,
        "maximum_rotation_radians": 0.25,
    }
    values.update(overrides)
    return continuous_collision.classify_update(**values)


def _hit(path: str, distance_m: float = 0.05) -> dict[str, object]:
    return {
        "rigid_body_path": path,
        "collider_path": f"{path}/collision",
        "distance_m": distance_m,
    }


def test_clean_motion_is_clear_and_json_serializable() -> None:
    result = _classify(current_filter=_pose(0.98))

    assert result["classification"] == "clear"
    assert result["passed"] is True
    assert result["complete"] is True
    assert result["failure_reasons"] == []
    assert result["relative_motion"] == {
        "translation_m": pytest.approx([0.12, 0.0, 0.0]),
        "direction_unit": pytest.approx([1.0, 0.0, 0.0]),
        "distance_m": pytest.approx(0.12),
    }
    json.dumps(result, allow_nan=False)


def test_paired_sweep_hit_without_endpoint_contact_is_tunneling() -> None:
    result = _classify(sweep_hits=[_hit("/World/cube")])

    assert result["classification"] == "paired_tunneling"
    assert result["passed"] is False
    assert result["complete"] is True
    assert result["tunneling_detected"] is True
    assert result["paired_hit_count"] == 1
    assert result["failure_reasons"] == [
        "paired_body_sweep_hit_without_endpoint_contact"
    ]


def test_unrelated_sweep_hit_does_not_trigger_tunneling() -> None:
    result = _classify(sweep_hits=[_hit("/World/table")])

    assert result["classification"] == "clear"
    assert result["passed"] is True
    assert result["paired_hit_count"] == 0
    assert result["sweep"]["captured_hit_count"] == 1


def test_endpoint_contact_accounts_for_paired_sweep_hit() -> None:
    result = _classify(
        current_endpoint_contact=True,
        sweep_hits=[_hit("/World/cube")],
    )

    assert result["classification"] == "clear"
    assert result["passed"] is True
    assert result["complete"] is True
    assert result["tunneling_detected"] is False
    assert result["paired_hit_count"] == 1


def test_rotation_above_translation_sweep_limit_fails_closed() -> None:
    angle = 0.3
    result = _classify(
        current_sensor=_pose(
            0.1,
            orientation_wxyz=[
                math.cos(angle / 2.0),
                0.0,
                0.0,
                math.sin(angle / 2.0),
            ],
        ),
        maximum_rotation_radians=0.2,
    )

    assert result["classification"] == "indeterminate"
    assert result["passed"] is False
    assert result["complete"] is False
    assert result["failure_reasons"] == ["rotation_limit_exceeded"]
    assert result["rotation_delta_radians"]["sensor"] == pytest.approx(angle)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"previous_sensor": None}, "pose_evidence_unavailable"),
        (
            {
                "sweep_hits": None,
                "sweep_query_available": False,
            },
            "sweep_query_unavailable",
        ),
    ],
)
def test_unavailable_pose_or_query_evidence_fails_closed(
    overrides: dict[str, object],
    reason: str,
) -> None:
    result = _classify(**overrides)

    assert result["classification"] == "indeterminate"
    assert result["passed"] is False
    assert result["complete"] is False
    assert reason in result["failure_reasons"]


def test_saturated_sweep_evidence_fails_closed_and_remains_bounded() -> None:
    result = _classify(
        sweep_hits=[
            _hit("/World/table", 0.01),
            _hit("/World/cube", 0.02),
        ],
        max_sweep_hits=1,
    )

    assert result["classification"] == "indeterminate"
    assert result["passed"] is False
    assert result["complete"] is False
    assert result["failure_reasons"] == ["sweep_hits_saturated"]
    assert result["sweep"]["saturated"] is True
    assert result["sweep"]["captured_hit_count"] == 1
    assert result["sweep"]["hits"] == [_hit("/World/table", 0.01)]


def test_quaternion_delta_uses_shortest_physical_rotation() -> None:
    assert continuous_collision.quaternion_angular_delta_radians(
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="must not be zero"):
        continuous_collision.RigidBodyPose.parse(
            {
                "position_m": [0.0, 0.0, 0.0],
                "orientation_wxyz": [0.0, 0.0, 0.0, 0.0],
            }
        )
