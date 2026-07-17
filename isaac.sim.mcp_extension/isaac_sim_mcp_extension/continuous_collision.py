"""Pure contracts for per-update continuous-collision evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, TypedDict


_SCHEMA_VERSION = 2
_MAX_SWEEP_HITS = 256
_QUATERNION_NORM_TOLERANCE = 1.0e-3
_MOTION_EPSILON_M = 1.0e-12


@dataclass(frozen=True)
class RigidBodyPose:
    """Validated world pose using Isaac's scalar-first quaternion convention."""

    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]

    @classmethod
    def parse(
        cls,
        value: RigidBodyPose | Mapping[str, Any] | Iterable[Any],
        orientation_wxyz: Iterable[Any] | None = None,
        *,
        field: str = "pose",
    ) -> RigidBodyPose:
        """Validate a pose mapping, raw component arrays, or a parsed pose."""

        if isinstance(value, cls) and orientation_wxyz is None:
            return value
        if orientation_wxyz is not None:
            position_value = value
            orientation_value = orientation_wxyz
        else:
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{field} must be a pose object or position/quaternion pair"
                )
            if "position_m" not in value or "orientation_wxyz" not in value:
                raise ValueError(
                    f"{field} must contain position_m and orientation_wxyz"
                )
            position_value = value["position_m"]
            orientation_value = value["orientation_wxyz"]
        position = _finite_components(
            position_value,
            length=3,
            field=f"{field}.position_m",
        )
        orientation = _validated_quaternion(
            orientation_value,
            field=f"{field}.orientation_wxyz",
        )
        return cls(
            (position[0], position[1], position[2]),
            (
                orientation[0],
                orientation[1],
                orientation[2],
                orientation[3],
            ),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Return a fresh JSON-safe representation."""

        return {
            "position_m": list(self.position_m),
            "orientation_wxyz": list(self.orientation_wxyz),
        }


@dataclass(frozen=True)
class RelativeMotion:
    """Sensor translation relative to the paired rigid body."""

    translation_m: tuple[float, float, float]
    direction_unit: tuple[float, float, float]
    distance_m: float

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-safe representation."""

        return {
            "translation_m": list(self.translation_m),
            "direction_unit": list(self.direction_unit),
            "distance_m": self.distance_m,
        }


class SweepHitRecord(TypedDict):
    """One bounded sweep hit copied into JSON-safe storage."""

    rigid_body_path: str
    collider_path: str | None
    distance_m: float


class SweepEvidence(TypedDict):
    """Bounded result of one scene-query sweep or overlap."""

    available: bool
    max_hits: int
    captured_hit_count: int
    saturated: bool
    hits: list[SweepHitRecord]


class ContinuousCollisionClassification(TypedDict):
    """JSON-safe verdict for one ordered rigid-body pair and one update."""

    schema_version: int
    classification: str
    passed: bool
    complete: bool
    swept_collision_risk_detected: bool
    tunneling_detected: bool
    failure_reasons: list[str]
    errors: list[str]
    sensor_path: str
    filter_path: str
    previous_endpoint_contact: bool | None
    current_endpoint_contact: bool | None
    poses: dict[str, dict[str, list[float]] | None]
    rotation_delta_radians: dict[str, float] | None
    maximum_rotation_radians: dict[str, float]
    relative_motion: dict[str, Any] | None
    sweep: SweepEvidence
    paired_hit_count: int
    exact_shape_sweep: SweepEvidence
    exact_paired_hit_count: int
    broad_phase_only: bool


PoseInput = RigidBodyPose | Mapping[str, Any]


__all__ = [
    "ContinuousCollisionClassification",
    "RelativeMotion",
    "RigidBodyPose",
    "SweepEvidence",
    "SweepHitRecord",
    "bounded_sweep_hits",
    "classify_update",
    "quaternion_angular_delta_radians",
    "rotate_vector_by_quaternion",
    "relative_rotation_delta_radians",
    "relative_motion",
]


def _finite_components(
    value: Iterable[Any],
    *,
    length: int,
    field: str,
) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{field} must contain exactly {length} numbers")
    try:
        components = list(value)
    except TypeError as exc:
        raise ValueError(f"{field} must contain exactly {length} numbers") from exc
    if len(components) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")

    result: list[float] = []
    for component in components:
        if isinstance(component, bool):
            raise ValueError(f"{field} must contain only finite numbers")
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain only finite numbers")
        result.append(number)
    return result


def _validated_quaternion(
    value: Iterable[Any],
    *,
    field: str,
) -> list[float]:
    quaternion = _finite_components(value, length=4, field=field)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm == 0.0:
        raise ValueError(f"{field} must not be zero")
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise ValueError(f"{field} must be a unit quaternion")
    return [component / norm for component in quaternion]


def _absolute_path(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or len(value) < 2
        or not value.startswith("/")
    ):
        raise ValueError(f"{field} must be an absolute prim path")
    return value


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _validated_max_hits(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_hits must be an integer")
    if not 1 <= value <= _MAX_SWEEP_HITS:
        raise ValueError(f"max_hits must be between 1 and {_MAX_SWEEP_HITS}")
    return value


def quaternion_angular_delta_radians(
    start_orientation_wxyz: Iterable[Any],
    end_orientation_wxyz: Iterable[Any],
) -> float:
    """Return the shortest physical rotation between two unit quaternions."""

    start = _validated_quaternion(
        start_orientation_wxyz,
        field="start_orientation_wxyz",
    )
    end = _validated_quaternion(
        end_orientation_wxyz,
        field="end_orientation_wxyz",
    )
    if sum(a * b for a, b in zip(start, end)) < 0.0:
        end = [-component for component in end]
    difference_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))
    sum_norm = math.sqrt(sum((a + b) ** 2 for a, b in zip(start, end)))
    return 4.0 * math.atan2(difference_norm, sum_norm)


def _quaternion_conjugate(
    orientation_wxyz: Iterable[Any],
    *,
    field: str,
) -> tuple[float, float, float, float]:
    w, x, y, z = _validated_quaternion(orientation_wxyz, field=field)
    return (w, -x, -y, -z)


def _quaternion_multiply(
    left_wxyz: Iterable[Any],
    right_wxyz: Iterable[Any],
    *,
    field: str,
) -> tuple[float, float, float, float]:
    left = _validated_quaternion(left_wxyz, field=f"{field}.left")
    right = _validated_quaternion(right_wxyz, field=f"{field}.right")
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    product = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    normalized = _validated_quaternion(product, field=field)
    return (normalized[0], normalized[1], normalized[2], normalized[3])


def rotate_vector_by_quaternion(
    orientation_wxyz: Iterable[Any],
    vector_xyz: Iterable[Any],
    *,
    field: str,
) -> tuple[float, float, float]:
    """Rotate a vector by a scalar-first unit quaternion."""

    w, x, y, z = _validated_quaternion(
        orientation_wxyz,
        field=f"{field}.orientation_wxyz",
    )
    vx, vy, vz = _finite_components(
        vector_xyz,
        length=3,
        field=f"{field}.vector_xyz",
    )
    twice_cross = (
        2.0 * (y * vz - z * vy),
        2.0 * (z * vx - x * vz),
        2.0 * (x * vy - y * vx),
    )
    cross_again = (
        y * twice_cross[2] - z * twice_cross[1],
        z * twice_cross[0] - x * twice_cross[2],
        x * twice_cross[1] - y * twice_cross[0],
    )
    return (
        vx + w * twice_cross[0] + cross_again[0],
        vy + w * twice_cross[1] + cross_again[1],
        vz + w * twice_cross[2] + cross_again[2],
    )


def relative_rotation_delta_radians(
    previous_sensor: PoseInput,
    current_sensor: PoseInput,
    previous_filter: PoseInput,
    current_filter: PoseInput,
) -> float:
    """Return sensor rotation relative to a frame rigidly attached to the filter."""

    previous_sensor_pose = RigidBodyPose.parse(
        previous_sensor,
        field="previous_sensor",
    )
    current_sensor_pose = RigidBodyPose.parse(
        current_sensor,
        field="current_sensor",
    )
    previous_filter_pose = RigidBodyPose.parse(
        previous_filter,
        field="previous_filter",
    )
    current_filter_pose = RigidBodyPose.parse(
        current_filter,
        field="current_filter",
    )
    previous_relative = _quaternion_multiply(
        _quaternion_conjugate(
            previous_filter_pose.orientation_wxyz,
            field="previous_filter.orientation_wxyz",
        ),
        previous_sensor_pose.orientation_wxyz,
        field="previous_relative_orientation",
    )
    current_relative = _quaternion_multiply(
        _quaternion_conjugate(
            current_filter_pose.orientation_wxyz,
            field="current_filter.orientation_wxyz",
        ),
        current_sensor_pose.orientation_wxyz,
        field="current_relative_orientation",
    )
    return quaternion_angular_delta_radians(previous_relative, current_relative)


def relative_motion(
    previous_sensor: PoseInput,
    current_sensor: PoseInput,
    previous_filter: PoseInput,
    current_filter: PoseInput,
) -> RelativeMotion:
    """Return sensor translation in coordinates rigidly attached to the filter."""

    previous_sensor_pose = RigidBodyPose.parse(
        previous_sensor,
        field="previous_sensor",
    )
    current_sensor_pose = RigidBodyPose.parse(
        current_sensor,
        field="current_sensor",
    )
    previous_filter_pose = RigidBodyPose.parse(
        previous_filter,
        field="previous_filter",
    )
    current_filter_pose = RigidBodyPose.parse(
        current_filter,
        field="current_filter",
    )
    previous_sensor_from_filter = tuple(
        previous_sensor_pose.position_m[axis] - previous_filter_pose.position_m[axis]
        for axis in range(3)
    )
    previous_sensor_in_filter = rotate_vector_by_quaternion(
        _quaternion_conjugate(
            previous_filter_pose.orientation_wxyz,
            field="previous_filter.orientation_wxyz",
        ),
        previous_sensor_from_filter,
        field="previous_sensor_in_filter",
    )
    current_sensor_from_filter = tuple(
        current_sensor_pose.position_m[axis] - current_filter_pose.position_m[axis]
        for axis in range(3)
    )
    current_sensor_in_filter = rotate_vector_by_quaternion(
        _quaternion_conjugate(
            current_filter_pose.orientation_wxyz,
            field="current_filter.orientation_wxyz",
        ),
        current_sensor_from_filter,
        field="current_sensor_in_filter",
    )
    translation = tuple(
        current_sensor_in_filter[axis] - previous_sensor_in_filter[axis]
        for axis in range(3)
    )
    distance = math.sqrt(sum(component * component for component in translation))
    if distance <= _MOTION_EPSILON_M:
        translation = (0.0, 0.0, 0.0)
        direction = (0.0, 0.0, 0.0)
        distance = 0.0
    else:
        direction = tuple(component / distance for component in translation)
    return RelativeMotion(
        (translation[0], translation[1], translation[2]),
        (direction[0], direction[1], direction[2]),
        distance,
    )


def _validate_sweep_hit(value: Mapping[str, Any]) -> SweepHitRecord:
    if not isinstance(value, Mapping):
        raise ValueError("each sweep hit must be an object")
    rigid_body_path = _absolute_path(
        value.get("rigid_body_path"),
        field="sweep hit rigid_body_path",
    )
    raw_collider_path = value.get("collider_path")
    collider_path = (
        None
        if raw_collider_path is None
        else _absolute_path(raw_collider_path, field="sweep hit collider_path")
    )
    return {
        "rigid_body_path": rigid_body_path,
        "collider_path": collider_path,
        "distance_m": _finite_nonnegative(
            value.get("distance_m"),
            field="sweep hit distance_m",
        ),
    }


def bounded_sweep_hits(
    hits: Iterable[Mapping[str, Any]] | None,
    *,
    max_hits: int,
    query_available: bool = True,
    saturated: bool = False,
) -> SweepEvidence:
    """Copy at most ``max_hits`` records and mark any observed overflow."""

    capacity = _validated_max_hits(max_hits)
    if not isinstance(query_available, bool):
        raise ValueError("query_available must be a boolean")
    if not isinstance(saturated, bool):
        raise ValueError("saturated must be a boolean")

    evidence: SweepEvidence = {
        "available": query_available and hits is not None,
        "max_hits": capacity,
        "captured_hit_count": 0,
        "saturated": saturated,
        "hits": [],
    }
    if not evidence["available"]:
        return evidence
    if isinstance(hits, (str, bytes, Mapping)):
        raise ValueError("sweep hits must be an iterable of objects")

    try:
        iterator = iter(hits)
    except TypeError as exc:
        raise ValueError("sweep hits must be an iterable of objects") from exc
    for index, hit in enumerate(iterator):
        if index >= capacity:
            evidence["saturated"] = True
            break
        evidence["hits"].append(_validate_sweep_hit(hit))
    evidence["captured_hit_count"] = len(evidence["hits"])
    return evidence


def classify_update(
    *,
    sensor_path: str,
    filter_path: str,
    previous_sensor: PoseInput | None,
    current_sensor: PoseInput | None,
    previous_filter: PoseInput | None,
    current_filter: PoseInput | None,
    previous_endpoint_contact: bool | None,
    current_endpoint_contact: bool | None,
    sweep_hits: Iterable[Mapping[str, Any]] | None,
    sweep_query_available: bool,
    maximum_sensor_rotation_radians: float,
    maximum_filter_rotation_radians: float,
    max_sweep_hits: int = 64,
    sweep_saturated: bool = False,
    exact_shape_hits: Iterable[Mapping[str, Any]] | None = None,
    exact_shape_query_available: bool | None = None,
    exact_shape_saturated: bool = False,
) -> ContinuousCollisionClassification:
    """Classify one update using exact-shape hits to confirm broad-phase risk.

    The rotation-safe OBB sweep is intentionally conservative and can intersect
    a paired body while the exact collision carriers do not. That condition is
    retained as diagnostic ``broad_phase_only`` evidence, but it is not a
    tunneling verdict. Exact-shape evidence is required to make the result
    complete when the mechanism provides it.
    """

    sensor = _absolute_path(sensor_path, field="sensor_path")
    filtered = _absolute_path(filter_path, field="filter_path")
    if sensor == filtered:
        raise ValueError("sensor_path and filter_path must differ")
    capacity = _validated_max_hits(max_sweep_hits)
    sensor_rotation_limit = _finite_nonnegative(
        maximum_sensor_rotation_radians,
        field="maximum_sensor_rotation_radians",
    )
    filter_rotation_limit = _finite_nonnegative(
        maximum_filter_rotation_radians,
        field="maximum_filter_rotation_radians",
    )
    if sensor_rotation_limit > math.pi:
        raise ValueError("maximum_sensor_rotation_radians must not exceed pi")
    if filter_rotation_limit > math.pi:
        raise ValueError("maximum_filter_rotation_radians must not exceed pi")

    failure_reasons: list[str] = []
    errors: list[str] = []

    def fail(reason: str, error: str | None = None) -> None:
        if reason not in failure_reasons:
            failure_reasons.append(reason)
        if error is not None:
            errors.append(error)

    raw_poses = {
        "previous_sensor": previous_sensor,
        "current_sensor": current_sensor,
        "previous_filter": previous_filter,
        "current_filter": current_filter,
    }
    poses: dict[str, RigidBodyPose | None] = {}
    for name, raw_pose in raw_poses.items():
        if raw_pose is None:
            poses[name] = None
            fail("pose_evidence_unavailable", f"{name} pose is unavailable")
            continue
        try:
            poses[name] = RigidBodyPose.parse(raw_pose, field=name)
        except ValueError as exc:
            poses[name] = None
            fail("pose_evidence_invalid", str(exc))

    previous_contact = (
        previous_endpoint_contact
        if isinstance(previous_endpoint_contact, bool)
        else None
    )
    current_contact = (
        current_endpoint_contact if isinstance(current_endpoint_contact, bool) else None
    )
    if previous_contact is None or current_contact is None:
        fail(
            "endpoint_contact_evidence_unavailable",
            "endpoint contact evidence must contain two booleans",
        )

    try:
        sweep = bounded_sweep_hits(
            sweep_hits,
            max_hits=capacity,
            query_available=sweep_query_available,
            saturated=sweep_saturated,
        )
    except ValueError as exc:
        sweep = {
            "available": False,
            "max_hits": capacity,
            "captured_hit_count": 0,
            "saturated": False,
            "hits": [],
        }
        fail("sweep_query_evidence_invalid", str(exc))
    if not sweep["available"]:
        fail("sweep_query_unavailable", "sweep query evidence is unavailable")
    if sweep["saturated"]:
        fail("sweep_hits_saturated")

    exact_shape_evidence_supplied = not (
        exact_shape_hits is None and exact_shape_query_available is None
    )
    if not exact_shape_evidence_supplied:
        exact_shape_hits = sweep_hits
        exact_shape_query_available = sweep_query_available
        exact_shape_saturated = sweep_saturated
    exact_shape_available = bool(exact_shape_query_available)
    try:
        exact_shape_sweep = bounded_sweep_hits(
            exact_shape_hits,
            max_hits=capacity,
            query_available=exact_shape_available,
            saturated=exact_shape_saturated,
        )
    except ValueError as exc:
        exact_shape_sweep = {
            "available": False,
            "max_hits": capacity,
            "captured_hit_count": 0,
            "saturated": False,
            "hits": [],
        }
        fail("exact_shape_sweep_query_invalid", str(exc))
    if (
        exact_shape_evidence_supplied
        and not exact_shape_sweep["available"]
        and sweep["captured_hit_count"] > 0
    ):
        fail(
            "exact_shape_sweep_query_unavailable",
            "exact collision-shape sweep evidence is unavailable",
        )
    if exact_shape_evidence_supplied and exact_shape_sweep["saturated"]:
        fail("exact_shape_sweep_hits_saturated")

    motion: RelativeMotion | None = None
    rotation_deltas: dict[str, float] | None = None
    if all(pose is not None for pose in poses.values()):
        previous_sensor_pose = poses["previous_sensor"]
        current_sensor_pose = poses["current_sensor"]
        previous_filter_pose = poses["previous_filter"]
        current_filter_pose = poses["current_filter"]
        assert previous_sensor_pose is not None
        assert current_sensor_pose is not None
        assert previous_filter_pose is not None
        assert current_filter_pose is not None

        motion = relative_motion(
            previous_sensor_pose,
            current_sensor_pose,
            previous_filter_pose,
            current_filter_pose,
        )
        sensor_rotation = quaternion_angular_delta_radians(
            previous_sensor_pose.orientation_wxyz,
            current_sensor_pose.orientation_wxyz,
        )
        filter_rotation = quaternion_angular_delta_radians(
            previous_filter_pose.orientation_wxyz,
            current_filter_pose.orientation_wxyz,
        )
        relative_rotation = relative_rotation_delta_radians(
            previous_sensor_pose,
            current_sensor_pose,
            previous_filter_pose,
            current_filter_pose,
        )
        rotation_deltas = {
            "sensor": sensor_rotation,
            "filter": filter_rotation,
            "relative": relative_rotation,
            "maximum": max(sensor_rotation, filter_rotation, relative_rotation),
        }
        if sensor_rotation > sensor_rotation_limit:
            fail("sensor_rotation_limit_exceeded")
        if filter_rotation > filter_rotation_limit:
            fail("filter_rotation_limit_exceeded")

    paired_hit_count = sum(hit["rigid_body_path"] == filtered for hit in sweep["hits"])
    exact_paired_hit_count = sum(
        hit["rigid_body_path"] == filtered for hit in exact_shape_sweep["hits"]
    )
    broad_phase_only = paired_hit_count > 0 and exact_paired_hit_count == 0
    tunneling_detected = (
        exact_paired_hit_count > 0
        and previous_contact is False
        and current_contact is False
    )
    if tunneling_detected:
        fail("paired_body_sweep_hit_without_endpoint_contact")

    incomplete_reasons = {
        "endpoint_contact_evidence_unavailable",
        "pose_evidence_invalid",
        "pose_evidence_unavailable",
        "filter_rotation_limit_exceeded",
        "sensor_rotation_limit_exceeded",
        "sweep_hits_saturated",
        "sweep_query_evidence_invalid",
        "sweep_query_unavailable",
        "exact_shape_sweep_hits_saturated",
        "exact_shape_sweep_query_invalid",
        "exact_shape_sweep_query_unavailable",
    }
    complete = not any(reason in incomplete_reasons for reason in failure_reasons)
    passed = complete and not tunneling_detected
    classification = (
        "paired_tunneling"
        if tunneling_detected
        else "conservative_envelope_only"
        if broad_phase_only and complete
        else "clear"
        if passed
        else "indeterminate"
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "classification": classification,
        "passed": passed,
        "complete": complete,
        "swept_collision_risk_detected": tunneling_detected,
        "tunneling_detected": tunneling_detected,
        "failure_reasons": failure_reasons,
        "errors": errors,
        "sensor_path": sensor,
        "filter_path": filtered,
        "previous_endpoint_contact": previous_contact,
        "current_endpoint_contact": current_contact,
        "poses": {
            name: pose.to_dict() if pose is not None else None
            for name, pose in poses.items()
        },
        "rotation_delta_radians": rotation_deltas,
        "maximum_rotation_radians": {
            "sensor": sensor_rotation_limit,
            "filter": filter_rotation_limit,
        },
        "relative_motion": motion.to_dict() if motion is not None else None,
        "sweep": sweep,
        "paired_hit_count": paired_hit_count,
        "exact_shape_sweep": exact_shape_sweep,
        "exact_paired_hit_count": exact_paired_hit_count,
        "broad_phase_only": broad_phase_only,
    }
