"""Bounded, fail-closed PhysX contact evidence for atomic simulation steps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


_SCHEMA_VERSION = 2
_MAX_PAIRS = 16
_MAX_CONTACTS_PER_PAIR = 256
_MAX_CONTACT_RECORD_SLOTS = 8192
_DEFAULT_CONTINUOUS_COLLISION_ROTATION_RADIANS = math.radians(5.0)


def _tensor_to_array(value: Any) -> np.ndarray:
    """Copy a tensor-like value before PhysX reuses its transient buffer."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _finite_float(value: Any, field: str) -> float:
    result = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(result):
        raise RuntimeError(f"contact telemetry emitted non-finite {field}")
    return result


def _finite_vector(value: Any, field: str) -> List[float]:
    result = [float(item) for item in np.asarray(value).reshape(-1)]
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"contact telemetry emitted invalid {field}")
    return result


@dataclass(frozen=True)
class ContactPair:
    """One ordered sensor/filter rigid-body pair."""

    label: str
    sensor_path: str
    filter_path: str
    sensor_collider_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContactIntegrityConfig:
    """Validated public configuration for one bounded contact trace."""

    pairs: tuple[ContactPair, ...]
    max_contacts_per_pair: int
    maximum_penetration_m: float | None
    maximum_normal_impulse_ns: float | None
    continuous_collision: Dict[str, Any] | None

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ContactIntegrityConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("contact_integrity must be an object")
        raw_pairs = raw.get("pairs")
        if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
            raise ValueError("contact_integrity.pairs must be a list")
        if not 1 <= len(raw_pairs) <= _MAX_PAIRS:
            raise ValueError(
                f"contact_integrity.pairs must contain between 1 and {_MAX_PAIRS} pairs"
            )

        pairs: list[ContactPair] = []
        labels: set[str] = set()
        for index, item in enumerate(raw_pairs):
            if not isinstance(item, Mapping):
                raise ValueError(f"contact_integrity.pairs[{index}] must be an object")
            sensor_path = item.get("sensor_path", item.get("a"))
            filter_path = item.get("filter_path", item.get("b"))
            label = item.get("label", f"pair-{index}")
            for field, value in (
                ("sensor_path", sensor_path),
                ("filter_path", filter_path),
                ("label", label),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"contact_integrity.pairs[{index}].{field} must be a non-empty string"
                    )
            if not sensor_path.startswith("/") or not filter_path.startswith("/"):
                raise ValueError("contact integrity paths must be absolute USD paths")
            if sensor_path == filter_path:
                raise ValueError(
                    "contact integrity sensor and filter paths must differ"
                )
            if label in labels:
                raise ValueError(f"duplicate contact integrity pair label: {label}")
            raw_colliders = item.get("sensor_collider_paths", ())
            if not isinstance(raw_colliders, Sequence) or isinstance(
                raw_colliders, (str, bytes)
            ):
                raise ValueError(
                    "contact_integrity.pairs"
                    f"[{index}].sensor_collider_paths must be a list"
                )
            if len(raw_colliders) > 32:
                raise ValueError(
                    "contact integrity sensor collider paths must contain at most 32 entries"
                )
            colliders: list[str] = []
            for collider_index, collider_path in enumerate(raw_colliders):
                if (
                    not isinstance(collider_path, str)
                    or not collider_path.startswith("/")
                    or not collider_path.strip()
                ):
                    raise ValueError(
                        "contact_integrity.pairs"
                        f"[{index}].sensor_collider_paths[{collider_index}] "
                        "must be an absolute USD path"
                    )
                if collider_path in colliders:
                    raise ValueError(
                        "duplicate contact integrity sensor collider path: "
                        f"{collider_path}"
                    )
                if not (
                    collider_path == sensor_path
                    or collider_path.startswith(sensor_path.rstrip("/") + "/")
                ):
                    raise ValueError(
                        "contact integrity sensor collider paths must be beneath "
                        f"the sensor rigid body: {collider_path}"
                    )
                colliders.append(collider_path)
            labels.add(label)
            pairs.append(ContactPair(label, sensor_path, filter_path, tuple(colliders)))

        max_contacts = raw.get("max_contacts_per_pair", 64)
        if isinstance(max_contacts, bool) or not isinstance(max_contacts, int):
            raise ValueError("max_contacts_per_pair must be an integer")
        if not 1 <= max_contacts <= _MAX_CONTACTS_PER_PAIR:
            raise ValueError(
                f"max_contacts_per_pair must be between 1 and {_MAX_CONTACTS_PER_PAIR}"
            )

        raw_limits = raw.get("limits", {})
        if not isinstance(raw_limits, Mapping):
            raise ValueError("contact_integrity.limits must be an object")
        supported_limits = {
            "maximum_penetration_m",
            "maximum_normal_impulse_ns",
        }
        unknown_limits = set(raw_limits) - supported_limits
        if unknown_limits:
            raise ValueError(
                "unsupported contact integrity limits: "
                + ", ".join(sorted(str(name) for name in unknown_limits))
            )

        def optional_limit(name: str) -> float | None:
            value = raw_limits.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"contact_integrity.limits.{name} must be a number")
            result = float(value)
            if not math.isfinite(result) or result < 0:
                raise ValueError(
                    f"contact_integrity.limits.{name} must be finite and non-negative"
                )
            return result

        raw_continuous = raw.get("continuous_collision")
        continuous: Dict[str, Any] | None = None
        if raw_continuous is not None:
            if not isinstance(raw_continuous, Mapping):
                raise ValueError(
                    "contact_integrity.continuous_collision must be an object"
                )
            supported = {
                "maximum_sensor_rotation_rad",
                "maximum_filter_rotation_rad",
                "max_hits_per_pair",
            }
            unknown = set(raw_continuous) - supported
            if unknown:
                raise ValueError(
                    "unsupported continuous collision options: "
                    + ", ".join(sorted(str(name) for name in unknown))
                )
            for name in (
                "maximum_sensor_rotation_rad",
                "maximum_filter_rotation_rad",
            ):
                value = raw_continuous.get(name)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"continuous_collision.{name} must be a number")
                if not math.isfinite(float(value)) or not 0 <= float(value) <= math.pi:
                    raise ValueError(
                        f"continuous_collision.{name} must be finite and between 0 and pi"
                    )
            max_hits = raw_continuous.get("max_hits_per_pair")
            if max_hits is not None and (
                isinstance(max_hits, bool)
                or not isinstance(max_hits, int)
                or not 1 <= max_hits <= 64
            ):
                raise ValueError(
                    "continuous_collision.max_hits_per_pair must be an integer "
                    "between 1 and 64"
                )
            continuous = {
                "maximum_sensor_rotation_rad": float(
                    raw_continuous.get(
                        "maximum_sensor_rotation_rad",
                        _DEFAULT_CONTINUOUS_COLLISION_ROTATION_RADIANS,
                    )
                ),
                "maximum_filter_rotation_rad": float(
                    raw_continuous.get(
                        "maximum_filter_rotation_rad",
                        _DEFAULT_CONTINUOUS_COLLISION_ROTATION_RADIANS,
                    )
                ),
                "max_hits_per_pair": int(raw_continuous.get("max_hits_per_pair", 16)),
            }

        return cls(
            tuple(pairs),
            max_contacts,
            optional_limit("maximum_penetration_m"),
            optional_limit("maximum_normal_impulse_ns"),
            continuous,
        )


class ContactIntegritySampler:
    """Own contact views for the lifetime of one atomic step request."""

    def __init__(self, stage: Any, config: Mapping[str, Any]) -> None:
        self.config = ContactIntegrityConfig.parse(config)
        self._stage = stage
        self._views: list[tuple[ContactPair, Any]] = []
        self._samples: list[Dict[str, Any]] = []
        self._errors: list[str] = []
        self._saturated_pairs: set[str] = set()
        self._violations: list[Dict[str, Any]] = []
        self._continuous_probe: Any | None = None
        self._continuous_incomplete_pairs: set[str] = set()

    def validate_request_size(self, requested_updates: int) -> None:
        """Reject traces whose worst-case response would exceed the hard budget."""

        if isinstance(requested_updates, bool) or not isinstance(
            requested_updates, int
        ):
            raise ValueError("contact integrity update count must be an integer")
        if requested_updates < 1:
            raise ValueError("contact integrity requires at least one update")
        record_slots = (
            requested_updates
            * len(self.config.pairs)
            * self.config.max_contacts_per_pair
        )
        sweep_slots = 0
        if self.config.continuous_collision is not None:
            max_hits = self.config.continuous_collision.get("max_hits_per_pair", 16)
            sweep_slots = requested_updates * len(self.config.pairs) * int(max_hits) * 2
        total_record_slots = record_slots + sweep_slots
        if total_record_slots > _MAX_CONTACT_RECORD_SLOTS:
            raise ValueError(
                "contact integrity request exceeds the bounded response budget: "
                f"{total_record_slots} > {_MAX_CONTACT_RECORD_SLOTS} total slots"
            )

    def prepare(self) -> None:
        """Validate exact paths and enable bounded contact reporting."""

        from isaacsim.core.experimental.prims import RigidPrim
        from pxr import PhysxSchema, UsdPhysics

        if self.config.continuous_collision is not None:
            from .continuous_collision_physx import (
                ContinuousCollisionSettings,
                PhysxContinuousCollisionProbe,
            )

            settings = ContinuousCollisionSettings.parse(
                self.config.continuous_collision
            )
            self._continuous_probe = PhysxContinuousCollisionProbe(
                self._stage,
                settings,
            )

        for pair in self.config.pairs:
            sensor = self._stage.GetPrimAtPath(pair.sensor_path)
            filtered = self._stage.GetPrimAtPath(pair.filter_path)
            for role, path, prim in (
                ("sensor", pair.sensor_path, sensor),
                ("filter", pair.filter_path, filtered),
            ):
                if not prim.IsValid():
                    raise ValueError(f"contact integrity {role} prim not found: {path}")
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(
                        f"contact integrity {role} is not a rigid body: {path}"
                    )
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

            view = RigidPrim(
                pair.sensor_path,
                contact_filter_paths=[pair.filter_path],
                max_contact_count=self.config.max_contacts_per_pair,
            )
            view.set_enabled_contact_tracking([True])
            self._views.append((pair, view))
            if self._continuous_probe is not None:
                self._continuous_probe.prepare_pair(
                    label=pair.label,
                    sensor_path=pair.sensor_path,
                    filter_path=pair.filter_path,
                    sensor_collider_paths=pair.sensor_collider_paths,
                )

    def sample(self, *, update_index: int, physics_dt_seconds: float) -> None:
        """Copy the transient contact manifold after one requested update."""

        if not math.isfinite(physics_dt_seconds) or physics_dt_seconds <= 0:
            raise ValueError("physics_dt_seconds must be finite and positive")

        pair_records: list[Dict[str, Any]] = []
        for pair, view in self._views:
            try:
                record = self._sample_pair(pair, view, physics_dt_seconds)
                self._record_limit_violations(update_index, pair, record)
            except Exception as exc:
                message = f"{pair.label}: {exc}"
                self._errors.append(message)
                record = {
                    "label": pair.label,
                    "sensor_path": pair.sensor_path,
                    "filter_path": pair.filter_path,
                    "complete": False,
                    "error": str(exc),
                    "contacts": [],
                    "friction_contacts": [],
                }
            if self._continuous_probe is not None:
                try:
                    manifold_contact = (
                        bool(record["contacts"])
                        if record.get("complete") is True
                        else None
                    )
                    continuous = self._continuous_probe.sample_pair(
                        label=pair.label,
                        current_manifold_contact=manifold_contact,
                    )
                except Exception as exc:
                    continuous = {
                        "schema_version": 2,
                        "complete": False,
                        "passed": False,
                        "swept_collision_risk_detected": False,
                        "tunneling_detected": False,
                        "failure_reasons": ["continuous_collision_probe_failed"],
                        "errors": [str(exc)],
                    }
                record["continuous_collision"] = continuous
                record["complete"] = bool(record.get("complete")) and bool(
                    continuous.get("complete")
                )
                self._record_continuous_collision(
                    update_index,
                    pair,
                    continuous,
                )
            pair_records.append(record)
        self._samples.append(
            {
                "update_index": update_index,
                "physics_dt_seconds": physics_dt_seconds,
                "pairs": pair_records,
            }
        )

    def _sample_pair(
        self, pair: ContactPair, view: Any, physics_dt_seconds: float
    ) -> Dict[str, Any]:
        contact_data = view.get_contact_force_data(dt=1.0)
        if contact_data is None:
            raise RuntimeError("contact tensor view is unavailable")
        forces, points, normals, distances, counts, starts = (
            _tensor_to_array(value) for value in contact_data
        )
        count = int(counts.reshape(-1)[0])
        start = int(starts.reshape(-1)[0])
        contact_start, captured_count, saturated = self._bounded_tensor_window(
            kind="contact",
            count=count,
            start=start,
            buffer_lengths=(len(forces), len(points), len(normals), len(distances)),
        )
        if saturated:
            self._saturated_pairs.add(pair.label)

        contacts: list[Dict[str, Any]] = []
        total_normal_impulse = 0.0
        maximum_penetration = 0.0
        maximum_normal_impulse = 0.0
        for index in range(contact_start, contact_start + captured_count):
            impulse = _finite_float(forces[index], "normal impulse")
            if impulse < 0:
                raise RuntimeError("contact telemetry emitted negative normal impulse")
            separation = _finite_float(distances[index], "signed separation")
            penetration = max(0.0, -separation)
            total_normal_impulse += impulse
            maximum_penetration = max(maximum_penetration, penetration)
            maximum_normal_impulse = max(maximum_normal_impulse, impulse)
            contacts.append(
                {
                    "point_m": _finite_vector(points[index], "contact point"),
                    "normal_filter_to_sensor": _finite_vector(
                        normals[index], "contact normal"
                    ),
                    "signed_separation_m": separation,
                    "penetration_m": penetration,
                    "normal_impulse_ns": impulse,
                    "normal_force_n": impulse / physics_dt_seconds,
                }
            )

        friction_contacts: list[Dict[str, Any]] = []
        friction_data = view.get_friction_data(dt=1.0)
        if count and friction_data is None:
            raise RuntimeError("friction tensor view is unavailable for active contact")
        if friction_data is not None:
            impulses, friction_points, friction_counts, friction_starts = (
                _tensor_to_array(value) for value in friction_data
            )
            friction_count = int(friction_counts.reshape(-1)[0])
            friction_start = int(friction_starts.reshape(-1)[0])
            (
                friction_start,
                captured_friction_count,
                friction_saturated,
            ) = self._bounded_tensor_window(
                kind="friction",
                count=friction_count,
                start=friction_start,
                buffer_lengths=(len(impulses), len(friction_points)),
            )
            if friction_saturated:
                saturated = True
                self._saturated_pairs.add(pair.label)
            for index in range(
                friction_start,
                friction_start + captured_friction_count,
            ):
                vector = _finite_vector(impulses[index], "friction impulse")
                friction_contacts.append(
                    {
                        "point_m": _finite_vector(
                            friction_points[index], "friction point"
                        ),
                        "impulse_vector_ns": vector,
                        "impulse_magnitude_ns": math.sqrt(
                            sum(component * component for component in vector)
                        ),
                    }
                )

        return {
            "label": pair.label,
            "sensor_path": pair.sensor_path,
            "filter_path": pair.filter_path,
            "complete": not saturated,
            "buffer_saturated": saturated,
            "contact_count": count,
            "contacts": contacts,
            "friction_contacts": friction_contacts,
            "maximum_penetration_m": maximum_penetration,
            "maximum_normal_impulse_ns": maximum_normal_impulse,
            "total_normal_impulse_ns": total_normal_impulse,
        }

    def _bounded_tensor_window(
        self,
        *,
        kind: str,
        count: int,
        start: int,
        buffer_lengths: tuple[int, ...],
    ) -> tuple[int, int, bool]:
        """Return the readable prefix while distinguishing overflow from corruption."""

        if count < 0 or not buffer_lengths or min(buffer_lengths) < 0:
            raise RuntimeError(f"{kind} tensor returned invalid buffer indexes")
        shortest_buffer = min(buffer_lengths)
        saturated = count >= self.config.max_contacts_per_pair
        if start < 0:
            if saturated:
                return 0, 0, True
            raise RuntimeError(f"{kind} tensor returned invalid buffer indexes")
        available = shortest_buffer - start
        if available < 0:
            raise RuntimeError(f"{kind} tensor returned invalid buffer indexes")
        if count <= available:
            return start, count, saturated
        if saturated:
            return start, available, True
        raise RuntimeError(f"{kind} tensor returned invalid buffer indexes")

    def _record_limit_violations(
        self,
        update_index: int,
        pair: ContactPair,
        record: Mapping[str, Any],
    ) -> None:
        for metric, limit in (
            ("maximum_penetration_m", self.config.maximum_penetration_m),
            (
                "maximum_normal_impulse_ns",
                self.config.maximum_normal_impulse_ns,
            ),
        ):
            observed = float(record[metric])
            if limit is not None and observed > limit:
                self._violations.append(
                    {
                        "update_index": update_index,
                        "pair_label": pair.label,
                        "metric": metric,
                        "observed": observed,
                        "limit": limit,
                    }
                )

    def _record_continuous_collision(
        self,
        update_index: int,
        pair: ContactPair,
        evidence: Mapping[str, Any],
    ) -> None:
        if evidence.get("complete") is not True:
            self._continuous_incomplete_pairs.add(pair.label)
            reasons = evidence.get("failure_reasons")
            detail = ", ".join(str(item) for item in reasons or [])
            self._errors.append(
                f"{pair.label}: continuous collision evidence incomplete"
                + (f" ({detail})" if detail else "")
            )
        swept_collision_risk = evidence.get(
            "swept_collision_risk_detected",
            evidence.get("tunneling_detected"),
        )
        if swept_collision_risk is True:
            observed = int(evidence.get("paired_hit_count", 1))
            self._violations.append(
                {
                    "update_index": update_index,
                    "pair_label": pair.label,
                    "metric": "unreported_swept_collision",
                    "observed": observed,
                    "limit": 0,
                }
            )

    def result(
        self, *, requested_updates: int, physics_dt_seconds: float
    ) -> Dict[str, Any]:
        maximum_penetration = 0.0
        maximum_normal_impulse = 0.0
        contact_updates = 0
        unreported_swept_collisions = 0
        maximum_relative_translation = 0.0
        maximum_sensor_rotation = 0.0
        maximum_filter_rotation = 0.0
        maximum_relative_rotation = 0.0
        maximum_rotation_envelope_inflation = 0.0
        for sample in self._samples:
            sample_has_contact = False
            for pair in sample["pairs"]:
                maximum_penetration = max(
                    maximum_penetration,
                    float(pair.get("maximum_penetration_m", 0.0)),
                )
                maximum_normal_impulse = max(
                    maximum_normal_impulse,
                    float(pair.get("maximum_normal_impulse_ns", 0.0)),
                )
                sample_has_contact = sample_has_contact or bool(pair.get("contacts"))
                continuous = pair.get("continuous_collision")
                if isinstance(continuous, Mapping):
                    swept_collision_risk = continuous.get(
                        "swept_collision_risk_detected",
                        continuous.get("tunneling_detected"),
                    )
                    unreported_swept_collisions += int(swept_collision_risk is True)
                    motion = continuous.get("relative_motion")
                    if isinstance(motion, Mapping):
                        maximum_relative_translation = max(
                            maximum_relative_translation,
                            float(motion.get("distance_m", 0.0)),
                        )
                    rotations = continuous.get("rotation_delta_radians")
                    if isinstance(rotations, Mapping):
                        maximum_sensor_rotation = max(
                            maximum_sensor_rotation,
                            float(rotations.get("sensor", 0.0)),
                        )
                        maximum_filter_rotation = max(
                            maximum_filter_rotation,
                            float(rotations.get("filter", 0.0)),
                        )
                        maximum_relative_rotation = max(
                            maximum_relative_rotation,
                            float(rotations.get("relative", 0.0)),
                        )
                    envelope = continuous.get("rotation_envelope")
                    if isinstance(envelope, Mapping):
                        maximum_rotation_envelope_inflation = max(
                            maximum_rotation_envelope_inflation,
                            float(envelope.get("inflation_m", 0.0)),
                        )
            contact_updates += int(sample_has_contact)
        complete = (
            len(self._samples) == requested_updates
            and not self._errors
            and not self._saturated_pairs
            and not self._continuous_incomplete_pairs
        )
        limits = {
            key: value
            for key, value in (
                ("maximum_penetration_m", self.config.maximum_penetration_m),
                (
                    "maximum_normal_impulse_ns",
                    self.config.maximum_normal_impulse_ns,
                ),
            )
            if value is not None
        }
        if self.config.continuous_collision is not None:
            limits.update(
                {
                    "maximum_sensor_rotation_rad_per_update": (
                        self.config.continuous_collision["maximum_sensor_rotation_rad"]
                    ),
                    "maximum_filter_rotation_rad_per_update": (
                        self.config.continuous_collision["maximum_filter_rotation_rad"]
                    ),
                    "unreported_swept_collisions": 0,
                }
            )
        summary = {
            "updates_with_contact": contact_updates,
            "maximum_penetration_m": maximum_penetration,
            "maximum_normal_impulse_ns": maximum_normal_impulse,
        }
        if self.config.continuous_collision is not None:
            summary.update(
                {
                    "unreported_swept_collisions": unreported_swept_collisions,
                    "maximum_relative_translation_m": maximum_relative_translation,
                    "maximum_sensor_rotation_rad": maximum_sensor_rotation,
                    "maximum_filter_rotation_rad": maximum_filter_rotation,
                    "maximum_relative_rotation_rad": maximum_relative_rotation,
                    "maximum_rotation_envelope_inflation_m": (
                        maximum_rotation_envelope_inflation
                    ),
                }
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "capture_source": (
                "isaacsim.core.experimental.prims.RigidPrim"
                if self.config.continuous_collision is None
                else "RigidPrim_contact_tensors_and_PhysX_scene_queries"
            ),
            "sampling_semantics": (
                "after_each_requested_kit_update"
                if self.config.continuous_collision is None
                else "initial_endpoint_overlap_then_pose_contact_rotation_safe_obb_and_exact_shape_sweep_after_each_update"
            ),
            "physics_dt_seconds": physics_dt_seconds,
            "requested_updates": requested_updates,
            "captured_updates": len(self._samples),
            "complete": complete,
            "errors": list(self._errors),
            "saturated_pairs": sorted(self._saturated_pairs),
            "continuous_collision_incomplete_pairs": sorted(
                self._continuous_incomplete_pairs
            ),
            "limits": limits,
            "continuous_collision": self.config.continuous_collision,
            "within_configured_limits": (
                complete and not self._violations if limits else None
            ),
            "violations": list(self._violations),
            "summary": summary,
            "samples": self._samples,
        }

    def close(self) -> None:
        """Release PhysX tensor views even when stepping fails."""

        for _, view in self._views:
            destroy = getattr(view, "destroy", None)
            if callable(destroy):
                destroy()
        self._views.clear()
        if self._continuous_probe is not None:
            self._continuous_probe.close()
            self._continuous_probe = None
