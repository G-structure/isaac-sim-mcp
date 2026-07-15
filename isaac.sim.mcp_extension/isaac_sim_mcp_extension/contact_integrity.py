"""Bounded, fail-closed PhysX contact evidence for atomic simulation steps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


_SCHEMA_VERSION = 1
_MAX_PAIRS = 16
_MAX_CONTACTS_PER_PAIR = 256
_MAX_CONTACT_RECORD_SLOTS = 8192


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


@dataclass(frozen=True)
class ContactIntegrityConfig:
    """Validated public configuration for one bounded contact trace."""

    pairs: tuple[ContactPair, ...]
    max_contacts_per_pair: int
    maximum_penetration_m: float | None
    maximum_normal_impulse_ns: float | None

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
            labels.add(label)
            pairs.append(ContactPair(label, sensor_path, filter_path))

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

        return cls(
            tuple(pairs),
            max_contacts,
            optional_limit("maximum_penetration_m"),
            optional_limit("maximum_normal_impulse_ns"),
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
        if record_slots > _MAX_CONTACT_RECORD_SLOTS:
            raise ValueError(
                "contact integrity request exceeds the bounded response budget: "
                f"{record_slots} > {_MAX_CONTACT_RECORD_SLOTS} contact slots"
            )

    def prepare(self) -> None:
        """Validate exact paths and enable bounded contact reporting."""

        from isaacsim.core.experimental.prims import RigidPrim
        from pxr import PhysxSchema, UsdPhysics

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

    def sample(self, *, update_index: int, physics_dt_seconds: float) -> None:
        """Copy the transient contact manifold after one requested update."""

        if not math.isfinite(physics_dt_seconds) or physics_dt_seconds <= 0:
            raise ValueError("physics_dt_seconds must be finite and positive")

        pair_records: list[Dict[str, Any]] = []
        for pair, view in self._views:
            try:
                record = self._sample_pair(pair, view, physics_dt_seconds)
                pair_records.append(record)
                self._record_limit_violations(update_index, pair, record)
            except Exception as exc:
                message = f"{pair.label}: {exc}"
                self._errors.append(message)
                pair_records.append(
                    {
                        "label": pair.label,
                        "sensor_path": pair.sensor_path,
                        "filter_path": pair.filter_path,
                        "complete": False,
                        "error": str(exc),
                        "contacts": [],
                        "friction_contacts": [],
                    }
                )
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
        if count < 0 or start < 0 or start + count > len(points):
            raise RuntimeError("contact tensor returned invalid buffer indexes")
        saturated = count >= self.config.max_contacts_per_pair
        if saturated:
            self._saturated_pairs.add(pair.label)

        contacts: list[Dict[str, Any]] = []
        total_normal_impulse = 0.0
        maximum_penetration = 0.0
        maximum_normal_impulse = 0.0
        for index in range(start, start + count):
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
            if (
                friction_count < 0
                or friction_start < 0
                or friction_start + friction_count > len(friction_points)
            ):
                raise RuntimeError("friction tensor returned invalid buffer indexes")
            if friction_count >= self.config.max_contacts_per_pair:
                saturated = True
                self._saturated_pairs.add(pair.label)
            for index in range(friction_start, friction_start + friction_count):
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

    def result(
        self, *, requested_updates: int, physics_dt_seconds: float
    ) -> Dict[str, Any]:
        maximum_penetration = 0.0
        maximum_normal_impulse = 0.0
        contact_updates = 0
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
            contact_updates += int(sample_has_contact)
        complete = (
            len(self._samples) == requested_updates
            and not self._errors
            and not self._saturated_pairs
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
        return {
            "schema_version": _SCHEMA_VERSION,
            "capture_source": "isaacsim.core.experimental.prims.RigidPrim",
            "sampling_semantics": "after_each_requested_kit_update",
            "physics_dt_seconds": physics_dt_seconds,
            "requested_updates": requested_updates,
            "captured_updates": len(self._samples),
            "complete": complete,
            "errors": list(self._errors),
            "saturated_pairs": sorted(self._saturated_pairs),
            "limits": limits,
            "within_configured_limits": (
                complete and not self._violations if limits else None
            ),
            "violations": list(self._violations),
            "summary": {
                "updates_with_contact": contact_updates,
                "maximum_penetration_m": maximum_penetration,
                "maximum_normal_impulse_ns": maximum_normal_impulse,
            },
            "samples": self._samples,
        }

    def close(self) -> None:
        """Release PhysX tensor views even when stepping fails."""

        for _, view in self._views:
            destroy = getattr(view, "destroy", None)
            if callable(destroy):
                destroy()
        self._views.clear()
