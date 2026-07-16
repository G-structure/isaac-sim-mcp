"""PhysX mechanism for bounded continuous-collision evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from .continuous_collision import (
    RigidBodyPose,
    bounded_sweep_hits,
    classify_update,
    relative_motion,
    relative_rotation_delta_radians,
    rotate_vector_by_quaternion,
)


_DEFAULT_MAX_ROTATION_RADIANS = math.radians(5.0)
_MAX_COLLIDERS_PER_PAIR = 32
_MAX_HITS_PER_PAIR = 64
_MOTION_EPSILON_M = 1.0e-12
_ENVELOPE_METHOD = "body_centered_symmetric_obb_with_chord_inflation"


@dataclass(frozen=True)
class ContinuousCollisionSettings:
    """Validated policy for conservative swept-volume evidence."""

    maximum_sensor_rotation_rad: float
    maximum_filter_rotation_rad: float
    max_hits_per_pair: int

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ContinuousCollisionSettings":
        if not isinstance(value, Mapping):
            raise ValueError("contact_integrity.continuous_collision must be an object")
        supported = {
            "maximum_sensor_rotation_rad",
            "maximum_filter_rotation_rad",
            "max_hits_per_pair",
        }
        unknown = set(value) - supported
        if unknown:
            raise ValueError(
                "unsupported continuous collision options: "
                + ", ".join(sorted(str(name) for name in unknown))
            )

        def rotation(name: str) -> float:
            raw = value.get(name, _DEFAULT_MAX_ROTATION_RADIANS)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"continuous_collision.{name} must be a number")
            result = float(raw)
            if not math.isfinite(result) or not 0 <= result <= math.pi:
                raise ValueError(
                    f"continuous_collision.{name} must be finite and between 0 and pi"
                )
            return result

        max_hits = value.get("max_hits_per_pair", 16)
        if isinstance(max_hits, bool) or not isinstance(max_hits, int):
            raise ValueError(
                "continuous_collision.max_hits_per_pair must be an integer"
            )
        if not 1 <= max_hits <= _MAX_HITS_PER_PAIR:
            raise ValueError(
                "continuous_collision.max_hits_per_pair must be between "
                f"1 and {_MAX_HITS_PER_PAIR}"
            )
        return cls(
            maximum_sensor_rotation_rad=rotation("maximum_sensor_rotation_rad"),
            maximum_filter_rotation_rad=rotation("maximum_filter_rotation_rad"),
            max_hits_per_pair=max_hits,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_sensor_rotation_rad": self.maximum_sensor_rotation_rad,
            "maximum_filter_rotation_rad": self.maximum_filter_rotation_rad,
            "max_hits_per_pair": self.max_hits_per_pair,
        }


@dataclass
class _PairState:
    label: str
    sensor_path: str
    filter_path: str
    sensor_collider_paths: tuple[str, ...]
    sensor_half_extents_m: tuple[float, float, float]
    previous_sensor_pose: RigidBodyPose | None
    previous_filter_pose: RigidBodyPose | None
    previous_endpoint_contact: bool | None
    previous_errors: list[str]


class PhysxContinuousCollisionProbe:
    """Own endpoint, exact-shape, and rotation-safe envelope evidence."""

    def __init__(
        self,
        stage: Any,
        settings: ContinuousCollisionSettings,
        *,
        physx_interface: Any | None = None,
        scene_query_interface: Any | None = None,
        meters_per_unit: float | None = None,
        collider_resolver: Callable[[str], Sequence[str]] | None = None,
        collider_validator: Callable[[str, Sequence[str]], None] | None = None,
        envelope_resolver: Callable[[str, Sequence[str]], Sequence[float]]
        | None = None,
        path_encoder: Callable[[str], tuple[int, int]] | None = None,
        vector_factory: Callable[[float, float, float], Any] | None = None,
        quaternion_factory: Callable[[float, float, float, float], Any] | None = None,
    ) -> None:
        self._stage = stage
        self.settings = settings
        self._physx = physx_interface
        self._scene_query = scene_query_interface
        self._meters_per_unit = meters_per_unit
        self._collider_resolver = collider_resolver
        self._collider_validator = collider_validator
        self._envelope_resolver = envelope_resolver
        self._path_encoder = path_encoder
        self._vector_factory = vector_factory
        self._quaternion_factory = quaternion_factory
        self._pairs: dict[str, _PairState] = {}

    def prepare_pair(
        self,
        *,
        label: str,
        sensor_path: str,
        filter_path: str,
        sensor_collider_paths: Sequence[str] = (),
    ) -> None:
        if label in self._pairs:
            raise ValueError(f"continuous collision pair already prepared: {label}")
        self._ensure_interfaces()
        colliders = tuple(sensor_collider_paths) or tuple(
            self._resolve_sensor_colliders(sensor_path)
        )
        if not colliders:
            raise ValueError(
                f"continuous collision sensor has no collision GPrims: {sensor_path}"
            )
        if len(colliders) > _MAX_COLLIDERS_PER_PAIR:
            raise ValueError(
                "continuous collision sensor exceeds collider budget: "
                f"{len(colliders)} > {_MAX_COLLIDERS_PER_PAIR}"
            )
        if self._collider_validator is not None:
            self._collider_validator(sensor_path, colliders)
        else:
            self._validate_sensor_colliders(sensor_path, colliders)
        half_extents = self._sensor_half_extents_m(sensor_path, colliders)

        errors: list[str] = []
        previous_sensor = self._try_pose(sensor_path, errors)
        previous_filter = self._try_pose(filter_path, errors)
        previous_overlap = self._try_overlap(colliders, filter_path, errors)
        self._pairs[label] = _PairState(
            label=label,
            sensor_path=sensor_path,
            filter_path=filter_path,
            sensor_collider_paths=colliders,
            sensor_half_extents_m=half_extents,
            previous_sensor_pose=previous_sensor,
            previous_filter_pose=previous_filter,
            previous_endpoint_contact=previous_overlap,
            previous_errors=errors,
        )

    def sample_pair(
        self,
        *,
        label: str,
        current_manifold_contact: bool | None,
    ) -> dict[str, Any]:
        state = self._pairs.get(label)
        if state is None:
            raise RuntimeError(f"continuous collision pair was not prepared: {label}")
        if current_manifold_contact is not None and not isinstance(
            current_manifold_contact,
            bool,
        ):
            raise ValueError("current_manifold_contact must be a boolean or null")

        errors = list(state.previous_errors)
        current_sensor = self._try_pose(state.sensor_path, errors)
        current_filter = self._try_pose(state.filter_path, errors)
        current_overlap = self._try_overlap(
            state.sensor_collider_paths,
            state.filter_path,
            errors,
        )
        current_endpoint = (
            True
            if current_manifold_contact or current_overlap is True
            else False
            if current_manifold_contact is False and current_overlap is False
            else None
        )

        envelope_hits: list[dict[str, Any]] | None = None
        envelope_available = False
        envelope_saturated = False
        envelope_metadata: dict[str, Any] | None = None
        shape_hits: list[dict[str, Any]] | None = None
        shape_available = False
        shape_saturated = False
        diagnostic_errors: list[str] = []
        if all(
            pose is not None
            for pose in (
                state.previous_sensor_pose,
                current_sensor,
                state.previous_filter_pose,
                current_filter,
            )
        ):
            assert state.previous_sensor_pose is not None
            assert current_sensor is not None
            assert state.previous_filter_pose is not None
            assert current_filter is not None
            motion = relative_motion(
                state.previous_sensor_pose,
                current_sensor,
                state.previous_filter_pose,
                current_filter,
            )
            relative_rotation = relative_rotation_delta_radians(
                state.previous_sensor_pose,
                current_sensor,
                state.previous_filter_pose,
                current_filter,
            )
            world_direction = rotate_vector_by_quaternion(
                current_filter.orientation_wxyz,
                motion.direction_unit,
                field="relative_motion_world_direction",
            )
            try:
                envelope_hits, envelope_saturated, envelope_metadata = (
                    self._query_rotation_envelope(
                        state,
                        current_sensor=current_sensor,
                        direction_unit=tuple(-value for value in world_direction),
                        distance_m=motion.distance_m,
                        relative_rotation_rad=relative_rotation,
                    )
                )
                envelope_available = True
            except Exception as exc:
                errors.append(f"rotation-safe envelope query failed: {exc}")
            try:
                shape_hits, shape_saturated = self._sweep_exact_shapes(
                    state,
                    direction_unit=tuple(-value for value in world_direction),
                    distance_m=motion.distance_m,
                )
                shape_available = True
            except Exception as exc:
                diagnostic_errors.append(f"exact shape sweep failed: {exc}")

        result = classify_update(
            sensor_path=state.sensor_path,
            filter_path=state.filter_path,
            previous_sensor=state.previous_sensor_pose,
            current_sensor=current_sensor,
            previous_filter=state.previous_filter_pose,
            current_filter=current_filter,
            previous_endpoint_contact=state.previous_endpoint_contact,
            current_endpoint_contact=current_endpoint,
            sweep_hits=envelope_hits,
            sweep_query_available=envelope_available,
            sweep_saturated=envelope_saturated,
            max_sweep_hits=self.settings.max_hits_per_pair,
            maximum_sensor_rotation_radians=(self.settings.maximum_sensor_rotation_rad),
            maximum_filter_rotation_radians=(self.settings.maximum_filter_rotation_rad),
        )
        result["errors"].extend(errors)
        result["sensor_collider_paths"] = list(state.sensor_collider_paths)
        result["endpoint_evidence"] = {
            "previous_contact_or_overlap": state.previous_endpoint_contact,
            "current_overlap": current_overlap,
            "current_manifold_contact": current_manifold_contact,
            "current_contact_or_overlap": current_endpoint,
        }
        result["sweep_semantics"] = (
            "rotation_safe_sensor_body_obb_backward_in_current_filter_frame"
        )
        result["rotation_envelope"] = envelope_metadata
        result["translation_shape_sweep"] = bounded_sweep_hits(
            shape_hits,
            max_hits=self.settings.max_hits_per_pair,
            query_available=shape_available,
            saturated=shape_saturated,
        )
        result["translation_shape_sweep_semantics"] = (
            "current_sensor_collision_shapes_backward_through_relative_translation"
        )
        result["diagnostic_errors"] = diagnostic_errors

        state.previous_sensor_pose = current_sensor
        state.previous_filter_pose = current_filter
        state.previous_endpoint_contact = current_endpoint
        state.previous_errors = []
        return result

    def close(self) -> None:
        self._pairs.clear()

    def _ensure_interfaces(self) -> None:
        if self._physx is None or self._scene_query is None:
            import omni.physx

            self._physx = self._physx or omni.physx.get_physx_interface()
            self._scene_query = (
                self._scene_query or omni.physx.get_physx_scene_query_interface()
            )
        if self._meters_per_unit is None:
            from pxr import UsdGeom

            self._meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(self._stage))
        if not math.isfinite(self._meters_per_unit) or self._meters_per_unit <= 0:
            raise RuntimeError("stage meters-per-unit must be finite and positive")

    def _try_pose(self, path: str, errors: list[str]) -> RigidBodyPose | None:
        try:
            return self._read_pose(path)
        except Exception as exc:
            errors.append(f"{path} pose unavailable: {exc}")
            return None

    def _read_pose(self, path: str) -> RigidBodyPose:
        payload = self._physx.get_rigidbody_transformation(path)
        if not isinstance(payload, Mapping) or payload.get("ret_val") is not True:
            raise RuntimeError("PhysX did not return a rigid-body transform")
        position = self._components(payload.get("position"), 3, "position")
        rotation_xyzw = self._components(payload.get("rotation"), 4, "rotation")
        scale = float(self._meters_per_unit)
        return RigidBodyPose.parse(
            [component * scale for component in position],
            [
                rotation_xyzw[3],
                rotation_xyzw[0],
                rotation_xyzw[1],
                rotation_xyzw[2],
            ],
            field=path,
        )

    @staticmethod
    def _components(value: Any, count: int, field: str) -> list[float]:
        try:
            result = [float(value[index]) for index in range(count)]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid PhysX {field}") from exc
        if not all(math.isfinite(item) for item in result):
            raise RuntimeError(f"non-finite PhysX {field}")
        return result

    def _try_overlap(
        self,
        collider_paths: Sequence[str],
        filter_path: str,
        errors: list[str],
    ) -> bool | None:
        try:
            return self._pair_overlaps(collider_paths, filter_path)
        except Exception as exc:
            errors.append(f"endpoint overlap query failed: {exc}")
            return None

    def _pair_overlaps(
        self,
        collider_paths: Sequence[str],
        filter_path: str,
    ) -> bool:
        for collider_path in collider_paths:
            encoded = self._encode_path(collider_path)
            paired_hit = False

            def report(hit: Any) -> bool:
                nonlocal paired_hit
                if str(getattr(hit, "rigid_body", "")) == filter_path:
                    paired_hit = True
                    return False
                return True

            self._scene_query.overlap_shape(
                encoded[0],
                encoded[1],
                report,
                False,
            )
            if paired_hit:
                return True
        return False

    def _sweep_exact_shapes(
        self,
        state: _PairState,
        *,
        direction_unit: tuple[float, float, float],
        distance_m: float,
    ) -> tuple[list[dict[str, Any]], bool]:
        if distance_m <= _MOTION_EPSILON_M:
            return [], False
        direction = self._vec3(*direction_unit)
        distance_stage_units = distance_m / float(self._meters_per_unit)
        hits: list[dict[str, Any]] = []
        saturated = False
        for collider_path in state.sensor_collider_paths:
            encoded = self._encode_path(collider_path)

            def report(hit: Any) -> bool:
                nonlocal saturated
                if str(getattr(hit, "rigid_body", "")) != state.filter_path:
                    return True
                if len(hits) >= self.settings.max_hits_per_pair:
                    saturated = True
                    return False
                distance = float(getattr(hit, "distance"))
                if not math.isfinite(distance) or distance < 0:
                    raise RuntimeError("sweep returned an invalid hit distance")
                collision = str(getattr(hit, "collision", "")) or None
                hits.append(
                    {
                        "rigid_body_path": state.filter_path,
                        "collider_path": collision,
                        "distance_m": distance * float(self._meters_per_unit),
                    }
                )
                return True

            self._scene_query.sweep_shape_all(
                encoded[0],
                encoded[1],
                direction,
                distance_stage_units,
                report,
                False,
            )
            if saturated:
                break
        return hits, saturated

    def _query_rotation_envelope(
        self,
        state: _PairState,
        *,
        current_sensor: RigidBodyPose,
        direction_unit: tuple[float, float, float],
        distance_m: float,
        relative_rotation_rad: float,
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        radius_m = math.sqrt(
            sum(value * value for value in state.sensor_half_extents_m)
        )
        inflation_m = 2.0 * radius_m * math.sin(relative_rotation_rad / 2.0)
        query_half_extents_m = tuple(
            value + inflation_m for value in state.sensor_half_extents_m
        )
        scale = float(self._meters_per_unit)
        half_extents = self._vec3(*(value / scale for value in query_half_extents_m))
        position = self._vec3(*(value / scale for value in current_sensor.position_m))
        w, x, y, z = current_sensor.orientation_wxyz
        rotation = self._quat_xyzw(x, y, z, w)
        hits: list[dict[str, Any]] = []
        saturated = False

        def report(hit: Any) -> bool:
            nonlocal saturated
            if str(getattr(hit, "rigid_body", "")) != state.filter_path:
                return True
            if len(hits) >= self.settings.max_hits_per_pair:
                saturated = True
                return False
            raw_distance = getattr(hit, "distance", 0.0)
            distance = float(raw_distance)
            if not math.isfinite(distance) or distance < 0:
                raise RuntimeError("rotation envelope returned an invalid hit distance")
            collision = str(getattr(hit, "collision", "")) or None
            hits.append(
                {
                    "rigid_body_path": state.filter_path,
                    "collider_path": collision,
                    "distance_m": distance * scale,
                }
            )
            return True

        if distance_m > _MOTION_EPSILON_M:
            query_kind = "sweep_box_all"
            self._scene_query.sweep_box_all(
                half_extents,
                position,
                rotation,
                self._vec3(*direction_unit),
                distance_m / scale,
                report,
                False,
            )
        else:
            query_kind = "overlap_box"
            self._scene_query.overlap_box(
                half_extents,
                position,
                rotation,
                report,
                False,
            )

        return (
            hits,
            saturated,
            {
                "method": _ENVELOPE_METHOD,
                "base_half_extents_m": list(state.sensor_half_extents_m),
                "radius_m": radius_m,
                "relative_rotation_rad": relative_rotation_rad,
                "inflation_m": inflation_m,
                "query_half_extents_m": list(query_half_extents_m),
                "query_kind": query_kind,
            },
        )

    def _sensor_half_extents_m(
        self,
        sensor_path: str,
        collider_paths: Sequence[str],
    ) -> tuple[float, float, float]:
        if self._envelope_resolver is not None:
            raw_extents = self._envelope_resolver(sensor_path, collider_paths)
        else:
            raw_extents = self._resolve_sensor_half_extents_m(
                sensor_path,
                collider_paths,
            )
        try:
            half_extents = tuple(float(raw_extents[index]) for index in range(3))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "continuous collision sensor envelope must contain three extents"
            ) from exc
        if not all(math.isfinite(value) and value > 0 for value in half_extents):
            raise ValueError(
                "continuous collision sensor envelope extents must be finite and positive"
            )
        return (half_extents[0], half_extents[1], half_extents[2])

    def _resolve_sensor_half_extents_m(
        self,
        sensor_path: str,
        collider_paths: Sequence[str],
    ) -> tuple[float, float, float]:
        from pxr import Usd, UsdGeom

        sensor = self._stage.GetPrimAtPath(sensor_path)
        sensor_world_transform = UsdGeom.XformCache(
            Usd.TimeCode.Default()
        ).GetLocalToWorldTransform(sensor)
        if not self._has_rigid_linear_transform(sensor_world_transform):
            raise ValueError(
                "continuous collision sensor world transform contains "
                f"scale, shear, or reflection: {sensor_path}"
            )
        purposes = [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ]
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            purposes,
            False,
            True,
        )
        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        for collider_path in collider_paths:
            collider = self._stage.GetPrimAtPath(collider_path)
            current = collider
            while current.IsValid() and current != sensor:
                xformable = UsdGeom.Xformable(current)
                if xformable and xformable.TransformMightBeTimeVarying():
                    raise ValueError(
                        "continuous collision sensor envelope contains a "
                        f"time-varying descendant transform: {current.GetPath()}"
                    )
                current = current.GetParent()
            if current != sensor:
                raise ValueError(
                    "continuous collision collider is not beneath the sensor: "
                    f"{collider_path}"
                )
            aligned_range = cache.ComputeRelativeBound(
                collider,
                sensor,
            ).ComputeAlignedRange()
            if aligned_range.IsEmpty():
                raise ValueError(
                    f"continuous collision collider bound is empty: {collider_path}"
                )
            lower = aligned_range.GetMin()
            upper = aligned_range.GetMax()
            for axis in range(3):
                lower_value = float(lower[axis])
                upper_value = float(upper[axis])
                if not math.isfinite(lower_value) or not math.isfinite(upper_value):
                    raise ValueError(
                        "continuous collision collider bound is non-finite: "
                        f"{collider_path}"
                    )
                minimum[axis] = min(minimum[axis], lower_value)
                maximum[axis] = max(maximum[axis], upper_value)
        scale = float(self._meters_per_unit)
        return tuple(
            max(abs(minimum[axis]), abs(maximum[axis])) * scale for axis in range(3)
        )

    @staticmethod
    def _has_rigid_linear_transform(matrix: Any) -> bool:
        rows = [
            tuple(float(value) for value in matrix.GetRow3(axis)) for axis in range(3)
        ]
        if not all(math.isfinite(value) for row in rows for value in row):
            return False
        tolerance = 1.0e-6
        for row in rows:
            norm = math.sqrt(sum(value * value for value in row))
            if abs(norm - 1.0) > tolerance:
                return False
        for left in range(3):
            for right in range(left + 1, 3):
                dot = sum(rows[left][axis] * rows[right][axis] for axis in range(3))
                if abs(dot) > tolerance:
                    return False
        determinant = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        return abs(determinant - 1.0) <= tolerance

    def _resolve_sensor_colliders(self, sensor_path: str) -> Sequence[str]:
        if self._collider_resolver is not None:
            return self._collider_resolver(sensor_path)

        from pxr import UsdPhysics

        root = self._stage.GetPrimAtPath(sensor_path)
        if not root.IsValid():
            raise ValueError(
                f"continuous collision sensor prim not found: {sensor_path}"
            )
        colliders: list[str] = []

        def visit(prim: Any) -> None:
            if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return
            # Imported robot assets may put CollisionAPI on an Xform shape
            # carrier while a descendant Mesh supplies the geometry.
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_enabled = (
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                )
                if collision_enabled is True:
                    colliders.append(str(prim.GetPath()))
            for child in prim.GetChildren():
                visit(child)

        visit(root)
        return colliders

    def _validate_sensor_colliders(
        self,
        sensor_path: str,
        collider_paths: Sequence[str],
    ) -> None:
        from pxr import UsdPhysics

        seen: set[str] = set()
        descendant_prefix = sensor_path.rstrip("/") + "/"
        for collider_path in collider_paths:
            if collider_path in seen:
                raise ValueError(
                    f"continuous collision collider is duplicated: {collider_path}"
                )
            seen.add(collider_path)
            if collider_path != sensor_path and not collider_path.startswith(
                descendant_prefix
            ):
                raise ValueError(
                    "continuous collision collider must be beneath the sensor "
                    f"rigid body: {collider_path}"
                )
            prim = self._stage.GetPrimAtPath(collider_path)
            if not prim.IsValid():
                raise ValueError(
                    f"continuous collision collider prim not found: {collider_path}"
                )
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(
                    "continuous collision collider must have a direct "
                    f"CollisionAPI: {collider_path}"
                )
            collision_enabled = (
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            )
            if collision_enabled is not True:
                raise ValueError(
                    f"continuous collision collider is disabled: {collider_path}"
                )

            current = prim
            closest_rigid_body: str | None = None
            while current.IsValid():
                if current.HasAPI(UsdPhysics.RigidBodyAPI):
                    closest_rigid_body = str(current.GetPath())
                    break
                parent = current.GetParent()
                if not parent.IsValid() or parent == current:
                    break
                current = parent
            if closest_rigid_body != sensor_path:
                raise ValueError(
                    "continuous collision collider's closest rigid body must be "
                    f"the declared sensor: {collider_path}"
                )

    def _encode_path(self, path: str) -> tuple[int, int]:
        if self._path_encoder is not None:
            return self._path_encoder(path)
        from pxr import PhysicsSchemaTools, Sdf

        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(path))
        return int(encoded[0]), int(encoded[1])

    def _vec3(self, x: float, y: float, z: float) -> Any:
        if self._vector_factory is not None:
            return self._vector_factory(x, y, z)
        import carb

        return carb.Float3(x, y, z)

    def _quat_xyzw(self, x: float, y: float, z: float, w: float) -> Any:
        if self._quaternion_factory is not None:
            return self._quaternion_factory(x, y, z, w)
        import carb

        return carb.Float4(x, y, z, w)
