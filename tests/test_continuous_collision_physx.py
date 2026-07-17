"""Tests for the PhysX continuous-collision mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "isaac.sim.mcp_extension"
    / "isaac_sim_mcp_extension"
)
PACKAGE_NAME = "isaac_sim_mcp_extension"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules[PACKAGE_NAME] = package


def _load_module(name: str):
    path = PACKAGE_ROOT / f"{name}.py"
    qualified = f"{PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


_load_module("continuous_collision")
physx_module = _load_module("continuous_collision_physx")


def _transform(
    x: float,
    *,
    y: float = 0.0,
    rotation_xyzw: list[float] | None = None,
) -> dict[str, object]:
    return {
        "ret_val": True,
        "position": [x, y, 0.0],
        "rotation": rotation_xyzw or [0.0, 0.0, 0.0, 1.0],
    }


class FakePhysx:
    def __init__(self, transforms: dict[str, list[dict[str, object]]]) -> None:
        self.transforms = transforms

    def get_rigidbody_transformation(self, path: str) -> dict[str, object]:
        return self.transforms[path].pop(0)


@dataclass
class FakeHit:
    rigid_body: str
    collision: str
    distance: float


class FakeSceneQuery:
    def __init__(
        self,
        *,
        overlap: list[bool],
        shape_sweep_hits: list[FakeHit],
        box_sweep_hits: list[FakeHit] | None = None,
        box_overlap_hits: list[FakeHit] | None = None,
    ) -> None:
        self.overlap = overlap
        self.shape_sweep_hits = shape_sweep_hits
        self.box_sweep_hits = list(box_sweep_hits or [])
        self.box_overlap_hits = list(box_overlap_hits or [])
        self.shape_sweeps: list[tuple[object, float]] = []
        self.box_sweeps: list[tuple[object, object, object, object, float]] = []
        self.box_overlaps: list[tuple[object, object, object]] = []

    def overlap_shape(self, _a, _b, report, _both_sides) -> int:
        paired = self.overlap.pop(0)
        if paired:
            report(FakeHit("/World/cube", "/World/cube/collision", 0.0))
        return int(paired)

    def sweep_shape_all(self, _a, _b, direction, distance, report, _both_sides):
        self.shape_sweeps.append((direction, distance))
        for hit in self.shape_sweep_hits:
            if report(hit) is False:
                break
        return bool(self.shape_sweep_hits)

    def sweep_box_all(
        self,
        half_extents,
        position,
        rotation,
        direction,
        distance,
        report,
        _both_sides,
    ):
        self.box_sweeps.append((half_extents, position, rotation, direction, distance))
        for hit in self.box_sweep_hits:
            if report(hit) is False:
                break
        return bool(self.box_sweep_hits)

    def overlap_box(
        self,
        half_extents,
        position,
        rotation,
        report,
        _any_hit,
    ):
        self.box_overlaps.append((half_extents, position, rotation))
        for hit in self.box_overlap_hits:
            if report(hit) is False:
                break
        return bool(self.box_overlap_hits)


def _probe(
    *,
    sensor_positions: tuple[float, float] = (0.0, 0.3),
    filter_positions: tuple[float, float] = (0.15, 0.15),
    overlap: list[bool] | None = None,
    sweep_hits: list[FakeHit] | None = None,
    box_sweep_hits: list[FakeHit] | None = None,
    box_overlap_hits: list[FakeHit] | None = None,
    sensor_transforms: list[dict[str, object]] | None = None,
    filter_transforms: list[dict[str, object]] | None = None,
    max_hits_per_pair: int = 16,
):
    query = FakeSceneQuery(
        overlap=list(overlap or [False, False]),
        shape_sweep_hits=list(sweep_hits or []),
        box_sweep_hits=list(
            box_sweep_hits if box_sweep_hits is not None else sweep_hits or []
        ),
        box_overlap_hits=box_overlap_hits,
    )
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=None,
        settings=physx_module.ContinuousCollisionSettings.parse(
            {"max_hits_per_pair": max_hits_per_pair}
        ),
        physx_interface=FakePhysx(
            {
                "/World/finger": [
                    *(
                        sensor_transforms
                        or [
                            _transform(sensor_positions[0]),
                            _transform(sensor_positions[1]),
                        ]
                    ),
                ],
                "/World/cube": [
                    *(
                        filter_transforms
                        or [
                            _transform(filter_positions[0]),
                            _transform(filter_positions[1]),
                        ]
                    ),
                ],
            }
        ),
        scene_query_interface=query,
        meters_per_unit=1.0,
        collider_resolver=lambda _path: ["/World/finger/collision"],
        collider_validator=lambda _sensor, _colliders: None,
        envelope_resolver=lambda _sensor, _colliders: (0.05, 0.02, 0.02),
        path_encoder=lambda _path: (1, 2),
        vector_factory=lambda x, y, z: (x, y, z),
        quaternion_factory=lambda x, y, z, w: (x, y, z, w),
    )
    probe.prepare_pair(
        label="finger-cube",
        sensor_path="/World/finger",
        filter_path="/World/cube",
    )
    return probe, query


def test_backward_relative_sweep_detects_endpoint_clean_crossing() -> None:
    probe, query = _probe(
        sweep_hits=[FakeHit("/World/cube", "/World/cube/collision", 0.1)]
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["classification"] == "paired_tunneling"
    assert result["tunneling_detected"] is True
    assert result["tunneling_detected"] is True
    assert result["complete"] is True
    assert query.box_sweeps[0][3:] == (
        (-1.0, -0.0, -0.0),
        pytest.approx(0.3),
    )
    assert query.shape_sweeps == [((-1.0, -0.0, -0.0), pytest.approx(0.3))]
    assert result["relative_motion"]["distance_m"] == pytest.approx(0.3)
    assert result["schema_version"] == 2
    assert result["sweep_semantics"] == (
        "rotation_safe_sensor_body_obb_backward_in_current_filter_frame"
    )
    assert result["translation_shape_sweep"]["captured_hit_count"] == 1


def test_current_manifold_contact_accounts_for_sweep_hit() -> None:
    probe, _ = _probe(sweep_hits=[FakeHit("/World/cube", "/World/cube/collision", 0.1)])

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=True,
    )

    assert result["classification"] == "clear"
    assert result["endpoint_evidence"]["current_manifold_contact"] is True


def test_rotation_envelope_only_hit_is_not_reported_as_tunneling() -> None:
    probe, _ = _probe(
        box_sweep_hits=[FakeHit("/World/cube", "/World/cube/collision", 0.1)],
        sweep_hits=[],
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["classification"] == "conservative_envelope_only"
    assert result["tunneling_detected"] is False
    assert result["broad_phase_only"] is True
    assert result["translation_shape_sweep"]["captured_hit_count"] == 0


def test_filter_motion_is_removed_from_relative_translation() -> None:
    probe, query = _probe(
        sensor_positions=(0.0, 0.3),
        filter_positions=(0.15, 0.25),
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["relative_motion"]["distance_m"] == pytest.approx(0.2)
    assert query.box_sweeps[0][4] == pytest.approx(0.2)


def test_filter_corotation_is_removed_before_scene_query() -> None:
    quarter_turn = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    probe, query = _probe(
        sensor_transforms=[
            _transform(1.0),
            _transform(0.0, y=1.0, rotation_xyzw=quarter_turn),
        ],
        filter_transforms=[
            _transform(0.0),
            _transform(0.0, rotation_xyzw=quarter_turn),
        ],
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["relative_motion"]["distance_m"] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert result["rotation_delta_radians"]["relative"] == pytest.approx(
        0.0,
        abs=1.0e-12,
    )
    assert query.box_sweeps == []
    assert len(query.box_overlaps) == 1


def test_missing_endpoint_query_fails_closed() -> None:
    probe, query = _probe()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("scene query unavailable")

    query.overlap_shape = unavailable
    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["complete"] is False
    assert "endpoint_contact_evidence_unavailable" in result["failure_reasons"]
    assert any("scene query unavailable" in error for error in result["errors"])


def test_unknown_manifold_and_clean_overlap_remain_indeterminate() -> None:
    probe, _ = _probe(overlap=[False, False])

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=None,
    )

    assert result["current_endpoint_contact"] is None
    assert result["endpoint_evidence"]["current_manifold_contact"] is None
    assert result["classification"] == "indeterminate"
    assert result["complete"] is False
    assert "endpoint_contact_evidence_unavailable" in result["failure_reasons"]


def test_sweep_collects_until_the_bounded_buffer_saturates() -> None:
    probe, _ = _probe(
        sweep_hits=[
            FakeHit("/World/cube", "/World/cube/collision-a", 0.05),
            FakeHit("/World/cube", "/World/cube/collision-b", 0.10),
        ],
        max_hits_per_pair=1,
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["complete"] is False
    assert result["sweep"]["saturated"] is True
    assert result["sweep"]["captured_hit_count"] == 1
    assert "sweep_hits_saturated" in result["failure_reasons"]


def test_rotation_only_motion_uses_inflated_overlap_envelope() -> None:
    angle = 0.05
    rotation_xyzw = [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]
    probe, query = _probe(
        sensor_transforms=[
            _transform(0.0),
            _transform(0.0, rotation_xyzw=rotation_xyzw),
        ],
        box_overlap_hits=[FakeHit("/World/cube", "/World/cube/collision", 0.0)],
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    radius = math.sqrt(0.05**2 + 0.02**2 + 0.02**2)
    expected_inflation = 2.0 * radius * math.sin(angle / 2.0)
    assert result["classification"] == "indeterminate"
    assert result["tunneling_detected"] is False
    assert result["broad_phase_only"] is True
    assert result["complete"] is False
    assert "exact_shape_sweep_query_unavailable" in result["failure_reasons"]
    assert result["rotation_envelope"] == {
        "method": "body_centered_symmetric_obb_with_chord_inflation",
        "base_half_extents_m": pytest.approx([0.05, 0.02, 0.02]),
        "radius_m": pytest.approx(radius),
        "relative_rotation_rad": pytest.approx(angle),
        "inflation_m": pytest.approx(expected_inflation),
        "query_half_extents_m": pytest.approx(
            [
                0.05 + expected_inflation,
                0.02 + expected_inflation,
                0.02 + expected_inflation,
            ]
        ),
        "query_kind": "overlap_box",
    }
    assert len(query.box_overlaps) == 1
    assert query.shape_sweeps == []


def test_rotation_only_motion_without_envelope_hit_is_clear() -> None:
    angle = 0.05
    rotation_xyzw = [0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)]
    probe, query = _probe(
        sensor_transforms=[
            _transform(0.0),
            _transform(0.0, rotation_xyzw=rotation_xyzw),
        ],
    )

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=False,
    )

    assert result["classification"] == "clear"
    assert result["complete"] is True
    assert result["rotation_delta_radians"]["relative"] == pytest.approx(angle)
    assert len(query.box_overlaps) == 1


class FakeGprim:
    pass


class FakeRigidBodyAPI:
    pass


class FakeCollisionAPI:
    def __init__(self, prim) -> None:
        self.prim = prim

    def GetCollisionEnabledAttr(self):
        return types.SimpleNamespace(Get=lambda: self.prim.collision_enabled)


class FakePrim:
    def __init__(
        self,
        path: str,
        *,
        gprim: bool = False,
        rigid_body: bool = False,
        collision: bool = False,
        collision_enabled: bool = True,
    ) -> None:
        self.path = path
        self.gprim = gprim
        self.rigid_body = rigid_body
        self.collision = collision
        self.collision_enabled = collision_enabled
        self.parent = None
        self.children: list[FakePrim] = []

    def add(self, child: "FakePrim") -> "FakePrim":
        child.parent = self
        self.children.append(child)
        return child

    def IsValid(self) -> bool:
        return True

    def IsA(self, schema) -> bool:
        return schema is FakeGprim and self.gprim

    def HasAPI(self, schema) -> bool:
        if schema is FakeRigidBodyAPI:
            return self.rigid_body
        if schema is FakeCollisionAPI:
            return self.collision
        return False

    def GetPath(self) -> str:
        return self.path

    def GetChildren(self) -> list["FakePrim"]:
        return list(self.children)

    def GetParent(self):
        return self.parent or FakeInvalidPrim()


class FakeInvalidPrim:
    def IsValid(self) -> bool:
        return False


class FakeStage:
    def __init__(self, prims: list[FakePrim]) -> None:
        self.prims = {prim.path: prim for prim in prims}

    def GetPrimAtPath(self, path: str):
        return self.prims.get(path, FakeInvalidPrim())


def _collision_stage() -> tuple[FakeStage, FakePrim]:
    sensor = FakePrim("/World/finger", rigid_body=True)
    enabled = sensor.add(
        FakePrim(
            "/World/finger/collision",
            gprim=True,
            collision=True,
        )
    )
    visual = sensor.add(FakePrim("/World/finger/visual", gprim=True))
    disabled = sensor.add(
        FakePrim(
            "/World/finger/disabled",
            gprim=True,
            collision=True,
            collision_enabled=False,
        )
    )
    nested = sensor.add(FakePrim("/World/finger/nested", rigid_body=True))
    nested_collision = nested.add(
        FakePrim(
            "/World/finger/nested/collision",
            gprim=True,
            collision=True,
        )
    )
    filtered = FakePrim("/World/cube", rigid_body=True)
    return (
        FakeStage(
            [
                sensor,
                enabled,
                visual,
                disabled,
                nested,
                nested_collision,
                filtered,
            ]
        ),
        sensor,
    )


def _install_fake_pxr(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        types.SimpleNamespace(
            UsdGeom=types.SimpleNamespace(Gprim=FakeGprim),
            UsdPhysics=types.SimpleNamespace(
                CollisionAPI=FakeCollisionAPI,
                RigidBodyAPI=FakeRigidBodyAPI,
            ),
        ),
    )


def test_collider_discovery_uses_only_enabled_direct_collision_prims(
    monkeypatch,
) -> None:
    _install_fake_pxr(monkeypatch)
    stage, _ = _collision_stage()
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=stage,
        settings=physx_module.ContinuousCollisionSettings.parse({}),
        physx_interface=FakePhysx(
            {
                "/World/finger": [_transform(0.0)],
                "/World/cube": [_transform(1.0)],
            }
        ),
        scene_query_interface=FakeSceneQuery(
            overlap=[False],
            shape_sweep_hits=[],
        ),
        meters_per_unit=1.0,
        envelope_resolver=lambda _sensor, _colliders: (0.05, 0.02, 0.02),
        path_encoder=lambda _path: (1, 2),
        vector_factory=lambda x, y, z: (x, y, z),
    )

    probe.prepare_pair(
        label="finger-cube",
        sensor_path="/World/finger",
        filter_path="/World/cube",
    )

    assert probe._pairs["finger-cube"].sensor_collider_paths == (
        "/World/finger/collision",
    )


def test_collider_discovery_accepts_xform_shape_carriers(monkeypatch) -> None:
    _install_fake_pxr(monkeypatch)
    sensor = FakePrim("/World/finger", rigid_body=True)
    carrier = sensor.add(
        FakePrim(
            "/World/finger/collision-carrier",
            collision=True,
        )
    )
    mesh = carrier.add(
        FakePrim(
            "/World/finger/collision-carrier/mesh",
            gprim=True,
        )
    )
    filtered = FakePrim("/World/cube", rigid_body=True)
    stage = FakeStage([sensor, carrier, mesh, filtered])
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=stage,
        settings=physx_module.ContinuousCollisionSettings.parse({}),
        physx_interface=FakePhysx(
            {
                "/World/finger": [_transform(0.0)],
                "/World/cube": [_transform(1.0)],
            }
        ),
        scene_query_interface=FakeSceneQuery(
            overlap=[False],
            shape_sweep_hits=[],
        ),
        meters_per_unit=1.0,
        envelope_resolver=lambda _sensor, _colliders: (0.05, 0.02, 0.02),
        path_encoder=lambda _path: (1, 2),
        vector_factory=lambda x, y, z: (x, y, z),
    )

    probe.prepare_pair(
        label="finger-cube",
        sensor_path="/World/finger",
        filter_path="/World/cube",
    )

    assert probe._pairs["finger-cube"].sensor_collider_paths == (
        "/World/finger/collision-carrier",
    )


@pytest.mark.parametrize(
    "collider_path",
    [
        "/World/finger/visual",
        "/World/finger/disabled",
        "/World/finger/nested/collision",
    ],
)
def test_explicit_collider_overrides_reject_nonphysical_geometry(
    monkeypatch,
    collider_path: str,
) -> None:
    _install_fake_pxr(monkeypatch)
    stage, _ = _collision_stage()
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=stage,
        settings=physx_module.ContinuousCollisionSettings.parse({}),
        physx_interface=FakePhysx({}),
        scene_query_interface=FakeSceneQuery(
            overlap=[],
            shape_sweep_hits=[],
        ),
        meters_per_unit=1.0,
    )

    with pytest.raises(ValueError, match="continuous collision collider"):
        probe.prepare_pair(
            label="finger-cube",
            sensor_path="/World/finger",
            filter_path="/World/cube",
            sensor_collider_paths=[collider_path],
        )


class FakeAlignedRange:
    def __init__(self, lower, upper) -> None:
        self.lower = lower
        self.upper = upper

    def GetMin(self):
        return self.lower

    def GetMax(self):
        return self.upper

    def IsEmpty(self) -> bool:
        return False


class FakeBBox:
    def __init__(self, lower, upper) -> None:
        self.aligned_range = FakeAlignedRange(lower, upper)

    def ComputeAlignedRange(self):
        return self.aligned_range


@pytest.mark.parametrize(
    ("world_rows", "expected_half_extents"),
    [
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.04, 0.03, 0.0075),
        ),
        (
            ((0.0, -2.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 0.5)),
            (0.08, 0.09, 0.00375),
        ),
    ],
)
def test_sensor_envelope_unions_relative_bounds_and_applies_world_scale(
    monkeypatch,
    world_rows,
    expected_half_extents,
) -> None:
    sensor = FakePrim("/World/finger", rigid_body=True)
    first = sensor.add(FakePrim("/World/finger/first", gprim=True, collision=True))
    second = sensor.add(FakePrim("/World/finger/second", gprim=True, collision=True))
    stage = FakeStage([sensor, first, second])
    bounds = {
        first.path: ([-2.0, -1.0, -0.5], [1.0, 2.0, 0.5]),
        second.path: ([-1.0, -3.0, -0.25], [4.0, 1.0, 0.75]),
    }

    class FakeBBoxCache:
        def __init__(self, *_args) -> None:
            pass

        def ComputeRelativeBound(self, collider, relative_to):
            assert relative_to is sensor
            return FakeBBox(*bounds[collider.path])

    class FakeMatrix:
        def GetRow3(self, axis):
            return world_rows[axis]

    class FakeXformable:
        def __init__(self, prim) -> None:
            self.prim = prim

        def GetLocalTransformation(self):
            return FakeMatrix()

        def TransformMightBeTimeVarying(self) -> bool:
            return False

    class FakeXformCache:
        def __init__(self, *_args) -> None:
            pass

        def GetLocalToWorldTransform(self, prim):
            assert prim is sensor
            return FakeMatrix()

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        types.SimpleNamespace(
            Usd=types.SimpleNamespace(
                TimeCode=types.SimpleNamespace(Default=lambda: object())
            ),
            UsdGeom=types.SimpleNamespace(
                BBoxCache=FakeBBoxCache,
                XformCache=FakeXformCache,
                Xformable=FakeXformable,
                Tokens=types.SimpleNamespace(
                    default_="default",
                    render="render",
                    proxy="proxy",
                    guide="guide",
                ),
            ),
        ),
    )
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=stage,
        settings=physx_module.ContinuousCollisionSettings.parse({}),
        physx_interface=FakePhysx({}),
        scene_query_interface=FakeSceneQuery(
            overlap=[],
            shape_sweep_hits=[],
        ),
        meters_per_unit=0.01,
    )

    half_extents = probe._resolve_sensor_half_extents_m(
        sensor.path,
        [first.path, second.path],
    )

    assert half_extents == pytest.approx(expected_half_extents)


def test_sensor_envelope_accepts_positive_orthogonal_scale_only() -> None:
    class Matrix:
        def __init__(self, rows) -> None:
            self.rows = rows

        def GetRow3(self, axis):
            return self.rows[axis]

    rigid = Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    scaled = Matrix(((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    rotated_scaled = Matrix(((0.0, -2.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 0.5)))
    sheared = Matrix(((1.0, 0.25, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    reflected = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    singular = Matrix(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    assert (
        physx_module.PhysxContinuousCollisionProbe._positive_orthogonal_scale_components(
            rigid
        )
        == pytest.approx((1.0, 1.0, 1.0))
    )
    assert (
        physx_module.PhysxContinuousCollisionProbe._positive_orthogonal_scale_components(
            scaled
        )
        == pytest.approx((2.0, 1.0, 1.0))
    )
    assert (
        physx_module.PhysxContinuousCollisionProbe._positive_orthogonal_scale_components(
            rotated_scaled
        )
        == pytest.approx((2.0, 3.0, 0.5))
    )
    for invalid in (sheared, reflected, singular):
        assert (
            physx_module.PhysxContinuousCollisionProbe._positive_orthogonal_scale_components(
                invalid
            )
            is None
        )
