"""Tests for the PhysX continuous-collision mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
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


def _transform(x: float) -> dict[str, object]:
    return {
        "ret_val": True,
        "position": [x, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
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
    def __init__(self, *, overlap: list[bool], sweep_hits: list[FakeHit]) -> None:
        self.overlap = overlap
        self.sweep_hits = sweep_hits
        self.sweeps: list[tuple[object, float]] = []

    def overlap_shape(self, _a, _b, report, _both_sides) -> int:
        paired = self.overlap.pop(0)
        if paired:
            report(FakeHit("/World/cube", "/World/cube/collision", 0.0))
        return int(paired)

    def sweep_shape_all(self, _a, _b, direction, distance, report, _both_sides):
        self.sweeps.append((direction, distance))
        for hit in self.sweep_hits:
            if report(hit) is False:
                break
        return bool(self.sweep_hits)


def _probe(
    *,
    sensor_positions: tuple[float, float] = (0.0, 0.3),
    filter_positions: tuple[float, float] = (0.15, 0.15),
    overlap: list[bool] | None = None,
    sweep_hits: list[FakeHit] | None = None,
    max_hits_per_pair: int = 16,
):
    query = FakeSceneQuery(
        overlap=list(overlap or [False, False]),
        sweep_hits=list(sweep_hits or []),
    )
    probe = physx_module.PhysxContinuousCollisionProbe(
        stage=None,
        settings=physx_module.ContinuousCollisionSettings.parse(
            {"max_hits_per_pair": max_hits_per_pair}
        ),
        physx_interface=FakePhysx(
            {
                "/World/finger": [
                    _transform(sensor_positions[0]),
                    _transform(sensor_positions[1]),
                ],
                "/World/cube": [
                    _transform(filter_positions[0]),
                    _transform(filter_positions[1]),
                ],
            }
        ),
        scene_query_interface=query,
        meters_per_unit=1.0,
        collider_resolver=lambda _path: ["/World/finger/collision"],
        collider_validator=lambda _sensor, _colliders: None,
        path_encoder=lambda _path: (1, 2),
        vector_factory=lambda x, y, z: (x, y, z),
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
    assert result["complete"] is True
    assert query.sweeps == [((-1.0, -0.0, -0.0), pytest.approx(0.3))]
    assert result["relative_motion"]["distance_m"] == pytest.approx(0.3)


def test_current_manifold_contact_accounts_for_sweep_hit() -> None:
    probe, _ = _probe(sweep_hits=[FakeHit("/World/cube", "/World/cube/collision", 0.1)])

    result = probe.sample_pair(
        label="finger-cube",
        current_manifold_contact=True,
    )

    assert result["classification"] == "clear"
    assert result["endpoint_evidence"]["current_manifold_contact"] is True


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
    assert query.sweeps[0][1] == pytest.approx(0.2)


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


def test_collider_discovery_uses_only_enabled_direct_collision_gprims(
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
        scene_query_interface=FakeSceneQuery(overlap=[False], sweep_hits=[]),
        meters_per_unit=1.0,
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
        scene_query_interface=FakeSceneQuery(overlap=[], sweep_hits=[]),
        meters_per_unit=1.0,
    )

    with pytest.raises(ValueError, match="continuous collision collider"):
        probe.prepare_pair(
            label="finger-cube",
            sensor_path="/World/finger",
            filter_path="/World/cube",
            sensor_collider_paths=[collider_path],
        )
