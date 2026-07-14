"""Focused lifecycle tests for the Isaac Sim 5 adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


EXTENSION_ROOT = (
    Path(__file__).resolve().parents[1]
    / "isaac.sim.mcp_extension"
    / "isaac_sim_mcp_extension"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package, path in (
    ("isaac_sim_mcp_extension", EXTENSION_ROOT),
    ("isaac_sim_mcp_extension.adapters", EXTENSION_ROOT / "adapters"),
):
    module = types.ModuleType(package)
    module.__path__ = [str(path)]
    sys.modules.setdefault(package, module)

if "isaac_sim_mcp_extension.adapters.base" not in sys.modules:
    _load_module(
        "isaac_sim_mcp_extension.adapters.base",
        EXTENSION_ROOT / "adapters" / "base.py",
    )
v5 = _load_module(
    "isaac_sim_mcp_extension.adapters.v5_lifecycle_test",
    EXTENSION_ROOT / "adapters" / "v5.py",
)


class FakeXformOp:
    def __init__(self, name: str, value: object = None) -> None:
        self.name = name
        self.value = value

    def GetOpName(self) -> str:
        return self.name

    def Set(self, value: object) -> None:
        self.value = value


class FakePrim:
    def __init__(self, ops: list[FakeXformOp] | None = None) -> None:
        self.ops = list(ops or [])
        self.reset_xform_stack = True

    @staticmethod
    def IsValid() -> bool:
        return True


class FakeStage:
    def __init__(self) -> None:
        self.prims: dict[str, FakePrim] = {}

    def GetPrimAtPath(self, prim_path: str) -> FakePrim:
        return self.prims[prim_path]


class FakeXformable:
    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def GetOrderedXformOps(self) -> list[FakeXformOp]:
        return list(self.prim.ops)

    def GetResetXformStack(self) -> bool:
        return self.prim.reset_xform_stack

    def SetXformOpOrder(
        self, ops: list[FakeXformOp], reset_xform_stack: bool = False
    ) -> None:
        self.prim.ops = list(ops)
        self.prim.reset_xform_stack = reset_xform_stack

    def _add_op(self, name: str) -> FakeXformOp:
        op = FakeXformOp(name)
        self.prim.ops.append(op)
        return op

    def AddTranslateOp(self, **_kwargs: object) -> FakeXformOp:
        return self._add_op("xformOp:translate")

    def AddRotateXYZOp(self, **_kwargs: object) -> FakeXformOp:
        return self._add_op("xformOp:rotateXYZ")

    def AddOrientOp(self, **_kwargs: object) -> FakeXformOp:
        return self._add_op("xformOp:orient")

    def AddScaleOp(self, **_kwargs: object) -> FakeXformOp:
        return self._add_op("xformOp:scale")


class FakeAttribute:
    def __init__(self) -> None:
        self.value = None

    def Set(self, value: object) -> None:
        self.value = value


class FakeCameraPrim:
    def __init__(self) -> None:
        self.focal_length = FakeAttribute()
        self.focus_distance = FakeAttribute()
        self.horizontal_aperture = FakeAttribute()
        self.vertical_aperture = FakeAttribute()
        self.clipping_range = FakeAttribute()

    def GetFocalLengthAttr(self) -> FakeAttribute:
        return self.focal_length

    def GetFocusDistanceAttr(self) -> FakeAttribute:
        return self.focus_distance

    def GetHorizontalApertureAttr(self) -> FakeAttribute:
        return self.horizontal_aperture

    def GetVerticalApertureAttr(self) -> FakeAttribute:
        return self.vertical_aperture

    def GetClippingRangeAttr(self) -> FakeAttribute:
        return self.clipping_range


def _install_fake_pxr(monkeypatch, stage: FakeStage) -> None:
    class Gf:
        @staticmethod
        def Vec3d(*values: float) -> tuple[float, ...]:
            return tuple(values)

        @staticmethod
        def Vec2f(*values: float) -> tuple[float, ...]:
            return tuple(values)

        @staticmethod
        def Quatd(real: float, imaginary: tuple[float, ...]) -> tuple[float, ...]:
            return (real, *imaginary)

    class XformOp:
        class PrecisionDouble:
            pass

    class Camera:
        @staticmethod
        def Define(_stage: FakeStage, prim_path: str) -> FakeCameraPrim:
            stage.prims.setdefault(prim_path, FakePrim())
            return FakeCameraPrim()

    usd_geom = types.SimpleNamespace(
        Camera=Camera,
        Xformable=FakeXformable,
        XformOp=XformOp,
    )
    pxr = types.ModuleType("pxr")
    pxr.Gf = Gf
    pxr.UsdGeom = usd_geom
    monkeypatch.setitem(sys.modules, "pxr", pxr)


def test_partial_transform_update_preserves_omitted_components(monkeypatch) -> None:
    stage = FakeStage()
    prim_path = "/World/camera"
    translate = FakeXformOp("xformOp:translate", (1.0, 2.0, 3.0))
    rotation = FakeXformOp("xformOp:rotateXYZ", (10.0, 20.0, 30.0))
    scale = FakeXformOp("xformOp:scale", (2.0, 3.0, 4.0))
    stage.prims[prim_path] = FakePrim([translate, rotation, scale])
    _install_fake_pxr(monkeypatch, stage)

    adapter = v5.IsaacAdapterV5()
    adapter.get_stage = lambda: stage
    adapter.set_prim_transform(
        prim_path,
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    ops = stage.prims[prim_path].ops
    assert [op.name for op in ops] == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ]
    assert ops[0] is translate and ops[0].value == (1.0, 2.0, 3.0)
    assert ops[1].value == (1.0, 0.0, 0.0, 0.0)
    assert ops[2] is scale and ops[2].value == (2.0, 3.0, 4.0)
    assert stage.prims[prim_path].reset_xform_stack is True

    before = list(ops)
    adapter.set_prim_transform(prim_path)
    assert stage.prims[prim_path].ops == before


def test_camera_resolution_survives_stop_play_recreation(monkeypatch) -> None:
    stage = FakeStage()
    prim_path = "/World/external_camera"
    stage.prims[prim_path] = FakePrim()
    _install_fake_pxr(monkeypatch, stage)

    class RuntimeCamera:
        instances: list[RuntimeCamera] = []

        def __init__(self, *, prim_path: str, resolution: tuple[int, int]) -> None:
            self.prim_path = prim_path
            self.resolution = tuple(resolution)
            self.initialized = False
            self.destroyed = False
            self.instances.append(self)

        def initialize(self) -> None:
            self.initialized = True

        def destroy(self) -> None:
            self.destroyed = True

        def get_rgba(self) -> list[list[list[int]]]:
            return [[[0, 0, 0, 255]]]

    camera_module = types.ModuleType("isaacsim.sensors.camera")
    camera_module.Camera = RuntimeCamera
    sensors_module = types.ModuleType("isaacsim.sensors")
    sensors_module.__path__ = []
    sensors_module.camera = camera_module
    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_module.__path__ = []
    isaacsim_module.sensors = sensors_module

    timeline_events = []
    timeline = types.SimpleNamespace(
        stop=lambda: timeline_events.append("stop"),
        play=lambda: timeline_events.append("play"),
    )
    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline
    omni_module = types.ModuleType("omni")
    omni_module.__path__ = []
    omni_module.timeline = timeline_module

    for name, module in {
        "isaacsim": isaacsim_module,
        "isaacsim.sensors": sensors_module,
        "isaacsim.sensors.camera": camera_module,
        "omni": omni_module,
        "omni.timeline": timeline_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    adapter = v5.IsaacAdapterV5()
    adapter.get_stage = lambda: stage
    adapter._ensure_physics_world = lambda: None
    created = adapter.create_camera(prim_path, resolution=(640, 360))

    adapter.stop()
    assert created.destroyed is True
    assert prim_path not in adapter._camera_cache
    assert adapter._camera_resolutions[prim_path] == (640, 360)

    adapter.play()
    image = adapter.capture_camera_image(prim_path)
    recreated = RuntimeCamera.instances[-1]
    assert recreated is not created
    assert recreated.initialized is True
    assert recreated.resolution == (640, 360)
    assert image.shape == (1, 1, 4)
    assert timeline_events == ["stop", "play"]
