"""Contract tests for fully conditioned hosted camera creation."""

from __future__ import annotations

import math
import importlib.util
import sys
import types
from pathlib import Path


EXTENSION_PARENT = Path(__file__).resolve().parents[1] / "isaac.sim.mcp_extension"
EXTENSION_ROOT = EXTENSION_PARENT / "isaac_sim_mcp_extension"


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
    ("isaac_sim_mcp_extension.handlers", EXTENSION_ROOT / "handlers"),
):
    module = types.ModuleType(package)
    module.__path__ = [str(path)]
    sys.modules.setdefault(package, module)

_load_module(
    "isaac_sim_mcp_extension.adapters.base", EXTENSION_ROOT / "adapters" / "base.py"
)
sensors = _load_module(
    "isaac_sim_mcp_extension.handlers.sensors",
    EXTENSION_ROOT / "handlers" / "sensors.py",
)

create_camera = sensors.create_camera
set_active_camera = sensors.set_active_camera


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_camera(self, prim_path: str, **kwargs: object) -> object:
        self.calls.append((prim_path, kwargs))
        return object()

    def set_active_camera(self, prim_path: str) -> dict[str, object]:
        self.calls.append((prim_path, {"active": True}))
        return {
            "previous_camera": "/OmniverseKit_Persp",
            "active_camera": prim_path,
            "resolution": [1280, 720],
        }


def test_create_camera_forwards_full_conditioning_contract() -> None:
    adapter = FakeAdapter()

    result = create_camera(
        adapter,  # type: ignore[arg-type]
        prim_path="/World/droid/camera",
        position=[0.05, 0.57, 0.66],
        orientation=[2.0, 0.0, 0.0, 0.0],
        resolution=[640, 360],
        focal_length=2.1,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
        clipping_range=[0.05, 100.0],
    )

    assert result == {
        "status": "success",
        "message": "Camera created at /World/droid/camera",
        "prim_path": "/World/droid/camera",
        "resolution": [640, 360],
    }
    assert len(adapter.calls) == 1
    prim_path, arguments = adapter.calls[0]
    assert prim_path == "/World/droid/camera"
    assert arguments["position"] == (0.05, 0.57, 0.66)
    assert arguments["orientation"] == (1.0, 0.0, 0.0, 0.0)
    assert arguments["resolution"] == (640, 360)
    assert arguments["clipping_range"] == (0.05, 100.0)
    assert math.isclose(arguments["focal_length"], 2.1)  # type: ignore[arg-type]


def test_create_camera_rejects_ambiguous_or_invalid_projection() -> None:
    adapter = FakeAdapter()

    ambiguous = create_camera(
        adapter,  # type: ignore[arg-type]
        rotation=[0.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],
    )
    invalid_clip = create_camera(
        adapter,  # type: ignore[arg-type]
        clipping_range=[1.0, 0.1],
    )

    assert ambiguous["status"] == "error"
    assert "mutually exclusive" in ambiguous["message"]
    assert invalid_clip["status"] == "error"
    assert "0 < near < far" in invalid_clip["message"]
    assert adapter.calls == []


def test_set_active_camera_returns_viewport_transition() -> None:
    adapter = FakeAdapter()

    result = set_active_camera(
        adapter,  # type: ignore[arg-type]
        prim_path="/World/droid/external_cam",
    )

    assert result == {
        "status": "success",
        "message": "Active viewport camera set to /World/droid/external_cam",
        "previous_camera": "/OmniverseKit_Persp",
        "active_camera": "/World/droid/external_cam",
        "resolution": [1280, 720],
    }
    assert adapter.calls == [("/World/droid/external_cam", {"active": True})]
