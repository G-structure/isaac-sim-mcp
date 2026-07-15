"""Focused contract tests for atomic Isaac simulation stepping."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = REPO_ROOT / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_extension_modules() -> tuple[types.ModuleType, types.ModuleType]:
    for package, path in (
        ("isaac_sim_mcp_extension", EXTENSION_ROOT),
        ("isaac_sim_mcp_extension.adapters", EXTENSION_ROOT / "adapters"),
        ("isaac_sim_mcp_extension.handlers", EXTENSION_ROOT / "handlers"),
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules.setdefault(package, module)

    if "isaac_sim_mcp_extension.adapters.base" not in sys.modules:
        _load_module(
            "isaac_sim_mcp_extension.adapters.base",
            EXTENSION_ROOT / "adapters" / "base.py",
        )

    adapter_module = _load_module(
        "isaac_sim_mcp_extension.adapters.v5_atomic_step_test",
        EXTENSION_ROOT / "adapters" / "v5.py",
    )
    handler_module = _load_module(
        "isaac_sim_mcp_extension.handlers.simulation_atomic_step_test",
        EXTENSION_ROOT / "handlers" / "simulation.py",
    )
    return adapter_module, handler_module


v5, simulation_handler = _load_extension_modules()


class RecordingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, name: str):
        def register(function):
            self.tools[name] = function
            return function

        return register


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def send_command(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((command, params))
        return {"status": "success"}


def _load_simulation_tool(monkeypatch) -> types.ModuleType:
    mcp_module = types.ModuleType("mcp")
    mcp_module.__path__ = []
    server_module = types.ModuleType("mcp.server")
    server_module.__path__ = []
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = RecordingMCP
    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    return _load_module(
        "isaac_mcp_simulation_atomic_step_test",
        REPO_ROOT / "isaac_mcp" / "tools" / "simulation.py",
    )


def _install_fake_omni(
    monkeypatch, *, fail_update: int | None = None
) -> tuple[list[str], Any]:
    events: list[str] = []

    class Timeline:
        def __init__(self) -> None:
            self.state = "playing"
            self.pending_state: str | None = None

        def pause(self) -> None:
            events.append("pause")
            self.pending_state = "paused"

        def play(self) -> None:
            events.append("play")
            self.pending_state = "playing"

        def commit(self) -> None:
            events.append("commit")
            if self.pending_state is not None:
                self.state = self.pending_state
                self.pending_state = None

    class App:
        updates = 0

        def update(self) -> None:
            self.updates += 1
            events.append("update")
            if self.updates == fail_update:
                raise RuntimeError("update failed")

    timeline = Timeline()
    app = App()
    omni_module = types.ModuleType("omni")
    omni_module.__path__ = []
    kit_module = types.ModuleType("omni.kit")
    kit_module.__path__ = []
    app_module = types.ModuleType("omni.kit.app")
    app_module.get_app = lambda: app
    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline
    omni_module.kit = kit_module
    omni_module.timeline = timeline_module
    kit_module.app = app_module

    for name, module in {
        "omni": omni_module,
        "omni.kit": kit_module,
        "omni.kit.app": app_module,
        "omni.timeline": timeline_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    return events, timeline


def test_step_tool_propagates_pause_after(monkeypatch) -> None:
    tool_module = _load_simulation_tool(monkeypatch)
    mcp = RecordingMCP()
    connection = RecordingConnection()
    tool_module.register_tools(mcp, lambda: connection)

    response = json.loads(mcp.tools["step_simulation"](num_steps=3, pause_after=True))

    assert response == {"status": "success"}
    assert connection.calls == [
        ("simulation.step", {"num_steps": 3, "pause_after": True})
    ]


def test_step_handler_propagates_pause_after() -> None:
    class Adapter:
        received: dict[str, Any] | None = None

        def step(self, **params: Any) -> dict[str, Any]:
            self.received = params
            return {"stepped": 2}

    adapter = Adapter()

    response = simulation_handler.step(adapter, num_steps=2, pause_after=True)

    assert response == {
        "status": "success",
        "message": "Stepped 2 frames",
        "stepped": 2,
    }
    assert adapter.received == {
        "num_steps": 2,
        "observe_prims": None,
        "observe_joints": None,
        "budget_ms": None,
        "observe_cap": None,
        "pause_after": True,
        "contact_integrity": None,
    }


def test_step_tool_propagates_contact_integrity(monkeypatch) -> None:
    tool_module = _load_simulation_tool(monkeypatch)
    mcp = RecordingMCP()
    connection = RecordingConnection()
    tool_module.register_tools(mcp, lambda: connection)
    contact_integrity = {
        "pairs": [
            {
                "label": "left-cube",
                "sensor_path": "/World/left_finger",
                "filter_path": "/World/cube",
            }
        ]
    }

    response = json.loads(
        mcp.tools["step_simulation"](
            num_steps=2,
            pause_after=True,
            contact_integrity=contact_integrity,
        )
    )

    assert response == {"status": "success"}
    assert connection.calls == [
        (
            "simulation.step",
            {
                "num_steps": 2,
                "pause_after": True,
                "contact_integrity": contact_integrity,
            },
        )
    ]


def test_step_handler_propagates_contact_integrity() -> None:
    class Adapter:
        received: dict[str, Any] | None = None

        def step(self, **params: Any) -> dict[str, Any]:
            self.received = params
            return {"stepped": 1}

    adapter = Adapter()
    contact_integrity = {"pairs": [{"a": "/World/left_finger", "b": "/World/cube"}]}

    response = simulation_handler.step(
        adapter,
        num_steps=1,
        pause_after=True,
        contact_integrity=contact_integrity,
    )

    assert response["status"] == "success"
    assert adapter.received is not None
    assert adapter.received["contact_integrity"] == contact_integrity


def test_contact_integrity_requires_atomic_step(monkeypatch) -> None:
    _install_fake_omni(monkeypatch)
    adapter = v5.IsaacAdapterV5()

    with pytest.raises(ValueError, match="requires pause_after=true"):
        adapter.step(
            num_steps=1,
            contact_integrity={"pairs": [{"a": "/World/a", "b": "/World/b"}]},
        )


def test_atomic_step_samples_and_closes_contact_trace(monkeypatch) -> None:
    events, _ = _install_fake_omni(monkeypatch)
    samples: list[int] = []

    class FakeSampler:
        def __init__(self, stage: Any, config: dict[str, Any]) -> None:
            assert stage == "stage"
            assert config["pairs"][0]["label"] == "left-cube"

        def prepare(self) -> None:
            events.append("contact_prepare")

        def validate_request_size(self, requested_updates: int) -> None:
            assert requested_updates == 3

        def sample(self, *, update_index: int, physics_dt_seconds: float) -> None:
            assert physics_dt_seconds == pytest.approx(1.0 / 120.0)
            samples.append(update_index)

        def close(self) -> None:
            events.append("contact_close")

        def result(
            self, *, requested_updates: int, physics_dt_seconds: float
        ) -> dict[str, Any]:
            return {
                "complete": samples == list(range(requested_updates)),
                "physics_dt_seconds": physics_dt_seconds,
                "samples": list(samples),
            }

    contact_module = types.ModuleType("isaac_sim_mcp_extension.contact_integrity")
    contact_module.ContactIntegritySampler = FakeSampler
    monkeypatch.setitem(
        sys.modules, "isaac_sim_mcp_extension.contact_integrity", contact_module
    )
    adapter = v5.IsaacAdapterV5()
    adapter._ensure_physics_world = lambda: events.append("ensure")
    adapter.get_stage = lambda: "stage"
    adapter.get_resources = lambda compact=False: {"available_ram_mb": 4096}
    adapter.get_simulation_state = lambda: {"physics_dt": 1.0 / 120.0}

    result = adapter.step(
        num_steps=3,
        pause_after=True,
        contact_integrity={
            "pairs": [
                {
                    "label": "left-cube",
                    "sensor_path": "/World/left_finger",
                    "filter_path": "/World/cube",
                }
            ]
        },
    )

    assert samples == [0, 1, 2]
    assert result["contact_integrity"] == {
        "complete": True,
        "physics_dt_seconds": pytest.approx(1.0 / 120.0),
        "samples": [0, 1, 2],
    }
    assert events.index("contact_prepare") < events.index("play")
    assert events[-1] == "contact_close"


def test_physics_state_uses_exact_runtime_velocity_and_marks_contacts_incomplete(
    monkeypatch,
) -> None:
    class Attr:
        def __init__(self, value: Any) -> None:
            self.value = value

        def Get(self) -> Any:
            return self.value

        def IsValid(self) -> bool:
            return True

    class RigidBodyAPI:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

        def GetKinematicEnabledAttr(self) -> Attr:
            return Attr(False)

    class MassAPI:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

        def GetMassAttr(self) -> Attr:
            return Attr(0.04)

    class CollisionAPI:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

        def GetCollisionEnabledAttr(self) -> Attr:
            return Attr(True)

    class Prim:
        def IsValid(self) -> bool:
            return True

        def HasAPI(self, api: Any) -> bool:
            return api in {RigidBodyAPI, MassAPI, CollisionAPI}

    class Stage:
        def GetPrimAtPath(self, path: str) -> Prim:
            assert path == "/World/cube"
            return Prim()

    class RigidView:
        count = 1
        prim_paths = ["/World/cube"]

        def check(self) -> bool:
            return True

        def get_velocities(self) -> np.ndarray:
            return np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    class PhysicsView:
        def create_rigid_body_view(self, path: str) -> RigidView:
            assert path == "/World/cube"
            return RigidView()

    class SimulationManager:
        @staticmethod
        def get_physics_sim_view() -> PhysicsView:
            return PhysicsView()

    pxr_module = types.ModuleType("pxr")
    usd_physics_module = types.ModuleType("pxr.UsdPhysics")
    usd_physics_module.RigidBodyAPI = RigidBodyAPI
    usd_physics_module.MassAPI = MassAPI
    usd_physics_module.CollisionAPI = CollisionAPI
    pxr_module.UsdPhysics = usd_physics_module
    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_module.__path__ = []
    isaacsim_core_module = types.ModuleType("isaacsim.core")
    isaacsim_core_module.__path__ = []
    simulation_manager_module = types.ModuleType("isaacsim.core.simulation_manager")
    simulation_manager_module.SimulationManager = SimulationManager
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", usd_physics_module)
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core_module)
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.simulation_manager", simulation_manager_module
    )
    adapter = v5.IsaacAdapterV5()
    adapter.get_stage = lambda: Stage()

    result = adapter.get_physics_state("/World/cube")

    assert result.get("velocity_error") is None, result.get("velocity_error")
    assert result["linear_velocity"] == [1.0, 2.0, 3.0]
    assert result["angular_velocity"] == [4.0, 5.0, 6.0]
    assert result["velocity_source"] == "physics_tensor"
    assert result["velocity_complete"] is True
    assert result["contacts"] == []
    assert result["contacts_complete"] is False
    assert result["contacts_source"] == "use_simulation_step_contact_integrity"


def test_atomic_step_runs_exact_updates_and_pauses(monkeypatch) -> None:
    events, timeline = _install_fake_omni(monkeypatch)
    adapter = v5.IsaacAdapterV5()
    adapter._ensure_physics_world = lambda: events.append("ensure")
    adapter.get_resources = lambda compact=False: {"available_ram_mb": 4096}

    result = adapter.step(num_steps=3, budget_ms=1, pause_after=True)

    assert events == [
        "pause",
        "commit",
        "ensure",
        "play",
        "commit",
        "update",
        "update",
        "update",
        "pause",
        "commit",
    ]
    assert timeline.state == "paused"
    assert result == {
        "stepped": 3,
        "resources": {"available_ram_mb": 4096},
        "requested_steps": 3,
        "pause_after": True,
        "exact_step_completed": True,
        "timeline_state": "paused",
    }


def test_atomic_step_pauses_when_update_fails(monkeypatch) -> None:
    events, timeline = _install_fake_omni(monkeypatch, fail_update=2)
    adapter = v5.IsaacAdapterV5()
    adapter._ensure_physics_world = lambda: events.append("ensure")

    with pytest.raises(RuntimeError, match="update failed"):
        adapter.step(num_steps=3, pause_after=True)

    assert events == [
        "pause",
        "commit",
        "ensure",
        "play",
        "commit",
        "update",
        "update",
        "pause",
        "commit",
    ]
    assert timeline.state == "paused"


def test_default_step_does_not_change_timeline(monkeypatch) -> None:
    events, timeline = _install_fake_omni(monkeypatch)
    adapter = v5.IsaacAdapterV5()
    adapter.get_resources = lambda compact=False: {"available_ram_mb": 4096}

    result = adapter.step(num_steps=2)

    assert events == ["update", "update"]
    assert timeline.state == "playing"
    assert result == {
        "stepped": 2,
        "resources": {"available_ram_mb": 4096},
    }
