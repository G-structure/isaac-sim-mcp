"""Focused contract tests for atomic Isaac simulation stepping."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

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
    }


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
