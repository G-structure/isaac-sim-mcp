"""Focused tests for fail-closed runtime articulation measurements."""

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

v5 = _load_module(
    "isaac_sim_mcp_extension.adapters.v5_runtime_measurement_test",
    EXTENSION_ROOT / "adapters" / "v5.py",
)
robot_handlers = _load_module(
    "isaac_sim_mcp_extension.handlers.robots_runtime_measurement_test",
    EXTENSION_ROOT / "handlers" / "robots.py",
)


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


class RuntimeArticulation:
    def __init__(
        self,
        names: list[str] | None,
        positions: np.ndarray | None,
        *,
        physics_handle_valid: bool = True,
    ) -> None:
        self.dof_names = names
        self._positions = positions
        self._physics_handle_valid = physics_handle_valid

    def is_physics_handle_valid(self) -> bool:
        return self._physics_handle_valid

    def get_joint_positions(self) -> np.ndarray | None:
        return self._positions


def _load_robot_tools(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
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
        "isaac_mcp_robot_runtime_measurement_test",
        REPO_ROOT / "isaac_mcp" / "tools" / "robots.py",
    )


def test_robot_tools_forward_require_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_module = _load_robot_tools(monkeypatch)
    mcp = RecordingMCP()
    connection = RecordingConnection()
    tool_module.register_tools(mcp, lambda: connection)

    assert json.loads(
        mcp.tools["get_robot_info"]("/World/Robot", require_runtime=True)
    ) == {"status": "success"}
    assert json.loads(
        mcp.tools["get_joint_positions"]("/World/Robot", require_runtime=True)
    ) == {"status": "success"}
    mcp.tools["get_robot_info"]("/World/Legacy")
    mcp.tools["get_joint_positions"]("/World/Legacy")

    assert connection.calls == [
        (
            "robots.get_info",
            {"prim_path": "/World/Robot", "require_runtime": True},
        ),
        (
            "robots.get_joints",
            {"prim_path": "/World/Robot", "require_runtime": True},
        ),
        (
            "robots.get_info",
            {"prim_path": "/World/Legacy", "require_runtime": False},
        ),
        (
            "robots.get_joints",
            {"prim_path": "/World/Legacy", "require_runtime": False},
        ),
    ]


def test_handlers_standardize_runtime_provenance_and_preserve_default_shapes() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, bool]] = []

        def get_robot_joint_info(
            self, prim_path: str, require_runtime: bool = False
        ) -> dict[str, Any]:
            self.calls.append(("info", prim_path, require_runtime))
            return {
                "joint_names": ["joint_a"],
                "num_dof": 1,
                "joint_limits": [],
            }

        def get_joint_positions(
            self, prim_path: str, require_runtime: bool = False
        ) -> list[float]:
            self.calls.append(("positions", prim_path, require_runtime))
            return [0.25]

    adapter = Adapter()

    assert robot_handlers.get_info(adapter, "/World/Robot", require_runtime=True) == {
        "status": "success",
        "joint_names": ["joint_a"],
        "num_dof": 1,
        "joint_limits": [],
        "measurement_source": "runtime_articulation",
    }
    assert robot_handlers.get_joints(adapter, "/World/Robot", require_runtime=True) == {
        "status": "success",
        "joint_positions": [0.25],
        "measurement_source": "runtime_articulation",
    }
    assert robot_handlers.get_info(adapter, "/World/Legacy") == {
        "status": "success",
        "joint_names": ["joint_a"],
        "num_dof": 1,
        "joint_limits": [],
    }
    assert robot_handlers.get_joints(adapter, "/World/Legacy") == {
        "status": "success",
        "joint_positions": [0.25],
    }
    assert adapter.calls == [
        ("info", "/World/Robot", True),
        ("positions", "/World/Robot", True),
        ("info", "/World/Legacy", False),
        ("positions", "/World/Legacy", False),
    ]


def test_v5_required_reads_use_runtime_articulation_without_usd() -> None:
    adapter = v5.IsaacAdapterV5()
    articulation = RuntimeArticulation(["joint_a", "joint_b"], np.array([0.25, -0.5]))
    adapter._get_cached_articulation = lambda _path: articulation
    adapter.get_stage = lambda: pytest.fail("required runtime read traversed USD")

    assert adapter.get_robot_joint_info("/World/Robot", require_runtime=True) == {
        "joint_names": ["joint_a", "joint_b"],
        "num_dof": 2,
        "joint_limits": [{"name": "joint_a"}, {"name": "joint_b"}],
    }
    assert adapter.get_joint_positions("/World/Robot", require_runtime=True) == [
        0.25,
        -0.5,
    ]


@pytest.mark.parametrize("method_name", ["get_robot_joint_info", "get_joint_positions"])
def test_v5_required_reads_fail_closed_without_live_articulation(
    method_name: str,
) -> None:
    adapter = v5.IsaacAdapterV5()

    def unavailable(_path: str) -> Any:
        raise RuntimeError("physics view is not initialized")

    adapter._get_cached_articulation = unavailable
    adapter.get_stage = lambda: pytest.fail("required runtime read traversed USD")

    with pytest.raises(
        RuntimeError,
        match=r"Runtime articulation unavailable.*require_runtime=True forbids USD fallback",
    ):
        getattr(adapter, method_name)("/World/Robot", require_runtime=True)


@pytest.mark.parametrize(
    ("articulation", "message"),
    [
        (
            RuntimeArticulation([], np.array([], dtype=float)),
            "Runtime joint names unavailable",
        ),
        (
            RuntimeArticulation(["joint_a"], None),
            "Runtime joint positions unavailable",
        ),
        (
            RuntimeArticulation(["joint_a", "joint_b"], np.array([0.25])),
            "Runtime articulation data incomplete",
        ),
        (
            RuntimeArticulation(
                ["joint_a"], np.array([0.25]), physics_handle_valid=False
            ),
            "Runtime articulation unavailable",
        ),
    ],
)
def test_v5_required_reads_reject_incomplete_runtime_state(
    articulation: RuntimeArticulation, message: str
) -> None:
    adapter = v5.IsaacAdapterV5()
    adapter._get_cached_articulation = lambda _path: articulation
    adapter.get_stage = lambda: pytest.fail("required runtime read traversed USD")

    with pytest.raises(RuntimeError, match=message):
        adapter.get_joint_positions("/World/Robot", require_runtime=True)


def _install_legacy_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    class Attribute:
        def __init__(self, value: float) -> None:
            self._value = value

        def Get(self) -> float:
            return self._value

    class RevoluteJoint:
        def __init__(self, prim: Any) -> None:
            self._prim = prim

        def GetLowerLimitAttr(self) -> Attribute:
            return Attribute(self._prim.lower)

        def GetUpperLimitAttr(self) -> Attribute:
            return Attribute(self._prim.upper)

    class PrismaticJoint:
        pass

    class Drive:
        def __init__(self, target: float) -> None:
            self._target = target

        def GetTargetPositionAttr(self) -> Attribute:
            return Attribute(self._target)

    class DriveAPI:
        @staticmethod
        def Get(prim: Any, _drive_type: str) -> Drive:
            return Drive(prim.target)

    class JointPrim:
        lower = -90.0
        upper = 90.0
        target = 90.0

        def IsA(self, schema: Any) -> bool:
            return schema is RevoluteJoint

        @staticmethod
        def GetName() -> str:
            return "joint_a"

    class RootPrim:
        descendants = [JointPrim()]

        @staticmethod
        def IsValid() -> bool:
            return True

    root_prim = RootPrim()
    usd = types.SimpleNamespace(PrimRange=lambda prim: list(prim.descendants))
    usd_physics = types.SimpleNamespace(
        RevoluteJoint=RevoluteJoint,
        PrismaticJoint=PrismaticJoint,
        DriveAPI=DriveAPI,
    )
    pxr_module = types.ModuleType("pxr")
    pxr_module.Usd = usd
    pxr_module.UsdPhysics = usd_physics
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    class SingleArticulation:
        def __init__(self, *, prim_path: str) -> None:
            self.prim_path = prim_path

        @staticmethod
        def initialize() -> None:
            raise RuntimeError("simulation is stopped")

    prims_module = types.ModuleType("isaacsim.core.prims")
    prims_module.SingleArticulation = SingleArticulation
    core_module = types.ModuleType("isaacsim.core")
    core_module.__path__ = []
    core_module.prims = prims_module
    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_module.__path__ = []
    isaacsim_module.core = core_module
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims_module)
    return root_prim, usd_physics


def test_v5_default_reads_preserve_usd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_prim, _usd_physics = _install_legacy_modules(monkeypatch)
    stage = types.SimpleNamespace(GetPrimAtPath=lambda _path: root_prim)
    adapter = v5.IsaacAdapterV5()
    adapter.get_stage = lambda: stage

    def unavailable(_path: str) -> Any:
        raise RuntimeError("simulation is stopped")

    adapter._get_cached_articulation = unavailable

    assert adapter.get_robot_joint_info("/World/Robot") == {
        "joint_names": ["joint_a"],
        "num_dof": 1,
        "joint_limits": [
            {
                "name": "joint_a",
                "type": "revolute",
                "lower": -90.0,
                "upper": 90.0,
                "units": "degrees",
            }
        ],
    }
    assert adapter.get_joint_positions("/World/Robot") == pytest.approx([np.pi / 2.0])


def test_v5_simulation_state_reads_typed_scene_at_120_hz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Scene:
        pass

    class Attribute:
        reads = 0

        def Get(self) -> float:
            self.reads += 1
            return 120.0

    class Prim:
        def __init__(self, is_scene: bool) -> None:
            self.is_scene = is_scene
            self.attribute = Attribute()

        def IsA(self, schema: Any) -> bool:
            assert schema is Scene
            return self.is_scene

        @staticmethod
        def HasAPI(_schema: Any) -> bool:
            pytest.fail("typed scene detection used HasAPI")

        def GetAttribute(self, name: str) -> Attribute:
            assert name == "physxScene:timeStepsPerSecond"
            return self.attribute

    physics_prim = Prim(is_scene=True)
    stage = types.SimpleNamespace(Traverse=lambda: [Prim(is_scene=False), physics_prim])

    timeline = types.SimpleNamespace(
        is_playing=lambda: True,
        is_stopped=lambda: False,
        get_current_time=lambda: 1.5,
    )
    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline
    omni_module = types.ModuleType("omni")
    omni_module.__path__ = []
    omni_module.timeline = timeline_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline_module)

    pxr_module = types.ModuleType("pxr")
    pxr_module.UsdPhysics = types.SimpleNamespace(Scene=Scene)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    adapter = v5.IsaacAdapterV5()
    adapter.get_stage = lambda: stage

    assert adapter.get_simulation_state() == {
        "timeline_state": "playing",
        "current_time": 1.5,
        "physics_dt": pytest.approx(1.0 / 120.0),
    }
    assert physics_prim.attribute.reads == 1
