"""Contract tests for runtime-required robot actuation."""

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
    "isaac_sim_mcp_extension.adapters.v5_runtime_actuation_test",
    EXTENSION_ROOT / "adapters" / "v5.py",
)
robot_handlers = _load_module(
    "isaac_sim_mcp_extension.handlers.robots_runtime_actuation_test",
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
        return {"status": "success", "control_source": "runtime_articulation"}


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
        "isaac_mcp_robot_runtime_actuation_test",
        REPO_ROOT / "isaac_mcp" / "tools" / "robots.py",
    )


def _install_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialize_error: Exception | None = None,
    apply_error: Exception | None = None,
) -> tuple[list[Any], list[Any]]:
    articulations: list[Any] = []
    actions: list[Any] = []

    class Controller:
        def apply_action(self, action: Any) -> None:
            if apply_error is not None:
                raise apply_error
            actions.append(action)

    class SingleArticulation:
        def __init__(self, *, prim_path: str) -> None:
            self.prim_path = prim_path
            self.controller = Controller()
            articulations.append(self)

        def initialize(self) -> None:
            if initialize_error is not None:
                raise initialize_error

        def get_articulation_controller(self) -> Controller:
            return self.controller

    class ArticulationAction:
        def __init__(self, *, joint_positions: Any, joint_indices: Any) -> None:
            self.joint_positions = joint_positions
            self.joint_indices = joint_indices

    prims_module = types.ModuleType("isaacsim.core.prims")
    prims_module.SingleArticulation = SingleArticulation
    types_module = types.ModuleType("isaacsim.core.utils.types")
    types_module.ArticulationAction = ArticulationAction
    utils_module = types.ModuleType("isaacsim.core.utils")
    utils_module.__path__ = []
    utils_module.types = types_module
    core_module = types.ModuleType("isaacsim.core")
    core_module.__path__ = []
    core_module.prims = prims_module
    core_module.utils = utils_module
    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_module.__path__ = []
    isaacsim_module.core = core_module
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.types", types_module)
    return articulations, actions


def test_robot_tool_forwards_runtime_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_module = _load_robot_tools(monkeypatch)
    mcp = RecordingMCP()
    connection = RecordingConnection()
    tool_module.register_tools(mcp, lambda: connection)

    assert json.loads(
        mcp.tools["set_joint_positions"](
            "/World/Robot",
            [0.1, 0.2],
            joint_indices=[1, 2],
            require_runtime=True,
        )
    ) == {"status": "success", "control_source": "runtime_articulation"}
    mcp.tools["set_joint_positions"]("/World/Legacy", [0.3])

    assert connection.calls == [
        (
            "robots.set_joints",
            {
                "prim_path": "/World/Robot",
                "joint_positions": [0.1, 0.2],
                "require_runtime": True,
                "joint_indices": [1, 2],
            },
        ),
        (
            "robots.set_joints",
            {
                "prim_path": "/World/Legacy",
                "joint_positions": [0.3],
                "require_runtime": False,
            },
        ),
    ]


def test_handler_returns_adapter_control_source() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[float], list[int] | None, bool]] = []

        def set_joint_positions(
            self,
            prim_path: str,
            positions: list[float],
            joint_indices: list[int] | None = None,
            require_runtime: bool = False,
        ) -> str:
            self.calls.append((prim_path, positions, joint_indices, require_runtime))
            if require_runtime:
                return "runtime_articulation"
            return "usd_drive_target"

    adapter = Adapter()

    assert robot_handlers.set_joints(
        adapter,
        "/World/Robot",
        [0.1],
        [2],
        require_runtime=True,
    ) == {
        "status": "success",
        "message": "Set joint positions on /World/Robot",
        "control_source": "runtime_articulation",
    }
    assert robot_handlers.set_joints(adapter, "/World/Legacy", [0.2]) == {
        "status": "success",
        "message": "Set joint positions on /World/Legacy",
        "control_source": "usd_drive_target",
    }
    assert adapter.calls == [
        ("/World/Robot", [0.1], [2], True),
        ("/World/Legacy", [0.2], None, False),
    ]


@pytest.mark.parametrize("require_runtime", [False, True])
def test_v5_reports_runtime_articulation_control(
    monkeypatch: pytest.MonkeyPatch,
    require_runtime: bool,
) -> None:
    articulations, actions = _install_runtime_modules(monkeypatch)
    adapter = v5.IsaacAdapterV5()
    adapter._set_joint_drive_targets = lambda *_args: pytest.fail(
        "runtime actuation used USD fallback"
    )

    assert (
        adapter.set_joint_positions(
            "/World/Robot",
            [0.1, 0.2],
            [1, 2],
            require_runtime=require_runtime,
        )
        == "runtime_articulation"
    )
    assert [articulation.prim_path for articulation in articulations] == [
        "/World/Robot"
    ]
    assert len(actions) == 1
    assert actions[0].joint_positions.tolist() == pytest.approx([0.1, 0.2])
    assert actions[0].joint_indices.tolist() == [1, 2]


def test_v5_default_actuation_preserves_usd_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime_modules(
        monkeypatch,
        initialize_error=RuntimeError("physics view is not initialized"),
    )
    adapter = v5.IsaacAdapterV5()
    fallback_calls: list[tuple[str, list[float], list[int] | None]] = []
    adapter._set_joint_drive_targets = lambda prim_path, positions, joint_indices: (
        fallback_calls.append((prim_path, list(positions), joint_indices))
    )

    assert adapter.set_joint_positions("/World/Robot", [0.1], [3]) == (
        "usd_drive_target"
    )
    assert fallback_calls == [("/World/Robot", [0.1], [3])]


@pytest.mark.parametrize(
    ("initialize_error", "apply_error", "failure_message"),
    [
        (RuntimeError("physics view is not initialized"), None, "physics view"),
        (None, RuntimeError("controller rejected action"), "controller rejected"),
    ],
)
def test_v5_required_actuation_fails_closed_without_usd(
    monkeypatch: pytest.MonkeyPatch,
    initialize_error: Exception | None,
    apply_error: Exception | None,
    failure_message: str,
) -> None:
    _install_runtime_modules(
        monkeypatch,
        initialize_error=initialize_error,
        apply_error=apply_error,
    )
    adapter = v5.IsaacAdapterV5()
    adapter._set_joint_drive_targets = lambda *_args: pytest.fail(
        "required runtime actuation traversed USD"
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Runtime articulation control unavailable for /World/Robot; "
            r"require_runtime=True forbids USD fallback: "
            rf"{failure_message}"
        ),
    ):
        adapter.set_joint_positions(
            "/World/Robot",
            [0.1],
            require_runtime=True,
        )


def test_handler_preserves_required_runtime_failure() -> None:
    class Adapter:
        @staticmethod
        def set_joint_positions(
            _prim_path: str,
            _positions: list[float],
            _joint_indices: list[int] | None = None,
            require_runtime: bool = False,
        ) -> str:
            assert require_runtime is True
            raise RuntimeError(
                "Runtime articulation control unavailable; "
                "require_runtime=True forbids USD fallback"
            )

    assert robot_handlers.set_joints(
        Adapter(),
        "/World/Robot",
        [0.1],
        require_runtime=True,
    ) == {
        "status": "error",
        "message": (
            "Runtime articulation control unavailable; "
            "require_runtime=True forbids USD fallback"
        ),
    }
