# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Behavioural proof that the RCE surface is CLOSED in session mode.

These tests do NOT grep source — they drive the *live* enforcement code:

  * the two OmniGraph handlers (``graphs.create_action_graph`` /
    ``graphs.edit_action_graph``) are called directly and must refuse any
    ScriptNode node, any inputs:script/scriptPath/usePath write or connection,
    and the script_file shortcut;
  * the object handlers (``objects.create`` / ``objects.clone``) are called
    directly and must refuse moving/overwriting a robot articulation-root while
    the timeline is live — the sink is blocked (create_prim/clone_prim never
    runs), not merely reported;
  * the extension's real ``_execute_command`` dispatch is exercised end-to-end to
    prove execute_script / reload_script / omini_kit_command / edit_action_graph are
    REFUSED ("forbidden in session mode") under a SESSION-capability command while
    they REACH the handler under an OPERATOR-capability command, and that a
    ScriptNode create is rejected for session but allowed for operator. The registry
    is kept full (no process-global pop): the split is per-command capability.

Isaac Sim is not present offline, so ``numpy`` / ``carb`` / ``omni.*`` are stubbed
into ``sys.modules`` *before* the real extension package is imported. The stubs
only satisfy import-time references; every assertion is against the real handler
and dispatch logic. TODO(probe:guard-live): re-run the dispatch rejections on a
live box to confirm the session token is refused with "forbidden in session mode"
while the operator token reaches the handler.
"""

import contextlib
import os
import sys
import types
from unittest import mock

# ── stub the Isaac/omni import surface before importing the real package ──────

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension")
)


def _install_stubs() -> None:
    # numpy: adapters.base does `import numpy as np` at module top (only used in
    # method bodies), so a bare mock module is enough to import the package.
    sys.modules.setdefault("numpy", mock.MagicMock())

    if "carb" not in sys.modules:
        carb = types.ModuleType("carb")
        carb.settings = types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(get=lambda *a, **k: None)
        )
        sys.modules["carb"] = carb

    omni = sys.modules.get("omni") or types.ModuleType("omni")
    sys.modules["omni"] = omni

    def _sub(parent_name: str, parent_mod, child: str):
        full = f"{parent_name}.{child}"
        mod = sys.modules.get(full) or types.ModuleType(full)
        sys.modules[full] = mod
        setattr(parent_mod, child, mod)
        return mod

    ext = _sub("omni", omni, "ext")
    if not hasattr(ext, "IExt"):
        class _IExt:  # `class MCPExtension(omni.ext.IExt)` needs a real base class
            def __init__(self, *a, **k):
                pass

        ext.IExt = _IExt

    kit = _sub("omni", omni, "kit")
    _sub("omni.kit", kit, "app")
    commands = _sub("omni.kit", kit, "commands")
    if not hasattr(commands, "execute"):
        commands.execute = lambda *a, **k: None
    _sub("omni", omni, "usd")

    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)


_install_stubs()

from isaac_sim_mcp_extension.extension import MCPExtension  # noqa: E402
from isaac_sim_mcp_extension.handlers import _guards, graphs, objects  # noqa: E402

REJECTED = _guards.REJECTED_BY  # "session_guard"
_ROBOT = "/World/G1"


@contextlib.contextmanager
def _session_mode(value):
    """Force MCP_SESSION_MODE for the duration of a test, then restore it."""
    old = os.environ.get("MCP_SESSION_MODE")
    if value is None:
        os.environ.pop("MCP_SESSION_MODE", None)
    else:
        os.environ["MCP_SESSION_MODE"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MCP_SESSION_MODE", None)
        else:
            os.environ["MCP_SESSION_MODE"] = old


class FakeAdapter:
    """Records every sink call so tests can assert the mutation never happened."""

    def __init__(self, timeline="playing", gravity=9.81, dt=1.0 / 60.0, roots=(_ROBOT,)):
        self._timeline = timeline
        self._gravity = gravity
        self._dt = dt
        self._roots = roots
        self.created = []
        self.transform_calls = []
        self.clone_calls = []

    # world-state hooks used by the guard
    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": self._dt}

    def list_articulation_roots(self):
        return self._roots

    def get_gravity_magnitude(self):
        return self._gravity

    def get_prim_transform(self, prim_path):
        return {"position": [0.0, 0.0, 1.0]}

    # mutation sinks — must NOT be reached when a guard rejects
    def create_prim(self, prim_path, prim_type="Xform", **kwargs):
        self.created.append(prim_path)
        return object()

    def set_prim_transform(self, prim_path, **kwargs):
        self.transform_calls.append((prim_path, kwargs))

    def clone_prim(self, source_path, target_path):
        self.clone_calls.append((source_path, target_path))

    def get_stage(self):  # pragma: no cover - only the auto-name path reaches this
        raise AssertionError("get_stage should not be reached in these tests")

    def get_prim_actual_size(self, prim_path):
        raise Exception("size unavailable offline")

    def _ensure_physics_world(self, *a, **k):
        return None


def _is_reject(d):
    return d is not None and d.get("status") == "error" and d.get("rejected_by") == REJECTED


# ══ graphs.create_action_graph — full ScriptNode surface refused ══════════════


def test_create_action_graph_rejects_scriptnode_node_type():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            nodes=[{"path": "S", "type": "omni.graph.scriptnode.ScriptNode"}],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_scriptnode_node_type_case_insensitive():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            nodes=[{"path": "S", "type": "OmNi.Graph.ScriptNode.ScriptNode"}],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_script_value_write():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            values=[{"attr": "ScriptNode.inputs:script", "value": "import os; os.system('x')"}],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_script_path_value_write():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            values=[{"attr": "ScriptNode.inputs:scriptPath", "value": "/evil.py"}],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_use_path_value_write():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            values=[{"attr": "ScriptNode.inputs:usePath", "value": True}],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_script_connection():
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            connections=[["Src.outputs:out", "ScriptNode.inputs:script"]],
        )
    assert _is_reject(d), d


def test_create_action_graph_rejects_script_file_shortcut():
    with _session_mode("1"):
        d = graphs.create_action_graph(FakeAdapter(), script_file="/tmp/controller.py")
    assert _is_reject(d), d


def test_create_action_graph_benign_wiring_not_rejected_by_guard():
    # Pure wiring (no ScriptNode) must pass the RCE guard; it then fails on the
    # (absent) omni.graph.core import — proving the guard let it through.
    with _session_mode("1"):
        d = graphs.create_action_graph(
            FakeAdapter(),
            nodes=[{"path": "tick", "type": "omni.graph.action.OnPlaybackTick"}],
        )
    assert d.get("rejected_by") != REJECTED, d


def test_create_action_graph_scriptnode_allowed_in_operator_mode():
    # Session mode OFF is the operator/legacy path: the ScriptNode surface is NOT
    # refused (it fails later on the absent omni import instead).
    with _session_mode("0"):
        d = graphs.create_action_graph(FakeAdapter(), script_file="/tmp/controller.py")
    assert d.get("rejected_by") != REJECTED, d


# ══ graphs.edit_action_graph — script writes/connections refused ══════════════


def test_edit_action_graph_rejects_script_value_write():
    with _session_mode("1"):
        d = graphs.edit_action_graph(
            FakeAdapter(),
            values=[{"attr": "ScriptNode.inputs:script", "value": "os.system('x')"}],
        )
    assert _is_reject(d), d


def test_edit_action_graph_rejects_script_path_connection():
    with _session_mode("1"):
        d = graphs.edit_action_graph(
            FakeAdapter(),
            connections=[["Src.outputs:out", "ScriptNode.inputs:scriptPath"]],
        )
    assert _is_reject(d), d


# ══ objects sink lock — create/clone cannot teleport a robot ══════════════════


def test_objects_create_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.create(adapter, prim_path=_ROBOT, position=[0.0, 0.0, 5.0])
    assert _is_reject(d), d
    assert adapter.created == [], "create_prim sink must not run when the guard rejects"
    assert adapter.transform_calls == [], "set_prim_transform sink must not run"


def test_objects_create_over_robot_descendant_blocked_while_paused():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="paused")
        d = objects.create(adapter, prim_path=_ROBOT + "/pelvis", position=[0.0, 0.0, 5.0])
    assert _is_reject(d), d
    assert adapter.created == []


def test_objects_create_over_robot_allowed_while_stopped():
    # Authoring (timeline stopped) is the legitimate window to (re)place a prim.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        objects.create(adapter, prim_path=_ROBOT, position=[0.0, 0.0, 5.0])
    assert adapter.created == [_ROBOT], "guard must allow authoring while stopped"


def test_objects_create_non_robot_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        objects.create(adapter, prim_path="/World/Cube", position=[1.0, 1.0, 1.0])
    assert adapter.created == ["/World/Cube"], "non-robot create must pass the guard"


def test_objects_create_fails_closed_when_robot_discovery_fails():
    # roots=None => discovery failed => cannot prove target is not a robot => reject.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = objects.create(adapter, prim_path="/World/Cube", position=[1.0, 1.0, 1.0])
    assert _is_reject(d), d
    assert adapter.created == []


def test_objects_clone_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.clone(adapter, source_path="/World/box", target_path=_ROBOT, position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.clone_calls == [], "clone_prim sink must not run when the guard rejects"
    assert adapter.transform_calls == []


def test_objects_clone_over_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        objects.clone(adapter, source_path="/World/box", target_path="/World/copy")
    assert adapter.clone_calls == [("/World/box", "/World/copy")]


def test_objects_create_over_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        objects.create(adapter, prim_path=_ROBOT, position=[0.0, 0.0, 5.0])
    assert adapter.created == [_ROBOT], "operator mode must not apply the session lock"


# ══ guard_pose_write helper (used by the sinks) ═══════════════════════════════


def test_guard_pose_write_rejects_robot_while_playing():
    assert _is_reject(_guards.guard_pose_write(FakeAdapter(timeline="playing"), _ROBOT, session_mode=True))


def test_guard_pose_write_allows_robot_while_stopped():
    assert _guards.guard_pose_write(FakeAdapter(timeline="stopped"), _ROBOT, session_mode=True) is None


def test_guard_pose_write_allows_non_robot_while_playing():
    assert _guards.guard_pose_write(FakeAdapter(timeline="playing"), "/World/Cube", session_mode=True) is None


def test_guard_pose_write_disabled_in_operator_mode():
    assert _guards.guard_pose_write(FakeAdapter(timeline="playing"), _ROBOT, session_mode=False) is None


# ══ end-to-end dispatch: RCE aliases UNREACHABLE on a session token ═══════════


def _make_extension():
    adapter = FakeAdapter(timeline="playing")
    ext = MCPExtension()
    ext._adapter = adapter
    ext._session_mode = True
    reg = ext._registry
    graphs.register(reg, adapter)
    objects.register(reg, adapter)
    # The FULL registry is kept (no process-global pop): the operator-only exec
    # aliases stay registered and dispatch enforces the capability split. Stub them
    # so an operator-capability dispatch can be observed reaching the handler.
    reg["simulation.execute_script"] = lambda **k: {"status": "success", "result": {"ran": True}}
    reg["simulation.reload_script"] = lambda **k: {"status": "success", "result": {"ran": True}}
    reg["execute_script"] = lambda **k: {"status": "success", "result": {"ran": True}}
    reg["omini_kit_command"] = lambda **k: {"status": "success", "result": {"ran": True}}
    # A benign read-only command that must survive the hardening.
    reg["scene.get_info"] = lambda **k: {"status": "success", "result": {"ok": True}}
    ext._guard = _guards.SessionGuard(adapter, session_mode=True)
    return ext, adapter


def _dispatch(ext, cmd_type, params=None, capability="session"):
    # capability mirrors the tag socket_server sets after authenticating the token.
    return ext._execute_command(
        {"type": cmd_type, "params": params or {}, "capability": capability}
    )


def _forbidden(r):
    return r["status"] == "error" and "forbidden in session mode" in r["message"]


def test_dispatch_execute_script_forbidden_for_session():
    with _session_mode("1"):
        ext, _ = _make_extension()
        for alias in ("simulation.execute_script", "execute_script"):
            r = _dispatch(ext, alias, {"code": "import os; os.system('id')"})
            assert _forbidden(r), (alias, r)


def test_dispatch_execute_script_allowed_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        for alias in ("simulation.execute_script", "execute_script"):
            r = _dispatch(ext, alias, {"code": "print(1)"}, capability="operator")
            # Reaches the stubbed handler and runs (operator = trusted bootstrap).
            assert r["status"] == "success" and r["result"]["result"]["ran"] is True, (alias, r)


def test_dispatch_reload_script_forbidden_for_session_allowed_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        assert _forbidden(_dispatch(ext, "simulation.reload_script", {}))
        r = _dispatch(ext, "simulation.reload_script", {}, capability="operator")
        assert r["status"] == "success", r


def test_dispatch_omini_kit_command_forbidden_for_session_allowed_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        assert _forbidden(_dispatch(ext, "omini_kit_command", {"command": "DeletePrim"}))
        r = _dispatch(ext, "omini_kit_command", {"command": "DeletePrim"}, capability="operator")
        assert r["status"] == "success", r


def test_dispatch_edit_action_graph_forbidden_for_session():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext,
            "graphs.edit_action_graph",
            {"values": [{"attr": "ScriptNode.inputs:script", "value": "os.system('x')"}]},
        )
        assert _forbidden(r), r


def test_dispatch_edit_action_graph_reaches_handler_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext,
            "graphs.edit_action_graph",
            {"values": [{"attr": "ScriptNode.inputs:script", "value": "print(1)"}]},
            capability="operator",
        )
    # Not forbidden and not RCE-rejected: it reached the real handler, which then
    # fails on the absent omni.graph.core import offline.
    assert r["status"] == "error"
    assert "forbidden in session mode" not in r["message"] and REJECTED not in r["message"], r


def test_dispatch_create_action_graph_scriptnode_rejected_for_session():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext,
            "graphs.create_action_graph",
            {"nodes": [{"path": "S", "type": "omni.graph.scriptnode.ScriptNode"}]},
        )
        assert r["status"] == "error" and REJECTED in r["message"], r


def test_dispatch_create_action_graph_scriptnode_allowed_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext,
            "graphs.create_action_graph",
            {"nodes": [{"path": "S", "type": "omni.graph.scriptnode.ScriptNode"}]},
            capability="operator",
        )
    # Operator keeps the ScriptNode surface: not rejected as RCE; it reaches the
    # handler and fails only on the absent omni.graph.core import offline.
    assert r["status"] == "error"
    assert REJECTED not in r["message"] and "forbidden in session mode" not in r["message"], r


def test_dispatch_create_action_graph_script_file_rejected_for_session():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(ext, "graphs.create_action_graph", {"script_file": "/tmp/x.py"})
        assert r["status"] == "error" and REJECTED in r["message"], r


def test_dispatch_create_action_graph_script_file_allowed_for_operator():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext, "graphs.create_action_graph", {"script_file": "/tmp/x.py"}, capability="operator"
        )
    assert r["status"] == "error"
    assert REJECTED not in r["message"] and "forbidden in session mode" not in r["message"], r


def test_dispatch_create_action_graph_benign_reaches_handler():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(
            ext,
            "graphs.create_action_graph",
            {"nodes": [{"path": "tick", "type": "omni.graph.action.OnPlaybackTick"}]},
        )
    # Not blocked as RCE and not forbidden: it reached the real handler (which then
    # fails on the absent omni.graph.core import offline).
    assert r["status"] == "error"
    assert REJECTED not in r["message"] and "forbidden in session mode" not in r["message"], r


def test_dispatch_objects_create_robot_teleport_rejected_and_not_executed():
    with _session_mode("1"):
        ext, adapter = _make_extension()
        r = _dispatch(ext, "objects.create", {"prim_path": _ROBOT, "position": [0, 0, 5]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.created == [], "the teleport-via-create must never touch the sink"


def test_dispatch_objects_create_robot_teleport_allowed_for_operator():
    # Operator = trusted bootstrap: the sink-level pose lock is bypassed, so the
    # create reaches the sink (create_prim runs) even at a robot root while playing.
    with _session_mode("1"):
        ext, adapter = _make_extension()
        _dispatch(ext, "objects.create", {"prim_path": _ROBOT, "position": [0, 0, 5]}, capability="operator")
    assert adapter.created == [_ROBOT], "operator must reach the create sink (guard bypassed)"


def test_dispatch_objects_clone_over_robot_rejected_and_not_executed():
    with _session_mode("1"):
        ext, adapter = _make_extension()
        r = _dispatch(ext, "objects.clone", {"source_path": "/World/box", "target_path": _ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.clone_calls == [], "the clone-over-robot must never touch the sink"


def test_dispatch_pose_write_to_robot_rejected_for_session():
    # The pre-existing dispatch-level pose guard still fires for session (regression).
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(ext, "objects.transform", {"prim_path": _ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r


def test_dispatch_pose_write_to_robot_allowed_for_operator():
    with _session_mode("1"):
        ext, adapter = _make_extension()
        r = _dispatch(ext, "objects.transform", {"prim_path": _ROBOT}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.transform_calls, "operator pose write must reach the sink (guard bypassed)"


def test_dispatch_readonly_command_still_works():
    with _session_mode("1"):
        ext, _ = _make_extension()
        r = _dispatch(ext, "scene.get_info", {})
    assert r["status"] == "success", r


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
