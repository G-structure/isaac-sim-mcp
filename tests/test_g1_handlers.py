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

"""Dispatch-level tests for the robots.g1.* low-level driver command surface.

Two layers are checked without importing the omni/numpy handler module:

  * the fail-closed ACTUATION gate over the g1 per-motor commands, exercised
    against the pure ``evaluate_command`` decision core (the same core the socket
    dispatch runs before any handler);
  * the STRUCTURE of the wiring — that spawn is operator-only, that the g1
    commands are registered, and that the honest-actuation invariants (no
    teleport, stopped-only rate-limited reset) hold — asserted from source so the
    checks run under plain ``python3`` (no isaacsim, omni, numpy, or pxr).
"""

import ast
import importlib.util
import os
import sys
import types

_EXT_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
)
_GUARDS_PATH = os.path.join(_EXT_ROOT, "handlers", "_guards.py")
_ROBOTS_PATH = os.path.join(_EXT_ROOT, "handlers", "robots.py")
_EXTENSION_PATH = os.path.join(_EXT_ROOT, "extension.py")

_SET_JOINT_CMD = "robots.g1.set_joint_command"
_SET_GAINS = "robots.g1.set_gains"
_SPAWN = "robots.g1.spawn"
_G1_COMMANDS = (
    _SPAWN,
    _SET_JOINT_CMD,
    "robots.g1.get_lowstate",
    _SET_GAINS,
    "robots.g1.set_control_mode",
    "robots.g1.reset",
)


def _load_guards():
    spec = importlib.util.spec_from_file_location("isaac_mcp_guards_g1_test", _GUARDS_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolve string annotations via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_guards()


def _state(timeline="playing", gravity=9.81, dt=1.0 / 200.0, roots=("/World/G1",)):
    # WorldState.gravity is the full VECTOR the gate checks; map a scalar strength to
    # straight-down (-Z) so the scalar-parametrised gate tests below stay expressive.
    gravity_vec = None if gravity is None else (0.0, 0.0, -float(gravity))
    return G.WorldState(
        timeline_state=timeline,
        gravity=gravity_vec,
        physics_dt=dt,
        robot_roots=roots,
    )


def _is_reject(decision):
    return decision is not None and decision.get("status") == "error"


def _read(path):
    with open(path) as f:
        return f.read()


def _func_src(path, name):
    src = _read(path)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {path}")


def _string_constants_of_assignment(path, target_name):
    """All str constants inside the value assigned to ``target_name`` at top level."""
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == target_name for t in node.targets
        ):
            return {
                sub.value
                for sub in ast.walk(node.value)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            }
    raise AssertionError(f"assignment to {target_name} not found in {path}")


# ── ACTUATION gate over the g1 per-motor surface ─────────────────────────────


def test_g1_actuation_commands_registered_in_gate_set():
    """The g1 per-motor commands must be in ACTUATION_COMMANDS so the gate applies."""
    assert _SET_JOINT_CMD in G.ACTUATION_COMMANDS
    assert _SET_GAINS in G.ACTUATION_COMMANDS
    # And therefore reachable by the guard fast-path (not silently skipped).
    assert _SET_JOINT_CMD in G._ALL_GUARDED_COMMANDS
    assert _SET_GAINS in G._ALL_GUARDED_COMMANDS


def test_g1_set_joint_command_rejected_when_stopped():
    d = G.evaluate_command(_SET_JOINT_CMD, {"prim_path": "/World/G1"}, _state(timeline="stopped"))
    assert _is_reject(d)


def test_g1_set_joint_command_rejected_when_timeline_unknown():
    d = G.evaluate_command(_SET_JOINT_CMD, {}, _state(timeline=None))
    assert _is_reject(d)


def test_g1_set_joint_command_rejected_when_gravity_zero_or_unknown():
    assert _is_reject(G.evaluate_command(_SET_JOINT_CMD, {}, _state(gravity=0.0)))
    assert _is_reject(G.evaluate_command(_SET_JOINT_CMD, {}, _state(gravity=None)))


def test_g1_set_joint_command_rejected_when_physics_dt_nonpositive():
    assert _is_reject(G.evaluate_command(_SET_JOINT_CMD, {}, _state(dt=0.0)))
    assert _is_reject(G.evaluate_command(_SET_JOINT_CMD, {}, _state(dt=None)))


def test_g1_set_joint_command_allowed_when_physics_live():
    assert G.evaluate_command(_SET_JOINT_CMD, {"prim_path": "/World/G1"}, _state()) is None


def test_g1_set_gains_gated_like_actuation():
    assert _is_reject(G.evaluate_command(_SET_GAINS, {}, _state(timeline="stopped")))
    assert G.evaluate_command(_SET_GAINS, {}, _state()) is None


def test_g1_session_guard_gates_set_joint_command_but_operator_bypasses():
    """End-to-end through the SessionGuard wrapper: session gated, operator free."""

    class _Adapter:
        def get_simulation_state(self):
            return {"timeline_state": "stopped", "physics_dt": 0.0}

        def list_articulation_roots(self):
            return ("/World/G1",)

        def get_gravity_magnitude(self):
            return 0.0

    guard = G.SessionGuard(_Adapter(), session_mode=True)
    # Session: physics not live -> rejected.
    assert _is_reject(guard.guard(_SET_JOINT_CMD, {}, capability=G.SESSION_CAPABILITY))
    # Operator (trusted bootstrap): the gate is bypassed entirely.
    assert guard.guard(_SET_JOINT_CMD, {}, capability=G.OPERATOR_CAPABILITY) is None


# ── spawn is operator-only ────────────────────────────────────────────────────


def test_g1_spawn_operator_only_at_dispatch():
    """robots.g1.spawn must be listed in the extension's operator-only set."""
    op_only = _string_constants_of_assignment(_EXTENSION_PATH, "_OPERATOR_ONLY_COMMANDS")
    assert _SPAWN in op_only
    # sanity: the existing exec surface is still there (we only added to the set).
    assert "simulation.execute_script" in op_only
    # The agent-facing g1 commands must NOT be operator-only.
    assert _SET_JOINT_CMD not in op_only
    assert _SET_GAINS not in op_only


def test_g1_spawn_handler_rechecks_operator_capability():
    """Defense-in-depth: the handler itself refuses a session capability."""
    body = _func_src(_ROBOTS_PATH, "g1_spawn")
    assert "current_capability()" in body
    assert "OPERATOR_CAPABILITY" in body
    assert "session_capability" in body


# ── registration + honest-actuation structure ────────────────────────────────


def test_all_g1_commands_registered():
    body = _func_src(_ROBOTS_PATH, "register")
    for cmd in _G1_COMMANDS:
        assert cmd in body, f"{cmd} not registered in robots.register()"


def test_set_joint_command_moves_only_through_driver_no_teleport():
    """The RAW per-motor path must submit to the driver, never teleport joints."""
    body = _func_src(_ROBOTS_PATH, "g1_set_joint_command")
    assert "submit_lowcmd(" in body
    # It must NOT reach a direct DOF-state / pose write sink.
    assert "set_joint_positions" not in body
    assert "set_prim_transform" not in body
    assert "set_world_pose" not in body


def test_reset_is_stopped_only_and_rate_limited():
    body = _func_src(_ROBOTS_PATH, "g1_reset")
    assert 'timeline != "stopped"' in body
    assert "_g1_last_reset" in body
    assert "_G1_RESET_MIN_INTERVAL_S" in body
    # The neutral pose is authored via the adapter (only reachable while stopped).
    assert "set_joint_positions" in body


def test_get_lowstate_reads_driver_seqlock():
    body = _func_src(_ROBOTS_PATH, "g1_get_lowstate")
    assert "latest_lowstate" in body
    # Falls back to the honest runtime read when no driver is installed.
    assert "get_lowstate_fields" in body


def test_driver_live_probe_todo_present():
    """The on-box RAW validation is still open; keep its probe marker."""
    assert "TODO(probe:driver-live)" in _read(_ROBOTS_PATH)


# ── behavioral: the LowCmd frame builder against the real codec ───────────────
#
# The g1 codec (unitree_hg) has a relative import (from .crc32), so it is loaded
# through a synthetic ``g1pkg`` package whose __path__ points at the g1/ dir. The
# pure frame-building helpers from robots.py are then exec'd in a namespace holding
# the real codec classes (they touch no numpy/omni), so we can prove a frame built
# from arrays is exactly what the driver's inbound verify accepts.

_G1_DIR = os.path.join(_EXT_ROOT, "g1")


def _load_g1_pkg():
    if "g1pkg" not in sys.modules:
        pkg = types.ModuleType("g1pkg")
        pkg.__path__ = [_G1_DIR]
        sys.modules["g1pkg"] = pkg
        for name in ("crc32", "gains", "motor_index_map", "unitree_hg", "g1_lowcmd_driver"):
            spec = importlib.util.spec_from_file_location(
                f"g1pkg.{name}", os.path.join(_G1_DIR, f"{name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"g1pkg.{name}"] = module
            spec.loader.exec_module(module)
    return sys.modules


def _pure_helpers():
    mods = _load_g1_pkg()
    hg = mods["g1pkg.unitree_hg"]
    mim = mods["g1pkg.motor_index_map"]
    drv = mods["g1pkg.g1_lowcmd_driver"]

    src = _read(_ROBOTS_PATH)
    tree = ast.parse(src)
    wanted = {"_coerce_frame_bytes", "_build_lowcmd_frame", "_decode_lowstate", "_parse_control_mode"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    # Prepend `from __future__ import annotations` so the string annotations
    # (Optional[Sequence[float]], G1LowCmdDriver, ...) are never evaluated at def
    # time — the helpers themselves reference only the codec names below.
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    module = ast.Module(body=[future] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "LowCmd": hg.LowCmd,
        "ModePR": hg.ModePR,
        "ControlMode": drv.ControlMode,
        "UNITREE_G1_JOINT_NAMES": mim.UNITREE_G1_JOINT_NAMES,
        "_N_BODY": len(mim.UNITREE_G1_JOINT_NAMES),
    }
    exec(compile(module, _ROBOTS_PATH, "exec"), ns)
    ns["_LOWCMD_SIZE"] = hg.LOWCMD_SIZE
    ns["_PR"] = int(hg.ModePR.PR)
    return ns


class _FakeDriver:
    def __init__(self, mode_machine=0):
        self._mm = mode_machine

    def stats(self):
        return {"mode_machine": self._mm}


def _frame_accepted_by_driver(h, frame, mode_machine):
    """Replicate G1LowCmdDriver._validated_cmd acceptance: length, CRC, mode."""
    assert len(frame) == h["_LOWCMD_SIZE"]
    lc = h["LowCmd"].unpack(frame)
    assert lc.verify_crc()
    assert lc.mode_pr == h["_PR"]
    assert lc.mode_machine == mode_machine
    return lc


def test_build_lowcmd_frame_from_full_arrays_is_driver_valid():
    h = _pure_helpers()
    q = [0.01 * i for i in range(29)]
    frame = h["_build_lowcmd_frame"](
        _FakeDriver(mode_machine=3), None, q, None, None, None, None, None, None, None
    )
    lc = _frame_accepted_by_driver(h, frame, mode_machine=3)
    # Every body slot carried its q; hand/unused motor slots stay zero.
    for u in range(29):
        assert abs(lc.motor_cmd[u].q - q[u]) < 1e-6
    assert lc.motor_cmd[29].q == 0.0


def test_build_lowcmd_frame_joint_indices_subset():
    h = _pure_helpers()
    frame = h["_build_lowcmd_frame"](
        _FakeDriver(), None, [1.5], None, [7.0], None, None, [3], None, None
    )
    lc = _frame_accepted_by_driver(h, frame, mode_machine=0)
    assert abs(lc.motor_cmd[3].q - 1.5) < 1e-6
    assert abs(lc.motor_cmd[3].tau - 7.0) < 1e-6
    assert lc.motor_cmd[3].mode == 1
    # Untouched slots stay zero.
    assert lc.motor_cmd[0].q == 0.0 and lc.motor_cmd[0].mode == 0


def test_build_lowcmd_frame_explicit_mode_machine_override():
    h = _pure_helpers()
    frame = h["_build_lowcmd_frame"](
        _FakeDriver(mode_machine=0), None, [0.0], None, None, None, None, [0], None, 9
    )
    _frame_accepted_by_driver(h, frame, mode_machine=9)


def test_build_lowcmd_frame_rejects_out_of_range_index():
    h = _pure_helpers()
    try:
        h["_build_lowcmd_frame"](_FakeDriver(), None, [0.0], None, None, None, None, [99], None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range joint index must raise")


def test_build_lowcmd_frame_rejects_length_mismatch():
    h = _pure_helpers()
    try:
        h["_build_lowcmd_frame"](_FakeDriver(), None, [1.0, 2.0, 3.0], None, None, None, None, [0, 1], None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("q/joint_indices length mismatch must raise")


def test_build_lowcmd_frame_requires_some_payload():
    h = _pure_helpers()
    try:
        h["_build_lowcmd_frame"](_FakeDriver(), None, None, None, None, None, None, None, None, None)
    except ValueError:
        pass
    else:
        raise AssertionError("empty command (no lowcmd, no arrays) must raise")


def test_coerce_frame_bytes_hex_and_list_and_bytes():
    h = _pure_helpers()
    assert h["_coerce_frame_bytes"](b"\x01\x02\x03") == b"\x01\x02\x03"
    assert h["_coerce_frame_bytes"]("010203") == b"\x01\x02\x03"
    assert h["_coerce_frame_bytes"]([1, 2, 3]) == b"\x01\x02\x03"


def test_prebuilt_lowcmd_frame_passthrough():
    """A prebuilt CRC'd frame is returned verbatim (agent-built LowCmd path)."""
    h = _pure_helpers()
    lc = h["LowCmd"](mode_pr=h["_PR"], mode_machine=2)
    lc.motor_cmd[5].q = 0.25
    lc.with_crc()
    raw = lc.pack()
    assert h["_build_lowcmd_frame"](_FakeDriver(), raw, None, None, None, None, None, None, None, None) == raw
    assert h["_build_lowcmd_frame"](_FakeDriver(), raw.hex(), None, None, None, None, None, None, None, None) == raw


def test_parse_control_mode_names_and_ints():
    h = _pure_helpers()
    CM = h["ControlMode"]
    assert h["_parse_control_mode"]("raw") == CM.RAW
    assert h["_parse_control_mode"]("ASSISTED") == CM.ASSISTED
    assert h["_parse_control_mode"](1) == CM.ASSISTED
    assert h["_parse_control_mode"](None) == CM.RAW
    try:
        h["_parse_control_mode"]("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown control mode must raise")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
