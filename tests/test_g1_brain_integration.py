# MIT License
#
# Copyright (c) 2026 whats2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the conditions in the MIT
# License. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

"""Dispatch-level tests for the ASSISTED G1 brain integration in robots.py.

Two layers, both under plain ``python3`` (no isaacsim / omni / numpy / pxr):

  * SOURCE / STRUCTURE checks (AST + string): the capability split (spawn is
    operator-only; set_control_mode / set_command / get_status are session-allowed
    high-level intent), command registration, and the ASSISTED->RAW block wiring.

  * BEHAVIORAL checks: the brain-lifecycle handlers (g1_set_control_mode,
    g1_set_command, g1_get_status, g1_set_joint_command) are AST-extracted from
    robots.py and exec'd against the REAL pure-Python ``g1_brain`` package with a
    fake driver/adapter, proving:
      - set_control_mode(ASSISTED) starts the in-process brain producer, RAW stops it;
      - set_command updates the brain setpoint (and latches before ASSISTED);
      - damp mode -> the brain emits a kp=0 LowCmd (no holding torque) AND the fall
        monitor fires on a tilted LowState (damp leads to a fall, detected);
      - RAW per-motor set_joint_command is rejected for a SESSION while ASSISTED,
        but an OPERATOR may still submit.
"""

import ast
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(__file__)
_EXT_ROOT = os.path.join(
    _HERE, "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension"
)
_GUARDS_PATH = os.path.join(_EXT_ROOT, "handlers", "_guards.py")
_ROBOTS_PATH = os.path.join(_EXT_ROOT, "handlers", "robots.py")
_EXTENSION_PATH = os.path.join(_EXT_ROOT, "extension.py")
_G1_DIR = os.path.join(_EXT_ROOT, "g1")
# The pure-Python brain package lives under the repo's rl/ tree.
_RL_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "rl"))

_SET_JOINT_CMD = "robots.g1.set_joint_command"
_SET_CONTROL_MODE = "robots.g1.set_control_mode"
_SET_COMMAND = "robots.g1.set_command"
_GET_STATUS = "robots.g1.get_status"
_SPAWN = "robots.g1.spawn"


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


def _load_guards():
    spec = importlib.util.spec_from_file_location("isaac_mcp_guards_brain_test", _GUARDS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_guards()


# ── SOURCE / STRUCTURE: capability split + registration + wiring ──────────────


def test_capability_split_mode_and_command_are_session_allowed():
    """spawn is operator-only; the high-level intent surface is NOT (session)."""
    op_only = _string_constants_of_assignment(_EXTENSION_PATH, "_OPERATOR_ONLY_COMMANDS")
    assert _SPAWN in op_only  # driver install stays operator-only
    # mode-switch + setpoint + status are session-allowed high-level intent.
    assert _SET_CONTROL_MODE not in op_only
    assert _SET_COMMAND not in op_only
    assert _GET_STATUS not in op_only


def test_new_g1_commands_registered():
    body = _func_src(_ROBOTS_PATH, "register")
    for cmd in (_SET_COMMAND, _GET_STATUS, _SET_CONTROL_MODE):
        assert cmd in body, f"{cmd} not registered"


def test_set_control_mode_starts_and_stops_brain_in_source():
    body = _func_src(_ROBOTS_PATH, "g1_set_control_mode")
    assert "_g1_start_brain(" in body
    assert "_g1_stop_brain(" in body
    assert "ControlMode.ASSISTED" in body
    # fail-closed: an ASSISTED start failure reverts to RAW (never producerless).
    assert "ControlMode.RAW" in body


def test_set_joint_command_blocks_session_while_assisted_in_source():
    body = _func_src(_ROBOTS_PATH, "g1_set_joint_command")
    assert "ControlMode.ASSISTED" in body
    assert "assisted_mode" in body
    assert "OPERATOR_CAPABILITY" in body
    # still the honest driver path, still no teleport sink.
    assert "submit_lowcmd(" in body
    assert "set_joint_positions" not in body
    assert "set_world_pose" not in body


def test_get_status_is_read_only_in_source():
    body = _func_src(_ROBOTS_PATH, "g1_get_status")
    # read-only: reads LowState + adapter pose/contacts, feeds the monitor. It must
    # NOT submit a LowCmd or write joints (never auto-catches a fall).
    assert "latest_lowstate_bytes(" in body
    assert "update_bytes(" in body
    assert "submit_lowcmd(" not in body
    assert "set_joint_positions" not in body
    assert "set_joint_command" not in body


def test_brain_live_probe_todo_present():
    assert "TODO(probe:brain-live)" in _read(os.path.join(_HERE, "..", "..",
        "isaac-session-neko", "scripts", "g1_brain_node.py"))


# ── BEHAVIORAL: exec the extracted handlers against the real g1_brain ─────────

_BEHAVIOR_FUNCS = (
    "_g1_brain_module", "_coerce_brain_mode", "_g1_make_brain", "_g1_start_brain",
    "_g1_stop_brain", "_g1_brain_running", "_g1_monitor", "_g1_contact_magnitude",
    "_parse_control_mode", "_coerce_frame_bytes", "_build_lowcmd_frame",
    "g1_set_command", "g1_set_control_mode", "g1_get_status", "g1_set_joint_command",
)


def _load_g1_codec():
    """Load the pure codec leaf modules (unitree_hg / motor_index_map / driver)."""
    if "g1pkg_brain" not in sys.modules:
        pkg = types.ModuleType("g1pkg_brain")
        pkg.__path__ = [_G1_DIR]
        sys.modules["g1pkg_brain"] = pkg
        for name in ("crc32", "gains", "motor_index_map", "unitree_hg", "g1_lowcmd_driver"):
            spec = importlib.util.spec_from_file_location(
                f"g1pkg_brain.{name}", os.path.join(_G1_DIR, f"{name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"g1pkg_brain.{name}"] = module
            spec.loader.exec_module(module)
    return sys.modules


def _handlers():
    """AST-extract the brain-lifecycle handlers + helpers and exec them in a
    namespace holding the real codec, real capability contextvar, and empty
    per-prim state dicts. Returns (ns, guards_module)."""
    if _RL_DIR not in sys.path:
        sys.path.insert(0, _RL_DIR)
    os.environ.setdefault("G1_CODEC_DIR", _G1_DIR)
    mods = _load_g1_codec()
    hg = mods["g1pkg_brain.unitree_hg"]
    mim = mods["g1pkg_brain.motor_index_map"]
    drv = mods["g1pkg_brain.g1_lowcmd_driver"]

    src = _read(_ROBOTS_PATH)
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _BEHAVIOR_FUNCS]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    module = ast.Module(body=[future] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "__file__": _ROBOTS_PATH,
        "os": os,
        "math": __import__("math"),
        "ControlMode": drv.ControlMode,
        "LowCmd": hg.LowCmd,
        "ModePR": hg.ModePR,
        "UNITREE_G1_JOINT_NAMES": mim.UNITREE_G1_JOINT_NAMES,
        "_N_BODY": len(mim.UNITREE_G1_JOINT_NAMES),
        "current_capability": G.current_capability,
        "OPERATOR_CAPABILITY": G.OPERATOR_CAPABILITY,
        "SESSION_CAPABILITY": G.SESSION_CAPABILITY,
        "_g1_drivers": {},
        "_g1_brains": {},
        "_g1_setpoints": {},
        "_g1_monitors": {},
        "_G1_BRAIN_PATHS": ("/session", "/session/g1_brain"),
    }
    exec(compile(module, _ROBOTS_PATH, "exec"), ns)
    ns["_codec"] = (hg, mim)
    return ns, G


class _FakeDriver:
    """Minimal G1LowCmdDriver stand-in exposing exactly what the brain path uses."""

    def __init__(self, hg, mode_machine=0, control_mode=None):
        self._hg = hg
        self._mm = mode_machine
        self.installed = True
        self._control_mode = control_mode
        self.submitted = []

    @property
    def control_mode(self):
        return self._control_mode

    def set_control_mode(self, cm):
        self._control_mode = cm

    def stats(self):
        return {"mode_machine": self._mm}

    def latest_lowstate_bytes(self):
        ls = self._hg.LowState()
        ls.mode_pr = int(self._hg.ModePR.PR)
        ls.mode_machine = self._mm
        return ls.with_crc().pack()

    def submit_lowcmd(self, frame):
        self.submitted.append(frame)
        return True


def _stop_all_brains(ns):
    for prim in list(ns["_g1_brains"]):
        try:
            ns["_g1_brains"][prim]["runner"].stop()
        except Exception:
            pass
    ns["_g1_brains"].clear()


def test_set_control_mode_assisted_starts_brain_raw_stops():
    ns, _ = _handlers()
    hg, _mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)
    ns["_g1_drivers"]["/World/G1"] = drv
    try:
        r = ns["g1_set_control_mode"](None, prim_path="/World/G1", mode="ASSISTED")
        assert r["status"] == "success"
        assert r["control_mode"] == "ASSISTED"
        assert r["brain_running"] is True
        assert drv.control_mode == ns["ControlMode"].ASSISTED

        r2 = ns["g1_set_control_mode"](None, prim_path="/World/G1", mode="RAW")
        assert r2["status"] == "success"
        assert r2["brain_running"] is False
        assert r2["brain_stopped"] is True
    finally:
        _stop_all_brains(ns)


def test_set_command_updates_setpoint_and_latches_before_assisted():
    ns, _ = _handlers()
    hg, _mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)
    ns["_g1_drivers"]["/World/G1"] = drv
    try:
        # (a) set_command BEFORE ASSISTED latches the setpoint.
        r = ns["g1_set_command"](None, prim_path="/World/G1", mode="squat")
        assert r["status"] == "success"
        assert r["setpoint"]["mode"] == "squat"
        assert ns["_g1_setpoints"]["/World/G1"].mode.value == "squat"

        # (b) starting ASSISTED applies the latched setpoint to the live brain.
        ns["g1_set_control_mode"](None, prim_path="/World/G1", mode="ASSISTED")
        brain = ns["_g1_brains"]["/World/G1"]["brain"]
        assert brain.setpoint.mode.value == "squat"

        # (c) a new set_command forwards to the running brain immediately.
        r2 = ns["g1_set_command"](None, prim_path="/World/G1", mode="walk", vx=0.4)
        assert r2["setpoint"] == {"mode": "walk", "vx": 0.4, "vy": 0.0, "wz": 0.0}
        assert brain.setpoint.mode.value == "walk"
        assert brain.setpoint.vx == 0.4
    finally:
        _stop_all_brains(ns)


def test_set_command_rejects_unknown_mode():
    ns, _ = _handlers()
    hg, _mim = ns["_codec"]
    ns["_g1_drivers"]["/World/G1"] = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)
    r = ns["g1_set_command"](None, prim_path="/World/G1", mode="moonwalk")
    assert r["status"] == "error"
    assert "unknown mode" in r["message"]


def test_damp_mode_leads_to_fall_and_detector_fires():
    ns, _ = _handlers()
    hg, mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)
    ns["_g1_drivers"]["/World/G1"] = drv
    try:
        ns["g1_set_command"](None, prim_path="/World/G1", mode="damp")
        ns["g1_set_control_mode"](None, prim_path="/World/G1", mode="ASSISTED")
        brain = ns["_g1_brains"]["/World/G1"]["brain"]

        # damp -> the brain emits a kp=0 LowCmd (no holding torque -> it will fall).
        lc = hg.LowCmd.unpack(brain.step(None))
        n_body = len(mim.UNITREE_G1_JOINT_NAMES)
        assert all(lc.motor_cmd[u].kp == 0.0 for u in range(n_body))
        assert any(lc.motor_cmd[u].kd > 0.0 for u in range(n_body))  # kd-only, honest fall

        # the fall detector fires on the resulting tilted-over state.
        import math

        from g1_brain.monitor import G1FallMonitor

        h = math.radians(90.0) / 2.0
        fallen = hg.LowState()
        fallen.mode_pr = int(hg.ModePR.PR)
        fallen.imu_state.quaternion = (math.cos(h), math.sin(h), 0.0, 0.0)  # rolled 90deg
        st = G1FallMonitor().update_bytes(
            fallen.with_crc().pack(), root_position=[0.0, 0.0, 0.15]
        )
        assert st.fell is True
        assert st.upright is False
    finally:
        _stop_all_brains(ns)


def test_raw_set_joint_command_blocked_for_session_while_assisted():
    ns, guards = _handlers()
    hg, _mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].ASSISTED)
    ns["_g1_drivers"]["/World/G1"] = drv

    tok = guards.set_current_capability(guards.SESSION_CAPABILITY)
    try:
        r = ns["g1_set_joint_command"](None, prim_path="/World/G1", q=[0.0] * 29)
        assert r["status"] == "error"
        assert r["rejected_by"] == "assisted_mode"
        assert drv.submitted == []  # never reached the driver
    finally:
        guards.reset_current_capability(tok)


def test_raw_set_joint_command_allowed_for_operator_while_assisted():
    ns, guards = _handlers()
    hg, _mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].ASSISTED)
    ns["_g1_drivers"]["/World/G1"] = drv

    tok = guards.set_current_capability(guards.OPERATOR_CAPABILITY)
    try:
        r = ns["g1_set_joint_command"](None, prim_path="/World/G1", q=[0.0] * 29)
        assert r["status"] == "success"  # operator (external producer/debug) passes
        assert len(drv.submitted) == 1
    finally:
        guards.reset_current_capability(tok)


def test_raw_set_joint_command_allowed_for_session_in_raw_mode():
    ns, guards = _handlers()
    hg, _mim = ns["_codec"]
    drv = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)
    ns["_g1_drivers"]["/World/G1"] = drv

    tok = guards.set_current_capability(guards.SESSION_CAPABILITY)
    try:
        r = ns["g1_set_joint_command"](None, prim_path="/World/G1", q=[0.0] * 29)
        assert r["status"] == "success"
        assert len(drv.submitted) == 1
    finally:
        guards.reset_current_capability(tok)


def test_get_status_reports_upright_from_lowstate():
    ns, _ = _handlers()
    hg, _mim = ns["_codec"]
    ns["_g1_drivers"]["/World/G1"] = _FakeDriver(hg, control_mode=ns["ControlMode"].RAW)

    class _Adapter:
        def get_prim_transform(self, p):
            return {"position": [0.0, 0.0, 0.8]}

        def get_physics_state(self, p):
            return {"contacts": [{"force": [0.0, 0.0, 250.0]}]}

    r = ns["g1_get_status"](_Adapter(), prim_path="/World/G1")
    assert r["status"] == "success"
    assert r["state_available"] is True
    detail = r["status_detail"]
    assert detail["upright"] is True and detail["fell"] is False
    assert detail["tainted"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
