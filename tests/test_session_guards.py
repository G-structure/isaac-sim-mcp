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

"""Fail-closed session-guard reject cases.

The guard decision core is a pure function over a WorldState snapshot, so these
run under plain ``python3`` — no isaacsim, no omni, no numpy, no pxr. The
``SessionGuard`` wrapper is exercised with a fake adapter that supplies the
duck-typed ``list_articulation_roots`` / ``get_gravity_magnitude`` hooks so the
USD-scanning fallback (TODO(probe:guard-live)) is never touched offline.
"""

import importlib.util
import os
import sys

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
    "handlers",
    "_guards.py",
)
EXTENSION_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
    "extension.py",
)


def _load_guards():
    spec = importlib.util.spec_from_file_location("isaac_mcp_guards_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: on Python 3.9, dataclasses resolve string annotations
    # (from __future__ import annotations) via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_guards()

_ROBOT = "/World/G1"
_ROOTS = (_ROBOT,)


def _grav_vec(gravity, up_axis="Z"):
    """Map a scalar strength to a straight-down gravity VECTOR along ``up_axis``
    (``None`` stays ``None`` = unreadable). Lets the scalar-parametrised gate tests
    below feed WorldState.gravity, which is now the full vector the gate checks."""
    if gravity is None:
        return None
    down = 1 if str(up_axis).upper() == "Y" else 2
    vec = [0.0, 0.0, 0.0]
    vec[down] = -float(gravity)
    return tuple(vec)


def _state(timeline="playing", gravity=9.81, dt=1.0 / 60.0, roots=_ROOTS, up_axis="Z"):
    return G.WorldState(
        timeline_state=timeline,
        gravity=_grav_vec(gravity, up_axis),
        physics_dt=dt,
        robot_roots=roots,
        up_axis=up_axis,
    )


def _is_reject(decision):
    return decision is not None and decision["status"] == "error" and decision["rejected_by"] == "session_guard"


# ── (B) pose / DOF-state writes to a robot root while timeline is locked ──────


def test_pose_write_to_robot_root_rejected_while_playing():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT}, _state(timeline="playing"))
    assert _is_reject(d)


def test_pose_write_to_robot_root_rejected_while_paused():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT}, _state(timeline="paused"))
    assert _is_reject(d)


def test_pose_write_to_robot_descendant_rejected_while_paused():
    d = G.evaluate_command("transform", {"prim_path": _ROBOT + "/pelvis"}, _state(timeline="paused"))
    assert _is_reject(d)


def test_pose_write_to_robot_root_allowed_while_stopped():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT}, _state(timeline="stopped"))
    assert d is None


def test_pose_write_to_non_robot_allowed_while_playing():
    d = G.evaluate_command("objects.transform", {"prim_path": "/World/Cube"}, _state(timeline="playing"))
    assert d is None


def test_pose_write_fails_closed_when_roots_unknown():
    # Discovery failed (roots=None) => cannot prove target is not a robot => reject.
    d = G.evaluate_command("objects.transform", {"prim_path": "/World/Cube"}, _state(timeline="playing", roots=None))
    assert _is_reject(d)


def test_pose_write_fails_closed_when_timeline_unknown():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT}, _state(timeline=None))
    assert _is_reject(d)


def test_dof_state_write_rejected_while_paused():
    d = G.evaluate_command("robots.set_dof_state", {"prim_path": _ROBOT}, _state(timeline="paused"))
    assert _is_reject(d)


# ── (A) hard-block is_kinematic / disableGravity on a robot body ──────────────


def test_is_kinematic_true_on_robot_rejected_even_when_stopped():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT, "is_kinematic": True}, _state(timeline="stopped"))
    assert _is_reject(d)
    assert "is_kinematic" in d["message"]


def test_disable_gravity_on_robot_rejected():
    d = G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT + "/base", "disableGravity": True}, _state())
    assert _is_reject(d)
    assert "disableGravity" in d["message"]


def test_kinematic_true_on_non_robot_allowed():
    d = G.evaluate_command("objects.transform", {"prim_path": "/World/Cube", "is_kinematic": True}, _state(timeline="stopped"))
    assert d is None


def test_kinematic_flag_fails_closed_without_target():
    d = G.evaluate_command("objects.transform", {"is_kinematic": True}, _state(timeline="stopped"))
    assert _is_reject(d)


def test_kinematic_string_true_is_truthy():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT, "kinematic_enabled": "true"}, _state(timeline="stopped"))
    assert _is_reject(d)


def test_kinematic_false_is_not_blocked():
    d = G.evaluate_command("objects.transform", {"prim_path": _ROBOT, "is_kinematic": False}, _state(timeline="stopped"))
    assert d is None  # timeline stopped + no truthy danger flag + pose allowed when stopped


# ── (C) actuation / step gate ─────────────────────────────────────────────────


def test_step_rejected_when_stopped():
    assert _is_reject(G.evaluate_command("simulation.step", {}, _state(timeline="stopped")))


def test_step_rejected_when_timeline_unknown():
    assert _is_reject(G.evaluate_command("simulation.step", {}, _state(timeline=None)))


def test_actuation_rejected_when_gravity_zero():
    assert _is_reject(G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT}, _state(gravity=0.0)))


def test_actuation_rejected_when_gravity_unknown():
    assert _is_reject(G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT}, _state(gravity=None)))


def test_actuation_rejected_when_physics_dt_nonpositive():
    assert _is_reject(G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT}, _state(dt=0.0)))
    assert _is_reject(G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT}, _state(dt=None)))


def test_actuation_allowed_when_physics_live():
    # NB: set_joints on a robot root while playing is a PD *target* (probe:effort-additive),
    # so once physics is live it is permitted.
    assert G.evaluate_command("robots.set_joints", {"prim_path": _ROBOT}, _state()) is None


def test_step_allowed_when_physics_live():
    assert G.evaluate_command("simulation.step", {"num_steps": 4}, _state()) is None


# ── uncategorised commands are never rejected by the pure core ────────────────


def test_readonly_command_allowed():
    assert G.evaluate_command("scene.get_info", {}, _state(timeline=None, gravity=None, dt=None, roots=None)) is None


# ── pure pose-jump helper ─────────────────────────────────────────────────────


def test_max_position_delta_detects_jump():
    before = {"/World/G1": (0.0, 0.0, 1.0)}
    after = {"/World/G1": (0.0, 0.0, 1.5)}
    assert abs(G.max_position_delta(before, after) - 0.5) < 1e-9


def test_max_position_delta_ignores_unmatched_roots():
    assert G.max_position_delta({"/a": (0, 0, 0)}, {"/b": (9, 9, 9)}) == 0.0


# ── SessionGuard wrapper + stateful pause/play continuity ─────────────────────


class _FakeAdapter:
    def __init__(self, timeline="playing", gravity=9.81, dt=1.0 / 60.0, roots=(_ROBOT,), poses=None):
        self._timeline = timeline
        self._gravity = gravity
        self._dt = dt
        self._roots = roots
        self._poses = poses or {_ROBOT: [0.0, 0.0, 1.0]}

    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": self._dt}

    def list_articulation_roots(self):
        return self._roots

    def get_gravity_magnitude(self):
        return self._gravity

    def get_prim_transform(self, prim_path):
        return {"position": self._poses[prim_path]}


def test_session_guard_disabled_allows_everything():
    guard = G.SessionGuard(_FakeAdapter(timeline="stopped"), session_mode=False)
    assert guard.guard("objects.transform", {"prim_path": _ROBOT}) is None
    assert guard.guard("simulation.step", {}) is None


def test_session_guard_rejects_teleport_while_playing():
    guard = G.SessionGuard(_FakeAdapter(timeline="playing"))
    assert _is_reject(guard.guard("objects.transform", {"prim_path": _ROBOT}))


def test_session_guard_pause_then_teleport_then_play_rejected():
    adapter = _FakeAdapter(timeline="playing")
    guard = G.SessionGuard(adapter)
    # Pause snapshots the current root pose.
    assert guard.guard("simulation.pause", {}) is None
    adapter._timeline = "paused"
    # Someone moves the robot behind the guard's back (e.g. via an unclassified path).
    adapter._poses[_ROBOT] = [0.0, 0.0, 5.0]
    # Resume must be refused because the root jumped while paused.
    assert _is_reject(guard.guard("simulation.play", {}))


def test_session_guard_pause_then_play_ok_when_still():
    adapter = _FakeAdapter(timeline="playing")
    guard = G.SessionGuard(adapter)
    assert guard.guard("simulation.pause", {}) is None
    adapter._timeline = "paused"
    assert guard.guard("simulation.play", {}) is None


def test_session_guard_play_rejected_when_root_unreadable_on_resume():
    # A root captured at pause that cannot be re-read on resume (deleted/renamed/read
    # failure) must FAIL CLOSED — the old code skipped it via `continue`, dropping it
    # from the delta comparison and masking a possible teleport.
    adapter = _FakeAdapter(timeline="playing")
    guard = G.SessionGuard(adapter)
    assert guard.guard("simulation.pause", {}) is None
    adapter._timeline = "paused"
    adapter._poses = {}  # get_prim_transform(_ROBOT) now raises -> unreadable
    assert _is_reject(guard.guard("simulation.play", {}))


def test_session_guard_play_rejected_when_snapshot_shrank():
    # The resume snapshot has fewer roots than the pause snapshot => tamper.
    adapter = _FakeAdapter(timeline="playing")
    guard = G.SessionGuard(adapter)
    assert guard.guard("simulation.pause", {}) is None
    adapter._timeline = "paused"
    adapter._roots = ()  # the paused root is no longer discoverable
    assert _is_reject(guard.guard("simulation.play", {}))


def test_session_guard_play_stays_rejected_on_retry_until_stop():
    # A rejected resume keeps the snapshot so retries also fail; a stop clears it.
    adapter = _FakeAdapter(timeline="playing")
    guard = G.SessionGuard(adapter)
    assert guard.guard("simulation.pause", {}) is None
    adapter._timeline = "paused"
    adapter._poses[_ROBOT] = [0.0, 0.0, 5.0]
    assert _is_reject(guard.guard("simulation.play", {}))
    assert _is_reject(guard.guard("simulation.play", {}))  # retry still closed
    assert guard.guard("simulation.stop", {}) is None  # stop resets the snapshot


def test_session_guard_internal_error_fails_closed():
    class _Boom:
        def get_simulation_state(self):
            raise RuntimeError("boom")

        def list_articulation_roots(self):
            raise RuntimeError("boom")

        def get_gravity_magnitude(self):
            raise RuntimeError("boom")

    # A totally broken adapter must not fall open on a guarded command.
    guard = G.SessionGuard(_Boom())
    assert _is_reject(guard.guard("simulation.step", {}))


# ── actuation gate reads the LIVE gravity VECTOR (not just a magnitude) ───────
#
# The gate must reject the SAME non-physical gravities the write-guard rejects —
# zero / tiny / sideways / upward / diagonal — applied to the gravity READ back off
# the live PhysicsContext, so a session cannot set a fake-balance gravity and then
# actuate to produce honest-looking telemetry. Earth gravity is allowed.
# TODO(probe:guard-live): re-run each of these against a real articulation root.

_ACTUATION = ("robots.set_joints", "robots.g1.set_joint_command", "robots.g1.set_gains", "simulation.step")

_EARTH_VEC = (0.0, 0.0, -9.81)
_NON_PHYSICAL_VECS = {
    "zero": (0.0, 0.0, 0.0),
    "tiny": (0.0, 0.0, -0.5),
    "sideways": (9.81, 0.0, 0.0),
    "upward": (0.0, 0.0, 9.81),
    "diagonal": (7.0, 0.0, -7.0),  # ~9.9 magnitude but only ~45deg from horizontal
    "too_strong": (0.0, 0.0, -20.0),
    "too_weak": (0.0, 0.0, -8.0),
}


class _GravityAdapter:
    """Adapter that reports a gravity VECTOR via the get_gravity_vector() hook the
    live guard prefers — the on-box actuation READ path, exercised offline."""

    def __init__(self, gravity_vector, up_axis="Z", timeline="playing", dt=1.0 / 200.0):
        self._g = gravity_vector
        self._up = up_axis
        self._timeline = timeline
        self._dt = dt

    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": self._dt}

    def list_articulation_roots(self):
        return (_ROBOT,)

    def get_gravity_vector(self):
        return self._g

    def get_stage_up_axis(self):
        return self._up

    def get_prim_transform(self, prim_path):
        return {"position": [0.0, 0.0, 1.0]}


def test_actuation_gate_rejects_non_physical_live_gravity():
    for name, g in _NON_PHYSICAL_VECS.items():
        guard = G.SessionGuard(_GravityAdapter(g))
        for cmd in _ACTUATION:
            assert _is_reject(guard.guard(cmd, {"prim_path": _ROBOT})), (name, cmd)


def test_actuation_gate_allows_earth_live_gravity():
    guard = G.SessionGuard(_GravityAdapter(_EARTH_VEC))
    for cmd in _ACTUATION:
        assert guard.guard(cmd, {"prim_path": _ROBOT}) is None, cmd
    # A small off-axis tilt is still allowed.
    tilted = G.SessionGuard(_GravityAdapter((0.3, -0.2, -9.8)))
    assert tilted.guard("robots.g1.set_joint_command", {"prim_path": _ROBOT}) is None


def test_actuation_gate_honours_y_up_for_live_gravity():
    # On a Y-up stage straight-down is -Y; a -Z earth vector is now "sideways".
    down_y = G.SessionGuard(_GravityAdapter((0.0, -9.81, 0.0), up_axis="Y"))
    assert down_y.guard("robots.g1.set_joint_command", {"prim_path": _ROBOT}) is None
    down_z = G.SessionGuard(_GravityAdapter(_EARTH_VEC, up_axis="Y"))
    assert _is_reject(down_z.guard("robots.g1.set_joint_command", {"prim_path": _ROBOT}))


def test_actuation_gate_fails_closed_when_live_gravity_unreadable():
    guard = G.SessionGuard(_GravityAdapter(None))
    assert _is_reject(guard.guard("simulation.step", {}))


def test_actuation_gate_and_write_guard_share_one_envelope():
    # The gate reason for a read gravity is produced by the SAME predicate the
    # write-guard uses, so the two can never drift apart.
    for _name, g in _NON_PHYSICAL_VECS.items():
        assert G._gravity_envelope_problem(g, "Z") is not None
        assert G._gravity_rejection(g, "Z") is not None
    assert G._gravity_envelope_problem(_EARTH_VEC, "Z") is None
    assert G._gravity_rejection(_EARTH_VEC, "Z") is None


# ── session-mode env selection ────────────────────────────────────────────────


def test_session_mode_default_on(monkeypatch=None):
    old = os.environ.pop("MCP_SESSION_MODE", None)
    try:
        assert G.session_mode_enabled() is True
    finally:
        if old is not None:
            os.environ["MCP_SESSION_MODE"] = old


def test_session_mode_explicit_off():
    old = os.environ.get("MCP_SESSION_MODE")
    try:
        os.environ["MCP_SESSION_MODE"] = "0"
        assert G.session_mode_enabled() is False
        os.environ["MCP_SESSION_MODE"] = "false"
        assert G.session_mode_enabled() is False
    finally:
        if old is None:
            os.environ.pop("MCP_SESSION_MODE", None)
        else:
            os.environ["MCP_SESSION_MODE"] = old


def test_session_token_reads_env():
    old = os.environ.get("MCP_SESSION_TOKEN")
    try:
        os.environ.pop("MCP_SESSION_TOKEN", None)
        assert G.session_token() is None
        os.environ["MCP_SESSION_TOKEN"] = "s3cret"
        assert G.session_token() == "s3cret"
    finally:
        if old is None:
            os.environ.pop("MCP_SESSION_TOKEN", None)
        else:
            os.environ["MCP_SESSION_TOKEN"] = old


def test_operator_token_reads_env():
    old = os.environ.get("MCP_OPERATOR_TOKEN")
    try:
        os.environ.pop("MCP_OPERATOR_TOKEN", None)
        assert G.operator_token() is None
        os.environ["MCP_OPERATOR_TOKEN"] = "op-secret"
        assert G.operator_token() == "op-secret"
    finally:
        if old is None:
            os.environ.pop("MCP_OPERATOR_TOKEN", None)
        else:
            os.environ["MCP_OPERATOR_TOKEN"] = old


# ── per-command capability (operator bypasses the session pose/actuation lock) ──


def test_current_capability_defaults_to_session():
    assert G.current_capability() == G.SESSION_CAPABILITY


def test_set_current_capability_normalises_unknown_to_session():
    token = G.set_current_capability("bogus")
    try:
        assert G.current_capability() == G.SESSION_CAPABILITY
    finally:
        G.reset_current_capability(token)


def test_set_and_reset_current_capability_operator():
    token = G.set_current_capability("operator")
    try:
        assert G.current_capability() == G.OPERATOR_CAPABILITY
    finally:
        G.reset_current_capability(token)
    assert G.current_capability() == G.SESSION_CAPABILITY


def test_guard_operator_capability_bypasses_pose_lock_while_playing():
    guard = G.SessionGuard(_FakeAdapter(timeline="playing"))
    assert guard.guard("objects.transform", {"prim_path": _ROBOT}, capability="operator") is None


def test_guard_session_capability_enforces_pose_lock_while_playing():
    guard = G.SessionGuard(_FakeAdapter(timeline="playing"))
    assert _is_reject(guard.guard("objects.transform", {"prim_path": _ROBOT}, capability="session"))


def test_guard_operator_capability_bypasses_actuation_gate_when_stopped():
    # Operator (trusted bootstrap) may actuate/step regardless of physics liveness.
    guard = G.SessionGuard(_FakeAdapter(timeline="stopped", gravity=0.0, dt=0.0))
    assert guard.guard("simulation.step", {}, capability="operator") is None


def test_guard_reads_current_capability_when_arg_omitted():
    guard = G.SessionGuard(_FakeAdapter(timeline="playing"))
    token = G.set_current_capability("operator")
    try:
        # No explicit capability arg: it must fall back to the in-flight contextvar.
        assert guard.guard("objects.transform", {"prim_path": _ROBOT}) is None
    finally:
        G.reset_current_capability(token)


def test_pose_write_operator_capability_bypasses_sink_lock():
    # guard_pose_write is the sink-level lock used by objects.create/clone.
    assert (
        G.guard_pose_write(_FakeAdapter(timeline="playing"), _ROBOT, capability="operator") is None
    )


def test_pose_write_session_capability_enforces_sink_lock():
    assert _is_reject(
        G.guard_pose_write(_FakeAdapter(timeline="playing"), _ROBOT, capability="session")
    )


# ── live-probe marker + extension hardening (source-level) ────────────────────


def test_guard_module_marks_probe_todo():
    with open(_MODULE_PATH) as f:
        assert "TODO(probe:guard-live)" in f.read()


def test_extension_enforces_capability_at_dispatch():
    with open(EXTENSION_PATH) as f:
        src = f.read()
    # The process-global startup pop is gone; the full registry is kept and the
    # operator-only exec surface is refused at dispatch for session capability.
    assert "_apply_session_mode_restrictions" not in src
    assert "_OPERATOR_ONLY_COMMANDS" in src
    assert "simulation.execute_script" in src
    assert "simulation.reload_script" in src
    assert "forbidden in session mode" in src
    # create_action_graph ScriptNode/script-input surface is refused for session.
    assert "script_file" in src
    # The capability is threaded into the guard before dispatch.
    assert "self._guard.guard(" in src
    assert "capability" in src


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
