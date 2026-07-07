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

"""No-GPU checks for the v5 LowCmd per-motor actuation contract + observability.

Validates the pure logic the new v5 methods rely on (g1.gains unit conversion,
effort/velocity clamps, fail-closed effort) and structurally asserts that
set_gains / set_joint_velocities / set_joint_efforts / get_lowstate_fields /
detect_motion / the contact-report reader exist, that the runtime-probe TODOs
still open (imu, motor-space, contacts) are present while the CONFIRMED ones
(effort-additive, gain-units) are resolved, and that the safety guards
(fail-closed effort, velocity clamp wiring, non-stubbed contacts) are wired in
by code path rather than advisory check. Runs under plain ``python3`` — no
isaacsim, no omni, no numpy.
"""

import ast
import importlib.util
import math
import os
import sys
import types

EXTENSION_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
)
V5_PATH = os.path.join(EXTENSION_ROOT, "adapters", "v5.py")
GAINS_PATH = os.path.join(EXTENSION_ROOT, "g1", "gains.py")
G1_DIR = os.path.join(EXTENSION_ROOT, "g1")


def _load_g1_pkg():
    """Load the g1 codec + driver under a synthetic ``g1pkg`` package (relative
    imports resolve) so the driver runs under plain python3 -- no omni/numpy."""
    if "g1pkg" not in sys.modules:
        pkg = types.ModuleType("g1pkg")
        pkg.__path__ = [G1_DIR]
        sys.modules["g1pkg"] = pkg
        for name in ("crc32", "gains", "motor_index_map", "unitree_hg", "g1_lowcmd_driver"):
            spec = importlib.util.spec_from_file_location(
                f"g1pkg.{name}", os.path.join(G1_DIR, f"{name}.py")
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"g1pkg.{name}"] = module
            spec.loader.exec_module(module)
    return sys.modules


def _load_gains():
    """Import g1/gains.py directly (bypasses the omni-importing package __init__)."""
    spec = importlib.util.spec_from_file_location("_g1_gains_under_test", GAINS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _v5_source():
    with open(V5_PATH) as f:
        return f.read()


def _v5_functions():
    return {
        n.name: n
        for n in ast.walk(ast.parse(_v5_source()))
        if isinstance(n, ast.FunctionDef)
    }


def _v5_function_src(name):
    """Return the exact source text of a v5 method (for code-path assertions)."""
    src = _v5_source()
    return ast.get_source_segment(src, _v5_functions()[name])


def _extract_pure_function(name):
    """Return the callable for a self-contained (import-free) v5 function/staticmethod."""
    import typing

    node = _v5_functions()[name]
    node.decorator_list = []
    # Annotations reference typing names (Sequence/List) evaluated at def time.
    ns: dict = {"Sequence": typing.Sequence, "List": typing.List, "Optional": typing.Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), V5_PATH, "exec"), ns)
    return ns[name]


# ── g1.gains: the load-bearing numeric core of set_gains / set_joint_efforts ──


def test_gain_degree_conversion_matches_pi_over_180():
    g = _load_gains()
    # USD angular DriveAPI is per-DEGREE: stiffness_deg = kp_rad * (pi/180).
    assert math.isclose(g.kp_to_drive_stiffness(100.0), 100.0 * math.pi / 180.0)
    assert math.isclose(g.kd_to_drive_damping(5.0), 5.0 * math.pi / 180.0)
    # The tensor/controller path is radian-based -> identity.
    assert g.kp_to_tensor_stiffness(100.0) == 100.0
    assert g.kd_to_tensor_damping(5.0) == 5.0


def test_effort_clamp_uses_official_limits():
    g = _load_gains()
    assert g.clamp_effort("left_knee", 500.0) == 139.0
    assert g.clamp_effort("right_wrist_pitch", -9.0) == -5.0
    assert g.clamp_effort("left_hip_pitch", 10.0) == 10.0  # within limit, untouched
    assert len(g.G1_JOINT_LIMITS) == 29


def test_effort_clamp_name_normalisation():
    """v5._g1_effort_clamp strips a trailing _joint; verify the canonical name maps."""
    g = _load_gains()
    dof_name = "left_knee_joint"
    canonical = dof_name[: -len("_joint")]
    assert canonical in g.G1_JOINT_LIMITS
    assert g.clamp_effort(canonical, 1e9) == g.G1_JOINT_LIMITS[canonical].effort


def test_effort_clamp_fail_closed_zeros_unknown_joints():
    """Prep-fix 3: unmapped joints are ZEROED, mapped joints clamp normally."""
    g = _load_gains()
    # Mapped joint: normal symmetric clamp.
    assert g.clamp_effort_fail_closed("left_knee", 500.0) == 139.0
    assert g.clamp_effort_fail_closed("right_wrist_pitch", -9.0) == -5.0
    assert g.clamp_effort_fail_closed("left_hip_pitch", 10.0) == 10.0
    # Unmapped joints (misnamed, or a Dex3 hand DoF absent from the table) -> 0.0,
    # never a raw pass-through. Fail CLOSED.
    assert g.clamp_effort_fail_closed("nonexistent_joint", 999.0) == 0.0
    assert g.clamp_effort_fail_closed("left_hand_index_0", 42.0) == 0.0
    # clamp_effort itself still fails loud (KeyError) on unknown names.
    try:
        g.clamp_effort("nonexistent_joint", 1.0)
    except KeyError:
        pass
    else:
        raise AssertionError("clamp_effort must raise KeyError on unknown joint")


def test_velocity_clamp_uses_official_limits():
    """Prep-fix 2 numeric core: clamp_velocity honours the per-joint ceiling."""
    g = _load_gains()
    assert g.clamp_velocity("left_hip_yaw", 50.0) == 32.0
    assert g.clamp_velocity("left_hip_yaw", -50.0) == -32.0
    assert g.clamp_velocity("left_knee", 100.0) == 20.0  # knee ceiling is 20 rad/s
    assert g.clamp_velocity("left_wrist_pitch", 5.0) == 5.0  # within limit, untouched


def test_hip_roll_reconciled_to_usd_maxforce():
    """probe-results.md: hip_roll USD maxForce = 139 (was 88 in the Phase-0 URDF)."""
    g = _load_gains()
    assert g.G1_JOINT_LIMITS["left_hip_roll"].effort == 139.0
    assert g.G1_JOINT_LIMITS["right_hip_roll"].effort == 139.0
    assert g.clamp_effort("left_hip_roll", 200.0) == 139.0


# ── v5 structure + probe TODOs ───────────────────────────────────────────────


def test_new_actuation_methods_exist():
    fns = _v5_functions()
    for name in (
        "set_gains",
        "set_joint_velocities",
        "set_joint_efforts",
        "get_lowstate_fields",
        "_g1_effort_clamp",
        "_g1_velocity_clamp",
        "_projected_gravity",
        "detect_motion",
        "_read_contact_forces",
        "_ensure_contact_reporting",
    ):
        assert name in fns, f"v5 missing {name}"


def test_open_probe_todos_present():
    """Still-unconfirmed runtime items MUST keep their TODO(probe:) marker."""
    src = _v5_source()
    for marker in (
        "TODO(probe:motor-space)",
        "TODO(probe:imu)",
        "TODO(probe:contacts)",
    ):
        assert marker in src, f"missing {marker}"


def test_confirmed_probes_resolved():
    """Items CONFIRMED by the live probe must NOT carry a TODO(probe:) marker."""
    src = _v5_source()
    for marker in ("TODO(probe:effort-additive)", "TODO(probe:gain-units)"):
        assert marker not in src, f"{marker} should be resolved after probe"
    g_src = open(GAINS_PATH).read()
    assert "TODO(probe:gain-units)" not in g_src


def test_effort_additive_no_override_hedging():
    """The 'override' hedging is gone; the additive fact is cited from the probe."""
    src = _v5_source()
    assert "OVERRIDE" not in src
    assert "must be redesigned" not in src
    # get_lowstate_fields + set_joint_efforts now assert additivity, not doubt.
    efforts = _v5_function_src("set_joint_efforts")
    assert "CONFIRMED" in efforts
    lowstate = _v5_function_src("get_lowstate_fields")
    assert "UNCONFIRMED" not in lowstate
    assert "additive" in lowstate.lower()


def test_effort_clamp_fail_closed_by_code_path():
    """_g1_effort_clamp must zero unmapped joints (no raw ``return tau``)."""
    body = _v5_function_src("_g1_effort_clamp")
    assert "clamp_effort_fail_closed" in body
    assert "return tau" not in body  # the old fail-open pass-through is gone
    assert "return 0.0" in body


def test_set_joint_velocities_clamped_by_code_path():
    """Prep-fix 2 wiring: velocities pass through _g1_velocity_clamp before apply."""
    body = _v5_function_src("set_joint_velocities")
    assert "_g1_velocity_clamp" in body
    # both the tensor action and the USD fallback must use the clamped values.
    assert "joint_velocities=np.array(dq)" in body
    assert "_set_joint_drive_velocities(prim_path, dq" in body


def test_contacts_not_stubbed():
    """get_physics_state must read real PhysX contacts, not return the [] stub."""
    body = _v5_function_src("get_physics_state")
    assert "self._read_contact_forces(prim_path)" in body
    reader = _v5_function_src("_read_contact_forces")
    assert "get_contact_report" in reader


def test_watchdog_uses_runtime_reads_only():
    """detect_motion must read PhysX runtime state, never authored USD targets."""
    body = _v5_function_src("detect_motion")
    assert "get_joint_positions" in body
    assert "get_rigidbody_transformation" in body
    # It must not consult drive targets / the USD authoring path.
    assert "GetTargetPositionAttr" not in body
    assert "_set_joint_drive_targets" not in body
    assert "physx_runtime" in body


def test_position_only_behavior_preserved():
    """set_joint_positions must still exist untouched (additive-only change)."""
    fns = _v5_functions()
    assert "set_joint_positions" in fns
    assert "_set_joint_drive_targets" in fns


# ── pure projected-gravity math ──────────────────────────────────────────────


def test_projected_gravity_upright_and_flipped():
    pg = _extract_pure_function("_projected_gravity")
    # Identity orientation -> gravity stays down in the body frame.
    up = pg((1.0, 0.0, 0.0, 0.0))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(up, (0.0, 0.0, -1.0)))
    # 180 deg about x -> body-frame gravity flips to +z.
    flipped = pg((0.0, 1.0, 0.0, 0.0))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(flipped, (0.0, 0.0, 1.0)))
    # 90 deg about y (w=cos45, y=sin45) -> world down maps to +x of the body.
    s = math.sqrt(0.5)
    ninety = pg((s, 0.0, s, 0.0))
    assert all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(ninety, (1.0, 0.0, 0.0))
    )


# ── driver: GIL/perf split (physics tick pays no pack/CRC) + inbound validation ──


def test_submit_lowcmd_validates_on_producer_thread():
    """FIX 2/3: submit_lowcmd decodes + CRC/mode-verifies (off the physics thread),
    rejects invalid frames, publishes a DECODED LowCmd, and holds a producer lock."""
    mods = _load_g1_pkg()
    drv_mod = mods["g1pkg.g1_lowcmd_driver"]
    hg = mods["g1pkg.unitree_hg"]

    drv = drv_mod.G1LowCmdDriver(enable_dds=False)
    # The two inbound producers are serialized by a real lock (single-writer seqlock).
    assert hasattr(drv._submit_lock, "acquire") and hasattr(drv._submit_lock, "release")

    lc = hg.LowCmd(mode_pr=int(hg.ModePR.PR), mode_machine=0)
    lc.with_crc()
    assert drv.submit_lowcmd(lc.pack()) is True
    _seq, cmd = drv._inbound.read()
    # The seqlock now carries a DECODED command, not raw bytes (codec off the tick).
    assert isinstance(cmd, hg.LowCmd)

    assert drv.submit_lowcmd(b"\x00" * 10) is False           # bad length
    bad = bytearray(lc.pack())
    bad[-1] ^= 0xFF
    assert drv.submit_lowcmd(bytes(bad)) is False             # bad CRC
    ab = hg.LowCmd(mode_pr=int(hg.ModePR.AB), mode_machine=0)
    ab.with_crc()
    assert drv.submit_lowcmd(ab.pack()) is False              # wrong mode_pr
    mm = hg.LowCmd(mode_pr=int(hg.ModePR.PR), mode_machine=7)
    mm.with_crc()
    assert drv.submit_lowcmd(mm.pack()) is False              # wrong mode_machine


def test_physics_callback_does_no_pack_or_crc():
    """FIX 1: the physics tick applies + snapshots only -- it must NOT struct.pack or
    CRC. The pack+CRC is proven to live on the packer path instead (counter spies)."""
    mods = _load_g1_pkg()
    drv_mod = mods["g1pkg.g1_lowcmd_driver"]
    hg = mods["g1pkg.unitree_hg"]

    drv = drv_mod.G1LowCmdDriver(enable_dds=False)

    class _FakeNp:
        float64 = float
        int64 = int
        nan = float("nan")

        @staticmethod
        def array(seq, dtype=None):
            return list(seq)

        @staticmethod
        def array_equal(a, b):
            return True  # keep the set-gains-on-change branch closed (no scatter)

        @staticmethod
        def asarray(seq, dtype=None):
            return list(seq)

    class _FakeArt:
        def get_joint_positions(self):
            return [0.1 * i for i in range(29)]

        def get_joint_velocities(self):
            return [0.0] * 29

        def get_measured_joint_efforts(self):
            return [0.0] * 29

        def get_world_pose(self):
            return ([0.0, 0.0, 0.0], (1.0, 0.0, 0.0, 0.0))

        def get_angular_velocity(self):
            return (0.0, 0.0, 0.0)

        def get_linear_velocity(self):
            return (0.0, 0.0, 0.0)

    class _FakeController:
        def apply_action(self, action):
            pass

    drv._np = _FakeNp()
    drv._ArticulationAction = lambda **kw: kw
    drv._art = _FakeArt()
    drv._controller = _FakeController()
    drv._physx = None
    drv._prim_path = None
    drv._u2i = tuple(range(29))
    drv._joint_indices = list(range(29))
    drv._q = [0.0] * 29
    drv._dq = [0.0] * 29
    drv._tau = [0.0] * 29
    drv._kp = [0.0] * 29
    drv._kd = [0.0] * 29
    drv._applied_kp = [0.0] * 29
    drv._applied_kd = [0.0] * 29
    drv._full_kp = None
    drv._full_kd = None
    drv._ls = hg.LowState()
    drv._installed = True
    drv._started = True
    drv._estopped = False

    # Valid inbound command, submitted (with its CRC) on the PRODUCER thread.
    lc = hg.LowCmd(mode_pr=int(hg.ModePR.PR), mode_machine=0)
    for u in range(29):
        lc.motor_cmd[u].q = 0.01 * u
    lc.with_crc()
    assert drv.submit_lowcmd(lc.pack()) is True

    # Spy the codec AFTER the producer-side submit so we only measure the tick.
    counts = {"crc": 0, "pack": 0}
    real_crc = drv_mod.crc32_of_struct
    real_struct = drv_mod.struct

    def _crc_spy(buf):
        counts["crc"] += 1
        return real_crc(buf)

    class _StructSpy:
        def pack(self, *a, **k):
            counts["pack"] += 1
            return real_struct.pack(*a, **k)

        def __getattr__(self, name):
            return getattr(real_struct, name)

    drv_mod.crc32_of_struct = _crc_spy
    drv_mod.struct = _StructSpy()
    try:
        for _ in range(5):
            drv._on_physics_step(0.0)
        # The physics tick paid NO pack/CRC cost and never wrote the outbound frame.
        assert counts == {"crc": 0, "pack": 0}, counts
        assert drv.latest_lowstate_bytes() is None
        assert drv._n_applied == 1  # one new frame applied, then latched

        # The packer path (off the physics thread) is where pack + CRC actually run.
        _seq, snap = drv._state.read()
        assert snap is not None
        packed = drv._pack_lowstate_from_snapshot(snap)
        assert counts["crc"] >= 1 and counts["pack"] >= 1, counts
        assert len(packed) == hg.LOWSTATE_SIZE
        drv._outbound.write(packed)
        assert drv.latest_lowstate_bytes() is not None
    finally:
        drv_mod.crc32_of_struct = real_crc
        drv_mod.struct = real_struct


def test_imu_gyro_and_rpy_are_body_frame():
    """FIX 4: gyroscope = world angular velocity rotated into the base frame; rpy is
    derived from the quaternion (pure math, matches v5 _projected_gravity convention)."""
    mods = _load_g1_pkg()
    drv_mod = mods["g1pkg.g1_lowcmd_driver"]
    rot = drv_mod._quat_rotate_inverse
    rpy = drv_mod._quat_to_rpy

    def _close(a, b):
        return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))

    # Upright base -> world angular velocity passes through unchanged.
    assert _close(rot((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), (1.0, 2.0, 3.0))
    # 180 deg about x -> y/z components flip in the body frame.
    assert _close(rot((0.0, 1.0, 0.0, 0.0), (1.0, 2.0, 3.0)), (1.0, -2.0, -3.0))
    # rpy: identity is level; 180 about x is roll = ±pi with zero pitch/yaw.
    assert _close(rpy((1.0, 0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
    r = rpy((0.0, 1.0, 0.0, 0.0))
    assert math.isclose(abs(r[0]), math.pi, abs_tol=1e-9)
    assert math.isclose(r[1], 0.0, abs_tol=1e-9)
    assert math.isclose(r[2], 0.0, abs_tol=1e-9)


def test_gain_seed_fallback_never_zeros_hand_dofs():
    """FIX 5: when get_gains() is unavailable, _write_gains must scatter ONLY the 29
    body indices (indexed set_gains) -- never a fabricated full array over the hands."""
    mods = _load_g1_pkg()
    drv_mod = mods["g1pkg.g1_lowcmd_driver"]

    drv = drv_mod.G1LowCmdDriver(enable_dds=False)

    class _FakeNp:
        float64 = float

        @staticmethod
        def array(seq, dtype=None):
            return list(seq)

        @staticmethod
        def asarray(seq, dtype=None):
            return list(seq)

    calls = {}

    class _FakeController:
        def get_gains(self):
            raise RuntimeError("gains unavailable")  # force the fallback

        def set_gains(self, kps=None, kds=None, joint_indices=None):
            calls["kps"] = kps
            calls["joint_indices"] = joint_indices

    drv._np = _FakeNp()
    drv._controller = _FakeController()
    drv._joint_indices = list(range(29))
    drv._full_kp = None  # install-time seed unavailable
    drv._full_kd = None

    body_kp = [10.0] * 29
    drv._write_gains(body_kp, [1.0] * 29)
    # Only the 29 body indices were written; NO full-length (43-wide) zeros array
    # that would have clobbered the 14 hand-DoF gains.
    assert calls["joint_indices"] == list(range(29))
    assert len(calls["kps"]) == 29


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
