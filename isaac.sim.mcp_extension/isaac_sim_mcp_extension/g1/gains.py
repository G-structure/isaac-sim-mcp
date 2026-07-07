"""
MIT License

Copyright (c) 2023-2025 omni-mcp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# G1 gain calibration + per-joint effort/velocity limit table.
#
# Two independent concerns live here, both keyed by canonical joint NAME
# (never by DoF/motor index -- the index map is runtime-gated, Phase 2):
#
#   1. Unit conversion between the Unitree control convention (radians) and
#      the USD/PhysX DriveAPI angular convention (DEGREES).
#   2. G1_JOINT_LIMITS: peak effort (N.m) and velocity (rad/s) per joint,
#      sourced from the official G1 URDF, with clamp helpers.
#
# ---------------------------------------------------------------------------
# Angular-unit convention (why the pi/180 factor exists)
# ---------------------------------------------------------------------------
# Unitree LowCmd.MotorCmd carries the PD law in RADIAN units:
#
#       tau = kp * (q_des - q) + kd * (dq_des - dq)      # kp: N.m/rad, kd: N.m.s/rad
#
# A USD ``UsdPhysics.DriveAPI`` of type "angular" evaluates the SAME PD law but
# in DEGREES for target/position and DEGREES/s for target-vel/velocity (this is
# the USD angular convention; confirmed in adapters/v5.py which does
# ``np.degrees(value)`` when authoring a target and ``np.radians(target)`` when
# reading one back). Its stiffness/damping therefore have per-DEGREE units:
#
#       tau = stiffness_deg * (target_deg - pos_deg)
#           + damping_deg   * (targetvel_deg - vel_deg)
#
# Since (target_deg - pos_deg) = (180/pi) * (target_rad - pos_rad), matching the
# produced torque requires:
#
#       stiffness_deg = kp * (pi/180)
#       damping_deg   = kd * (pi/180)
#
# This is exactly what the Isaac Sim URDF importer does when it authors DriveAPI
# gains from a radian-convention URDF.
#
# CONFIRMED (docs/src/design/g1-drive/probe-results.md, Isaac Sim 6.0.0-rc.22):
# the runtime control path is the PhysX TENSOR API
# (articulation_controller.set_gains / apply_action), which is RADIAN-based --
# set_gains(kps, kds) takes N.m/rad and N.m.s/rad DIRECTLY (see examples/g1.py:
# set_gains(kps=[100]*n)). So v5.set_gains passes Unitree kp/kd through unscaled
# (kp_to_tensor_stiffness / kd_to_tensor_damping are identity). The pi/180 DEGREE
# conversion below is used ONLY when authoring the RAW USD DriveAPI
# stiffness/damping attributes (the stopped-sim fallback v5._set_drive_gains);
# it must NEVER be applied on the tensor path or every gain is mis-scaled by
# 180/pi.

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Gain conversions for RAW USD DriveAPI attribute writes ONLY
# (Unitree radian convention -> USD DriveAPI degree convention).
# The runtime tensor/controller path is SI/radian and must NOT use these --
# see kp_to_tensor_stiffness below and the module header.
# ---------------------------------------------------------------------------
def kp_to_drive_stiffness(kp_nm_per_rad: float) -> float:
    """Unitree kp [N.m/rad] -> USD angular DriveAPI stiffness [N.m/deg].

    RAW USD DriveAPI authoring only (v5._set_drive_gains). Never on the tensor
    path (articulation_controller.set_gains), which is radian-based.
    """
    return kp_nm_per_rad * DEG_TO_RAD


def kd_to_drive_damping(kd_nms_per_rad: float) -> float:
    """Unitree kd [N.m.s/rad] -> USD angular DriveAPI damping [N.m.s/deg]."""
    return kd_nms_per_rad * DEG_TO_RAD


def drive_stiffness_to_kp(stiffness_nm_per_deg: float) -> float:
    """USD angular DriveAPI stiffness [N.m/deg] -> Unitree kp [N.m/rad] (readback)."""
    return stiffness_nm_per_deg * RAD_TO_DEG


def drive_damping_to_kd(damping_nms_per_deg: float) -> float:
    """USD angular DriveAPI damping [N.m.s/deg] -> Unitree kd [N.m.s/rad] (readback)."""
    return damping_nms_per_deg * RAD_TO_DEG


def kp_to_tensor_stiffness(kp_nm_per_rad: float) -> float:
    """Unitree kp [N.m/rad] -> PhysX tensor-API stiffness (identity; radian-based).

    The PhysX tensor API (Articulation.set_gains / apply_action) works in
    radians, so no scaling is needed -- this is the CONFIRMED runtime control
    path (probe-results.md). Provided so the caller picks the path explicitly
    (identity) rather than accidentally reusing the DriveAPI degree conversion.
    """
    return kp_nm_per_rad


def kd_to_tensor_damping(kd_nms_per_rad: float) -> float:
    """Unitree kd [N.m.s/rad] -> PhysX tensor-API damping (identity; radian-based)."""
    return kd_nms_per_rad


# ---------------------------------------------------------------------------
# Position / velocity conversions for RAW USD DriveAPI attribute writes ONLY
# (Unitree radians <-> DriveAPI degrees). Runtime tensor reads/writes are radian.
# ---------------------------------------------------------------------------
def q_rad_to_deg(q_rad: float) -> float:
    """Joint position [rad] -> DriveAPI target position [deg]."""
    return q_rad * RAD_TO_DEG


def q_deg_to_rad(q_deg: float) -> float:
    """DriveAPI position [deg] -> joint position [rad] (readback)."""
    return q_deg * DEG_TO_RAD


def dq_rad_to_deg(dq_rad_per_s: float) -> float:
    """Joint velocity [rad/s] -> DriveAPI target velocity [deg/s]."""
    return dq_rad_per_s * RAD_TO_DEG


def dq_deg_to_rad(dq_deg_per_s: float) -> float:
    """DriveAPI velocity [deg/s] -> joint velocity [rad/s] (readback)."""
    return dq_deg_per_s * DEG_TO_RAD


# ---------------------------------------------------------------------------
# Per-joint limit table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JointLimit:
    """Peak actuator limits for one G1 joint (official G1 URDF).

    effort   -- peak torque, N.m
    velocity -- peak joint velocity, rad/s

    Position (angle) limits are intentionally absent: they are not in the
    verified ground truth. TODO(probe:usd-jointlimit) read lower/upper from the
    on-image USD RevoluteJoint (note: USD authors those in DEGREES).
    """

    effort: float
    velocity: float


# Limit spec per joint "leaf" name. Leg + arm groups are mirrored L/R below;
# the waist group has no side prefix. Values are effort[N.m] / velocity[rad/s].
#
# SOURCE: official G1 URDF (see VERIFIED GROUND TRUTH). Do NOT source these from
# design/robot_params.yaml (wrong robot) or the 300/20/300 clamps in
# VisionProTeleop/examples/13_g1_freefall.py (those are PhysX solver clamps).
_LEG_LEAVES: Dict[str, JointLimit] = {
    "hip_pitch": JointLimit(effort=88.0, velocity=32.0),
    # hip_roll effort = 139 (USD maxForce), NOT the 88 the Phase-0 URDF note
    # claimed. Reconciled to the on-image USD, which is authoritative for the sim
    # (probe-results.md, 6.0.0-rc.22: "hip_roll USD = 139 ... USD is authoritative").
    "hip_roll": JointLimit(effort=139.0, velocity=32.0),
    "hip_yaw": JointLimit(effort=88.0, velocity=32.0),
    "knee": JointLimit(effort=139.0, velocity=20.0),
    "ankle_pitch": JointLimit(effort=50.0, velocity=37.0),
    "ankle_roll": JointLimit(effort=50.0, velocity=37.0),
}

_ARM_LEAVES: Dict[str, JointLimit] = {
    "shoulder_pitch": JointLimit(effort=25.0, velocity=37.0),
    "shoulder_roll": JointLimit(effort=25.0, velocity=37.0),
    "shoulder_yaw": JointLimit(effort=25.0, velocity=37.0),
    "elbow": JointLimit(effort=25.0, velocity=37.0),
    "wrist_roll": JointLimit(effort=25.0, velocity=37.0),
    "wrist_pitch": JointLimit(effort=5.0, velocity=22.0),
    "wrist_yaw": JointLimit(effort=5.0, velocity=22.0),
}

_WAIST_JOINTS: Dict[str, JointLimit] = {
    "waist_yaw": JointLimit(effort=88.0, velocity=32.0),
    "waist_roll": JointLimit(effort=50.0, velocity=37.0),
    "waist_pitch": JointLimit(effort=50.0, velocity=37.0),
}


def _build_limits() -> Dict[str, JointLimit]:
    table: Dict[str, JointLimit] = {}
    for side in ("left", "right"):
        for leaf, lim in _LEG_LEAVES.items():
            table[f"{side}_{leaf}"] = lim
        for leaf, lim in _ARM_LEAVES.items():
            table[f"{side}_{leaf}"] = lim
    table.update(_WAIST_JOINTS)
    return table


G1_JOINT_LIMITS: Dict[str, JointLimit] = _build_limits()

# Canonical joint names (alphabetical). This is NOT the motor/DoF index order --
# the ordered motor_index_map is runtime-gated (Phase 2). Key everything by name.
G1_JOINT_NAMES: Tuple[str, ...] = tuple(sorted(G1_JOINT_LIMITS))

# Joints whose per-joint (serial, mode_pr=PR) clamp is only an APPROXIMATION of
# the true actuator-space limit because the physical joint is a parallel linkage.
# The ankle is a confirmed parallel A/B linkage; the waist is INFERRED parallel.
#
# TODO(motor-space): effort/velocity clamps for these must ultimately be applied
# in MOTOR (A/B) space via the linkage Jacobian, not in serial pitch/roll space.
# The coupling matrix has to be extracted from the on-image USD before this can
# be done correctly; until then clamp_effort/clamp_velocity clamp per-joint.
PARALLEL_LINKAGE_JOINTS = frozenset(
    {
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_ankle_pitch",
        "right_ankle_roll",
    }
)
INFERRED_PARALLEL_JOINTS = frozenset({"waist_roll", "waist_pitch", "waist_yaw"})

# The table above is reconciled to the on-image USD drive ``maxForce`` (targetType
# "force") spot-checked by the live probe (probe-results.md): hip_roll=139 (fixed
# from the Phase-0 URDF's 88), knee=139, hip_pitch/hip_yaw/waist_yaw=88,
# ankle/waist_roll/pitch=50, shoulder/elbow/wrist_roll=25, wrist_pitch/yaw=5.
# USD is authoritative for the sim, so it wins on any mismatch. Any per-joint
# override still surfaces through effort_limit() (min of override and table).
USD_MAXFORCE_OVERRIDES: Dict[str, float] = {}  # populated by the runtime probe


# ---------------------------------------------------------------------------
# Lookup + clamp helpers
# ---------------------------------------------------------------------------
def get_limits(joint_name: str) -> JointLimit:
    """Return the JointLimit for a canonical joint name. Raises KeyError if unknown."""
    try:
        return G1_JOINT_LIMITS[joint_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown G1 joint '{joint_name}'. Known joints: {G1_JOINT_NAMES}"
        ) from exc


def effort_limit(joint_name: str) -> float:
    """Peak torque [N.m] for a joint, honoring a probed USD maxForce override."""
    override = USD_MAXFORCE_OVERRIDES.get(joint_name)
    if override is not None:
        return min(override, get_limits(joint_name).effort)
    return get_limits(joint_name).effort


def clamp_effort(joint_name: str, tau: float) -> float:
    """Clamp a commanded torque [N.m] to the joint's symmetric effort limit.

    TODO(motor-space): for PARALLEL_LINKAGE_JOINTS this serial-space clamp is an
    approximation; the true clamp is in A/B motor space via the linkage Jacobian.
    """
    lim = effort_limit(joint_name)
    return max(-lim, min(lim, tau))


def clamp_velocity(joint_name: str, dq: float) -> float:
    """Clamp a commanded velocity [rad/s] to the joint's symmetric velocity limit.

    TODO(motor-space): same parallel-linkage caveat as clamp_effort.
    """
    lim = get_limits(joint_name).velocity
    return max(-lim, min(lim, dq))


def clamp_effort_fail_closed(joint_name: str, tau: float) -> float:
    """FAIL-CLOSED effort clamp: zero the torque for any joint NOT in the limit
    table instead of raising or passing the raw command through.

    A torque command on an unmapped / misnamed joint (e.g. a Dex3 hand DoF that
    carries no verified effort ceiling) is a safety hazard, so it is rejected to
    0.0 rather than trusted. Mapped joints go through the normal symmetric clamp.
    """
    if joint_name in G1_JOINT_LIMITS:
        return clamp_effort(joint_name, tau)
    return 0.0


def is_motor_space_clamp_pending(joint_name: str) -> bool:
    """True if this joint's clamp still needs to move to A/B motor space."""
    return (
        joint_name in PARALLEL_LINKAGE_JOINTS
        or joint_name in INFERRED_PARALLEL_JOINTS
    )


def calibrate_gains(
    kp_nm_per_rad: float, kd_nms_per_rad: float
) -> Tuple[float, float]:
    """Convenience: (kp, kd) radian gains -> (stiffness, damping) DriveAPI degree gains."""
    return kp_to_drive_stiffness(kp_nm_per_rad), kd_to_drive_damping(kd_nms_per_rad)


if __name__ == "__main__":
    # Self-test: runs with plain python3, no GPU / sim required.
    print("=== G1 gain calibration (Unitree radian -> USD DriveAPI degree) ===")
    print(f"RAD_TO_DEG = {RAD_TO_DEG:.6f}   DEG_TO_RAD = {DEG_TO_RAD:.6f}\n")

    for kp, kd in ((100.0, 2.0), (150.0, 4.0), (40.0, 1.0)):
        stiffness, damping = calibrate_gains(kp, kd)
        print(
            f"kp={kp:>6.1f} N.m/rad  kd={kd:>4.1f} N.m.s/rad  ->  "
            f"stiffness={stiffness:.5f} N.m/deg  damping={damping:.5f} N.m.s/deg"
        )
        # Round-trip must recover the radian gains.
        assert math.isclose(drive_stiffness_to_kp(stiffness), kp, rel_tol=1e-12)
        assert math.isclose(drive_damping_to_kd(damping), kd, rel_tol=1e-12)

    print("\n=== Position / velocity conversion round-trip ===")
    for q in (0.5, -1.2, 3.14159):
        assert math.isclose(q_deg_to_rad(q_rad_to_deg(q)), q, rel_tol=1e-12)
    for dq in (10.0, -20.0):
        assert math.isclose(dq_deg_to_rad(dq_rad_to_deg(dq)), dq, rel_tol=1e-12)
    print("q  0.500 rad ->", f"{q_rad_to_deg(0.5):.4f} deg")
    print("dq 20.00 rad/s ->", f"{dq_rad_to_deg(20.0):.4f} deg/s")

    print(f"\n=== G1_JOINT_LIMITS ({len(G1_JOINT_LIMITS)} joints) ===")
    assert len(G1_JOINT_LIMITS) == 29, "expected 29-DoF G1"
    for name in G1_JOINT_NAMES:
        lim = G1_JOINT_LIMITS[name]
        flag = "  [motor-space clamp pending]" if is_motor_space_clamp_pending(name) else ""
        print(f"  {name:<20} effort={lim.effort:>6.1f} N.m   velocity={lim.velocity:>5.1f} rad/s{flag}")

    print("\n=== Clamp examples ===")
    print("clamp_effort(left_knee, 200.0)   =", clamp_effort("left_knee", 200.0))
    print("clamp_effort(right_wrist_pitch, 9.0) =", clamp_effort("right_wrist_pitch", 9.0))
    print("clamp_velocity(left_hip_yaw, -50.0)  =", clamp_velocity("left_hip_yaw", -50.0))
    assert clamp_effort("left_knee", 200.0) == 139.0
    assert clamp_effort("right_wrist_pitch", 9.0) == 5.0
    assert clamp_velocity("left_hip_yaw", -50.0) == -32.0

    print("\nAll self-tests passed.")
