"""Bidirectional Unitree ``G1JointIndex`` <-> Isaac articulation DoF map.

The unitree_sdk2 ``G1JointIndex`` order (leg-L, leg-R, waist, arm-L, arm-R) and
Isaac Sim's articulation ``dof_names`` order (a USD tree traversal: interleaved
L/R with the waist injected early) DIFFER. See
``docs/src/design/g1-drive/probe-results.md``. Any map built by POSITION would be
silently wrong, so this module builds it BY JOINT NAME from the LIVE
``art.dof_names`` and asserts (fail-closed at spawn) that every one of the 29
Unitree body joints is present.

The 14 Dex3 hand DoF (Isaac indices 29-42 on the on-image ``g1.usd``) carry no
Unitree LowCmd motor slot; they are returned as the ignored set so the driver can
lock/skip them.

Names are the canonical g1.gains names (``left_hip_pitch`` ... ``waist_yaw``), so
this map keys straight into ``g1.gains.G1_JOINT_LIMITS``. Live ``dof_names`` may
carry a trailing ``_joint`` (URDF convention); it is normalised away on lookup,
mirroring ``adapters/v5.py`` ``_g1_effort_clamp``.

This module is pure Python (stdlib only); it imports no omni / isaacsim / numpy /
even the sibling codec, so it loads under plain ``python3`` (including via
``importlib`` file-location, the test-harness pattern) and runs its own self-test.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

# NB: intentionally self-contained (only stdlib) so this module imports/runs under
# plain python3 via file-location -- same standalone convention as crc32.py /
# gains.py. The 29 names below MUST stay 1:1 with unitree_hg.G1JointIndex (0-28);
# that coupling is asserted in the unitree_hg / driver import path, not here.

# 29 body-joint canonical names in unitree_sdk2 G1JointIndex (0-28) order. This is
# the authoritative Unitree motor-slot ordering; it is NOT the Isaac DoF order.
UNITREE_G1_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch",       # 0  G1JointIndex.LeftHipPitch
    "left_hip_roll",        # 1  LeftHipRoll
    "left_hip_yaw",         # 2  LeftHipYaw
    "left_knee",            # 3  LeftKnee
    "left_ankle_pitch",     # 4  LeftAnklePitch (parallel A/B: B)
    "left_ankle_roll",      # 5  LeftAnkleRoll  (parallel A/B: A)
    "right_hip_pitch",      # 6  RightHipPitch
    "right_hip_roll",       # 7  RightHipRoll
    "right_hip_yaw",        # 8  RightHipYaw
    "right_knee",           # 9  RightKnee
    "right_ankle_pitch",    # 10 RightAnklePitch (parallel A/B: B)
    "right_ankle_roll",     # 11 RightAnkleRoll  (parallel A/B: A)
    "waist_yaw",            # 12 WaistYaw
    "waist_roll",           # 13 WaistRoll  (inferred parallel A/B: A)
    "waist_pitch",          # 14 WaistPitch (inferred parallel A/B: B)
    "left_shoulder_pitch",  # 15 LeftShoulderPitch
    "left_shoulder_roll",   # 16 LeftShoulderRoll
    "left_shoulder_yaw",    # 17 LeftShoulderYaw
    "left_elbow",           # 18 LeftElbow
    "left_wrist_roll",      # 19 LeftWristRoll
    "left_wrist_pitch",     # 20 LeftWristPitch
    "left_wrist_yaw",       # 21 LeftWristYaw
    "right_shoulder_pitch",  # 22 RightShoulderPitch
    "right_shoulder_roll",  # 23 RightShoulderRoll
    "right_shoulder_yaw",   # 24 RightShoulderYaw
    "right_elbow",          # 25 RightElbow
    "right_wrist_roll",     # 26 RightWristRoll
    "right_wrist_pitch",    # 27 RightWristPitch
    "right_wrist_yaw",      # 28 RightWristYaw
)

# Sanity: exactly 29 body joints (must line up 1:1 with G1JointIndex 0-28).
assert len(UNITREE_G1_JOINT_NAMES) == 29
assert len(set(UNITREE_G1_JOINT_NAMES)) == 29

# Probe-captured Isaac articulation DoF order for the on-image
# ``/Isaac/Robots/Unitree/G1/g1.usd`` (probe-results.md, 6.0.0-rc.22). Body joints
# occupy indices 0-28; the 14 Dex3 hand DoF follow at 29-42. This is a REFERENCE
# for tests only -- the live map is always built from the runtime ``art.dof_names``
# via build_index_maps, never from this constant.
PROBE_ISAAC_DOF_ORDER: Tuple[str, ...] = (
    "left_hip_pitch",       # 0
    "right_hip_pitch",      # 1
    "waist_yaw",            # 2
    "left_hip_roll",        # 3
    "right_hip_roll",       # 4
    "waist_roll",           # 5
    "left_hip_yaw",         # 6
    "right_hip_yaw",        # 7
    "waist_pitch",          # 8
    "left_knee",            # 9
    "right_knee",           # 10
    "left_shoulder_pitch",  # 11
    "right_shoulder_pitch",  # 12
    "left_ankle_pitch",     # 13
    "right_ankle_pitch",    # 14
    "left_shoulder_roll",   # 15
    "right_shoulder_roll",  # 16
    "left_ankle_roll",      # 17
    "right_ankle_roll",     # 18
    "left_shoulder_yaw",    # 19
    "right_shoulder_yaw",   # 20
    "left_elbow",           # 21
    "right_elbow",          # 22
    "left_wrist_roll",      # 23
    "right_wrist_roll",     # 24
    "left_wrist_pitch",     # 25
    "right_wrist_pitch",    # 26
    "left_wrist_yaw",       # 27
    "right_wrist_yaw",      # 28
)

# Dex3 hand DoF leaf names (2x7), for reference. TODO(probe:hand-dof-names) the
# exact on-image spelling/order of the 14 hand DoF is not needed for locomotion
# (they are ignored by name), but confirm before any hand control.
_HAND_LEAVES: Tuple[str, ...] = (
    "hand_index_0", "hand_index_1", "hand_middle_0", "hand_middle_1",
    "hand_thumb_0", "hand_thumb_1", "hand_thumb_2",
)

_JOINT_SUFFIX = "_joint"


class DofLayoutError(ValueError):
    """Raised when a live ``dof_names`` list is missing / duplicates a G1 body joint."""


def _canonical(dof_name: str) -> str:
    """Strip a trailing ``_joint`` so a USD dof name matches a gains canonical name."""
    if dof_name.endswith(_JOINT_SUFFIX):
        return dof_name[: -len(_JOINT_SUFFIX)]
    return dof_name


@dataclass(frozen=True)
class IndexMaps:
    """Resolved maps between the 29 Unitree motor slots and Isaac DoF indices.

    unitree_to_isaac -- length-29 tuple; ``unitree_to_isaac[u]`` is the Isaac DoF
        index for Unitree motor slot ``u`` (== G1JointIndex ``u``). Use it BOTH as
        the ``joint_indices`` for a 29-wide ArticulationAction (scatter) AND as
        gather indices to pull the 29 body values, in Unitree order, out of a
        full-length Isaac state vector (``q_full[unitree_to_isaac[u]]``).
    isaac_to_unitree -- length-``num_dof`` tuple; ``isaac_to_unitree[i]`` is the
        Unitree slot for Isaac DoF ``i`` or ``-1`` if ``i`` is an ignored (hand) DoF.
    ignored_isaac_dofs -- Isaac DoF indices with no Unitree slot (the Dex3 hands).
    num_dof -- total Isaac articulation DoF count (body + hands).
    """

    unitree_to_isaac: Tuple[int, ...]
    isaac_to_unitree: Tuple[int, ...]
    ignored_isaac_dofs: FrozenSet[int]
    num_dof: int

    @property
    def body_isaac_indices(self) -> Tuple[int, ...]:
        """The 29 body Isaac DoF indices, in Unitree motor-slot order."""
        return self.unitree_to_isaac


def assert_dof_layout(dof_names: Sequence[str]) -> None:
    """Fail-closed spawn check: every Unitree G1 body joint must appear exactly once.

    Raises DofLayoutError with a readable diff (missing / duplicated names + the
    observed dof_names) if the live articulation cannot satisfy the 29-DoF Unitree
    body layout. Order is NOT checked -- the map is built by name, so any traversal
    order is fine, but a MISSING or AMBIGUOUS joint would corrupt the map, so it is
    rejected before the driver touches the robot.
    """
    names = list(dof_names)
    counts = Counter(_canonical(n) for n in names)
    missing = [n for n in UNITREE_G1_JOINT_NAMES if counts.get(n, 0) == 0]
    duplicated = [n for n in UNITREE_G1_JOINT_NAMES if counts.get(n, 0) > 1]
    if missing or duplicated:
        raise DofLayoutError(
            "G1 articulation DoF layout does not satisfy the Unitree 29-joint body "
            "contract (map is built by joint NAME).\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  duplicated ({len(duplicated)}): {duplicated}\n"
            f"  observed dof_names ({len(names)}): {names}\n"
            f"  expected body joints (29): {list(UNITREE_G1_JOINT_NAMES)}"
        )


def build_index_maps(dof_names: Sequence[str]) -> IndexMaps:
    """Build the bidirectional map from the LIVE ``art.dof_names`` (BY NAME).

    Calls assert_dof_layout first (fail-closed). Never assumes positional order --
    each Unitree joint name is looked up in dof_names (with ``_joint`` normalised).
    """
    names = list(dof_names)
    assert_dof_layout(names)

    canon_to_isaac: Dict[str, int] = {}
    for i, name in enumerate(names):
        canon_to_isaac.setdefault(_canonical(name), i)

    unitree_to_isaac = tuple(canon_to_isaac[n] for n in UNITREE_G1_JOINT_NAMES)
    slot_of = {isaac_idx: u for u, isaac_idx in enumerate(unitree_to_isaac)}
    isaac_to_unitree = tuple(slot_of.get(i, -1) for i in range(len(names)))
    ignored = frozenset(i for i in range(len(names)) if i not in slot_of)
    return IndexMaps(
        unitree_to_isaac=unitree_to_isaac,
        isaac_to_unitree=isaac_to_unitree,
        ignored_isaac_dofs=ignored,
        num_dof=len(names),
    )


__all__ = [
    "UNITREE_G1_JOINT_NAMES",
    "PROBE_ISAAC_DOF_ORDER",
    "IndexMaps",
    "DofLayoutError",
    "build_index_maps",
    "assert_dof_layout",
]


if __name__ == "__main__":
    # Dependency-free self-test (plain python3, no omni / isaacsim / numpy).
    # 1. Build from the probe Isaac order + fake Dex3 hand DoF and round-trip.
    fake_hands: List[str] = [
        f"{side}_{leaf}" for side in ("left", "right") for leaf in _HAND_LEAVES
    ]
    assert len(fake_hands) == 14
    # Exercise the ``_joint`` normalisation on half the names.
    live = [
        (n + "_joint" if i % 2 == 0 else n)
        for i, n in enumerate(PROBE_ISAAC_DOF_ORDER)
    ] + fake_hands

    maps = build_index_maps(live)
    assert maps.num_dof == 43, maps.num_dof
    assert len(maps.unitree_to_isaac) == 29
    assert len(maps.ignored_isaac_dofs) == 14

    # Every Unitree slot must resolve to the Isaac index whose canonical name matches.
    for u, name in enumerate(UNITREE_G1_JOINT_NAMES):
        isaac_idx = maps.unitree_to_isaac[u]
        assert _canonical(live[isaac_idx]) == name, (u, name, isaac_idx)
        assert maps.isaac_to_unitree[isaac_idx] == u

    # Ignored set == the hand DoF indices (29-42), none in the body map.
    assert maps.ignored_isaac_dofs == frozenset(range(29, 43))
    assert all(maps.isaac_to_unitree[i] == -1 for i in maps.ignored_isaac_dofs)

    # Spot-check a couple of the tricky interleaved entries against the probe order.
    assert maps.unitree_to_isaac[UNITREE_G1_JOINT_NAMES.index("waist_yaw")] == 2
    assert maps.unitree_to_isaac[UNITREE_G1_JOINT_NAMES.index("right_hip_pitch")] == 1
    assert maps.unitree_to_isaac[UNITREE_G1_JOINT_NAMES.index("left_ankle_roll")] == 17

    # Gather a fake full-length Isaac state vector into Unitree order and back.
    q_full = [float(i) for i in range(43)]
    q_unitree = [q_full[maps.unitree_to_isaac[u]] for u in range(29)]
    for u in range(29):
        assert q_unitree[u] == float(maps.unitree_to_isaac[u])

    # 2. assert_dof_layout must fail closed on a missing joint.
    broken = [n for n in PROBE_ISAAC_DOF_ORDER if n != "waist_pitch"]
    try:
        assert_dof_layout(broken)
    except DofLayoutError as exc:
        assert "waist_pitch" in str(exc)
    else:
        raise AssertionError("assert_dof_layout must raise on a missing joint")

    # 3. ... and on a duplicated joint.
    dup = list(PROBE_ISAAC_DOF_ORDER) + ["left_knee"]
    try:
        assert_dof_layout(dup)
    except DofLayoutError as exc:
        assert "left_knee" in str(exc)
    else:
        raise AssertionError("assert_dof_layout must raise on a duplicated joint")

    print("motor_index_map self-test OK (43 DoF, 29 body mapped, 14 hands ignored)")
