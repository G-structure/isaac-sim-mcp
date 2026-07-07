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

"""Robot creation and control command handlers."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..adapters.base import IsaacAdapterBase
from ..g1 import (
    UNITREE_G1_JOINT_NAMES,
    ControlMode,
    G1LowCmdDriver,
    LowCmd,
    ModePR,
)
from ._guards import OPERATOR_CAPABILITY, current_capability, guard_pose_write

_N_BODY = len(UNITREE_G1_JOINT_NAMES)  # 29 Unitree body motor slots

# Installed in-process G1 LowCmd drivers, keyed by articulation-root prim path.
# robots.g1.spawn (operator-only) installs one; the agent-facing g1 commands look
# theirs up here. Module-level (not adapter state) so it survives across the
# per-command handler calls, mirroring robots._discovered_robots above.
_g1_drivers: Dict[str, G1LowCmdDriver] = {}

# Per-prim last-reset monotonic timestamp for the robots.g1.reset rate limit.
_g1_last_reset: Dict[str, float] = {}

# Minimum seconds between robots.g1.reset calls (Phase-1 reset constraint: a
# fixed neutral re-pose, only while STOPPED, not spammable).
_G1_RESET_MIN_INTERVAL_S = 1.0

# ── ASSISTED-mode brain producer + status monitor state ───────────────────────
#
# In ASSISTED mode the in-container BRAIN is the LowCmd producer: it tracks a
# high-level setpoint (mode + vx/vy/wz) and submits LowCmd frames through the SAME
# driver->apply_action path a RAW per-motor command uses, so ASSISTED is exactly
# as physics-honest as RAW — the agent sends INTENT (robots.g1.set_command)
# instead of per-motor torques. The brain runs IN-PROCESS next to the driver
# (g1_brain.in_process_runner: source=driver.latest_lowstate_bytes,
# sink=driver.submit_lowcmd), so there is no socket hop and no second producer.
#
# Capability split (decided + TESTED in tests/test_g1_brain_integration.py):
#   * robots.g1.spawn (driver install) stays OPERATOR-only — it subscribes a
#     physics-step callback + DDS thread (trusted setup), enforced at dispatch.
#   * robots.g1.set_control_mode and robots.g1.set_command are SESSION-allowed
#     HIGH-LEVEL INTENT. Starting the brain producer is strictly LESS capable than
#     the RAW per-motor set_joint_command a session ALREADY has: the brain emits
#     trusted stand/policy LowCmds through the same gated apply path, so it grants
#     no new power and needs no operator token.
#   * While ASSISTED, a SESSION RAW per-motor set_joint_command is REJECTED — the
#     brain owns the producer; operator may still submit (external-producer/debug).
#
# Keyed by articulation-root prim path, mirroring _g1_drivers. The desired
# setpoint is stored separately so a set_command that arrives before ASSISTED is
# applied when the brain starts.
_g1_brains: Dict[str, Dict[str, Any]] = {}        # {prim: {"brain":..., "runner":...}}
_g1_setpoints: Dict[str, Any] = {}                # {prim: g1_brain.Setpoint}
_g1_monitors: Dict[str, Any] = {}                 # {prim: g1_brain.monitor.G1FallMonitor}

# Paths prepended to sys.path so the pure-Python g1_brain package imports inside
# kit's interpreter (the container copies it to /session/g1_brain). Overridable
# via G1_BRAIN_PATH.
_G1_BRAIN_PATHS = ("/session", "/session/g1_brain")

# Hardcoded fallback — used only if live discovery fails.
# Keys are lowercase robot names, asset_path is relative to the assets root.
FALLBACK_ROBOT_LIBRARY: Dict[str, Dict[str, str]] = {
    "frankapanda": {
        "asset_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "description": "FrankaRobotics FrankaPanda",
        "manufacturer": "FrankaRobotics",
    },
    "jetbot": {
        "asset_path": "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
        "description": "NVIDIA Jetbot",
        "manufacturer": "NVIDIA",
    },
    "carter_v1": {
        "asset_path": "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
        "description": "NVIDIA Carter",
        "manufacturer": "NVIDIA",
    },
    "novacarter": {
        "asset_path": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        "description": "NVIDIA NovaCarter",
        "manufacturer": "NVIDIA",
    },
    "g1": {"asset_path": "/Isaac/Robots/Unitree/G1/g1.usd", "description": "Unitree G1", "manufacturer": "Unitree"},
    "go1": {"asset_path": "/Isaac/Robots/Unitree/Go1/go1.usd", "description": "Unitree Go1", "manufacturer": "Unitree"},
    "spot": {
        "asset_path": "/Isaac/Robots/BostonDynamics/spot/spot.usd",
        "description": "BostonDynamics spot",
        "manufacturer": "BostonDynamics",
    },
}

# Cached discovered robots — populated on first call to list_robots.
_discovered_robots: Optional[Dict[str, Dict[str, str]]] = None


def _get_robot_library(adapter: IsaacAdapterBase) -> Dict[str, Dict[str, str]]:
    """Return the robot library, discovering from the asset server on first call.

    Falls back to FALLBACK_ROBOT_LIBRARY if discovery fails.
    """
    global _discovered_robots
    if _discovered_robots is not None:
        return _discovered_robots

    try:
        robots = adapter.discover_robots()
        if robots:
            _discovered_robots = robots
            print(f"Discovered {len(robots)} robots from asset server")
            return _discovered_robots
    except Exception as e:
        print(f"Robot discovery failed, using fallback: {e}")

    _discovered_robots = FALLBACK_ROBOT_LIBRARY
    return _discovered_robots


def _find_robot(adapter: IsaacAdapterBase, query: str) -> Optional[Dict[str, Any]]:
    """Find a robot by name. Tries exact key match, then partial match on key/description/manufacturer."""
    library = _get_robot_library(adapter)
    q = query.lower().strip()

    # Exact key match
    if q in library:
        return {"key": q, **library[q]}

    # Partial match on key, description, manufacturer
    matches = []
    for key, info in library.items():
        searchable = f"{key} {info.get('description', '')} {info.get('manufacturer', '')}".lower()
        if q in searchable:
            matches.append({"key": key, **info})

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return closest match (shortest key that contains the query)
        matches.sort(key=lambda m: len(m["key"]))
        return matches[0]

    return None


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["robots.create"] = lambda **p: create(adapter, **p)
    registry["robots.list"] = lambda **p: list_robots(adapter, **p)
    registry["robots.refresh"] = lambda **p: refresh_robots(adapter, **p)
    registry["robots.get_info"] = lambda **p: get_info(adapter, **p)
    registry["robots.set_joints"] = lambda **p: set_joints(adapter, **p)
    registry["robots.get_joints"] = lambda **p: get_joints(adapter, **p)
    # Unitree G1 low-level (LowCmd/LowState) driver surface. spawn is operator-only
    # (trusted driver install, enforced at dispatch in extension._OPERATOR_ONLY_COMMANDS
    # and re-checked in the handler); the rest are session-allowed honest actuation
    # gated by the ACTUATION gate in _guards.
    registry["robots.g1.spawn"] = lambda **p: g1_spawn(adapter, **p)
    registry["robots.g1.set_joint_command"] = lambda **p: g1_set_joint_command(adapter, **p)
    registry["robots.g1.get_lowstate"] = lambda **p: g1_get_lowstate(adapter, **p)
    registry["robots.g1.set_gains"] = lambda **p: g1_set_gains(adapter, **p)
    registry["robots.g1.set_control_mode"] = lambda **p: g1_set_control_mode(adapter, **p)
    # High-level ASSISTED setpoint (session intent) + read-only status monitor.
    registry["robots.g1.set_command"] = lambda **p: g1_set_command(adapter, **p)
    registry["robots.g1.get_status"] = lambda **p: g1_get_status(adapter, **p)
    registry["robots.g1.reset"] = lambda **p: g1_reset(adapter, **p)


def create(
    adapter: IsaacAdapterBase,
    robot_type: str = "franka",
    position: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
    prim_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        match = _find_robot(adapter, robot_type)
        if not match:
            library = _get_robot_library(adapter)
            available = list(library.keys())[:20]
            return {
                "status": "error",
                "message": f"Robot '{robot_type}' not found. Try robots.list to see available robots. Some options: {available}",
            }

        assets_root = adapter.get_assets_root_path()
        asset_path = assets_root + match["asset_path"]
        if prim_path is None:
            prim_name = name or match["key"].capitalize()
            prim_path = f"/{prim_name}"
        # Sink guard: spawning a robot reference AT (or under) an existing robot
        # articulation-root and then set_world_pose-ing it while the timeline is
        # not stopped is a disguised teleport/overwrite. Route it through the same
        # fail-closed pose lock the dispatch-level guard applies to
        # objects.transform, so create cannot smuggle a robot re-pose past it.
        rejection = guard_pose_write(adapter, prim_path)
        if rejection is not None:
            return rejection
        adapter.add_reference_to_stage(asset_path, prim_path)
        if position:
            xform = adapter.create_xform_prim(prim_path)
            xform.set_world_pose(position=np.array(position))
        # Author contact reporting on the collider subtree AT SPAWN (pre-play)
        # so foot-ground reaction forces fire on the first impact. Applying it
        # lazily in the get_physics_state read path is too late once the scene
        # is live — PhysX will not re-parse — so it must happen here.
        contact_reporting = "applied_at_spawn"
        try:
            adapter.ensure_contact_reporting(prim_path)
        except Exception as e:
            contact_reporting = f"not_applied: {e}"
        result = {
            "status": "success",
            "message": f"Created {match['description']} robot",
            "prim_path": prim_path,
            "robot_key": match["key"],
            "contact_reporting": contact_reporting,
        }
        try:
            info = adapter.get_robot_joint_info(prim_path)
            result["joint_names"] = info.get("joint_names", [])
            result["num_dof"] = info.get("num_dof", 0)
        except Exception:
            pass
        # Check for broken drive configs (zero stiffness + zero damping)
        try:
            joint_config = adapter.get_joint_config(prim_path)
            warnings = joint_config.get("warnings", [])
            if warnings:
                result["warnings"] = warnings
        except Exception:
            pass
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_robots(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    library = _get_robot_library(adapter)
    return {"status": "success", "robot_count": len(library), "robots": library}


def refresh_robots(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    """Force re-scan the asset server for available robots."""
    global _discovered_robots
    _discovered_robots = None
    library = _get_robot_library(adapter)
    return {
        "status": "success",
        "message": f"Refreshed robot library, found {len(library)} robots",
        "robot_count": len(library),
    }


def get_info(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        info = adapter.get_robot_joint_info(prim_path)
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def set_joints(
    adapter: IsaacAdapterBase,
    prim_path: Optional[str] = None,
    joint_positions: Optional[Sequence[float]] = None,
    joint_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    try:
        if not prim_path or joint_positions is None:
            return {"status": "error", "message": "prim_path and joint_positions are required"}
        adapter.set_joint_positions(prim_path, joint_positions, joint_indices)
        return {"status": "success", "message": f"Set joint positions on {prim_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_joints(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        positions = adapter.get_joint_positions(prim_path)
        return {"status": "success", "joint_positions": positions}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Unitree G1 low-level (LowCmd/LowState) driver commands ────────────────────
#
# These wire the standalone G1LowCmdDriver (g1/g1_lowcmd_driver.py) into the MCP
# command surface. The driver owns the physics-step callback and the DDS side
# thread; these handlers only marshal bytes/params in and out of it — they never
# touch PhysX directly, so a per-motor command moves joints ONLY through the
# driver -> ArticulationController.apply_action PD path (no teleport).


def _g1_prim_exists(adapter: IsaacAdapterBase, prim_path: str) -> bool:
    """True if a valid prim already lives at ``prim_path`` (spawn idempotency)."""
    try:
        stage = adapter.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        return bool(prim and prim.IsValid())
    except Exception:
        return False


def _parse_control_mode(mode: Any) -> ControlMode:
    """Coerce a RAW|ASSISTED name, a ControlMode, or an int into a ControlMode."""
    if mode is None:
        return ControlMode.RAW
    if isinstance(mode, ControlMode):
        return mode
    if isinstance(mode, str):
        key = mode.strip().upper()
        if key not in ControlMode.__members__:
            raise ValueError(
                f"unknown control_mode {mode!r}; expected one of "
                f"{list(ControlMode.__members__)}"
            )
        return ControlMode[key]
    return ControlMode(int(mode))


def _coerce_frame_bytes(lowcmd: Any) -> bytes:
    """Accept a prebuilt LowCmd frame as bytes, a hex string, or a byte list."""
    if isinstance(lowcmd, (bytes, bytearray)):
        return bytes(lowcmd)
    if isinstance(lowcmd, str):
        return bytes.fromhex(lowcmd.strip())
    if isinstance(lowcmd, (list, tuple)):
        return bytes(int(b) & 0xFF for b in lowcmd)
    raise ValueError(
        "lowcmd must be bytes, a hex string, or a list of byte values"
    )


def _build_lowcmd_frame(
    driver: G1LowCmdDriver,
    lowcmd: Any,
    q: Optional[Sequence[float]],
    dq: Optional[Sequence[float]],
    tau: Optional[Sequence[float]],
    kp: Optional[Sequence[float]],
    kd: Optional[Sequence[float]],
    joint_indices: Optional[Sequence[int]],
    mode_pr: Optional[int],
    mode_machine: Optional[int],
) -> bytes:
    """Return a packed, CRC'd LowCmd frame from either a prebuilt frame or arrays.

    The array path indexes by Unitree ``G1JointIndex`` slot (0-28), NOT by Isaac
    DoF index — the driver remaps to Isaac DoF by joint NAME. mode_pr defaults to
    PR (the serial ankle/waist space the driver enforces) and mode_machine to the
    driver's configured value, so the frame passes the driver's inbound verify.
    """
    if lowcmd is not None:
        return _coerce_frame_bytes(lowcmd)

    arrays = {"q": q, "dq": dq, "tau": tau, "kp": kp, "kd": kd}
    provided = {name: list(arr) for name, arr in arrays.items() if arr is not None}
    if not provided:
        raise ValueError(
            "provide either a prebuilt 'lowcmd' frame or at least one of "
            "q/dq/tau/kp/kd (per Unitree G1JointIndex slot)"
        )

    if joint_indices is not None:
        slots = [int(j) for j in joint_indices]
    else:
        slots = list(range(len(next(iter(provided.values())))))

    for name, arr in provided.items():
        if len(arr) != len(slots):
            raise ValueError(
                f"{name} length {len(arr)} != joint_indices length {len(slots)}"
            )

    stats = driver.stats()
    mm = int(mode_machine) if mode_machine is not None else int(stats.get("mode_machine") or 0)
    mp = int(mode_pr) if mode_pr is not None else int(ModePR.PR)
    lc = LowCmd(mode_pr=mp, mode_machine=mm)
    for pos, u in enumerate(slots):
        if not (0 <= u < _N_BODY):
            raise ValueError(
                f"joint index {u} out of range 0..{_N_BODY - 1} (Unitree G1JointIndex)"
            )
        m = lc.motor_cmd[u]
        m.mode = 1
        if q is not None:
            m.q = float(q[pos])
        if dq is not None:
            m.dq = float(dq[pos])
        if tau is not None:
            m.tau = float(tau[pos])
        if kp is not None:
            m.kp = float(kp[pos])
        if kd is not None:
            m.kd = float(kd[pos])
    lc.with_crc()
    return lc.pack()


def _decode_lowstate(ls: Any) -> Dict[str, Any]:
    """Decode a LowState into JSON-friendly fields (body slots in Unitree order)."""
    motor_state = []
    for u in range(_N_BODY):
        ms = ls.motor_state[u]
        motor_state.append(
            {
                "slot": u,
                "name": UNITREE_G1_JOINT_NAMES[u],
                "mode": ms.mode,
                "q": ms.q,
                "dq": ms.dq,
                "tau_est": ms.tau_est,
            }
        )
    imu = ls.imu_state
    return {
        "mode_pr": ls.mode_pr,
        "mode_machine": ls.mode_machine,
        "tick": ls.tick,
        "imu": {
            "quaternion": list(imu.quaternion),
            "gyroscope": list(imu.gyroscope),
            "accelerometer": list(imu.accelerometer),
            "rpy": list(imu.rpy),
        },
        "motor_state": motor_state,
    }


# ── ASSISTED brain producer helpers (lazy g1_brain import; in-process runner) ──


def _g1_brain_module():
    """Import the pure-Python ``g1_brain`` package (lazy, so robots.py stays
    importable off-GPU / without the brain deployed). Ensures the deploy path is on
    sys.path and points the brain's codec loader at THIS extension's already-imported
    g1 leaf modules (``_reuse_installed`` picks them up -> no duplicate codec load)."""
    import sys

    for p in (os.environ.get("G1_BRAIN_PATH"),) + _G1_BRAIN_PATHS:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault(
        "G1_CODEC_DIR", os.path.join(os.path.dirname(__file__), "..", "g1")
    )
    import g1_brain

    return g1_brain


def _coerce_brain_mode(gb: Any, mode: Any):
    """Coerce a mode name / BrainMode into a ``g1_brain.BrainMode`` (raises ValueError)."""
    BrainMode = gb.BrainMode
    if isinstance(mode, BrainMode):
        return mode
    if mode is None:
        return BrainMode.STAND
    return BrainMode(str(mode).strip().lower())


def _g1_make_brain(gb: Any, mode_machine: int):
    """Build the ASSISTED-mode brain: a unitree_rl_gym ``PolicyBrain`` when a
    checkpoint is configured (``G1_POLICY_CHECKPOINT``), else — or if the policy
    cannot load — the always-usable ``HandTunedStandBrain``."""
    checkpoint = os.environ.get("G1_POLICY_CHECKPOINT")
    if checkpoint:
        try:
            return gb.PolicyBrain(
                checkpoint_path=checkpoint,
                num_actions=int(os.environ.get("G1_POLICY_NUM_ACTIONS", str(_N_BODY))),
                backend=os.environ.get("G1_POLICY_BACKEND", "auto"),
                mode_machine=mode_machine,
            )
        except gb.PolicyUnavailableError as e:  # noqa: BLE001
            print(f"[g1-brain] policy unavailable ({e}); using HandTunedStandBrain")
    return gb.HandTunedStandBrain(mode_machine=mode_machine)


def _g1_start_brain(driver: G1LowCmdDriver, prim_path: str) -> Dict[str, Any]:
    """Start (or return the already-running) in-process brain producer for
    ``prim_path`` and apply any stored setpoint. Producer -> driver.submit_lowcmd,
    the SAME gated apply path as a RAW LowCmd."""
    gb = _g1_brain_module()
    entry = _g1_brains.get(prim_path)
    if entry is not None and entry["runner"].running:
        return entry
    mode_machine = int(driver.stats().get("mode_machine") or 0)
    brain = _g1_make_brain(gb, mode_machine)
    runner = gb.in_process_runner(driver, brain)
    sp = _g1_setpoints.get(prim_path)
    if sp is not None:
        runner.set_setpoint(setpoint=sp)
    runner.start()
    entry = {"brain": brain, "runner": runner}
    _g1_brains[prim_path] = entry
    return entry


def _g1_stop_brain(prim_path: str) -> bool:
    """Stop + drop the in-process brain producer for ``prim_path`` (idempotent)."""
    entry = _g1_brains.pop(prim_path, None)
    if entry is None:
        return False
    entry["runner"].stop()
    return True


def _g1_brain_running(prim_path: str) -> bool:
    entry = _g1_brains.get(prim_path)
    return bool(entry is not None and entry["runner"].running)


def _g1_monitor(prim_path: str):
    """Return the per-prim fall/TAINTED monitor, creating it on first use."""
    mon = _g1_monitors.get(prim_path)
    if mon is None:
        _g1_brain_module()  # ensure g1_brain is importable
        from g1_brain.monitor import G1FallMonitor

        mon = G1FallMonitor()
        _g1_monitors[prim_path] = mon
    return mon


def _g1_contact_magnitude(contacts: Any) -> float:
    """Sum contact force/impulse magnitudes from the adapter's get_physics_state
    ``contacts`` list into one scalar (0.0 when nothing is in contact)."""
    total = 0.0
    for c in contacts or []:
        if isinstance(c, dict):
            for k in ("magnitude", "force", "impulse", "normal_force", "normalForce"):
                v = c.get(k)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    total += math.sqrt(sum(float(x) * float(x) for x in v))
                else:
                    total += abs(float(v))
                break
        elif isinstance(c, (int, float)):
            total += abs(float(c))
    return total


def g1_spawn(
    adapter: IsaacAdapterBase,
    robot_type: str = "g1",
    prim_path: str = "/World/G1",
    position: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
    enable_dds: bool = True,
    mode_machine: int = 0,
    control_mode: Any = "RAW",
    lowstate_hz: float = 500.0,
    lowstate_publish_stride: int = 1,
    strict_mode_machine: bool = True,
    cmd_timeout_s: Optional[float] = None,
    start_brain: bool = False,
) -> Dict[str, Any]:
    """Spawn the G1 (idempotent) and install + start its LowCmd driver.

    TRUSTED setup: installing the driver subscribes a physics-step callback and
    spawns a DDS side thread, so this is operator-only. extension.py refuses it
    at dispatch for the session capability (_OPERATOR_ONLY_COMMANDS); the check
    here is defense-in-depth for any path that reaches the handler directly.

    Requires physics to be PLAYING (the articulation must be initialized so
    dof_names is populated and the fail-closed layout assert can run) — the
    autospawn flow spawns + grounds + plays before calling this.

    TODO(probe:driver-live): validate on a GPU box that after this call a raw
    LowCmd submitted through robots.g1.set_joint_command actually moves the joint
    it addresses and that robots.g1.get_lowstate reflects the motion.
    """
    if current_capability() != OPERATOR_CAPABILITY:
        return {
            "status": "error",
            "message": (
                "robots.g1.spawn installs the low-level driver (trusted setup) and "
                "is operator-only; forbidden in session mode"
            ),
            "rejected_by": "session_capability",
        }
    try:
        existing = _g1_drivers.get(prim_path)
        if existing is not None and existing.installed:
            return {
                "status": "success",
                "message": f"G1 driver already installed at {prim_path}",
                "prim_path": prim_path,
                "already_installed": True,
                "driver": existing.stats(),
            }

        # 1. Ensure the asset exists (reuse robots.create; idempotent).
        created_info: Dict[str, Any] = {}
        if not _g1_prim_exists(adapter, prim_path):
            created = create(
                adapter, robot_type=robot_type, position=position, name=name, prim_path=prim_path
            )
            if created.get("status") != "success":
                return created
            created_info = created

        # 2. Bind + initialize the articulation (dof_names populated once playing).
        art = adapter.create_articulation(prim_path, name=name or "g1")
        art.initialize()

        # 3. Install + start the driver (fail-closed DoF layout assert runs inside).
        driver = G1LowCmdDriver(
            control_mode=_parse_control_mode(control_mode),
            enable_dds=bool(enable_dds),
            mode_machine=int(mode_machine),
            lowstate_hz=float(lowstate_hz),
            lowstate_publish_stride=int(lowstate_publish_stride),
            strict_mode_machine=bool(strict_mode_machine),
            cmd_timeout_s=cmd_timeout_s,
        )
        driver.install(art, prim_path=prim_path)
        driver.start()
        _g1_drivers[prim_path] = driver

        # Optionally start the ASSISTED brain producer now (operator convenience:
        # a booted robot that stands itself). Best-effort — a brain-start failure
        # never fails the driver install; it leaves the driver in RAW.
        brain_running = False
        brain_error = None
        if start_brain:
            try:
                driver.set_control_mode(ControlMode.ASSISTED)
                _g1_start_brain(driver, prim_path)
                brain_running = _g1_brain_running(prim_path)
            except Exception as e:  # noqa: BLE001
                driver.set_control_mode(ControlMode.RAW)
                brain_error = str(e)

        result: Dict[str, Any] = {
            "status": "success",
            "message": f"Installed + started G1 LowCmd driver at {prim_path}",
            "prim_path": prim_path,
            "driver": driver.stats(),
            "brain_running": brain_running,
        }
        if brain_error is not None:
            result["brain_error"] = brain_error
        for key in ("robot_key", "num_dof", "warnings", "contact_reporting"):
            if key in created_info:
                result[key] = created_info[key]
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_set_joint_command(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
    lowcmd: Any = None,
    q: Optional[Sequence[float]] = None,
    dq: Optional[Sequence[float]] = None,
    tau: Optional[Sequence[float]] = None,
    kp: Optional[Sequence[float]] = None,
    kd: Optional[Sequence[float]] = None,
    joint_indices: Optional[Sequence[int]] = None,
    mode_pr: Optional[int] = None,
    mode_machine: Optional[int] = None,
) -> Dict[str, Any]:
    """RAW per-motor control: submit a Unitree LowCmd to the driver's inbound seqlock.

    This IS the "agent drives each motor" path. Accepts either a prebuilt CRC'd
    LowCmd frame (``lowcmd`` as bytes/hex/byte-list) or q/dq/tau/kp/kd arrays with
    optional ``joint_indices`` (Unitree G1JointIndex slots 0-28). The frame is
    only queued here; submit_lowcmd verifies CRC + mode on the producer thread and
    the physics-step callback consumes the already-validated command, applying it
    through the PhysX PD path. Gated by the ACTUATION gate in _guards
    (timeline PLAYING AND physics_dt > 0 AND the live-read gravity is a physical
    earth-like downward vector — the SAME envelope enforced on gravity writes), so
    it cannot run against a frozen, zero-gravity, or sideways/upward "fake-balance"
    world; session-allowed once that gate is met.
    """
    try:
        driver = _g1_drivers.get(prim_path)
        if driver is None or not driver.installed:
            return {
                "status": "error",
                "message": f"no G1 driver installed at {prim_path}; run robots.g1.spawn first",
            }
        # While ASSISTED, the in-process brain is the LowCmd producer; a SESSION
        # agent may only send high-level intent (robots.g1.set_command), not RAW
        # per-motor frames. Reject the session RAW path so two producers never
        # fight over the inbound seqlock. Operator (external-producer / debug) is
        # allowed through.
        if (
            driver.control_mode == ControlMode.ASSISTED
            and current_capability() != OPERATOR_CAPABILITY
        ):
            return {
                "status": "error",
                "message": (
                    "RAW per-motor set_joint_command is blocked while "
                    "control_mode=ASSISTED; the brain is the LowCmd producer — send "
                    "robots.g1.set_command (high-level setpoint) or switch to RAW"
                ),
                "rejected_by": "assisted_mode",
                "control_mode": driver.control_mode.name,
            }
        frame = _build_lowcmd_frame(
            driver, lowcmd, q, dq, tau, kp, kd, joint_indices, mode_pr, mode_machine
        )
        accepted = driver.submit_lowcmd(frame)
        if not accepted:
            return {
                "status": "error",
                "message": f"driver rejected LowCmd frame (length must be {len(frame)} == LOWCMD_SIZE)",
            }
        return {
            "status": "success",
            "message": "LowCmd submitted to G1 driver",
            "prim_path": prim_path,
            "frame_bytes": len(frame),
            "control_mode": driver.control_mode.name,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_get_lowstate(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
    include_runtime: bool = False,
) -> Dict[str, Any]:
    """Read the driver's outbound seqlock (latest CRC-valid LowState).

    Returns the decoded LowState fields plus the raw hex frame. When no driver is
    installed it falls back to a direct runtime read via the adapter's
    get_lowstate_fields. ``include_runtime`` also attaches that runtime read as a
    cross-check alongside the driver's published frame.
    """
    try:
        driver = _g1_drivers.get(prim_path)
        if driver is None or not driver.installed:
            fields = adapter.get_lowstate_fields(prim_path)
            return {
                "status": "success",
                "prim_path": prim_path,
                "source": "adapter_runtime",
                "driver_installed": False,
                "lowstate_fields": fields,
            }

        result: Dict[str, Any] = {
            "status": "success",
            "prim_path": prim_path,
            "source": "driver_seqlock",
            "driver_installed": True,
            "stats": driver.stats(),
        }
        ls = driver.latest_lowstate()
        if ls is not None:
            result["lowstate"] = _decode_lowstate(ls)
            result["crc_valid"] = ls.verify_crc()
        raw = driver.latest_lowstate_bytes()
        if raw is not None:
            result["raw_hex"] = raw.hex()
        if include_runtime:
            try:
                result["lowstate_fields"] = adapter.get_lowstate_fields(prim_path)
            except Exception as e:
                result["lowstate_fields_error"] = str(e)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_set_gains(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
    kp: Optional[Sequence[float]] = None,
    kd: Optional[Sequence[float]] = None,
    joint_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Set per-joint PD gains (SI/radian: kp N·m/rad, kd N·m·s/rad) via v5.set_gains.

    ``joint_indices`` are Isaac DoF indices (the adapter's tensor-API convention).
    Values must be finite and non-negative. Honest actuation, session-allowed, but
    gated by the ACTUATION gate in _guards (only meaningful against a live sim).
    """
    try:
        if kp is None or kd is None:
            return {
                "status": "error",
                "message": "kp and kd are required (SI/radian: kp [N·m/rad], kd [N·m·s/rad])",
            }
        kp = [float(v) for v in kp]
        kd = [float(v) for v in kd]
        if len(kp) != len(kd):
            return {
                "status": "error",
                "message": f"kp/kd length mismatch ({len(kp)} != {len(kd)})",
            }
        if joint_indices is not None and len(joint_indices) != len(kp):
            return {
                "status": "error",
                "message": f"joint_indices length {len(joint_indices)} must match kp/kd length {len(kp)}",
            }
        for label, arr in (("kp", kp), ("kd", kd)):
            for v in arr:
                if not math.isfinite(v) or v < 0.0:
                    return {
                        "status": "error",
                        "message": f"{label} values must be finite and >= 0 (got {v})",
                    }
        adapter.set_gains(prim_path, kp, kd, joint_indices)
        return {
            "status": "success",
            "message": f"Set PD gains on {prim_path}",
            "prim_path": prim_path,
            "num_joints": len(kp),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_set_control_mode(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
    mode: Any = None,
) -> Dict[str, Any]:
    """Set the driver control mode and start/stop the in-process brain producer.

    RAW -> the agent is the LowCmd producer (per-motor set_joint_command); the
    brain is stopped. ASSISTED -> the in-container BRAIN is the producer (it tracks
    the robots.g1.set_command setpoint and submits LowCmds through the same gated
    driver->apply_action path); the brain runner is started. SESSION-allowed
    high-level intent — starting the brain is strictly less capable than the RAW
    per-motor surface a session already has. Fail-closed: if the ASSISTED brain
    cannot start, the driver is reverted to RAW so ASSISTED never runs producerless.
    """
    try:
        driver = _g1_drivers.get(prim_path)
        if driver is None or not driver.installed:
            return {
                "status": "error",
                "message": f"no G1 driver installed at {prim_path}; run robots.g1.spawn first",
            }
        cm = _parse_control_mode(mode)
        driver.set_control_mode(cm)
        brain_started = False
        brain_stopped = False
        if cm == ControlMode.ASSISTED:
            try:
                _g1_start_brain(driver, prim_path)
                brain_started = True
            except Exception as e:  # noqa: BLE001
                driver.set_control_mode(ControlMode.RAW)
                _g1_stop_brain(prim_path)
                return {
                    "status": "error",
                    "message": (
                        f"failed to start ASSISTED brain producer ({e}); "
                        "reverted control_mode to RAW"
                    ),
                }
        else:
            brain_stopped = _g1_stop_brain(prim_path)
        return {
            "status": "success",
            "message": f"G1 control_mode set to {cm.name}",
            "prim_path": prim_path,
            "control_mode": cm.name,
            "brain_running": _g1_brain_running(prim_path),
            "brain_started": brain_started,
            "brain_stopped": brain_stopped,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_set_command(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
    mode: Any = "stand",
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
) -> Dict[str, Any]:
    """High-level ASSISTED setpoint: mode (stand|walk|damp|squat) + (vx,vy,wz).

    This is the ONLY control surface a SESSION agent uses in ASSISTED mode — pure
    INTENT, not actuation. It stores the setpoint and forwards it to the running
    brain; it never moves a joint itself (the brain producer does, through the
    gated driver->apply_action path). If ASSISTED is not active yet the setpoint is
    latched and applied when the brain starts. ``damp`` zeroes kp so the robot
    falls honestly under kd-only damping.
    """
    try:
        driver = _g1_drivers.get(prim_path)
        if driver is None or not driver.installed:
            return {
                "status": "error",
                "message": f"no G1 driver installed at {prim_path}; run robots.g1.spawn first",
            }
        gb = _g1_brain_module()
        try:
            bmode = _coerce_brain_mode(gb, mode)
        except ValueError:
            valid = [m.value for m in gb.BrainMode]
            return {
                "status": "error",
                "message": f"unknown mode {mode!r}; expected one of {valid}",
            }
        sp = gb.Setpoint(mode=bmode, vx=float(vx), vy=float(vy), wz=float(wz))
        _g1_setpoints[prim_path] = sp
        entry = _g1_brains.get(prim_path)
        if entry is not None:
            entry["runner"].set_setpoint(setpoint=sp)
        return {
            "status": "success",
            "message": "G1 setpoint updated",
            "prim_path": prim_path,
            "setpoint": {"mode": sp.mode.value, "vx": sp.vx, "vy": sp.vy, "wz": sp.wz},
            "brain_running": _g1_brain_running(prim_path),
            "control_mode": driver.control_mode.name,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_get_status(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
) -> Dict[str, Any]:
    """Derive ``/g1/status`` (upright/fell + a TAINTED teleport flag) from the
    driver's LowState (base tilt via projected-gravity z) plus the articulation
    root pose / contacts read from the adapter.

    READ-ONLY: it NEVER auto-catches a fall — it only reports. The TAINTED flag
    fires when the root pose jumps faster than physics allows (> v_max*dt) with no
    matching motor effort or ground contact, i.e. a ``set_world_pose`` teleport
    rather than solver-driven motion (the Phase-1 teleport detector, applied to a
    playing sim). Fail-soft: a missing root pose / contact read still returns the
    LowState-derived tilt.
    """
    try:
        driver = _g1_drivers.get(prim_path)
        lowstate_bytes = None
        if driver is not None and driver.installed:
            lowstate_bytes = driver.latest_lowstate_bytes()

        root_position = None
        try:
            tf = adapter.get_prim_transform(prim_path)
            pos = tf.get("position") if isinstance(tf, dict) else None
            if pos is not None:
                root_position = [float(c) for c in pos]
        except Exception:
            pass

        contact_force = None
        try:
            ps = adapter.get_physics_state(prim_path)
            contacts = ps.get("contacts") if isinstance(ps, dict) else None
            if contacts is not None:
                contact_force = _g1_contact_magnitude(contacts)
        except Exception:
            pass

        if lowstate_bytes is None:
            return {
                "status": "success",
                "prim_path": prim_path,
                "driver_installed": bool(driver is not None and driver.installed),
                "state_available": False,
                "message": "no LowState yet (driver not installed or not stepped)",
            }

        mon = _g1_monitor(prim_path)
        st = mon.update_bytes(
            lowstate_bytes, root_position=root_position, contact_force=contact_force
        )
        if st is None:
            return {"status": "error", "message": "could not decode LowState frame"}
        return {
            "status": "success",
            "prim_path": prim_path,
            "driver_installed": True,
            "state_available": True,
            "control_mode": driver.control_mode.name,
            "brain_running": _g1_brain_running(prim_path),
            "status_detail": st.to_dict(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def g1_reset(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/G1",
) -> Dict[str, Any]:
    """Re-author the G1 to a fixed neutral pose — only while the timeline is STOPPED.

    Reuses the Phase-1 reset constraint: a DOF re-pose is honest only while the
    sim is stopped (a playing/paused re-pose is a teleport the guards forbid), and
    is rate-limited so it cannot be spammed. The neutral pose is all-zeros.
    TODO(probe:neutral-pose): zeros is a straight-legged stand; replace with a
    measured crouch/home pose once available.
    """
    try:
        state = adapter.get_simulation_state()
        timeline = state.get("timeline_state")
        if timeline != "stopped":
            return {
                "status": "error",
                "message": (
                    f"robots.g1.reset requires timeline stopped (got {timeline!r}); "
                    "stop the sim to re-author the neutral pose"
                ),
                "rejected_by": "session_guard",
            }
        now = time.monotonic()
        last = _g1_last_reset.get(prim_path)
        if last is not None and (now - last) < _G1_RESET_MIN_INTERVAL_S:
            return {
                "status": "error",
                "message": (
                    f"robots.g1.reset rate-limited "
                    f"(min {_G1_RESET_MIN_INTERVAL_S}s between resets)"
                ),
            }
        info = adapter.get_robot_joint_info(prim_path)
        num_dof = int(info.get("num_dof") or 0)
        if num_dof <= 0:
            return {"status": "error", "message": f"could not resolve DoF count for {prim_path}"}
        adapter.set_joint_positions(prim_path, [0.0] * num_dof)
        _g1_last_reset[prim_path] = now
        # Forget teleport history so this legitimate stopped re-pose is not later
        # flagged TAINTED by robots.g1.get_status.
        mon = _g1_monitors.get(prim_path)
        if mon is not None:
            mon.reset()
        return {
            "status": "success",
            "message": f"Reset {prim_path} to neutral pose",
            "prim_path": prim_path,
            "num_dof": num_dof,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
