"""In-process Unitree G1 ``LowCmd`` -> Isaac Sim driver (physics-rate control).

Bridges the Unitree ``unitree_hg`` low-level interface (``rt/lowcmd`` /
``rt/lowstate``) onto a running Isaac Sim ``SingleArticulation`` on the on-image
``g1.usd`` (43 DoF: 29 body + 14 Dex3 hands).

Threading model (load-bearing -- see CLAUDE / probe-results.md):
  * The PHYSICS-STEP callback runs on Isaac's physics thread. It does the minimum
    RT work: read the latest ALREADY-VALIDATED inbound LowCmd from a seqlock,
    remap the 29 motor entries to Isaac DoF indices, clamp dq/tau, set gains on
    change, drive PhysX (additive apply_action), then SNAPSHOT the solved q/dq/
    measured-efforts + base pose/vel into a raw buffer (a cheap memcpy). It does
    NO struct.pack and NO CRC, never touches rclpy/DDS, and never blocks (any
    exception is swallowed so it cannot wedge the sim loop).
  * Inbound CRC32 + mode_pr/mode_machine verification happens in submit_lowcmd on
    the PRODUCER thread (DDS side thread / MCP handler), never on the physics tick.
    A threading.Lock serializes the two producers so the single-writer seqlock
    invariant holds.
  * A PACKER SIDE THREAD reads the latest raw snapshot and does the LowState
    struct.pack + CRC at the decimated publish rate, writing packed bytes to the
    outbound seqlock -- so the physics tick pays no pack/CRC cost.
  * All rclpy/DDS lives on a SIDE THREAD that only shuffles bytes through the two
    seqlocks at a decimated rate. rclpy must never be called from the physics
    callback (GIL contention would stall the sim).

Effort semantics (CONFIRMED, probe-results.md 6.0.0-rc.22): PhysX SUMS the
ArticulationAction feed-forward effort on top of the DriveAPI PD torque:
``total = kp*(q*-q) + kd*(dq*-dq) + tau_ff`` clamped to the USD maxForce. So a
LowCmd (q, dq, tau, kp, kd) maps to: ``set_gains(kp, kd)`` (SI/radian, on change) +
``apply_action(joint_positions=q, joint_velocities=dq, joint_efforts=tau)``.

omni / isaacsim / numpy are imported lazily inside methods so this module (and the
codec + index map it re-exports) import under plain ``python3`` for tests.

TODO(probe:dds-msgs): the DDS bridge needs the colcon-built ``unitree_hg`` python
msgs (built in the session Dockerfile, absent here) AND a recorded real capture to
validate ROS<->packed byte parity + QoS/topic names. It degrades to disabled if
they are unavailable.
TODO(probe:mode-machine): ``mode_machine`` is chosen by the robot and echoed by the
client; the real numeric value must be set by the operator / read from a live
LowState. Default 0 (sim); inbound frames whose mode_machine mismatches are rejected.
TODO(probe:imu): the LowState IMU quaternion component order/sign is unverified --
it is passed through as (w, x, y, z) from get_world_pose. The gyroscope is the
world angular velocity rotated into the BASE frame (quat inverse) and rpy is
derived from the quaternion; both inherit that order/sign caveat.
TODO(probe:imu-accel): the accelerometer is a kinematic finite-difference of the
base linear velocity rotated into the base frame -- it is NOT the mounted-link
specific force (a - g); the gravity term + mount offset are unverified.
"""
from __future__ import annotations

import logging
import math
import struct
import threading
import time
from enum import IntEnum
from typing import Any, Dict, Optional

from . import gains as g1_gains
from .crc32 import crc32_of_struct
from .motor_index_map import (
    UNITREE_G1_JOINT_NAMES,
    assert_dof_layout,
    build_index_maps,
)
from .unitree_hg import (
    LOWCMD_SIZE,
    NUM_MOTOR,
    G1JointIndex,
    LowCmd,
    LowState,
    ModePR,
)

logger = logging.getLogger(__name__)

_N_BODY = len(UNITREE_G1_JOINT_NAMES)  # 29
# Coupling guard: the by-name map must stay 1:1 with the packed codec's motor enum.
assert _N_BODY == len(G1JointIndex) == 29


class ControlMode(IntEnum):
    """Driver control mode."""

    RAW = 0       # LowCmd passed straight through to PhysX (default).
    ASSISTED = 1  # reserved for the Phase-3 brain (currently behaves as RAW).


class _SeqlockBuffer:
    """Single-writer / multi-reader seqlock over one immutable-by-convention payload.

    The payload is a single object that the writer never mutates after publishing
    (packed ``bytes`` on the outbound/inbound-command path, a freshly decoded
    ``LowCmd`` on the inbound path, or a fresh snapshot tuple on the state path), so
    under the GIL a reference swap is atomic and can never tear. The even/odd
    sequence counter is kept anyway so this stays correct if it is later ported to a
    real mmap ring, and so readers get a monotonically increasing completed-write
    count for dedup.
    """

    __slots__ = ("_seq", "_payload")

    def __init__(self) -> None:
        self._seq = 0
        self._payload: Optional[Any] = None

    def write(self, payload: Optional[Any]) -> None:
        self._seq += 1          # -> odd: write in progress
        self._payload = payload  # atomic ref swap (GIL)
        self._seq += 1          # -> even: write complete

    def read(self) -> "tuple[int, Optional[Any]]":
        for _ in range(8):
            s1 = self._seq
            if s1 & 1:           # writer mid-write; retry
                continue
            payload = self._payload
            if self._seq == s1:  # unchanged across the read -> consistent
                return s1 >> 1, payload
        return self._seq >> 1, self._payload


class G1LowCmdDriver:
    """Drive a running Isaac Sim G1 articulation from Unitree LowCmd frames.

    Public interface (what the MCP handlers / DDS thread call):
      install(articulation, prim_path=None) -> None
      start() -> None            begin applying commands + start the DDS thread
      stop() -> None             estop: hold + zero efforts, stop the DDS thread
      uninstall() -> None        full teardown (unsubscribes the physics callback)
      set_control_mode(mode: ControlMode) -> None
      submit_lowcmd(cmd_bytes: bytes) -> bool     write inbound seqlock (any thread)
      latest_lowstate_bytes() -> Optional[bytes]  packed, CRC-valid LowState
      latest_lowstate() -> Optional[LowState]     decoded convenience
      stats() -> dict
      properties: installed, running, control_mode
    """

    def __init__(
        self,
        *,
        control_mode: ControlMode = ControlMode.RAW,
        enable_dds: bool = True,
        lowcmd_topic: str = "rt/lowcmd",
        lowstate_topic: str = "rt/lowstate",
        lowstate_hz: float = 500.0,
        lowstate_publish_stride: int = 1,
        mode_machine: int = 0,
        strict_mode_machine: bool = True,
        cmd_timeout_s: Optional[float] = None,
    ) -> None:
        self._control_mode = ControlMode(control_mode)
        self._enable_dds = enable_dds
        self._lowcmd_topic = lowcmd_topic
        self._lowstate_topic = lowstate_topic
        self._lowstate_hz = float(lowstate_hz)
        self._lowstate_publish_stride = max(1, int(lowstate_publish_stride))
        # mode_machine echoed into outbound LowState AND required on inbound LowCmd.
        self._mode_machine = int(mode_machine)
        self._strict_mode_machine = bool(strict_mode_machine)
        self._cmd_timeout_s = cmd_timeout_s

        # Wiring, filled by install().
        self._np = None
        self._ArticulationAction = None
        self._art = None
        self._controller = None
        self._physx = None
        self._prim_path: Optional[str] = None
        self._step_sub = None
        self._maps = None
        self._u2i: tuple = ()
        self._joint_indices = None  # np.ndarray of the 29 body Isaac DoF indices
        self._installed = False
        self._started = False

        # Seqlocks: inbound carries a decoded+validated LowCmd, state carries the
        # physics-thread raw snapshot tuple, outbound carries the packed LowState.
        self._inbound = _SeqlockBuffer()
        self._state = _SeqlockBuffer()
        self._outbound = _SeqlockBuffer()
        # Serialize the two inbound producers (DDS side thread + MCP submit_lowcmd)
        # so the single-writer seqlock invariant holds. Never taken on the physics
        # thread (which only reads inbound).
        self._submit_lock = threading.Lock()

        # Physics-thread scratch (allocated in install()).
        self._q = self._dq = self._tau = self._kp = self._kd = None
        self._full_kp = self._full_kd = None
        self._applied_kp = self._applied_kd = None
        self._prev_lin_world = None  # for the accelerometer finite-difference
        # Gravity field vector (world frame) for IMU specific force (a - g); a real
        # accelerometer reads ~+g along body-up at rest, not 0. Z-up default;
        # TODO(probe:imu-accel): read the live gravity/up-axis from the scene.
        self._gravity_world = (0.0, 0.0, -9.81)
        self._ls: Optional[LowState] = None  # owned by the packer thread

        # Physics-thread state.
        self._applied_seq = -1
        self._tick = 0
        self._n_steps = 0
        self._last_cmd_step = -1
        self._had_cmd = False
        self._estopped = False

        # DDS + packer side threads.
        self._dds_thread: Optional[threading.Thread] = None
        self._dds_running = False
        self._packer_thread: Optional[threading.Thread] = None
        self._packer_running = False

        # Counters (read by stats(); ints -> atomic increments under GIL).
        self._n_applied = 0
        self._n_rejected = 0
        self._n_published = 0
        self._n_step_errors = 0
        self._n_dds_rx = 0
        self._n_dds_tx = 0
        self._last_error = ""

    # ── properties ──────────────────────────────────────────────────────────
    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def running(self) -> bool:
        return self._started

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    # ── lifecycle ───────────────────────────────────────────────────────────
    def install(self, articulation: Any, *, prim_path: Optional[str] = None) -> None:
        """Bind to a running SingleArticulation, build the index maps (fail-closed),
        allocate seqlock/scratch, and subscribe the physics-step callback.

        Must be called after the articulation is initialized (dof_names populated)
        and physics is playing. Registers via
        ``omni.physx.get_physx_interface().subscribe_physics_step_events`` -- NOT an
        OmniGraph OnPlaybackTick node.
        """
        import numpy as np
        import omni.physx
        from isaacsim.core.utils.types import ArticulationAction

        self._np = np
        self._ArticulationAction = ArticulationAction

        dof_names = list(articulation.dof_names) if articulation.dof_names else []
        # Fail-closed BEFORE binding: a bad layout must abort spawn, not mis-drive.
        assert_dof_layout(dof_names)
        self._maps = build_index_maps(dof_names)
        self._u2i = self._maps.unitree_to_isaac
        self._joint_indices = np.array(self._u2i, dtype=np.int64)

        self._art = articulation
        self._controller = articulation.get_articulation_controller()
        self._prim_path = (
            prim_path
            or getattr(articulation, "prim_path", None)
            or getattr(articulation, "_prim_path", None)
        )
        self._physx = omni.physx.get_physx_interface()

        num_dof = self._maps.num_dof
        # 29-wide command scratch (Unitree slot order).
        self._q = np.zeros(_N_BODY, dtype=np.float64)
        self._dq = np.zeros(_N_BODY, dtype=np.float64)
        self._tau = np.zeros(_N_BODY, dtype=np.float64)
        self._kp = np.zeros(_N_BODY, dtype=np.float64)
        self._kd = np.zeros(_N_BODY, dtype=np.float64)
        # Full-length gain arrays (seed from current controller gains so untouched
        # hand DoF keep theirs); set_gains takes full arrays for cross-version safety.
        # NB: a zeros seed here would be scattered over the 14 excluded Dex3 hand
        # DoF on the first set_gains, ZEROING their gains -- so if get_gains() is
        # unavailable we leave the seed UNSET and resolve it lazily from a live
        # get_gains() at the first gain change (_write_gains), scattering only the
        # 29 body indices. Hand-DoF gains are never fabricated/overwritten.
        try:
            cur = self._controller.get_gains()
        except Exception:
            cur = None
        if cur is not None and cur[0] is not None:
            self._full_kp = np.array(cur[0], dtype=np.float64)
            self._full_kd = np.array(cur[1], dtype=np.float64)
        else:
            self._full_kp = None
            self._full_kd = None
        self._applied_kp = np.full(_N_BODY, np.nan, dtype=np.float64)
        self._applied_kd = np.full(_N_BODY, np.nan, dtype=np.float64)

        # Preallocated LowState reused by the PACKER thread (mutated in place, then
        # packed to a fresh immutable bytes for the outbound seqlock). Only the
        # packer thread touches it after this seed.
        self._ls = LowState()
        self._ls.mode_pr = int(ModePR.PR)

        # Seed the outbound seqlock so a reader before the first pack gets a valid
        # (zeroed) CRC'd frame rather than None.
        self._outbound.write(self._crc_pack(self._ls))

        self._step_sub = self._physx.subscribe_physics_step_events(
            self._on_physics_step
        )
        # Packer thread runs from install to uninstall (observability continues
        # even when not started/estopped), off the physics thread.
        self._packer_running = True
        self._packer_thread = threading.Thread(
            target=self._packer_run, name="g1-lowstate-packer", daemon=True
        )
        self._packer_thread.start()
        self._installed = True
        logger.info(
            "G1LowCmdDriver installed on %s (num_dof=%d, 29 body mapped, %d hands ignored)",
            self._prim_path,
            num_dof,
            len(self._maps.ignored_isaac_dofs),
        )

    def start(self) -> None:
        """Begin applying inbound LowCmd and, if enabled, start the DDS side thread."""
        if not self._installed:
            raise RuntimeError("G1LowCmdDriver.start() before install()")
        self._estopped = False
        self._started = True
        if self._enable_dds and self._dds_thread is None:
            self._dds_running = True
            self._dds_thread = threading.Thread(
                target=self._dds_run, name="g1-lowcmd-dds", daemon=True
            )
            self._dds_thread.start()

    def stop(self) -> None:
        """Estop: stop applying commands, stop the DDS thread, and zero efforts once.

        PhysX latches the last actuation force, so a dropped/held command would keep
        pushing torque; zeroing efforts on stop clears that (probe-results.md item 5).
        """
        self._started = False
        self._dds_running = False
        thread = self._dds_thread
        self._dds_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._zero_efforts()
        # Drop the accel finite-difference baseline so the first post-resume/reset
        # snapshot does not divide a velocity discontinuity by dt into a spurious spike.
        self._prev_lin_world = None
        self._estopped = True

    def uninstall(self) -> None:
        """Full teardown: stop, unsubscribe the physics callback, drop references."""
        if self._installed:
            self.stop()
        # Stop the packer thread (it outlives stop() for observability).
        self._packer_running = False
        packer = self._packer_thread
        self._packer_thread = None
        if packer is not None and packer.is_alive():
            packer.join(timeout=2.0)
        sub = self._step_sub
        self._step_sub = None
        if sub is not None:
            unsub = getattr(sub, "unsubscribe", None)
            if callable(unsub):
                try:
                    unsub()
                except Exception:
                    pass
        self._installed = False
        self._art = None
        self._controller = None
        self._physx = None

    def set_control_mode(self, mode: ControlMode) -> None:
        self._control_mode = ControlMode(mode)

    # ── inbound / outbound (any thread) ───────────────────────────────────────
    def submit_lowcmd(self, cmd_bytes: bytes) -> bool:
        """Decode + verify a packed LowCmd frame and publish it to the inbound seqlock.

        Runs on the PRODUCER thread (DDS side thread / MCP handler), NOT the physics
        tick: length, CRC32 and mode_pr/mode_machine are all checked here so the
        physics callback consumes an already-validated command and never pays codec
        cost. Invalid frames are rejected (return False). A lock serializes the two
        producers, preserving the single-writer seqlock invariant.
        """
        if not isinstance(cmd_bytes, (bytes, bytearray)):
            return False
        if len(cmd_bytes) != LOWCMD_SIZE:
            return False
        raw = bytes(cmd_bytes)
        try:
            cmd = LowCmd.unpack(raw)
        except Exception:
            self._n_rejected += 1
            return False
        # CRC the RECEIVED bytes directly rather than re-packing the decoded object
        # (verify_crc() would pack the whole frame again -- wasted pure-Python work on
        # this producer thread). The trailing 4 bytes are the little-endian crc word.
        expected_crc = struct.unpack_from("<I", raw, LOWCMD_SIZE - 4)[0]
        if not (
            crc32_of_struct(raw) == expected_crc
            and cmd.mode_pr == int(ModePR.PR)
            and (not self._strict_mode_machine or cmd.mode_machine == self._mode_machine)
        ):
            self._n_rejected += 1
            return False
        # Publish the decoded, immutable-by-convention command. Serialize the two
        # inbound producers; the decode/verify above stays outside the lock.
        with self._submit_lock:
            self._inbound.write(cmd)
        return True

    def latest_lowstate_bytes(self) -> Optional[bytes]:
        """Latest packed, CRC-valid LowState frame (or None before the first step)."""
        _, buf = self._outbound.read()
        return buf

    def latest_lowstate(self) -> Optional[LowState]:
        buf = self.latest_lowstate_bytes()
        if buf is None:
            return None
        return LowState.unpack(buf)

    def stats(self) -> Dict[str, Any]:
        return {
            "installed": self._installed,
            "running": self._started,
            "estopped": self._estopped,
            "control_mode": self._control_mode.name,
            "prim_path": self._prim_path,
            "num_dof": self._maps.num_dof if self._maps else None,
            "mode_machine": self._mode_machine,
            "steps": self._n_steps,
            "applied": self._n_applied,
            "rejected": self._n_rejected,
            "published": self._n_published,
            "step_errors": self._n_step_errors,
            "dds_rx": self._n_dds_rx,
            "dds_tx": self._n_dds_tx,
            "last_error": self._last_error,
        }

    # ── physics-step callback (physics thread; MUST stay light, never raise) ──
    def _on_physics_step(self, dt: float) -> None:
        """Light RT work only: apply the latest command + snapshot solved state.

        NO struct.pack / NO CRC here -- the LowState pack+CRC is done off-thread by
        the packer thread reading the snapshot (see _packer_run / _snapshot_state).
        """
        self._n_steps += 1
        try:
            if self._started and not self._estopped:
                self._apply_latest_cmd()
            # Cheap raw snapshot of solved state for the packer thread (even when
            # not started -> observability). No pack, no CRC.
            self._snapshot_state(dt)
        except Exception as exc:  # never propagate out of the physics loop
            self._n_step_errors += 1
            self._last_error = repr(exc)

    def _apply_latest_cmd(self) -> None:
        seq, cmd = self._inbound.read()
        if cmd is None:
            return
        if seq == self._applied_seq:
            # No new frame -> PhysX latches the last targets/forces. Optional
            # staleness watchdog (real hardware); disabled by default in sim.
            self._maybe_watchdog(dt_steps=self._n_steps - self._last_cmd_step)
            return
        self._applied_seq = seq
        # cmd is already length/CRC/mode validated in submit_lowcmd (producer
        # thread) -> the tick pays no codec cost, only the light remap below.

        np = self._np
        q, dq, tau, kp, kd = self._q, self._dq, self._tau, self._kp, self._kd
        mc = cmd.motor_cmd
        names = UNITREE_G1_JOINT_NAMES
        for u in range(_N_BODY):
            m = mc[u]
            q[u] = m.q
            dq[u] = g1_gains.clamp_velocity(names[u], m.dq)
            # Fail-closed effort clamp (hands/unknown -> 0); serial-space PR clamp.
            # TODO(probe:motor-space): ankle (confirmed) + waist (inferred) are
            # parallel A/B linkages; the true clamp is in motor space via the
            # linkage Jacobian, not this per-joint PR scalar.
            tau[u] = g1_gains.clamp_effort_fail_closed(names[u], m.tau)
            kp[u] = m.kp
            kd[u] = m.kd

        # ASSISTED mode is reserved for the Phase-3 brain; today it is RAW.
        # TODO(phase3): blend a brain command from a third seqlock here.

        if not np.array_equal(kp, self._applied_kp) or not np.array_equal(
            kd, self._applied_kd
        ):
            self._write_gains(kp, kd)
            self._applied_kp[:] = kp
            self._applied_kd[:] = kd

        action = self._ArticulationAction(
            joint_positions=q,
            joint_velocities=dq,
            joint_efforts=tau,  # additive feed-forward (CONFIRMED)
            joint_indices=self._joint_indices,
        )
        self._controller.apply_action(action)
        self._n_applied += 1
        self._had_cmd = True
        self._last_cmd_step = self._n_steps

    def _write_gains(self, kp, kd) -> None:
        """set_gains scattering ONLY the 29 body indices; hand-DoF gains untouched.

        Uses the install-time full-length seed (which carried the real hand-DoF
        gains) so the scatter leaves the 14 excluded Dex3 hand DoF alone. If that
        seed was unavailable at install, re-read the live full gains now (so hand
        gains stay their current values) before scattering; only if even that fails
        do we fall back to an indexed set_gains over the body indices -- never a
        fabricated full array that would zero the hands.
        """
        np = self._np
        if self._full_kp is None:
            try:
                cur = self._controller.get_gains()
            except Exception:
                cur = None
            if cur is not None and cur[0] is not None:
                self._full_kp = np.array(cur[0], dtype=np.float64)
                self._full_kd = np.array(cur[1], dtype=np.float64)
        # SI/radian tensor path -- Unitree kp/kd pass through unscaled
        # (probe-results.md; g1.gains.kp_to_tensor_stiffness is identity).
        if self._full_kp is not None:
            self._full_kp[self._joint_indices] = kp
            self._full_kd[self._joint_indices] = kd
            self._controller.set_gains(kps=self._full_kp, kds=self._full_kd)
        else:
            self._controller.set_gains(
                kps=np.asarray(kp), kds=np.asarray(kd),
                joint_indices=self._joint_indices,
            )

    def _maybe_watchdog(self, dt_steps: int) -> None:
        if self._cmd_timeout_s is None or not self._had_cmd or self._estopped:
            return
        # Approximate step->time via the configured lowstate rate as a proxy; the
        # physics dt is passed to the callback but not stored to keep this light.
        # TODO(probe:watchdog) key this off the real physics dt for exactness.
        timeout_steps = max(1, int(self._cmd_timeout_s * self._lowstate_hz))
        if dt_steps > timeout_steps:
            self._zero_efforts()
            self._estopped = True
            self._last_error = "lowcmd watchdog timeout"

    def _zero_efforts(self) -> None:
        if self._controller is None or self._np is None:
            return
        try:
            action = self._ArticulationAction(
                joint_efforts=self._np.zeros(_N_BODY, dtype=self._np.float64),
                joint_indices=self._joint_indices,
            )
            self._controller.apply_action(action)
        except Exception as exc:
            self._last_error = f"zero_efforts failed: {exc!r}"

    # ── raw state snapshot (physics thread; cheap memcpy, NO pack / NO CRC) ────
    def _snapshot_state(self, dt: float) -> None:
        """Copy the solved joint state + base pose/vel into an immutable snapshot.

        Runs on the physics thread: it only READS PhysX (the solved state must be
        read on the physics thread) and COPIES into a fresh tuple published to the
        state seqlock. The expensive LowState struct.pack + CRC is done off-thread
        by the packer thread reading this snapshot -- the tick pays no codec cost.
        """
        art = self._art
        np = self._np
        qf = dqf = tauf = None
        try:
            qf = art.get_joint_positions()
            dqf = art.get_joint_velocities()
        except Exception:
            pass
        try:
            tauf = art.get_measured_joint_efforts()
        except Exception:
            tauf = None

        quat = ang = lin = None
        try:
            _pos, quat = art.get_world_pose()
        except Exception:
            quat = None
        try:
            ang = art.get_angular_velocity()
        except Exception:
            ang = None
        if ang is None and self._prim_path:
            try:
                rb = self._physx.get_rigidbody_transformation(self._prim_path)
                if rb and rb.get("ret_val", False):
                    ang = rb.get("angular_velocity", None)
            except Exception:
                ang = None
        try:
            lin = art.get_linear_velocity()
        except Exception:
            lin = None

        # World-frame base linear acceleration via finite difference (exact physics
        # dt; the packer rotates it into the base frame). See TODO(probe:imu-accel).
        accel = None
        if lin is not None and dt and dt > 0.0:
            cur = np.array(lin, dtype=np.float64)
            if self._prev_lin_world is not None:
                accel = (cur - self._prev_lin_world) / dt
            self._prev_lin_world = cur

        # TODO(probe:tick-timebase): this is a per-physics-step counter, NOT the real
        # robot's ~1 kHz millisecond tick -- a consumer deriving dt from tick deltas
        # gets the physics-step rate, not wall-ms. Confirm the real tick unit (ms vs
        # us) against a capture and key it off the accumulated physics time.
        self._tick = (self._tick + 1) & 0xFFFFFFFF
        snap = (
            self._tick,
            None if qf is None else np.array(qf, dtype=np.float64),
            None if dqf is None else np.array(dqf, dtype=np.float64),
            None if tauf is None else np.array(tauf, dtype=np.float64),
            None if quat is None
            else (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            None if ang is None
            else (float(ang[0]), float(ang[1]), float(ang[2])),
            None if accel is None
            else (float(accel[0]), float(accel[1]), float(accel[2])),
        )
        self._state.write(snap)

    # ── LowState pack + CRC (packer thread; owns self._ls) ─────────────────────
    @staticmethod
    def _crc_pack(ls: LowState) -> bytes:
        """Pack a LowState once and patch its trailing CRC-32 word."""
        raw = ls.pack()
        crc = crc32_of_struct(raw)
        return raw[:-4] + struct.pack("<I", crc)

    def _pack_lowstate_from_snapshot(self, snap: tuple) -> bytes:
        """Fill the reused LowState from a physics snapshot, pack + CRC (off-thread)."""
        tick, qf, dqf, tauf, quat, ang, accel = snap
        ls = self._ls
        ls.mode_pr = int(ModePR.PR)
        ls.mode_machine = self._mode_machine
        ls.tick = tick
        u2i = self._u2i
        for u in range(_N_BODY):
            ii = u2i[u]
            ms = ls.motor_state[u]
            ms.mode = 1
            ms.q = float(qf[ii]) if qf is not None else 0.0
            ms.dq = float(dqf[ii]) if dqf is not None else 0.0
            ms.tau_est = float(tauf[ii]) if tauf is not None else 0.0
            # TODO(probe:motor-ddq): motor_state.ddq stays 0 -- Isaac exposes no
            # per-joint acceleration and the physics tick does not finite-difference
            # dq. Real LowState reports a (filtered) ddq; a consumer reading it here
            # gets a constant 0. Populate + validate filtering/units against a capture.
        self._fill_imu(ls, quat, ang, accel)
        return self._crc_pack(ls)

    def _fill_imu(self, ls: LowState, quat, ang_world, accel_world) -> None:
        """Populate the IMU proxy from the snapshot's base pose/velocity.

        The IMU is body-fixed, so the world-frame angular velocity / linear
        acceleration are rotated into the BASE frame via the inverse of the base
        orientation quaternion; rpy is derived from that quaternion.
        TODO(probe:imu): the real IMU is on a mounted link with its own frame; this
        uses the articulation-root frame, and the (w, x, y, z) quaternion order/sign
        is unverified (both the gyro rotation and rpy inherit that caveat).
        """
        imu = ls.imu_state
        # Reset every field first: self._ls is reused in place, so on a read failure
        # a stale (last-good) IMU value would otherwise persist against a fresh tick.
        imu.quaternion = (1.0, 0.0, 0.0, 0.0)
        imu.rpy = (0.0, 0.0, 0.0)
        imu.gyroscope = (0.0, 0.0, 0.0)
        imu.accelerometer = (0.0, 0.0, 0.0)
        if quat is not None:
            imu.quaternion = (
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
            imu.rpy = _quat_to_rpy(quat)
        if ang_world is not None:
            imu.gyroscope = (
                _quat_rotate_inverse(quat, ang_world)
                if quat is not None
                else (float(ang_world[0]), float(ang_world[1]), float(ang_world[2]))
            )
        # Specific force a_proper = a_world - g_world -- what a real accelerometer
        # reports (~+g along body-up at rest), then rotated into the body frame.
        # Kinematic a_world defaults to 0 before the first finite-difference / right
        # after a reset. TODO(probe:imu-accel): mounted-link offset + exact live
        # gravity/up-axis are unverified.
        if quat is not None:
            ax, ay, az = accel_world if accel_world is not None else (0.0, 0.0, 0.0)
            gx, gy, gz = self._gravity_world
            imu.accelerometer = _quat_rotate_inverse(quat, (ax - gx, ay - gy, az - gz))

    # ── packer side thread (pack + CRC off the physics thread) ─────────────────
    def _packer_run(self) -> None:
        """Pack the latest snapshot -> outbound seqlock at the decimated publish rate.

        Runs from install to uninstall so observability continues even when not
        started. Dedups on the snapshot sequence so a paused sim is not re-packed.
        """
        period = (
            self._lowstate_publish_stride / self._lowstate_hz
            if self._lowstate_hz > 0
            else 0.002
        )
        last_seq = -1
        while self._packer_running:
            seq, snap = self._state.read()
            if snap is not None and seq != last_seq:
                last_seq = seq
                try:
                    self._outbound.write(self._pack_lowstate_from_snapshot(snap))
                    self._n_published += 1
                except Exception as exc:
                    self._n_step_errors += 1
                    self._last_error = f"packer: {exc!r}"
            time.sleep(period)

    # ── DDS side thread (rclpy; NEVER call anything here from the physics cb) ──
    def _dds_run(self) -> None:
        """Bridge rt/lowcmd -> inbound seqlock and outbound seqlock -> rt/lowstate.

        All rclpy work stays on this thread. If rclpy or the vendored ``unitree_hg``
        msgs are unavailable (they are colcon-built in the session image, not here),
        the bridge disables itself and the in-process seqlock path still works via
        submit_lowcmd / latest_lowstate_bytes.
        """
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from unitree_hg.msg import LowCmd as RosLowCmd  # noqa: F401
            from unitree_hg.msg import LowState as RosLowState  # noqa: F401
        except Exception as exc:
            logger.warning(
                "G1 DDS bridge disabled (rclpy/unitree_hg unavailable): %s", exc
            )
            self._enable_dds = False
            return

        # TODO(probe:dds-msgs): QoS + field mapping are the design's best guess;
        # validate against a recorded rt/lowcmd/rt/lowstate capture. High-rate
        # low-level control is typically best-effort, depth 1.
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        if not rclpy.ok():
            rclpy.init(args=None)
        node = Node("g1_lowcmd_driver_bridge")

        def _on_ros_lowcmd(msg: Any) -> None:
            try:
                self.submit_lowcmd(_ros_lowcmd_to_bytes(msg))
                self._n_dds_rx += 1
            except Exception as exc:
                self._last_error = f"dds rx: {exc!r}"

        node.create_subscription(RosLowCmd, self._lowcmd_topic, _on_ros_lowcmd, qos)
        pub = node.create_publisher(RosLowState, self._lowstate_topic, qos)

        period = 1.0 / self._lowstate_hz if self._lowstate_hz > 0 else 0.002
        next_pub = time.monotonic()
        while self._dds_running:
            rclpy.spin_once(node, timeout_sec=0.001)
            now = time.monotonic()
            if now >= next_pub:
                next_pub = now + period
                buf = self.latest_lowstate_bytes()
                if buf is not None:
                    try:
                        pub.publish(_lowstate_bytes_to_ros(buf, RosLowState))
                        self._n_dds_tx += 1
                    except Exception as exc:
                        self._last_error = f"dds tx: {exc!r}"
        node.destroy_node()  # do NOT rclpy.shutdown(): the ROS2 bridge may share it


# ── IMU frame math (pure; runs on the packer thread) ──────────────────────────
def _quat_rotate_inverse(quat_wxyz, v) -> "tuple[float, float, float]":
    """Rotate a WORLD-frame vector into the BASE frame via the inverse of the
    (w, x, y, z) base orientation (IsaacGym ``quat_rotate_inverse``). Matches the
    convention used by adapters/v5.py ``_projected_gravity``. TODO(probe:imu):
    correctness depends on the quaternion component order being (w, x, y, z)."""
    w, x, y, z = (
        float(quat_wxyz[0]), float(quat_wxyz[1]),
        float(quat_wxyz[2]), float(quat_wxyz[3]),
    )
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    scale = 2.0 * w * w - 1.0
    ax, ay, az = vx * scale, vy * scale, vz * scale
    cross_x = y * vz - z * vy
    cross_y = z * vx - x * vz
    cross_z = x * vy - y * vx
    two_w = 2.0 * w
    bx, by, bz = cross_x * two_w, cross_y * two_w, cross_z * two_w
    dot = 2.0 * (x * vx + y * vy + z * vz)
    cx, cy, cz = x * dot, y * dot, z * dot
    return (ax - bx + cx, ay - by + cy, az - bz + cz)


def _quat_to_rpy(quat_wxyz) -> "tuple[float, float, float]":
    """Roll/pitch/yaw (aerospace ZYX Tait-Bryan) from a (w, x, y, z) quaternion.
    TODO(probe:imu): quaternion order/sign AND the rpy convention are unverified."""
    w, x, y, z = (
        float(quat_wxyz[0]), float(quat_wxyz[1]),
        float(quat_wxyz[2]), float(quat_wxyz[3]),
    )
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = 1.0 if sinp > 1.0 else (-1.0 if sinp < -1.0 else sinp)
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


# ── ROS <-> packed conversion (used only by the DDS thread) ───────────────────
# TODO(probe:dds-msgs): field names below follow the unitree_hg .msg contract but
# are unvalidated against the colcon-built msg + a real capture.
def _ros_lowcmd_to_bytes(msg: Any) -> bytes:
    """Convert a ``unitree_hg/msg/LowCmd`` into our packed frame (CRC preserved)."""
    from .unitree_hg import MotorCmd

    lc = LowCmd(
        mode_pr=int(msg.mode_pr),
        mode_machine=int(msg.mode_machine),
        crc=int(msg.crc),
    )
    for i in range(NUM_MOTOR):
        m = msg.motor_cmd[i]
        lc.motor_cmd[i] = MotorCmd(
            mode=int(m.mode),
            q=float(m.q),
            dq=float(m.dq),
            tau=float(m.tau),
            kp=float(m.kp),
            kd=float(m.kd),
            reserve=int(m.reserve),
        )
    lc.reserve = tuple(int(r) for r in msg.reserve[:4])
    return lc.pack()


def _lowstate_bytes_to_ros(buf: bytes, ros_cls: Any) -> Any:
    """Convert our packed LowState frame into a ``unitree_hg/msg/LowState``."""
    ls = LowState.unpack(buf)
    msg = ros_cls()
    msg.mode_pr = ls.mode_pr
    msg.mode_machine = ls.mode_machine
    msg.tick = ls.tick
    msg.crc = ls.crc
    try:
        msg.version = list(ls.version)
    except Exception:
        pass
    imu = msg.imu_state
    imu.quaternion = list(ls.imu_state.quaternion)
    imu.gyroscope = list(ls.imu_state.gyroscope)
    imu.accelerometer = list(ls.imu_state.accelerometer)
    imu.rpy = list(ls.imu_state.rpy)
    for i in range(NUM_MOTOR):
        src = ls.motor_state[i]
        dst = msg.motor_state[i]
        dst.mode = src.mode
        dst.q = src.q
        dst.dq = src.dq
        dst.ddq = src.ddq
        dst.tau_est = src.tau_est
    return msg


__all__ = [
    "ControlMode",
    "G1LowCmdDriver",
]
