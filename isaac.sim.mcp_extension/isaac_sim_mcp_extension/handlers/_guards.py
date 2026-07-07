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

"""Fail-closed session guards for the Isaac Sim MCP bridge.

The session bridge listens on loopback ``:8766`` and bypasses the gateway's
per-scope authorization, so ANY process that can reach the socket could teleport
a robot, disable its gravity, or freeze it kinematic to fake a "balanced" pose
and then report honest-looking telemetry. These guards make such cheats
*structurally impossible* in session mode rather than merely discouraged:

  * pose / DOF-state writes to a robot articulation-root (or any prim that
    descends from one) are refused while the timeline is PLAYING **or PAUSED** —
    a robot may only be re-posed while the sim is STOPPED (authoring);
  * a robot root whose world pose jumped across a pause/play boundary is treated
    as tamper and the resume is refused;
  * setting ``is_kinematic=True`` or ``disableGravity=True`` on a robot body is
    hard-blocked in every timeline state;
  * actuation / step commands are gated behind
    (timeline PLAYING and physics_dt > 0 and gravity is a physical earth-like
    *downward* vector — the SAME envelope enforced on gravity writes, applied to
    the gravity READ from the live PhysicsContext) so telemetry can never be
    produced from a frozen, zero-gravity, or sideways/upward "fake-balance" world.

Every decision fails CLOSED: if the world state needed to clear a guarded command
cannot be determined, the command is REJECTED.

The decision core (:func:`evaluate_command`) is a pure function over a
:class:`WorldState` snapshot, so it is unit-testable without Omniverse; the thin
:class:`SessionGuard` wrapper materialises that snapshot from the live adapter.
TODO(probe:guard-live): validate every live rejection path (robot-root discovery,
gravity read, pose snapshot) on a real box.
"""

from __future__ import annotations

import contextvars
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

REJECTED_BY = "session_guard"

# Per-command capability, set by socket_server.py after it authenticates the
# presented token and threaded into dispatch. ``operator`` is trusted bootstrap
# (warm_slot_agent, g1_autospawn) with the full registry and unrestricted pose
# writes; ``session`` is the untrusted bridge->gateway agent path, which the guards
# below enforce and which cannot reach the exec/script commands. A missing/unknown
# token authenticates as ``session`` — fail-safe to the least-privilege level.
OPERATOR_CAPABILITY = "operator"
SESSION_CAPABILITY = "session"

# Robot world pose must not move by more than this (metres, L-infinity) across a
# pause/play boundary — a paused sim advances no physics, so any jump is tamper.
POSE_JUMP_TOLERANCE = 1.0e-3

# Physical-gravity envelope enforced for SESSION gravity writes (simulation.set_physics
# and scene.create_physics) AND — via the SAME predicate (_gravity_envelope_problem) —
# on the gravity READ back from the live PhysicsContext by the actuation gate. A gate
# that only required gravity != 0 still let a session set gravity=[0,0,0], a tiny,
# sideways, upward, or diagonal vector to trivially "balance" a robot and then report
# honest-looking telemetry; requiring the read gravity to clear this envelope closes
# that. A gravity write / live gravity must be earth-like in magnitude AND point
# predominantly down along the stage up-axis; OPERATOR capability may set anything.
GRAVITY_MIN_MAGNITUDE = 9.0
GRAVITY_MAX_MAGNITUDE = 10.5
# cos of the largest tilt from straight-down still accepted as "predominantly downward"
# (0.9 ≈ 25°): rejects horizontal, upward, and diagonal (e.g. equal down+sideways).
GRAVITY_DOWNWARD_COS = 0.9

# ── Command taxonomy ─────────────────────────────────────────────────────────

# Pose teleports (objects.transform == SingleXFormPrim.set_world_pose over the
# bridge, plus its legacy alias).
POSE_WRITE_COMMANDS = frozenset({"objects.transform", "transform"})

# Instantaneous DOF-STATE writes (set_dof_state / set_joint_positions-as-state).
# None are registered today; this is a forward guard so a future teleport-style
# command cannot silently bypass the lock.
DOF_STATE_WRITE_COMMANDS = frozenset(
    {
        "robots.set_dof_state",
        "objects.set_dof_state",
        "simulation.set_dof_state",
        "robots.set_joint_state",
    }
)

# Motor commands — only meaningful against a live physics world.
#
# The ``robots.g1.*`` per-motor actuation surface is gated here too: a LowCmd
# submitted to the in-process G1 driver (robots.g1.set_joint_command) and a
# per-joint PD gain write (robots.g1.set_gains) only move the robot through the
# PhysX PD solver, so — exactly like robots.set_joints — they must be refused
# unless the timeline is PLAYING with earth-like gravity and a positive physics
# dt. This keeps a session from producing "the robot moved" telemetry out of a
# frozen or zero-gravity world. They are HONEST actuation (a bad command makes
# the robot fall), so they stay session-allowed once the physics gate is met;
# only robots.g1.spawn (which installs the driver / subscribes a physics-step
# callback) is operator-only, enforced at dispatch in extension.py.
ACTUATION_COMMANDS = frozenset(
    {
        "robots.set_joints",
        "robots.set_gains",
        "robots.set_joint_velocities",
        "robots.set_joint_efforts",
        "robots.g1.set_joint_command",
        "robots.g1.set_gains",
    }
)

STEP_COMMANDS = frozenset({"simulation.step"})
PLAY_COMMANDS = frozenset({"simulation.play"})
PAUSE_COMMANDS = frozenset({"simulation.pause"})
STOP_COMMANDS = frozenset({"simulation.stop"})

_TIMELINE_COMMANDS = PLAY_COMMANDS | PAUSE_COMMANDS | STOP_COMMANDS
_STRUCTURAL_WRITE_COMMANDS = POSE_WRITE_COMMANDS | DOF_STATE_WRITE_COMMANDS
_GATED_COMMANDS = ACTUATION_COMMANDS | STEP_COMMANDS
_ALL_GUARDED_COMMANDS = _STRUCTURAL_WRITE_COMMANDS | _GATED_COMMANDS | _TIMELINE_COMMANDS

# Param keys that may carry a prim target, tried in priority order.
_TARGET_KEYS = ("prim_path", "path", "target_path", "robot_path")

# Physics-disable flags (compared case-insensitively) that must never be set on a
# robot body over the bridge.
_KINEMATIC_KEYS = frozenset(
    {"is_kinematic", "kinematic", "kinematic_enabled", "kinematicenabled", "set_kinematic"}
)
_DISABLE_GRAVITY_KEYS = frozenset(
    {"disable_gravity", "disablegravity", "gravity_disabled", "disable_gravities", "no_gravity", "nogravity"}
)


@dataclass(frozen=True)
class WorldState:
    """Immutable snapshot of everything a guard decision depends on.

    ``None`` means "could not be determined" and is treated as fail-closed by the
    decision logic. ``robot_roots=None`` specifically means discovery FAILED (as
    opposed to an empty tuple, which means "no robots present"). ``gravity`` is the
    full gravity VECTOR (m/s^2) read from the live PhysicsContext — not just a
    magnitude — so the actuation gate can enforce direction as well as strength;
    ``gravity=None`` means the read failed (fail-closed).
    """

    timeline_state: Optional[str]
    gravity: Optional[Tuple[float, ...]]
    physics_dt: Optional[float]
    robot_roots: Optional[Tuple[str, ...]]
    up_axis: str = "Z"


# ── Pure helpers ─────────────────────────────────────────────────────────────


def _reject(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message, "rejected_by": REJECTED_BY}


def _timeline_locked(timeline_state: Optional[str]) -> bool:
    """True unless the timeline is *definitively* stopped (fail-closed on unknown)."""
    return timeline_state != "stopped"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _flag_truthy(params: Mapping[str, Any], keys: frozenset) -> bool:
    for raw_key, value in params.items():
        if isinstance(raw_key, str) and raw_key.lower() in keys and _truthy(value):
            return True
    return False


def _extract_target(params: Mapping[str, Any]) -> Optional[str]:
    for key in _TARGET_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _robot_relation(target: Optional[str], roots: Optional[Tuple[str, ...]]) -> Optional[bool]:
    """Whether ``target`` collides with a robot articulation-root — it IS a root, a
    DESCENDANT of one, or an ANCESTOR of one.

    Returns ``True`` when it provably collides, ``False`` when it provably does not,
    and ``None`` when the relationship cannot be determined (discovery failed or no
    resolvable target) — callers treat ``None`` as fail-closed.

    An ANCESTOR collides too: a delete / clear / transform applied to a robot's
    parent (e.g. ``/World`` — or the pseudo-root ``/`` — while the robot lives at
    ``/World/G1``) wipes or teleports the robot *through its parent*, bypassing an
    otherwise-correct per-target sink guard. Checks use the ``/`` component boundary
    in BOTH directions so ``/World`` never falsely matches ``/Worldxyz``; the
    pseudo-root ``/`` normalises to ``""`` and so is an ancestor of every absolute
    root.
    """
    if roots is None or not target:
        return None
    t = target.rstrip("/")
    for root in roots:
        base = root.rstrip("/")
        if (
            t == base
            or base.startswith(t + "/")  # target is an ANCESTOR of the root
            or t.startswith(base + "/")  # target is a DESCENDANT of the root
        ):
            return True
    return False


def _gravity_envelope_problem(
    gravity: Sequence[float], up_axis: str
) -> Optional[str]:
    """The ONE physical-gravity predicate, shared by the gravity write-guard
    (:func:`_gravity_rejection`) and the actuation READ gate (:func:`_gate_problems`).

    Given a concrete gravity 3-vector (m/s^2) and the stage up-axis, return a
    human-readable reason string when it is NOT a physical gravity — else ``None``.
    A physical gravity is earth-like in magnitude (``[GRAVITY_MIN_MAGNITUDE,
    GRAVITY_MAX_MAGNITUDE]``) AND points predominantly *down* along the up-axis
    (``-Z`` for Z-up, ``-Y`` for Y-up; cos-tilt >= ``GRAVITY_DOWNWARD_COS``). This
    is what rejects a session's ``[0,0,0]`` / tiny / sideways / upward / diagonal
    fake-balance vector — whether it is being written or read back off the engine.
    Fail-closed: a malformed / non-3 vector yields a reason (never ``None``).
    """
    try:
        components = [float(c) for c in gravity]
    except (TypeError, ValueError):
        return f"unreadable gravity {gravity!r}"
    if len(components) != 3:
        return f"gravity {gravity!r} is not a 3-vector"
    magnitude = math.sqrt(sum(c * c for c in components))
    if not (GRAVITY_MIN_MAGNITUDE <= magnitude <= GRAVITY_MAX_MAGNITUDE):
        return (
            f"non-physical gravity magnitude {magnitude:.4f} m/s^2 "
            f"(need [{GRAVITY_MIN_MAGNITUDE}, {GRAVITY_MAX_MAGNITUDE}])"
        )
    down_index = 1 if str(up_axis).upper() == "Y" else 2
    axis_name = "Y" if down_index == 1 else "Z"
    axial = components[down_index]
    # Down along -up_axis => the up-axis component must be negative and dominate the
    # magnitude (a small off-axis tilt is fine; a sideways/diagonal vector is not).
    if axial >= 0 or (-axial / magnitude) < GRAVITY_DOWNWARD_COS:
        return (
            f"non-downward gravity {components!r} on a {axis_name}-up stage "
            f"(must point predominantly down -{axis_name})"
        )
    return None


def _gate_problems(state: WorldState) -> Tuple[str, ...]:
    problems = []
    if state.timeline_state != "playing":
        problems.append(f"timeline={state.timeline_state!r} (need 'playing')")
    gravity = state.gravity
    if gravity is None:
        # Read failed / no physics scene -> cannot prove gravity is physical -> reject.
        problems.append("gravity=None (unreadable)")
    else:
        envelope = _gravity_envelope_problem(gravity, state.up_axis)
        if envelope is not None:
            problems.append(envelope)
    dt = state.physics_dt
    if dt is None or dt <= 0.0:
        problems.append(f"physics_dt={dt!r} (need > 0)")
    return tuple(problems)


def max_position_delta(
    before: Mapping[str, Tuple[float, ...]], after: Mapping[str, Tuple[float, ...]]
) -> float:
    """Largest per-axis (L-infinity) position delta of roots present in both snapshots."""
    worst = 0.0
    for root, pos in before.items():
        cur = after.get(root)
        if cur is None or pos is None:
            continue
        for a, b in zip(pos, cur):
            worst = max(worst, abs(float(a) - float(b)))
    return worst


def _gravity_rejection(
    gravity: Optional[Sequence[float]], up_axis: str
) -> Optional[Dict[str, Any]]:
    """Return a rejection dict if ``gravity`` is not a physical gravity write, else
    ``None``. ``gravity=None`` (leave the scene's gravity unchanged) is always allowed.

    A physical write is earth-like in magnitude (``[GRAVITY_MIN_MAGNITUDE,
    GRAVITY_MAX_MAGNITUDE]`` m/s^2) and points predominantly *down* along the stage
    up-axis (``-Z`` for Z-up, ``-Y`` for Y-up). This closes the fake-balance vector
    where a session sets gravity=[0,0,0] (or a tiny/sideways/upward vector). Fail-closed:
    a malformed / non-3 vector rejects. The envelope itself is
    :func:`_gravity_envelope_problem`, shared verbatim with the actuation READ gate.
    """
    if gravity is None:
        return None
    problem = _gravity_envelope_problem(gravity, up_axis)
    if problem is None:
        return None
    return _reject(
        f"{REJECTED_BY}: refusing {problem}; a session may not disable, weaken, or "
        f"tilt gravity to fake balance"
    )


def evaluate_command(cmd_type: str, params: Optional[Mapping[str, Any]], state: WorldState) -> Optional[Dict[str, Any]]:
    """Return a rejection dict if ``cmd_type`` is forbidden by the snapshot, else ``None``."""
    params = params or {}
    target = _extract_target(params)
    roots = state.robot_roots

    # (A) Hard-block physics-disable flags on a robot body — any timeline state.
    kinematic = _flag_truthy(params, _KINEMATIC_KEYS)
    disable_gravity = _flag_truthy(params, _DISABLE_GRAVITY_KEYS)
    if kinematic or disable_gravity:
        if _robot_relation(target, roots) is not False:  # True or None (unknown) -> reject
            which = "is_kinematic=True" if kinematic else "disableGravity=True"
            return _reject(
                f"{REJECTED_BY}: refusing to set {which} on robot body {target or '<unknown>'!r}"
            )

    # (B) Pose / DOF-state writes to a robot root while the timeline is locked.
    if cmd_type in _STRUCTURAL_WRITE_COMMANDS and _timeline_locked(state.timeline_state):
        if _robot_relation(target, roots) is not False:  # True or None (unknown) -> reject
            kind = "pose" if cmd_type in POSE_WRITE_COMMANDS else "DOF-state"
            return _reject(
                f"{REJECTED_BY}: refusing {kind} write to robot {target or '<unknown>'!r} while "
                f"timeline={state.timeline_state!r}; stop the sim to re-author"
            )

    # (C) Actuation / step gate: physics must be live.
    if cmd_type in _GATED_COMMANDS:
        problems = _gate_problems(state)
        if problems:
            return _reject(
                f"{REJECTED_BY}: refusing {cmd_type!r} — physics not live ({'; '.join(problems)})"
            )

    return None


# ── Runtime wiring ───────────────────────────────────────────────────────────


def session_mode_enabled() -> bool:
    """Whether session-safe mode is active. Default ON; only an explicit falsey
    value for ``MCP_SESSION_MODE`` opts into the operator/legacy behaviour."""
    raw = os.environ.get("MCP_SESSION_MODE")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def session_token() -> Optional[str]:
    """Restricted per-session shared secret for the framed socket protocol (the
    untrusted bridge->gateway agent path), or ``None``. A request bearing this token
    — or no token, or an unknown one — authenticates as ``session`` capability."""
    token = os.environ.get("MCP_SESSION_TOKEN")
    return token or None


def operator_token() -> Optional[str]:
    """Full-capability shared secret for trusted bootstrap (warm_slot_agent,
    g1_autospawn), or ``None``. A request whose token matches this authenticates as
    ``operator`` capability: the exec/script commands are permitted and the pose
    guards below are bypassed. Kept distinct from :func:`session_token` so the same
    process can serve both the trusted bootstrap and the locked-down agent."""
    token = os.environ.get("MCP_OPERATOR_TOKEN")
    return token or None


# Capability of the command currently being dispatched. socket_server.py classifies
# it from the presented token and extension.py publishes it here around dispatch so
# the sink-level guards (objects.create/clone -> guard_pose_write) and the ScriptNode
# gate can honour operator/session without every handler growing a capability arg.
# Fail-safe default: the least-privilege ``session`` level.
_CURRENT_CAPABILITY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mcp_command_capability", default=SESSION_CAPABILITY
)


def current_capability() -> str:
    """The capability of the in-flight command; ``session`` when none is set."""
    return _CURRENT_CAPABILITY.get()


def set_current_capability(capability: Optional[str]) -> "contextvars.Token[str]":
    """Publish the in-flight command's capability, normalising anything that is not
    exactly ``operator`` to ``session`` (fail-safe). Returns a token for
    :func:`reset_current_capability`."""
    level = OPERATOR_CAPABILITY if capability == OPERATOR_CAPABILITY else SESSION_CAPABILITY
    return _CURRENT_CAPABILITY.set(level)


def reset_current_capability(token: "contextvars.Token[str]") -> None:
    """Restore the capability in effect before the matching :func:`set_current_capability`."""
    _CURRENT_CAPABILITY.reset(token)


class SessionGuard:
    """Materialises a :class:`WorldState` from the live adapter and applies the
    pure guard decisions, plus stateful pause/play pose-continuity checks.

    ``adapter`` may optionally expose ``list_articulation_roots()`` and
    ``get_gravity_vector()``; when absent the guard falls back to a direct USD
    scan (TODO(probe:guard-live)).
    """

    def __init__(
        self,
        adapter: Any,
        session_mode: bool = True,
        *,
        pose_jump_tolerance: float = POSE_JUMP_TOLERANCE,
    ) -> None:
        self._adapter = adapter
        self._session_mode = session_mode
        self._pose_jump_tolerance = pose_jump_tolerance
        self._pause_pose_snapshot: Optional[Dict[str, Tuple[float, ...]]] = None

    def guard(
        self,
        cmd_type: str,
        params: Optional[Mapping[str, Any]],
        capability: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a rejection dict to short-circuit dispatch, or ``None`` to allow.

        ``capability`` defaults to the in-flight :func:`current_capability`. The
        ``operator`` capability (trusted bootstrap) bypasses the pose/actuation lock
        entirely; ``session`` is enforced.
        """
        if not self._session_mode:
            return None
        if capability is None:
            capability = current_capability()
        if capability == OPERATOR_CAPABILITY:
            return None
        params = params or {}
        # Fast path: skip state collection for commands no guard cares about.
        if cmd_type not in _ALL_GUARDED_COMMANDS and not _has_danger_flag(params):
            return None
        try:
            state = self._collect_state()
            decision = evaluate_command(cmd_type, params, state)
            if decision is not None:
                return decision
            return self._check_pose_continuity(cmd_type, state)
        except Exception as exc:  # never let a guard bug fail OPEN
            return _reject(f"{REJECTED_BY}: internal error, failing closed: {exc}")

    def pose_write_rejection(
        self, target_path: Optional[str], capability: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Sink-level pose/topology guard for handlers whose command type is NOT in
        POSE_WRITE_COMMANDS but whose body still reaches a set_prim_transform /
        clone_prim sink (objects.create, objects.clone). Refuses moving or
        overwriting a robot articulation-root while the timeline is not stopped —
        the same rule evaluate_command applies to objects.transform — so a create
        or clone cannot smuggle a robot teleport past the dispatch-level guard.

        Returns a rejection dict, or ``None`` to allow. Fail-closed: an
        indeterminate world state or a discovery failure rejects. ``capability``
        defaults to the in-flight :func:`current_capability`; ``operator`` bypasses.
        """
        if not self._session_mode:
            return None
        if capability is None:
            capability = current_capability()
        if capability == OPERATOR_CAPABILITY:
            return None
        try:
            state = self._collect_state()
            return evaluate_command("objects.transform", {"prim_path": target_path}, state)
        except Exception as exc:  # never let a guard bug fail OPEN
            return _reject(f"{REJECTED_BY}: internal error, failing closed: {exc}")

    # ── State collection ─────────────────────────────────────────────────────

    def _collect_state(self) -> WorldState:
        timeline_state: Optional[str] = None
        physics_dt: Optional[float] = None
        try:
            sim = self._adapter.get_simulation_state()
            timeline_state = sim.get("timeline_state")
            physics_dt = sim.get("physics_dt")
        except Exception:
            pass  # unknown -> fail closed downstream
        up_axis = _resolve_up_axis(self._adapter)
        return WorldState(
            timeline_state=timeline_state,
            gravity=self._read_gravity_vector(up_axis),
            physics_dt=physics_dt,
            robot_roots=self._discover_robot_roots(),
            up_axis=up_axis,
        )

    def _discover_robot_roots(self) -> Optional[Tuple[str, ...]]:
        getter = getattr(self._adapter, "list_articulation_roots", None)
        if callable(getter):
            try:
                roots = getter()
                return tuple(roots) if roots is not None else None
            except Exception:
                return None
        # Fallback: scan the stage for UsdPhysics.ArticulationRootAPI.
        try:
            from pxr import UsdPhysics

            stage = self._adapter.get_stage()
            return tuple(
                prim.GetPath().pathString
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            )
        except Exception:
            return None  # discovery failed -> fail closed

    def _read_gravity_vector(self, up_axis: str) -> Optional[Tuple[float, ...]]:
        """The full gravity VECTOR (m/s^2) from the live PhysicsContext, or ``None``
        when it cannot be determined (fail-closed). Prefers the adapter's
        ``get_gravity_vector()`` hook (a live ``ctx.get_gravity()`` read); otherwise
        reconstructs direction * magnitude off the authored UsdPhysics.Scene.
        Returning a vector (not just a magnitude) is what lets the actuation gate
        reject a sideways / upward / diagonal fake-balance gravity, not only a zero one.
        """
        getter = getattr(self._adapter, "get_gravity_vector", None)
        if callable(getter):
            try:
                vec = getter()
                return tuple(float(c) for c in vec) if vec is not None else None
            except Exception:
                return None
        # Fallback: reconstruct the vector from the first UsdPhysics.Scene.
        # TODO(probe:guard-live): confirm this USD read equals the live PhysicsContext
        # gravity on a real box (authored USD can lag the running engine).
        try:
            from pxr import UsdPhysics

            stage = self._adapter.get_stage()
            down_index = 1 if str(up_axis).upper() == "Y" else 2
            for prim in stage.Traverse():
                if not (prim.IsA(UsdPhysics.Scene) or prim.GetTypeName() == "PhysicsScene"):
                    continue
                scene = UsdPhysics.Scene(prim)
                magnitude = scene.GetGravityMagnitudeAttr().Get()
                if magnitude is None or magnitude < 0:
                    magnitude = 9.81  # USD sentinel / unset => engine default (earth)
                direction = scene.GetGravityDirectionAttr().Get()
                if direction is None:
                    # Direction unset => engine applies default straight down the up-axis.
                    vec = [0.0, 0.0, 0.0]
                    vec[down_index] = -float(magnitude)
                    return tuple(vec)
                dvec = [float(direction[0]), float(direction[1]), float(direction[2])]
                if not any(abs(c) > 0 for c in dvec):
                    return (0.0, 0.0, 0.0)  # explicit zero direction => no effective gravity
                return tuple(c * float(magnitude) for c in dvec)
            return None  # no physics scene -> unknown -> fail closed
        except Exception:
            return None

    # ── Pause/play pose continuity ───────────────────────────────────────────

    def _check_pose_continuity(self, cmd_type: str, state: WorldState) -> Optional[Dict[str, Any]]:
        if cmd_type in PAUSE_COMMANDS:
            self._pause_pose_snapshot = self._snapshot_root_poses(state)
            return None
        if cmd_type in STOP_COMMANDS:
            self._pause_pose_snapshot = None
            return None
        if cmd_type in PLAY_COMMANDS and self._pause_pose_snapshot is not None:
            paused = self._pause_pose_snapshot
            current = self._snapshot_root_poses(state)
            # Fail CLOSED on any root we captured at pause that we cannot re-read on
            # resume (or a snapshot that shrank): a hidden delete/rename or a read
            # failure would otherwise drop that root from the delta comparison
            # (max_position_delta skips unmatched roots), masking a teleport. Keep the
            # snapshot so retries stay rejected until a stop resets it.
            missing = [root for root in paused if root not in current]
            if missing or len(current) < len(paused):
                return _reject(
                    f"{REJECTED_BY}: robot root(s) {missing or '<snapshot shrank>'} unreadable on "
                    f"resume (had {len(paused)}, now {len(current)}); refusing resume — reset the sim"
                )
            delta = max_position_delta(paused, current)
            if delta > self._pose_jump_tolerance:
                return _reject(
                    f"{REJECTED_BY}: robot root moved {delta:.4f} m while paused "
                    f"(tolerance {self._pose_jump_tolerance} m); refusing resume — reset the sim"
                )
            self._pause_pose_snapshot = None
        return None

    def _snapshot_root_poses(self, state: WorldState) -> Dict[str, Tuple[float, ...]]:
        snapshot: Dict[str, Tuple[float, ...]] = {}
        for root in state.robot_roots or ():
            try:
                transform = self._adapter.get_prim_transform(root)
                position = transform.get("position")
                if position is not None:
                    snapshot[root] = tuple(float(c) for c in position)
            except Exception:
                continue
        return snapshot


def _has_danger_flag(params: Mapping[str, Any]) -> bool:
    return _flag_truthy(params, _KINEMATIC_KEYS) or _flag_truthy(params, _DISABLE_GRAVITY_KEYS)


def _resolve_up_axis(adapter: Any) -> str:
    """Best-effort stage up-axis (``"Z"`` or ``"Y"``); defaults to Isaac's Z-up when it
    cannot be read. The default is deliberate: Isaac Sim stages are Z-up, so an unknown
    axis still enforces a straight-down-Z gravity for the common case."""
    getter = getattr(adapter, "get_stage_up_axis", None)
    if callable(getter):
        try:
            axis = getter()
            if axis:
                return str(axis)
        except Exception:
            pass
    try:
        from pxr import UsdGeom

        axis = UsdGeom.GetStageUpAxis(adapter.get_stage())
        if axis:
            return str(axis)
    except Exception:
        pass
    return "Z"


def guard_pose_write(
    adapter: Any,
    target_path: Optional[str],
    *,
    session_mode: Optional[bool] = None,
    capability: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Sink-level pose/topology guard for handlers (objects.create / objects.clone)
    whose mutation is not classified as a POSE_WRITE_COMMAND but still writes a prim.

    Builds a throwaway :class:`SessionGuard` (the pose-write check is stateless — it
    holds no pause snapshot) and applies :meth:`SessionGuard.pose_write_rejection`.
    ``session_mode`` defaults to the process setting; passing it explicitly is only
    for tests. ``capability`` defaults to the in-flight :func:`current_capability`,
    so a dispatch under the operator token bypasses this sink lock too. Returns a
    rejection dict, or ``None`` to allow.
    """
    if session_mode is None:
        session_mode = session_mode_enabled()
    return SessionGuard(adapter, session_mode=session_mode).pose_write_rejection(
        target_path, capability=capability
    )


def guard_gravity_write(
    adapter: Any,
    gravity: Optional[Sequence[float]],
    *,
    session_mode: Optional[bool] = None,
    capability: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Sink-level physical-gravity guard for handlers that accept a gravity vector
    (simulation.set_physics, scene.create_physics).

    Rejects a SESSION gravity write that is not earth-like and predominantly downward,
    so gravity=[0,0,0] (or a tiny/sideways/upward vector) can never be used to trivially
    fake a balance; ``gravity=None`` (leave gravity unchanged) is always allowed.
    ``capability`` defaults to the in-flight :func:`current_capability`, so a dispatch
    under the operator token bypasses this lock; ``session_mode`` defaults to the process
    setting (pass it only for tests). Returns a rejection dict, or ``None`` to allow.
    """
    if session_mode is None:
        session_mode = session_mode_enabled()
    if not session_mode:
        return None
    if capability is None:
        capability = current_capability()
    if capability == OPERATOR_CAPABILITY:
        return None
    if gravity is None:
        return None
    return _gravity_rejection(gravity, _resolve_up_axis(adapter))
