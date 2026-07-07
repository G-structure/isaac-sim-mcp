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

"""Abstract base adapter for Isaac Sim version-specific APIs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)


class IsaacAdapterBase(ABC):
    """Abstract interface that isolates all Isaac Sim version-specific API calls.

    Handler code should never import isaacsim.* directly — use this adapter instead.
    Each supported Isaac Sim version provides a concrete implementation.
    """

    # ── Scene ──────────────────────────────────────────────

    @abstractmethod
    def get_stage(self) -> Usd.Stage:
        """Return the current USD stage."""
        ...

    @abstractmethod
    def get_assets_root_path(self) -> str:
        """Return the root path for Isaac Sim built-in assets."""
        ...

    @abstractmethod
    def discover_environments(self) -> Dict[str, Dict[str, str]]:
        """Scan the asset server for available environment USD files."""
        ...

    @abstractmethod
    def load_environment(self, env_path: str, prim_path: str = "/Environment") -> None:
        """Load an environment USD into the stage."""
        ...

    # ── Prims ──────────────────────────────────────────────

    @abstractmethod
    def create_prim(self, prim_path: str, prim_type: str = "Xform", **kwargs) -> Usd.Prim:
        """Create a USD prim at the given path."""
        ...

    @abstractmethod
    def delete_prim(self, prim_path: str) -> bool:
        """Delete a prim from the stage. Returns True on success."""
        ...

    @abstractmethod
    def add_reference_to_stage(self, usd_path: str, prim_path: str) -> Usd.Prim:
        """Add a USD reference to the stage at prim_path."""
        ...

    @abstractmethod
    def set_prim_transform(
        self,
        prim_path: str,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        """Set position, rotation, and/or scale on a prim."""
        ...

    @abstractmethod
    def get_prim_transform(self, prim_path: str) -> Dict[str, Any]:
        """Return position, rotation, scale of a prim."""
        ...

    @abstractmethod
    def list_prims(self, root_path: str = "/", prim_type: Optional[str] = None) -> List[Dict[str, str]]:
        """List prims under root_path, optionally filtered by type."""
        ...

    @abstractmethod
    def get_prim_info(self, prim_path: str) -> Dict[str, Any]:
        """Return detailed info about a prim (type, transform, properties)."""
        ...

    @abstractmethod
    def get_prim_actual_size(self, prim_path: str) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Return actual dimensions and bounding box for a geometric prim.

        Returns:
            A tuple of (actual_size, (bbox_min, bbox_max)) where:
            - actual_size: [x, y, z] dimensions in meters (default_size * scale)
            - bbox_min: [x, y, z] world-space minimum corner
            - bbox_max: [x, y, z] world-space maximum corner
        """
        ...

    def ping(self) -> Dict[str, Any]:
        """Cheap main-thread health probe."""
        return {"ready": True}

    def get_resources(self) -> Dict[str, Any]:
        """Return runtime resource telemetry."""
        return {}

    # ── Robots ─────────────────────────────────────────────

    @abstractmethod
    def create_xform_prim(self, prim_path: str) -> Any:
        """Create an XFormPrim wrapper for positioning."""
        ...

    @abstractmethod
    def create_articulation(self, prim_path: str, name: str) -> Any:
        """Create an Articulation wrapper for a robot at prim_path."""
        ...

    @abstractmethod
    def discover_robots(self) -> Dict[str, Dict[str, str]]:
        """Scan the asset server for available robot USD files.

        Returns a dict mapping robot key to {"asset_path": ..., "description": ..., "manufacturer": ...}.
        """
        ...

    @abstractmethod
    def get_robot_joint_info(self, prim_path: str) -> Dict[str, Any]:
        """Return joint names, DOF count, and current positions for a robot."""
        ...

    @abstractmethod
    def set_joint_positions(
        self, prim_path: str, positions: Sequence[float], joint_indices: Optional[List[int]] = None
    ) -> None:
        """Set target joint positions on a robot articulation."""
        ...

    @abstractmethod
    def get_joint_positions(self, prim_path: str) -> List[float]:
        """Read current joint positions from a robot articulation."""
        ...

    @abstractmethod
    def get_joint_config(self, prim_path: str) -> Dict[str, Any]:
        """Return joint drive configuration: stiffness, damping, limits, target vs actual positions."""
        ...

    # ── Physics ────────────────────────────────────────────

    @abstractmethod
    def create_world(self, **kwargs) -> Any:
        """Create a World instance for simulation management."""
        ...

    @abstractmethod
    def create_simulation_context(self, **kwargs) -> Any:
        """Create a SimulationContext for physics stepping."""
        ...

    @abstractmethod
    def create_physics_scene(self, gravity: Optional[Sequence[float]] = None, scene_name: str = "PhysicsScene") -> str:
        """Create a physics scene prim with gravity settings."""
        ...

    @abstractmethod
    def get_physics_state(self, prim_path: str) -> Dict[str, Any]:
        """Return physics state for a prim: rigid body, mass, velocities, contacts."""
        ...

    # ── Sensors ────────────────────────────────────────────

    @abstractmethod
    def create_camera(self, prim_path: str, resolution: Tuple[int, int] = (1280, 720), **kwargs) -> Any:
        """Create a camera sensor at prim_path."""
        ...

    @abstractmethod
    def capture_camera_image(self, prim_path: str) -> np.ndarray:
        """Capture an RGB image from a camera. Returns image data."""
        ...

    @abstractmethod
    def create_lidar(self, prim_path: str, config: Optional[str] = None, **kwargs) -> Any:
        """Create a lidar sensor at prim_path."""
        ...

    @abstractmethod
    def get_lidar_point_cloud(self, prim_path: str) -> np.ndarray:
        """Get point cloud data from a lidar sensor."""
        ...

    # ── Materials ──────────────────────────────────────────

    @abstractmethod
    def create_pbr_material(
        self,
        prim_path: str,
        color: Optional[Sequence[float]] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> Any:
        """Create an OmniPBR material."""
        ...

    @abstractmethod
    def create_physics_material(
        self,
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
    ) -> Any:
        """Create a physics material with friction/restitution."""
        ...

    @abstractmethod
    def apply_material(self, material_path: str, target_prim_path: str) -> None:
        """Bind a material to a prim."""
        ...

    # ── Lighting ───────────────────────────────────────────

    @abstractmethod
    def create_light(
        self,
        light_type: str,
        prim_path: str,
        intensity: float = 1000.0,
        color: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> Any:
        """Create a light prim (Distant, Dome, Sphere, Rect, Disk, Cylinder)."""
        ...

    @abstractmethod
    def modify_light(
        self, prim_path: str, intensity: Optional[float] = None, color: Optional[Sequence[float]] = None
    ) -> None:
        """Modify properties of an existing light."""
        ...

    @abstractmethod
    def clone_prim(self, source_path: str, target_path: str) -> None:
        """Copy a prim from source_path to target_path."""
        ...

    # ── Assets ─────────────────────────────────────────────

    @abstractmethod
    def import_urdf(self, urdf_path: str, prim_path: str = "/World/robot", **kwargs) -> Any:
        """Import a robot from a URDF file."""
        ...

    # ── Simulation ─────────────────────────────────────────

    def _ensure_physics_world(self, physics_dt: Optional[float] = None) -> None:
        """Ensure a World with initialised physics exists.

        Called by play() and create_action_graph() to guarantee that
        SingleArticulation.initialize() works inside ScriptNode scripts.

        Physics is configured at construction because PhysX silently drops
        dt / GPU-dynamics / solver-iteration changes made after the scene goes
        live (see handlers.simulation.set_physics for the live-apply path).
        The World is built with a small fixed physics_dt (<= 1/200 s), raised
        solver iteration counts and GPU dynamics enabled so 29-DoF contact is
        stable from the first step. Callers may request a finer (smaller) dt;
        coarser requests are clamped to the 1/200 s ceiling.

        The default implementation uses isaacsim.core.api.World (Isaac Sim 5.x).
        Override in version-specific adapters if the API differs.
        """
        max_physics_dt = 1.0 / 200.0
        if physics_dt is None:
            physics_dt = max_physics_dt
        else:
            physics_dt = min(float(physics_dt), max_physics_dt)
        try:
            from isaacsim.core.api import World

            world = World.instance()
            if world is None:
                world = World(
                    physics_dt=physics_dt,
                    rendering_dt=1.0 / 60.0,
                    stage_units_in_meters=1.0,
                )
            if world.physics_sim_view is None:
                # Configure the context BEFORE physics init; post-hoc writes are
                # dropped by PhysX once the scene is live.
                ctx = world.get_physics_context()
                ctx.set_physics_dt(physics_dt)
                try:
                    ctx.enable_gpu_dynamics(True)
                except Exception as exc:
                    # Do not swallow: a dropped GPU-dynamics request silently
                    # falls back to the CPU solver and destabilises 29-DoF
                    # contact. Surface why it failed instead of hiding it.
                    logger.warning(
                        "physics-fidelity: enable_gpu_dynamics(True) raised: %s",
                        exc,
                    )
                # TODO(probe:solver-iters): tune iteration counts for stable
                #   29-DoF contact on a live box; 8/4 is a conservative floor.
                self._author_scene_solver_iters(ctx, 8, 4)
                self._author_articulation_solver_iters(8, 4)
                world.initialize_physics()
                # Verify the engine honored GPU dynamics; a silent CPU fallback
                # is a fidelity lie unless it is reported.
                try:
                    if not bool(ctx.is_gpu_dynamics_enabled()):
                        logger.warning(
                            "physics-fidelity: GPU dynamics requested but the "
                            "engine reports it disabled after init (CPU "
                            "fallback); 29-DoF contact may be unstable. "
                            "TODO(probe:gpu-dynamics-live)"
                        )
                except Exception as exc:
                    logger.warning(
                        "physics-fidelity: is_gpu_dynamics_enabled() readback "
                        "failed: %s",
                        exc,
                    )
        except ImportError:
            pass  # Non-v5 runtimes may not have isaacsim.core.api

    def _author_scene_solver_iters(
        self, ctx: Any, position_iters: int, velocity_iters: int
    ) -> None:
        """Author scene-wide min solver-iteration floors on the live physics
        scene prim, then read them back. Warns (never silently swallows) if the
        scene prim is missing or the counts did not stick."""
        try:
            from pxr import PhysxSchema

            scene_prim = ctx.get_current_physics_scene_prim()
        except Exception as exc:
            logger.warning(
                "physics-fidelity: could not author scene solver iters: %s", exc
            )
            return
        if scene_prim is None or not scene_prim.IsValid():
            logger.warning(
                "physics-fidelity: no live physics scene prim; scene solver "
                "iteration floors not authored"
            )
            return
        try:
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            scene_api.CreateMinPositionIterationCountAttr(int(position_iters))
            scene_api.CreateMinVelocityIterationCountAttr(int(velocity_iters))
            pos = scene_api.GetMinPositionIterationCountAttr()
            vel = scene_api.GetMinVelocityIterationCountAttr()
            if not (
                pos and pos.HasAuthoredValue() and vel and vel.HasAuthoredValue()
            ):
                logger.warning(
                    "physics-fidelity: scene solver-iteration counts did not "
                    "author; contact solve may be under-iterated"
                )
        except Exception as exc:
            logger.warning(
                "physics-fidelity: scene solver-iteration author failed: %s", exc
            )

    def _author_articulation_solver_iters(
        self, position_iters: Optional[int], velocity_iters: Optional[int]
    ) -> None:
        """Author PhysX per-articulation solver-iteration counts on every
        articulation root in the stage, reading them back and warning on any the
        engine did not accept.

        Scene min-counts are only a floor; PhysX drives the actual solver work
        from each articulation's own PhysxArticulationAPI counts, so a 29-DoF
        robot needs these set too — the scene minima alone do not raise its
        per-articulation solve.
        """
        if position_iters is None and velocity_iters is None:
            return
        try:
            from pxr import PhysxSchema, UsdPhysics

            stage = self.get_stage()
        except Exception as exc:
            logger.warning(
                "physics-fidelity: cannot author articulation solver iters: %s",
                exc,
            )
            return
        if stage is None:
            return
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            try:
                art_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                if position_iters is not None:
                    art_api.CreateSolverPositionIterationCountAttr(
                        int(position_iters)
                    )
                    pos = art_api.GetSolverPositionIterationCountAttr()
                    if not (pos and pos.HasAuthoredValue()):
                        logger.warning(
                            "physics-fidelity: articulation %s rejected solver "
                            "position iterations",
                            prim.GetPath(),
                        )
                if velocity_iters is not None:
                    art_api.CreateSolverVelocityIterationCountAttr(
                        int(velocity_iters)
                    )
                    vel = art_api.GetSolverVelocityIterationCountAttr()
                    if not (vel and vel.HasAuthoredValue()):
                        logger.warning(
                            "physics-fidelity: articulation %s rejected solver "
                            "velocity iterations",
                            prim.GetPath(),
                        )
            except Exception as exc:
                logger.warning(
                    "physics-fidelity: articulation %s solver-iter author "
                    "failed: %s",
                    prim.GetPath(),
                    exc,
                )

    def apply_physics_params(
        self,
        gravity: Optional[Sequence[float]] = None,
        time_step: Optional[float] = None,
        gpu_enabled: Optional[bool] = None,
        solver_position_iteration_count: Optional[int] = None,
        solver_velocity_iteration_count: Optional[int] = None,
        substeps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply physics parameters to the live PhysicsContext and read back the
        effective values from the running PhysX scene.

        Unlike create_physics_scene (which only authors a USD scene prim), this
        drives the live context so changes take effect on the next step:
        time_step -> set_physics_dt, gpu_enabled -> enable_gpu_dynamics, and the
        solver iteration counts scene-wide via PhysxSchema.PhysxSceneAPI (plus
        per-articulation counts). The live dt is clamped to the 1/200 s fidelity
        ceiling so set_physics can never raise PhysX above it. Fails closed
        unless exactly one UsdPhysics.PhysicsScene is present.

        The returned ``effective`` dict labels each value by provenance so a
        caller never mistakes a request for verified truth: ``*_requested`` /
        ``*_authored`` values only echo what was set / written to USD (PhysX
        drops dt / solver-iter writes made after the scene is live), whereas
        ``*_effective`` values come from live context getters.

        The default implementation uses isaacsim.core.api.World (Isaac Sim 5.x).
        Override in version-specific adapters if the API differs.
        """
        from isaacsim.core.api import World
        from pxr import Gf, PhysxSchema, UsdPhysics

        max_physics_dt = 1.0 / 200.0

        self._ensure_physics_world(physics_dt=time_step)

        stage = self.get_stage()
        scene_prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scene_prims) != 1:
            raise RuntimeError(
                f"expected exactly one UsdPhysics.PhysicsScene, found {len(scene_prims)}"
            )
        scene_prim = scene_prims[0]

        world = World.instance()
        ctx = world.get_physics_context()

        if time_step is not None or substeps is not None:
            dt = float(time_step) if time_step is not None else float(ctx.get_physics_dt())
            # Fidelity ceiling: set_physics must NEVER raise the LIVE PhysX dt
            # above 1/200 s (mirrors the _ensure_physics_world clamp so a coarse
            # dt request cannot degrade the running contact solve).
            dt = min(dt, max_physics_dt)
            ctx.set_physics_dt(dt, int(substeps) if substeps is not None else 1)
        if gpu_enabled is not None:
            ctx.enable_gpu_dynamics(bool(gpu_enabled))
        if gravity is not None:
            gx, gy, gz = float(gravity[0]), float(gravity[1]), float(gravity[2])
            if abs(gx) < 1e-9 and abs(gy) < 1e-9:
                ctx.set_gravity(gz)
            else:
                mag = float(np.linalg.norm([gx, gy, gz]))
                scene = UsdPhysics.Scene(scene_prim)
                scene.CreateGravityDirectionAttr().Set(
                    Gf.Vec3f(gx / mag, gy / mag, gz / mag)
                    if mag
                    else Gf.Vec3f(0.0, 0.0, -1.0)
                )
                scene.CreateGravityMagnitudeAttr().Set(mag)

        scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        # TODO(probe:solver-iters): tune iteration counts for stable 29-DoF
        #   contact on a live box once measured; 8/4 is a conservative floor.
        if solver_position_iteration_count is not None:
            scene_api.CreateMinPositionIterationCountAttr(
                int(solver_position_iteration_count)
            )
        if solver_velocity_iteration_count is not None:
            scene_api.CreateMinVelocityIterationCountAttr(
                int(solver_velocity_iteration_count)
            )
        # Scene minima are only a floor; mirror the request onto each
        # articulation's own counts so the robot's solve actually changes.
        self._author_articulation_solver_iters(
            solver_position_iteration_count, solver_velocity_iteration_count
        )

        # Read back, labelling by provenance. PhysX drops dt / solver-iter
        # writes made after the scene is live, so those are what we REQUESTED /
        # AUTHORED, not proof the running engine honored them this step. Gravity
        # and GPU-dynamics come from live context getters (ENGINE readback); the
        # GPU value is compared to the request to catch a silent fallback.
        warnings: List[str] = []
        effective: Dict[str, Any] = {
            # REQUESTED: echoes the value set on the context; not a live-step
            # proof. TODO(probe:dt-live): confirm on a live box that the stepped
            # PhysX dt equals this value.
            "physics_dt_requested": float(ctx.get_physics_dt()),
        }
        try:
            direction, magnitude = ctx.get_gravity()
            effective["gravity_effective"] = [
                float(direction[i]) * float(magnitude) for i in range(3)
            ]
        except Exception:
            effective["gravity_effective"] = None
        try:
            gpu_on = bool(ctx.is_gpu_dynamics_enabled())
            effective["gpu_dynamics_enabled_effective"] = gpu_on
            if gpu_enabled is not None and gpu_on != bool(gpu_enabled):
                msg = (
                    f"gpu_dynamics requested={bool(gpu_enabled)} but engine "
                    f"reports {gpu_on} (silent fallback)"
                )
                warnings.append(msg)
                logger.warning("physics-fidelity: %s", msg)
        except Exception:
            effective["gpu_dynamics_enabled_effective"] = None
        # AUTHORED on the USD scene prim — reflects what we wrote, NOT a live
        # engine readback. Named so callers never treat it as verified truth.
        pos_attr = scene_api.GetMinPositionIterationCountAttr()
        vel_attr = scene_api.GetMinVelocityIterationCountAttr()
        effective["solver_position_iteration_count_authored"] = (
            int(pos_attr.Get()) if pos_attr and pos_attr.HasAuthoredValue() else None
        )
        effective["solver_velocity_iteration_count_authored"] = (
            int(vel_attr.Get()) if vel_attr and vel_attr.HasAuthoredValue() else None
        )
        effective["physics_scene_count"] = len(scene_prims)
        if warnings:
            effective["warnings"] = warnings
        return effective

    def get_gravity_vector(self) -> Optional[List[float]]:
        """Live gravity VECTOR (m/s^2) read from the running PhysicsContext, or
        ``None`` if no world is instantiated.

        Consumed by the session guard's actuation READ gate to verify gravity was
        not tampered to a non-physical fake-balance vector (zero / tiny / sideways /
        upward / diagonal): the guard reuses the same earth-like-and-downward
        envelope it enforces on gravity WRITES. Reads the same live context getter
        as apply_physics_params' readback (direction * magnitude).
        TODO(probe:guard-live): confirm this equals ctx.get_gravity() on a live box.
        """
        from isaacsim.core.api import World

        world = World.instance()
        if world is None:
            return None
        ctx = world.get_physics_context()
        direction, magnitude = ctx.get_gravity()
        return [float(direction[i]) * float(magnitude) for i in range(3)]

    def ensure_contact_reporting(self, prim_path: str) -> None:
        """Apply contact-report APIs to the collider subtree under prim_path so
        PhysX emits contact / ground-reaction forces on first impact.

        Must be called at SPAWN / pre-play: PhysX may not honor a contact-report
        API applied after the scene is already live without a re-parse, so
        deferring this to the get_physics_state read path silently drops the
        first-impact forces. The default is a no-op; version adapters that
        support PhysX contact reports override this.
        """
        return None

    @abstractmethod
    def play(self) -> None:
        """Start the simulation."""
        ...

    @abstractmethod
    def pause(self) -> None:
        """Pause the simulation."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the simulation."""
        ...

    @abstractmethod
    def step(
        self,
        num_steps: int = 1,
        observe_prims: Optional[List[str]] = None,
        observe_joints: Optional[List[str]] = None,
        budget_ms: Optional[int] = None,
        observe_cap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Step the simulation forward and optionally observe prim/joint states.

        Args:
            num_steps: Number of frames to step.
            observe_prims: Prim paths to snapshot after stepping (transform + velocity).
            observe_joints: Articulation paths to snapshot (joint positions).
            budget_ms: Optional wall-clock budget.
            observe_cap: Optional cap on articulation observations.
        """
        ...

    @abstractmethod
    def get_simulation_state(self) -> Dict[str, Any]:
        """Return current timeline state, simulation time, physics dt, and step count."""
        ...

    @abstractmethod
    def execute_script(self, code: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute arbitrary Python code in the Isaac Sim context.

        Args:
            code: Python code to execute.
            cwd: Optional working directory to add to sys.path before execution.
        """
        ...

    @abstractmethod
    def reload_script(self, file_path: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Reload a Python script or module into the Isaac Sim runtime.

        Args:
            file_path: Path to the Python file.
            module_name: If provided, reload this module. Otherwise execute the file.
        """
        ...
