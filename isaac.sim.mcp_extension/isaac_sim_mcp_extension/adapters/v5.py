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

"""Isaac Sim 5.1.0 adapter implementation."""

from __future__ import annotations

import traceback
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import IsaacAdapterBase

if TYPE_CHECKING:
    from pxr import Usd


class IsaacAdapterV5(IsaacAdapterBase):
    """Adapter for Isaac Sim 5.1.0 (isaacsim.* namespace)."""

    def __init__(self) -> None:
        self._articulation_cache: Dict[str, Any] = {}
        self._joint_name_cache: Dict[str, List[str]] = {}

    # ── Scene ──────────────────────────────────────────────

    def get_stage(self) -> Usd.Stage:
        import omni.usd

        return omni.usd.get_context().get_stage()

    def get_assets_root_path(self) -> str:
        from isaacsim.storage.native import get_assets_root_path

        return get_assets_root_path()

    # ── Prims ──────────────────────────────────────────────

    def create_prim(
        self, prim_path: str, prim_type: str = "Xform", **kwargs
    ) -> Usd.Prim:
        from isaacsim.core.utils.prims import create_prim

        return create_prim(prim_path, prim_type, **kwargs)

    def delete_prim(self, prim_path: str) -> bool:
        import omni.kit.commands

        omni.kit.commands.execute("DeletePrims", paths=[prim_path])
        return True

    def discover_environments(self) -> Dict[str, Dict[str, str]]:
        """Scan the Isaac Sim asset server for available environment USD files."""
        import omni.client
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        discovered: Dict[str, Dict[str, str]] = {}

        search_bases = ["/Isaac/Environments/", "/NVIDIA/Assets/Scenes/Templates/"]
        for base in search_bases:
            result, entries = omni.client.list(root + base)
            if result != omni.client.Result.OK:
                continue
            for entry in entries:
                name = entry.relative_path.rstrip("/")
                dir_path = root + base + name + "/"
                r2, files = omni.client.list(dir_path)
                if r2 != omni.client.Result.OK:
                    continue
                # Find USD files at this level
                for f in files:
                    if f.relative_path.endswith(".usd") or f.relative_path.endswith(
                        ".usda"
                    ):
                        key = name.lower().replace(" ", "_")
                        if key not in discovered:
                            discovered[key] = {
                                "asset_path": base + name + "/" + f.relative_path,
                                "description": name.replace("_", " "),
                            }
                        break
                # Also check one level deeper for nested envs
                for f in files:
                    subname = f.relative_path.rstrip("/")
                    r3, subfiles = omni.client.list(dir_path + subname + "/")
                    if r3 != omni.client.Result.OK:
                        continue
                    for sf in subfiles:
                        if sf.relative_path.endswith(
                            ".usd"
                        ) or sf.relative_path.endswith(".usda"):
                            key = f"{name}_{subname}".lower().replace(" ", "_")
                            if key not in discovered:
                                discovered[key] = {
                                    "asset_path": base
                                    + name
                                    + "/"
                                    + subname
                                    + "/"
                                    + sf.relative_path,
                                    "description": f"{name} {subname}".replace(
                                        "_", " "
                                    ),
                                }
                            break
        return discovered

    def load_environment(self, env_path: str, prim_path: str = "/Environment") -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage

        add_reference_to_stage(env_path, prim_path)

    def add_reference_to_stage(self, usd_path: str, prim_path: str) -> Usd.Prim:
        from isaacsim.core.utils.stage import add_reference_to_stage

        return add_reference_to_stage(usd_path, prim_path)

    def set_prim_transform(
        self,
        prim_path: str,
        position: Optional[Sequence[float]] = None,
        rotation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
    ) -> None:
        from pxr import Gf, UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        if position is not None:
            xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*position)
            )
        if rotation is not None:
            xformable.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*rotation)
            )
        if scale is not None:
            xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(*scale)
            )

    def get_prim_transform(self, prim_path: str) -> Dict[str, Any]:
        from pxr import UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        translation = local_transform.ExtractTranslation()
        return {
            "position": [translation[0], translation[1], translation[2]],
        }

    def list_prims(
        self, root_path: str = "/", prim_type: Optional[str] = None
    ) -> List[Dict[str, str]]:
        stage = self.get_stage()
        root = stage.GetPrimAtPath(root_path)
        results: List[Dict[str, str]] = []
        for prim in root.GetAllChildren():
            ptype = prim.GetTypeName()
            if prim_type and ptype != prim_type:
                continue
            results.append({"path": str(prim.GetPath()), "type": ptype})
        return results

    def get_prim_info(self, prim_path: str) -> Dict[str, Any]:
        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")
        transform = self.get_prim_transform(prim_path)
        children = [str(c.GetPath()) for c in prim.GetAllChildren()]
        info: Dict[str, Any] = {
            "path": prim_path,
            "type": prim.GetTypeName(),
            "transform": transform,
            "children": children,
        }
        if prim.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Cone", "Capsule"):
            try:
                actual_size, _bbox = self.get_prim_actual_size(prim_path)
                info["actual_size"] = actual_size
            except Exception:
                pass
        return info

    def get_prim_actual_size(
        self, prim_path: str
    ) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Return actual dimensions and bounding box for a geometric prim."""
        from pxr import UsdGeom

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        prim_type = prim.GetTypeName()

        # Read scale from xform
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        # Extract scale from the matrix diagonal (assuming uniform or axis-aligned scale)
        scale = [
            float(local_transform.GetRow3(0).GetLength()),
            float(local_transform.GetRow3(1).GetLength()),
            float(local_transform.GetRow3(2).GetLength()),
        ]

        if prim_type == "Cube":
            geom = UsdGeom.Cube(prim)
            size_attr = geom.GetSizeAttr()
            size = (
                float(size_attr.Get())
                if size_attr and size_attr.Get() is not None
                else 1.0
            )
            dims = [size * scale[0], size * scale[1], size * scale[2]]
        elif prim_type == "Sphere":
            geom = UsdGeom.Sphere(prim)
            radius_attr = geom.GetRadiusAttr()
            radius = (
                float(radius_attr.Get())
                if radius_attr and radius_attr.Get() is not None
                else 0.5
            )
            diameter = radius * 2.0
            dims = [diameter * scale[0], diameter * scale[1], diameter * scale[2]]
        elif prim_type == "Cylinder":
            geom = UsdGeom.Cylinder(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            axis_attr = geom.GetAxisAttr()
            radius = (
                float(radius_attr.Get())
                if radius_attr and radius_attr.Get() is not None
                else 0.5
            )
            height = (
                float(height_attr.Get())
                if height_attr and height_attr.Get() is not None
                else 1.0
            )
            axis = axis_attr.Get() if axis_attr and axis_attr.Get() is not None else "Z"
            diameter = radius * 2.0
            if axis == "X":
                dims = [height * scale[0], diameter * scale[1], diameter * scale[2]]
            elif axis == "Y":
                dims = [diameter * scale[0], height * scale[1], diameter * scale[2]]
            else:  # Z (default)
                dims = [diameter * scale[0], diameter * scale[1], height * scale[2]]
        elif prim_type == "Cone":
            geom = UsdGeom.Cone(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            axis_attr = geom.GetAxisAttr()
            radius = (
                float(radius_attr.Get())
                if radius_attr and radius_attr.Get() is not None
                else 0.5
            )
            height = (
                float(height_attr.Get())
                if height_attr and height_attr.Get() is not None
                else 1.0
            )
            axis = axis_attr.Get() if axis_attr and axis_attr.Get() is not None else "Z"
            diameter = radius * 2.0
            if axis == "X":
                dims = [height * scale[0], diameter * scale[1], diameter * scale[2]]
            elif axis == "Y":
                dims = [diameter * scale[0], height * scale[1], diameter * scale[2]]
            else:  # Z (default)
                dims = [diameter * scale[0], diameter * scale[1], height * scale[2]]
        elif prim_type == "Capsule":
            geom = UsdGeom.Capsule(prim)
            radius_attr = geom.GetRadiusAttr()
            height_attr = geom.GetHeightAttr()
            radius = (
                float(radius_attr.Get())
                if radius_attr and radius_attr.Get() is not None
                else 0.5
            )
            height = (
                float(height_attr.Get())
                if height_attr and height_attr.Get() is not None
                else 1.0
            )
            total_height = height + 2.0 * radius
            diameter = radius * 2.0
            dims = [diameter * scale[0], diameter * scale[1], total_height * scale[2]]
        else:
            raise ValueError(f"Unsupported prim type for size calculation: {prim_type}")

        # Compute world-space position for bounding box
        from pxr import Usd

        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        pos = [float(translation[0]), float(translation[1]), float(translation[2])]
        half = [d / 2.0 for d in dims]
        bbox_min = [pos[0] - half[0], pos[1] - half[1], pos[2] - half[2]]
        bbox_max = [pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]]

        return dims, (bbox_min, bbox_max)

    # ── Robots ─────────────────────────────────────────────

    def discover_robots(self) -> Dict[str, Dict[str, str]]:
        """Scan the Isaac Sim asset server for all available robot USD files."""
        import omni.client
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        robots_base = root + "/Isaac/Robots/"
        discovered: Dict[str, Dict[str, str]] = {}

        result, manufacturers = omni.client.list(robots_base)
        if result != omni.client.Result.OK:
            return discovered

        for mfr_entry in manufacturers:
            mfr_name = mfr_entry.relative_path.rstrip("/")
            mfr_path = robots_base + mfr_name + "/"

            result2, models = omni.client.list(mfr_path)
            if result2 != omni.client.Result.OK:
                continue

            for model_entry in models:
                model_name = model_entry.relative_path.rstrip("/")
                model_path = mfr_path + model_name + "/"

                # Look for USD files directly in the model directory
                result3, files = omni.client.list(model_path)
                if result3 != omni.client.Result.OK:
                    continue

                for file_entry in files:
                    fname = file_entry.relative_path
                    if not (fname.endswith(".usd") or fname.endswith(".usda")):
                        continue
                    # Skip variants with suffixes like _physx_lidar, _with_arm
                    _base_name = fname.rsplit(".", 1)[0]
                    asset_rel = f"/Isaac/Robots/{mfr_name}/{model_name}/{fname}"

                    # Use lowercase model name as key, prefer shorter/simpler names
                    key = model_name.lower().replace(" ", "_")
                    if key in discovered:
                        # Keep the simpler filename (shorter name wins)
                        if len(fname) < len(
                            discovered[key]["asset_path"].split("/")[-1]
                        ):
                            discovered[key]["asset_path"] = asset_rel
                    else:
                        discovered[key] = {
                            "asset_path": asset_rel,
                            "description": f"{mfr_name} {model_name}",
                            "manufacturer": mfr_name,
                        }

        return discovered

    def create_xform_prim(self, prim_path: str) -> Any:
        from isaacsim.core.prims import SingleXFormPrim

        return SingleXFormPrim(prim_path=prim_path)

    def create_articulation(self, prim_path: str, name: str) -> Any:
        from isaacsim.core.prims import SingleArticulation

        return SingleArticulation(prim_path=prim_path, name=name)

    def get_robot_joint_info(self, prim_path: str) -> Dict[str, Any]:
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdPhysics

        # Try to get joint info via articulation API (requires running sim)
        joint_names: List[str] = []
        num_dof = 0
        art = SingleArticulation(prim_path=prim_path)
        try:
            art.initialize()
            joint_names = list(art.dof_names) if art.dof_names else []
            num_dof = art.num_dof if art.num_dof else 0
        except Exception:
            pass

        # Fallback: discover joints by traversing USD stage
        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not joint_names and root_prim.IsValid():
            for desc in Usd.PrimRange(root_prim):
                if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                    UsdPhysics.PrismaticJoint
                ):
                    joint_names.append(desc.GetName())
            num_dof = len(joint_names)

        joint_limits = []
        for jname in joint_names:
            limit_entry: Dict[str, Any] = {"name": jname}
            for desc in Usd.PrimRange(root_prim):
                if desc.GetName() != jname:
                    continue
                if desc.IsA(UsdPhysics.RevoluteJoint):
                    rev = UsdPhysics.RevoluteJoint(desc)
                    lo = rev.GetLowerLimitAttr().Get()
                    hi = rev.GetUpperLimitAttr().Get()
                    limit_entry["type"] = "revolute"
                    limit_entry["lower"] = float(lo) if lo is not None else None
                    limit_entry["upper"] = float(hi) if hi is not None else None
                    limit_entry["units"] = "degrees"
                    break
                if desc.IsA(UsdPhysics.PrismaticJoint):
                    pris = UsdPhysics.PrismaticJoint(desc)
                    lo = pris.GetLowerLimitAttr().Get()
                    hi = pris.GetUpperLimitAttr().Get()
                    limit_entry["type"] = "prismatic"
                    limit_entry["lower"] = float(lo) if lo is not None else None
                    limit_entry["upper"] = float(hi) if hi is not None else None
                    limit_entry["units"] = "meters"
                    break
            joint_limits.append(limit_entry)

        return {
            "joint_names": joint_names,
            "num_dof": num_dof,
            "joint_limits": joint_limits,
        }

    def set_joint_positions(
        self,
        prim_path: str,
        positions: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        art = SingleArticulation(prim_path=prim_path)
        try:
            art.initialize()
            action = ArticulationAction(
                joint_positions=np.array(positions),
                joint_indices=np.array(joint_indices) if joint_indices else None,
            )
            controller = art.get_articulation_controller()
            controller.apply_action(action)
        except Exception:
            # Fallback: set USD drive targets directly (works when sim is stopped)
            self._set_joint_drive_targets(prim_path, positions, joint_indices)

    def _set_joint_drive_targets(
        self,
        prim_path: str,
        positions: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Set joint drive targets via USD API — works regardless of simulation state."""
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        # Collect all joints under the articulation
        joints = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                UsdPhysics.PrismaticJoint
            ):
                joints.append(desc)

        if joint_indices is not None:
            targets = list(zip(joint_indices, positions))
        else:
            targets = list(enumerate(positions))

        for idx, value in targets:
            if idx >= len(joints):
                continue
            joint_prim = joints[idx]
            is_revolute = joint_prim.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
            if is_revolute:
                drive.GetTargetPositionAttr().Set(float(np.degrees(value)))
            else:
                # Prismatic joints: positions in meters, USD targets in cm
                drive.GetTargetPositionAttr().Set(float(value * 100.0))

    def set_gains(
        self,
        prim_path: str,
        kp: Sequence[float],
        kd: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Write PD gains (Unitree kp [N·m/rad], kd [N·m·s/rad]) to a robot's drives.

        Prefers the articulation controller (PhysX tensor API, SI/radian); falls
        back to authoring USD DriveAPI stiffness/damping (per-DEGREE) when no
        running articulation is available. The two paths use DIFFERENT units — the
        tensor path is the confirmed runtime path (probe-results.md).
        """
        from isaacsim.core.prims import SingleArticulation

        from ..g1 import gains as g1_gains

        kp = [float(v) for v in kp]
        kd = [float(v) for v in kd]
        idx = np.array(joint_indices) if joint_indices is not None else None

        try:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            controller = art.get_articulation_controller()
            # CONFIRMED (probe-results.md, 6.0.0-rc.22): the controller path is the
            # PhysX tensor API, which is RADIAN-based, so Unitree kp/kd pass through
            # to set_gains(kps, kds) UNSCALED (g1.gains.kp_to_tensor_stiffness /
            # kd_to_tensor_damping are identity). The per-DEGREE conversion
            # (kp_to_drive_stiffness) is applied ONLY on the raw USD DriveAPI
            # fallback below — using it here would mis-scale every gain by 180/pi.
            tensor_kp = np.array(
                [g1_gains.kp_to_tensor_stiffness(v) for v in kp], dtype=float
            )
            tensor_kd = np.array(
                [g1_gains.kd_to_tensor_damping(v) for v in kd], dtype=float
            )
            if idx is not None:
                cur_kp, cur_kd = controller.get_gains()
                cur_kp = np.array(cur_kp, dtype=float)
                cur_kd = np.array(cur_kd, dtype=float)
                cur_kp[idx] = tensor_kp
                cur_kd[idx] = tensor_kd
                controller.set_gains(kps=cur_kp, kds=cur_kd)
            else:
                controller.set_gains(kps=tensor_kp, kds=tensor_kd)
        except Exception:
            self._set_drive_gains(prim_path, kp, kd, joint_indices)

    def _set_drive_gains(
        self,
        prim_path: str,
        kp: Sequence[float],
        kd: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Author USD DriveAPI stiffness/damping — works when sim is stopped.

        Angular (revolute) drives take per-DEGREE units, so Unitree radian gains
        are scaled via g1.gains. G1 is all-revolute; a prismatic joint would need
        per-metre scaling instead (flagged inline).
        """
        from pxr import Usd, UsdPhysics

        from ..g1 import gains as g1_gains

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        joints = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                UsdPhysics.PrismaticJoint
            ):
                joints.append(desc)

        if joint_indices is not None:
            targets = list(zip(joint_indices, kp, kd))
        else:
            targets = [(i, kp[i], kd[i]) for i in range(min(len(kp), len(kd)))]

        for jidx, kp_val, kd_val in targets:
            if jidx >= len(joints):
                continue
            joint_prim = joints[jidx]
            is_revolute = joint_prim.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
            if is_revolute:
                stiffness = g1_gains.kp_to_drive_stiffness(float(kp_val))
                damping = g1_gains.kd_to_drive_damping(float(kd_val))
            else:
                # Prismatic drives use per-metre units, not the angular per-degree
                # scaling; G1 has no prismatic DoF so these pass through unscaled
                # until a linear joint appears.
                stiffness = float(kp_val)
                damping = float(kd_val)
            drive.GetStiffnessAttr().Set(stiffness)
            drive.GetDampingAttr().Set(damping)

    def set_joint_velocities(
        self,
        prim_path: str,
        velocities: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Command target joint velocities [rad/s] on a robot articulation.

        Each command is clamped to the G1 per-joint velocity ceiling
        (g1.gains.clamp_velocity) before it reaches either the tensor API or the
        USD fallback, so an over-speed request can never leave this method.
        """
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        names = self._get_joint_names(prim_path)
        if joint_indices is not None:
            pairs = zip(joint_indices, velocities)
        else:
            pairs = enumerate(velocities)
        dq = [
            self._g1_velocity_clamp(self._joint_name_at(names, jidx), float(v))
            for jidx, v in pairs
        ]

        art = SingleArticulation(prim_path=prim_path)
        try:
            art.initialize()
            action = ArticulationAction(
                joint_velocities=np.array(dq),
                joint_indices=np.array(joint_indices) if joint_indices else None,
            )
            controller = art.get_articulation_controller()
            controller.apply_action(action)
        except Exception:
            # Fallback: author USD angular drive target velocity (deg/s) so the
            # command survives with the sim stopped, mirroring the position path.
            self._set_joint_drive_velocities(prim_path, dq, joint_indices)

    def _set_joint_drive_velocities(
        self,
        prim_path: str,
        velocities: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Set joint drive target velocities via USD API — works when sim stopped."""
        from pxr import Usd, UsdPhysics

        from ..g1 import gains as g1_gains

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        joints = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                UsdPhysics.PrismaticJoint
            ):
                joints.append(desc)

        if joint_indices is not None:
            targets = list(zip(joint_indices, velocities))
        else:
            targets = list(enumerate(velocities))

        for jidx, value in targets:
            if jidx >= len(joints):
                continue
            joint_prim = joints[jidx]
            is_revolute = joint_prim.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
            if is_revolute:
                drive.GetTargetVelocityAttr().Set(g1_gains.dq_rad_to_deg(float(value)))
            else:
                # Prismatic joints: velocities in m/s, USD targets in cm/s
                drive.GetTargetVelocityAttr().Set(float(value) * 100.0)

    def set_joint_efforts(
        self,
        prim_path: str,
        efforts: Sequence[float],
        joint_indices: Optional[List[int]] = None,
    ) -> None:
        """Command feed-forward joint torques [N·m], clamped to G1 effort limits.

        CONFIRMED additive (docs/src/design/g1-drive/probe-results.md,
        6.0.0-rc.22): PhysX SUMS the ArticulationAction joint_efforts feed-forward
        ON TOP of the DriveAPI PD torque —
        ``total = kp·(q*−q) + kd·(dq*−dq) + tau_ff`` (clamped to USD maxForce;
        measured torque moved +60.9 for +60 applied while position held). So this
        LowCmd tau_ff path is additive as intended; no override / zero-gain
        redesign is needed. Unmapped joints are zeroed by _g1_effort_clamp
        (fail-closed).
        """
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        names = self._get_joint_names(prim_path)
        if joint_indices is not None:
            pairs = zip(joint_indices, efforts)
        else:
            pairs = enumerate(efforts)
        # TODO(probe:motor-space): the ankle (confirmed) and waist (inferred) are
        # parallel A/B linkages; in mode_pr=PR these serial-space clamps are only
        # approximations. The true clamp is in motor (A/B) space via the linkage
        # Jacobian (coupling matrix TBD from the on-image USD).
        tau = [
            self._g1_effort_clamp(self._joint_name_at(names, jidx), float(t))
            for jidx, t in pairs
        ]

        art = SingleArticulation(prim_path=prim_path)
        try:
            art.initialize()
            action = ArticulationAction(
                joint_efforts=np.array(tau),
                joint_indices=np.array(joint_indices) if joint_indices else None,
            )
            controller = art.get_articulation_controller()
            controller.apply_action(action)
        except Exception as exc:
            # Torque has no authored-USD analogue (unlike a position/velocity
            # target), so there is no stopped-sim fallback — fail loudly instead.
            raise RuntimeError(
                f"joint efforts require a running physics articulation at "
                f"{prim_path}: {exc}"
            ) from exc

    @staticmethod
    def _joint_name_at(names: List[str], idx: int) -> Optional[str]:
        return names[idx] if 0 <= idx < len(names) else None

    @staticmethod
    def _g1_effort_clamp(joint_name: Optional[str], tau: float) -> float:
        """FAIL-CLOSED torque clamp to the G1 effort limit for joint_name.

        Accepts either the g1.gains canonical name (e.g. ``left_knee``) or the USD
        dof name with a trailing ``_joint`` (e.g. ``left_knee_joint``). A joint
        with no verified limit — an unresolved name, or a DoF absent from
        G1_JOINT_LIMITS such as a Dex3 hand joint — is ZEROED rather than passed
        through: an unbounded torque on an unmapped joint is a safety hazard.
        """
        if not joint_name:
            return 0.0
        from ..g1 import gains as g1_gains

        candidates = [joint_name]
        if joint_name.endswith("_joint"):
            candidates.append(joint_name[: -len("_joint")])
        for name in candidates:
            if name in g1_gains.G1_JOINT_LIMITS:
                return g1_gains.clamp_effort_fail_closed(name, tau)
        return 0.0

    @staticmethod
    def _g1_velocity_clamp(joint_name: Optional[str], dq: float) -> float:
        """Clamp a commanded velocity [rad/s] to the G1 per-joint ceiling.

        Mapped joints go through g1.gains.clamp_velocity. Unmapped joints (e.g.
        Dex3 hand DoFs absent from the table) pass through: velocity is not the
        fail-closed safety axis — effort is (see _g1_effort_clamp).
        """
        if not joint_name:
            return dq
        from ..g1 import gains as g1_gains

        candidates = [joint_name]
        if joint_name.endswith("_joint"):
            candidates.append(joint_name[: -len("_joint")])
        for name in candidates:
            if name in g1_gains.G1_JOINT_LIMITS:
                return g1_gains.clamp_velocity(name, dq)
        return dq

    def _get_joint_names(self, prim_path: str) -> List[str]:
        """Get joint names, trying articulation API first then USD fallback."""
        cached = self._joint_name_cache.get(prim_path)
        if cached is not None:
            return cached
        try:
            art = self._get_cached_articulation(prim_path)
            if art.dof_names:
                names = list(art.dof_names)
                self._joint_name_cache[prim_path] = names
                return names
        except Exception:
            pass

        # Fallback: traverse USD
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return []
        names: List[str] = []
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                UsdPhysics.PrismaticJoint
            ):
                names.append(desc.GetName())
        # Only cache a non-empty result. Caching [] (prim not yet a valid
        # articulation / no joints authored yet) pinned it forever via the
        # `cached is not None` guard, so a later delete+recreate or a prim that
        # becomes articulated never re-resolved (F11).
        if names:
            self._joint_name_cache[prim_path] = names
        return names

    def get_joint_positions(self, prim_path: str) -> List[float]:
        try:
            art = self._get_cached_articulation(prim_path)
            positions = art.get_joint_positions()
            if positions is not None:
                return positions.tolist()
        except Exception:
            pass

        # Fallback: read drive target positions from USD
        # WARNING: these are authored targets, not actual physics positions
        from pxr import Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return []
        positions_list: List[float] = []
        for desc in Usd.PrimRange(root_prim):
            if not (
                desc.IsA(UsdPhysics.RevoluteJoint)
                or desc.IsA(UsdPhysics.PrismaticJoint)
            ):
                continue
            is_revolute = desc.IsA(UsdPhysics.RevoluteJoint)
            drive_type = "angular" if is_revolute else "linear"
            drive = UsdPhysics.DriveAPI.Get(desc, drive_type)
            if drive:
                target = drive.GetTargetPositionAttr().Get()
                if target is not None:
                    if is_revolute:
                        positions_list.append(float(np.radians(target)))
                    else:
                        positions_list.append(float(target / 100.0))
                else:
                    positions_list.append(0.0)
            else:
                positions_list.append(0.0)
        return positions_list

    def _get_cached_articulation(self, prim_path: str) -> Any:
        from isaacsim.core.prims import SingleArticulation

        self._ensure_physics_world()
        art = self._articulation_cache.get(prim_path)
        if art is None:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            self._articulation_cache[prim_path] = art
        return art

    def get_joint_config(self, prim_path: str) -> Dict[str, Any]:
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd, UsdPhysics

        self._ensure_physics_world()
        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        # Get current joint positions and names via articulation (requires running sim)
        joint_names = self._get_joint_names(prim_path)
        current_pos_list = self.get_joint_positions(prim_path)

        # Get runtime target positions (from applied actions, not USD defaults)
        art = SingleArticulation(prim_path=prim_path)
        runtime_targets: List[float] = []
        try:
            art.initialize()
            applied_action = art.get_applied_action()
            if applied_action and applied_action.joint_positions is not None:
                runtime_targets = applied_action.joint_positions.tolist()
        except Exception:
            pass  # Fall back to USD values if articulation controller unavailable

        joints_info = []

        # Walk descendants to find joint prims
        for desc in Usd.PrimRange(prim):
            if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(
                UsdPhysics.PrismaticJoint
            ):
                joint_data: Dict[str, Any] = {"name": desc.GetName()}

                if desc.IsA(UsdPhysics.RevoluteJoint):
                    joint_data["type"] = "revolute"
                    joint_api = UsdPhysics.RevoluteJoint(desc)
                    lower_attr = joint_api.GetLowerLimitAttr()
                    upper_attr = joint_api.GetUpperLimitAttr()
                else:
                    joint_data["type"] = "prismatic"
                    joint_api = UsdPhysics.PrismaticJoint(desc)
                    lower_attr = joint_api.GetLowerLimitAttr()
                    upper_attr = joint_api.GetUpperLimitAttr()

                joint_data["lower_limit"] = lower_attr.Get() if lower_attr else None
                joint_data["upper_limit"] = upper_attr.Get() if upper_attr else None

                # Get drive config
                for drive_type in ["angular", "linear"]:
                    drive_api = UsdPhysics.DriveAPI.Get(desc, drive_type)
                    if drive_api:
                        joint_data["drive_type"] = drive_type
                        stiffness_attr = drive_api.GetStiffnessAttr()
                        damping_attr = drive_api.GetDampingAttr()
                        target_attr = drive_api.GetTargetPositionAttr()
                        joint_data["stiffness"] = (
                            stiffness_attr.Get() if stiffness_attr else None
                        )
                        joint_data["damping"] = (
                            damping_attr.Get() if damping_attr else None
                        )
                        # USD default as fallback
                        joint_data["target_position"] = (
                            target_attr.Get() if target_attr else None
                        )
                        break

                # Match actual position from articulation if possible
                joint_name = desc.GetName()
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    if idx < len(current_pos_list):
                        joint_data["actual_position"] = current_pos_list[idx]

                    # Override target_position with runtime value if available
                    if idx < len(runtime_targets):
                        joint_data["target_position"] = float(runtime_targets[idx])

                    # Calculate position_error using (possibly runtime) target
                    if (
                        joint_data.get("target_position") is not None
                        and "actual_position" in joint_data
                    ):
                        joint_data["position_error"] = (
                            joint_data["target_position"]
                            - joint_data["actual_position"]
                        )

                joints_info.append(joint_data)

        # Warn about joints with zero stiffness (broken drive config)
        warnings = []
        for j in joints_info:
            stiff = j.get("stiffness")
            damp = j.get("damping")
            if stiff is not None and stiff == 0 and (damp is None or damp == 0):
                warnings.append(
                    f"Joint '{j['name']}' has stiffness=0 and damping=0 — "
                    f"its drive is effectively disabled and will not respond to position targets."
                )

        result: Dict[str, Any] = {
            "prim_path": prim_path,
            "joint_count": len(joints_info),
            "joints": joints_info,
        }
        if warnings:
            result["warnings"] = warnings
        return result

    def get_lowstate_fields(self, prim_path: str) -> Dict[str, Any]:
        """Assemble the runtime fields needed to populate a Unitree LowState.

        Every value is a PhysX RUNTIME read (never an authored USD target):
        measured joint efforts, joint positions/velocities, the floating-base
        world pose + spatial velocity, and an IMU proxy (base angular velocity +
        gravity projected into the base frame). Returns plain python containers.
        """
        result: Dict[str, Any] = {"prim_path": prim_path}

        def _as_list(x: Any) -> Optional[List[float]]:
            if x is None:
                return None
            if hasattr(x, "tolist"):
                x = x.tolist()
            return [float(v) for v in x]

        art = self._get_cached_articulation(prim_path)
        result["q"] = _as_list(art.get_joint_positions())
        result["dq"] = _as_list(art.get_joint_velocities())
        result["dof_names"] = list(art.dof_names) if art.dof_names else []

        # tau_est: measured joint efforts from the solver (the true total torque).
        # CONFIRMED (probe-results.md, 6.0.0-rc.22): an applied joint_efforts
        # feed-forward is ADDITIVE on top of the DriveAPI PD torque and shows up
        # in this measurement (moved +60.9 for +60 applied while position held),
        # so this readback reflects the total torque — no override hedging.
        try:
            result["tau_est"] = _as_list(art.get_measured_joint_efforts())
        except Exception:
            result["tau_est"] = None

        # Floating-base world pose (runtime physics read).
        quat: Optional[List[float]] = None
        try:
            pos, quat_wxyz = art.get_world_pose()
            result["root_position"] = _as_list(pos)
            quat = _as_list(quat_wxyz)
            result["root_quat_wxyz"] = quat
        except Exception:
            result["root_position"] = None
            result["root_quat_wxyz"] = None

        # Floating-base spatial velocity — prefer articulation runtime reads,
        # fall back to the PhysX rigid-body interface on the root prim.
        lin: Optional[List[float]] = None
        ang: Optional[List[float]] = None
        try:
            lin = _as_list(art.get_linear_velocity())
            ang = _as_list(art.get_angular_velocity())
        except Exception:
            pass
        if lin is None or ang is None:
            try:
                import omni.physx

                physx = omni.physx.get_physx_interface()
                rb = physx.get_rigidbody_transformation(prim_path)
                if rb and rb.get("ret_val", False):
                    lin = [float(v) for v in rb.get("linear_velocity", (0.0, 0.0, 0.0))]
                    ang = [float(v) for v in rb.get("angular_velocity", (0.0, 0.0, 0.0))]
            except Exception:
                pass
        result["root_linear_velocity"] = lin
        result["root_angular_velocity"] = ang

        # IMU proxy. TODO(probe:imu): the real IMU sits on a specific link with
        # its own mounting orientation — this uses the articulation-root frame.
        # The quaternion component order (w, x, y, z, a Unitree convention) AND
        # its sign must be runtime-verified against a real LowState capture
        # before this feeds any controller.
        result["imu"] = {
            "angular_velocity": ang,
            "projected_gravity": self._projected_gravity(quat) if quat else None,
        }
        return result

    @staticmethod
    def _projected_gravity(
        quat_wxyz: Sequence[float],
        gravity: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> List[float]:
        """Gravity direction expressed in the base frame (RL 'projected gravity').

        Rotates the world gravity direction into the body frame via the inverse of
        the (w, x, y, z) root orientation (IsaacGym quat_rotate_inverse). An
        upright base returns ~(0, 0, -1). TODO(probe:imu): correctness depends on
        the quaternion order being (w, x, y, z); confirm at runtime.
        """
        w, x, y, z = (
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        )
        gx, gy, gz = float(gravity[0]), float(gravity[1]), float(gravity[2])
        scale = 2.0 * w * w - 1.0
        ax, ay, az = gx * scale, gy * scale, gz * scale
        cross_x = y * gz - z * gy
        cross_y = z * gx - x * gz
        cross_z = x * gy - y * gx
        two_w = 2.0 * w
        bx, by, bz = cross_x * two_w, cross_y * two_w, cross_z * two_w
        dot = 2.0 * (x * gx + y * gy + z * gz)
        cx, cy, cz = x * dot, y * dot, z * dot
        return [ax - bx + cx, ay - by + cy, az - bz + cz]

    def detect_motion(
        self,
        prim_path: str,
        num_steps: int = 2,
        position_tol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Frozen-sim detector: step the sim and report whether DoFs actually moved.

        Compares PhysX RUNTIME reads — articulation joint positions
        (``get_joint_positions``) and the root rigid-body world transform — taken
        BEFORE and AFTER stepping. It NEVER consults authored USD drive targets,
        so a wedged physics pipeline (targets advancing while nothing integrates)
        is caught: ``advanced`` is False when the largest joint / root delta stays
        within ``position_tol`` across the steps. Runtime-only by construction.
        """
        import omni.kit.app

        def _runtime_q() -> Optional[List[float]]:
            try:
                art = self._get_cached_articulation(prim_path)
                q = art.get_joint_positions()
            except Exception:
                return None
            if q is None:
                return None
            return [float(v) for v in (q.tolist() if hasattr(q, "tolist") else q)]

        def _runtime_root() -> Optional[List[float]]:
            try:
                import omni.physx

                physx = omni.physx.get_physx_interface()
                rb = physx.get_rigidbody_transformation(prim_path)
                if rb and rb.get("ret_val", False):
                    pos = rb["position"]
                    return [float(pos[0]), float(pos[1]), float(pos[2])]
            except Exception:
                pass
            return None

        q_before = _runtime_q()
        root_before = _runtime_root()

        stepped = 0
        for _ in range(max(1, num_steps)):
            omni.kit.app.get_app().update()
            stepped += 1

        q_after = _runtime_q()
        root_after = _runtime_root()

        joint_delta: Optional[float] = None
        if (
            q_before is not None
            and q_after is not None
            and len(q_before) == len(q_after)
        ):
            joint_delta = max(
                (abs(a - b) for a, b in zip(q_before, q_after)), default=0.0
            )
        root_delta: Optional[float] = None
        if root_before is not None and root_after is not None:
            root_delta = max(abs(a - b) for a, b in zip(root_before, root_after))

        advanced = bool(
            (joint_delta is not None and joint_delta > position_tol)
            or (root_delta is not None and root_delta > position_tol)
        )
        return {
            "prim_path": prim_path,
            "advanced": advanced,
            "steps": stepped,
            "max_joint_delta": joint_delta,
            "max_root_delta": root_delta,
            "source": "physx_runtime",
        }

    # ── Physics ────────────────────────────────────────────

    def create_world(self, **kwargs) -> Any:
        from isaacsim.core.api import World

        return World(**kwargs)

    def create_simulation_context(self, **kwargs) -> Any:
        from isaacsim.core.api import SimulationContext

        return SimulationContext(**kwargs)

    def create_physics_scene(
        self,
        gravity: Optional[Sequence[float]] = None,
        scene_name: str = "PhysicsScene",
    ) -> str:
        import omni.kit.commands

        scene_path = f"/World/{scene_name}"
        omni.kit.commands.execute(
            "CreatePrim", prim_path=scene_path, prim_type="PhysicsScene"
        )
        return scene_path

    def get_physics_state(self, prim_path: str) -> Dict[str, Any]:
        from pxr import UsdPhysics

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found: {prim_path}")

        result: Dict[str, Any] = {"prim_path": prim_path}

        # Check rigid body API
        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        result["has_rigid_body"] = has_rb

        if has_rb:
            rb = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rb.GetKinematicEnabledAttr()
            result["is_kinematic"] = kinematic_attr.Get() if kinematic_attr else False

        # Check mass
        has_mass = prim.HasAPI(UsdPhysics.MassAPI)
        if has_mass:
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            result["mass"] = mass_attr.Get() if mass_attr else None

        # Check collision
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        result["collision_enabled"] = has_collision

        # Get velocities from PhysX runtime API (not USD attributes which may be stale)
        if has_rb:
            try:
                import omni.physx

                physx = omni.physx.get_physx_interface()
                rb_data = physx.get_rigidbody_transformation(prim_path)
                if rb_data and rb_data.get("ret_val", False):
                    vel = rb_data.get("linear_velocity", (0.0, 0.0, 0.0))
                    ang_vel = rb_data.get("angular_velocity", (0.0, 0.0, 0.0))
                    result["linear_velocity"] = [
                        float(vel[0]),
                        float(vel[1]),
                        float(vel[2]),
                    ]
                    result["angular_velocity"] = [
                        float(ang_vel[0]),
                        float(ang_vel[1]),
                        float(ang_vel[2]),
                    ]
                else:
                    result["linear_velocity"] = [0.0, 0.0, 0.0]
                    result["angular_velocity"] = [0.0, 0.0, 0.0]
            except Exception:
                result["linear_velocity"] = [0.0, 0.0, 0.0]
                result["angular_velocity"] = [0.0, 0.0, 0.0]

        # Contact reaction forces from the PhysX contact-report buffer (runtime
        # physics state, not authored USD). Nonempty on impact — e.g. a foot
        # striking the ground reports a nonzero contact impulse; [] otherwise.
        try:
            result["contacts"] = self._read_contact_forces(prim_path)
        except Exception:
            result["contacts"] = []

        return result

    def ensure_contact_reporting(self, prim_path: str) -> None:
        """Public spawn / pre-play entry point (overrides base). The extension
        calls this at robot spawn — via robots.create — rather than lazily in
        the get_physics_state read path, so PhysxContactReportAPI is authored
        before the scene goes live and foot-ground reaction forces fire on the
        first impact instead of being dropped by PhysX until a re-parse.
        """
        self._ensure_contact_reporting(prim_path)

    def _ensure_contact_reporting(self, prim_path: str) -> None:
        """Apply PhysxContactReportAPI to rigid bodies under prim_path so PhysX
        emits contact reports for them. Idempotent; safe to call repeatedly.

        TODO(probe:contacts): whether applying this on an already-playing sim
        takes effect without a physics re-parse is runtime-unconfirmed — hence
        it is authored at spawn (pre-play) via ensure_contact_reporting, not in
        the read path, for guaranteed reporting.
        """
        from pxr import PhysxSchema, Usd, UsdPhysics

        stage = self.get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return
        for desc in Usd.PrimRange(root_prim):
            if not desc.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if desc.HasAPI(PhysxSchema.PhysxContactReportAPI):
                continue
            api = PhysxSchema.PhysxContactReportAPI.Apply(desc)
            # threshold 0 -> report every contact (foot-ground impacts included).
            api.CreateThresholdAttr(0.0)

    def _read_contact_forces(self, prim_path: str) -> List[Dict[str, Any]]:
        """Read PhysX contact reports whose collider subtree touches prim_path.

        Uses the PhysX simulation contact-report buffer (RUNTIME physics state,
        never authored USD). Foot-ground reaction forces appear here as nonzero
        contact impulses; returns [] when nothing is in contact (or sim stopped).

        TODO(probe:contacts): the contact-report struct field names and impulse
        units (impulse [N·s] vs force = impulse/dt) are runtime-unconfirmed on
        6.0.0-rc.22; reported values are passed through as PhysX supplies them.
        """
        contacts: List[Dict[str, Any]] = []
        try:
            import omni.physx
            from pxr import PhysicsSchemaTools
        except Exception:
            return contacts

        # NB: contact reporting is authored at SPAWN (ensure_contact_reporting),
        # NOT here. Applying it lazily in this read path is too late once the
        # scene is live (PhysX will not re-parse), so it would silently miss the
        # first-impact forces. This reader only consumes the real PhysX buffer.
        try:
            sim = omni.physx.get_physx_simulation_interface()
            contact_headers, contact_data = sim.get_contact_report()
        except Exception:
            return contacts

        def _decode(encoded: Any) -> str:
            try:
                return str(PhysicsSchemaTools.intToSdfPath(encoded))
            except Exception:
                return ""

        def _vec3(v: Any) -> List[float]:
            try:
                return [float(v[0]), float(v[1]), float(v[2])]
            except Exception:
                try:
                    return [float(v.x), float(v.y), float(v.z)]
                except Exception:
                    return [0.0, 0.0, 0.0]

        for header in contact_headers or []:
            body0 = _decode(getattr(header, "collider0", getattr(header, "actor0", 0)))
            body1 = _decode(getattr(header, "collider1", getattr(header, "actor1", 0)))
            if prim_path not in body0 and prim_path not in body1:
                continue
            offset = int(getattr(header, "contact_data_offset", 0) or 0)
            count = int(getattr(header, "num_contact_data", 0) or 0)
            total = [0.0, 0.0, 0.0]
            points: List[Dict[str, Any]] = []
            for i in range(offset, offset + count):
                if i < 0 or i >= len(contact_data):
                    break
                cd = contact_data[i]
                impulse = _vec3(getattr(cd, "impulse", (0.0, 0.0, 0.0)))
                total = [
                    total[0] + impulse[0],
                    total[1] + impulse[1],
                    total[2] + impulse[2],
                ]
                points.append(
                    {
                        "position": _vec3(getattr(cd, "position", (0.0, 0.0, 0.0))),
                        "normal": _vec3(getattr(cd, "normal", (0.0, 0.0, 0.0))),
                        "impulse": impulse,
                        "separation": float(getattr(cd, "separation", 0.0) or 0.0),
                    }
                )
            contacts.append(
                {
                    "body0": body0,
                    "body1": body1,
                    "num_points": count,
                    "total_impulse": total,
                    "points": points,
                }
            )
        return contacts

    # ── Sensors ────────────────────────────────────────────

    def create_camera(
        self, prim_path: str, resolution: Tuple[int, int] = (1280, 720), **kwargs
    ) -> Any:
        from isaacsim.sensors.camera import Camera

        return Camera(prim_path=prim_path, resolution=resolution, **kwargs)

    def capture_camera_image(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.camera import Camera

        cam = Camera(prim_path=prim_path)
        return cam.get_rgba()

    def create_lidar(
        self, prim_path: str, config: Optional[str] = None, **kwargs
    ) -> Any:
        from isaacsim.sensors.rtx import LidarRtx

        return LidarRtx(
            prim_path=prim_path, config=config or "Example_Rotary", **kwargs
        )

    def get_lidar_point_cloud(self, prim_path: str) -> np.ndarray:
        from isaacsim.sensors.rtx import LidarRtx

        lidar = LidarRtx(prim_path=prim_path)
        return lidar.get_point_cloud()

    # ── Materials ──────────────────────────────────────────

    def create_pbr_material(
        self,
        prim_path: str,
        color: Optional[Sequence[float]] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
    ) -> Any:
        from pxr import Gf, Sdf, UsdShade

        stage = self.get_stage()
        material = UsdShade.Material.Define(stage, prim_path)
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        if color:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color[:3])
            )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
        return material

    def create_physics_material(
        self,
        prim_path: str,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        restitution: float = 0.0,
    ) -> Any:
        from pxr import UsdPhysics

        stage = self.get_stage()
        material = UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(prim_path))
        material.CreateStaticFrictionAttr(static_friction)
        material.CreateDynamicFrictionAttr(dynamic_friction)
        material.CreateRestitutionAttr(restitution)
        return material

    def apply_material(self, material_path: str, target_prim_path: str) -> None:
        from pxr import UsdShade

        stage = self.get_stage()
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        target = stage.GetPrimAtPath(target_prim_path)
        UsdShade.MaterialBindingAPI(target).Bind(material)

    # ── Lighting ───────────────────────────────────────────

    def create_light(
        self,
        light_type: str,
        prim_path: str,
        intensity: float = 1000.0,
        color: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> Any:
        from pxr import Gf, UsdLux

        stage = self.get_stage()
        light_classes = {
            "DistantLight": UsdLux.DistantLight,
            "DomeLight": UsdLux.DomeLight,
            "SphereLight": UsdLux.SphereLight,
            "RectLight": UsdLux.RectLight,
            "DiskLight": UsdLux.DiskLight,
            "CylinderLight": UsdLux.CylinderLight,
        }
        cls = light_classes.get(light_type)
        if not cls:
            raise ValueError(
                f"Unknown light type: {light_type}. Options: {list(light_classes.keys())}"
            )
        light = cls.Define(stage, prim_path)
        light.CreateIntensityAttr(intensity)
        if color:
            light.CreateColorAttr(Gf.Vec3f(*color[:3]))
        position = kwargs.get("position")
        if position:
            self.set_prim_transform(prim_path, position=position)
        rotation = kwargs.get("rotation")
        if rotation:
            self.set_prim_transform(prim_path, rotation=rotation)
        return light

    def modify_light(
        self,
        prim_path: str,
        intensity: Optional[float] = None,
        color: Optional[Sequence[float]] = None,
    ) -> None:
        from pxr import Gf

        stage = self.get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Light not found: {prim_path}")
        if intensity is not None:
            prim.GetAttribute("inputs:intensity").Set(intensity)
        if color is not None:
            prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color[:3]))

    # ── Assets ─────────────────────────────────────────────

    def clone_prim(self, source_path: str, target_path: str) -> None:
        import omni.kit.commands

        omni.kit.commands.execute(
            "CopyPrim", path_from=source_path, path_to=target_path
        )

    def import_urdf(
        self, urdf_path: str, prim_path: str = "/World/robot", **kwargs
    ) -> Any:
        import os

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status or import_config is None:
            raise RuntimeError("URDFCreateImportConfig failed")
        parse_result = omni.kit.commands.execute(
            "URDFParseFile", urdf_path=urdf_path, import_config=import_config
        )
        if (
            isinstance(parse_result, tuple)
            and parse_result
            and parse_result[0] is False
        ):
            raise RuntimeError(f"URDFParseFile failed: {parse_result}")
        result = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            import_config=import_config,
            dest_path=prim_path,
        )
        if isinstance(result, tuple) and result and result[0] is False:
            self.delete_prim(prim_path)
            raise RuntimeError(f"URDFImportRobot failed: {result}")
        prim = self.get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            self.delete_prim(prim_path)
            raise RuntimeError(
                f"URDF importer completed but produced no valid prim at {prim_path}: {result}"
            )
        return result

    # ── Simulation ─────────────────────────────────────────

    def play(self) -> None:
        import omni.timeline

        self._ensure_physics_world()
        omni.timeline.get_timeline_interface().play()

    def pause(self) -> None:
        import omni.timeline

        omni.timeline.get_timeline_interface().pause()

    def stop(self) -> None:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
        self._articulation_cache.clear()
        self._joint_name_cache.clear()

    def ping(self) -> Dict[str, Any]:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        return {
            "ready": stage is not None,
            "stage_url": omni.usd.get_context().get_stage_url() or "",
            "resources": self.get_resources(compact=True),
        }

    def get_resources(self, compact: bool = False) -> Dict[str, Any]:
        mem: Dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    key, value = line.split(":", 1)
                    parts = value.strip().split()
                    if parts:
                        mem[key] = int(parts[0])
        except Exception:
            mem = {}

        result: Dict[str, Any] = {
            "available_ram_mb": mem.get("MemAvailable", 0) // 1024 if mem else None,
            "total_ram_mb": mem.get("MemTotal", 0) // 1024 if mem else None,
            "swap_free_mb": mem.get("SwapFree", 0) // 1024 if mem else None,
            "swap_total_mb": mem.get("SwapTotal", 0) // 1024 if mem else None,
        }
        if not compact:
            try:
                import omni.timeline

                timeline = omni.timeline.get_timeline_interface()
                result["timeline_state"] = (
                    "playing"
                    if timeline.is_playing()
                    else "stopped"
                    if timeline.is_stopped()
                    else "paused"
                )
            except Exception:
                result["timeline_state"] = None

            # nvidia-smi is a ~2s subprocess on Isaac's MAIN thread. The compact path
            # is the high-frequency health probe (simulation.ping on every save +
            # readiness + bridge heartbeat); running nvidia-smi there adds repeated
            # main-thread stalls — the opposite of the #155 goal — so VRAM telemetry
            # is gathered ONLY for the explicit get_resources call (F5). Also: the
            # default NVIDIA_VISIBLE_DEVICES="all" is not a valid `-i` selector, so
            # only pass -i for a numeric index; otherwise query all and take the
            # first row (F14).
            try:
                vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").split(",")[0].strip()
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ]
                if vis.isdigit():
                    cmd += ["-i", vis]
                proc = subprocess.run(
                    cmd, text=True, capture_output=True, timeout=2, check=False
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    free, total = [
                        int(part.strip())
                        for part in proc.stdout.strip().splitlines()[0].split(",")[:2]
                    ]
                    result["vram_free_mb"] = free
                    result["vram_total_mb"] = total
                else:
                    result["vram_free_mb"] = None
                    result["vram_total_mb"] = None
            except Exception:
                result["vram_free_mb"] = None
                result["vram_total_mb"] = None

        return result

    def step(
        self,
        num_steps: int = 1,
        observe_prims: Optional[List[str]] = None,
        observe_joints: Optional[List[str]] = None,
        budget_ms: Optional[int] = None,
        observe_cap: Optional[int] = None,
    ) -> Dict[str, Any]:
        import omni.kit.app

        start = time.monotonic()
        # Treat budget_ms<=0 (and None) as the default budget — NOT as "unbounded".
        # A caller passing budget_ms=0 previously disabled the wall-clock guard
        # entirely, re-opening the exact OOM/main-thread wedge #151 asked to fix.
        # A hard frame ceiling is a second guard against a runaway num_steps (F2).
        effective_budget_ms = (
            budget_ms if (budget_ms is not None and budget_ms > 0) else 8000
        )
        max_step_frames = 4000
        stepped = 0
        timed_out = False
        for _ in range(min(max(0, num_steps), max_step_frames)):
            if (time.monotonic() - start) * 1000 >= effective_budget_ms:
                timed_out = True
                break
            omni.kit.app.get_app().update()
            stepped += 1

        result: Dict[str, Any] = {
            "stepped": stepped,
            "resources": self.get_resources(compact=True),
        }
        if timed_out:
            result["timed_out"] = True

        # Observe prim states
        if observe_prims:
            from pxr import UsdPhysics

            prim_states = []
            stage = self.get_stage()
            for path in observe_prims:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    prim_states.append({"prim_path": path, "error": "Prim not found"})
                    continue
                state: Dict[str, Any] = {"prim_path": path}
                # Prefer PhysX runtime position for rigid bodies (always current)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        import omni.physx

                        physx = omni.physx.get_physx_interface()
                        rb_data = physx.get_rigidbody_transformation(path)
                        if rb_data and rb_data.get("ret_val", False):
                            pos = rb_data["position"]
                            state["position"] = [
                                float(pos[0]),
                                float(pos[1]),
                                float(pos[2]),
                            ]
                        else:
                            transform = self.get_prim_transform(path)
                            state["position"] = transform.get("position", [0, 0, 0])
                    except Exception:
                        transform = self.get_prim_transform(path)
                        state["position"] = transform.get("position", [0, 0, 0])
                else:
                    transform = self.get_prim_transform(path)
                    state["position"] = transform.get("position", [0, 0, 0])
                # Add velocity if rigid body
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        physics_state = self.get_physics_state(path)
                        state["linear_velocity"] = physics_state.get(
                            "linear_velocity", [0, 0, 0]
                        )
                        state["angular_velocity"] = physics_state.get(
                            "angular_velocity", [0, 0, 0]
                        )
                    except Exception:
                        pass
                prim_states.append(state)
            result["prim_states"] = prim_states

        # Observe joint states
        if observe_joints:
            joint_states = []
            resources = result.get("resources") or {}
            available_ram_mb = resources.get("available_ram_mb")
            effective_observe_cap = observe_cap
            if (
                effective_observe_cap is None
                and isinstance(available_ram_mb, int)
                and available_ram_mb < 3000
            ):
                effective_observe_cap = 1
            observed_paths = list(observe_joints)
            skipped_paths: List[str] = []
            if effective_observe_cap is not None and effective_observe_cap >= 0:
                skipped_paths = observed_paths[effective_observe_cap:]
                observed_paths = observed_paths[:effective_observe_cap]
            for path in observed_paths:
                try:
                    positions = self.get_joint_positions(path)
                    names = self._get_joint_names(path)
                    joints_dict = (
                        dict(zip(names, positions))
                        if names
                        else {"positions": positions}
                    )
                    joint_states.append({"prim_path": path, "joints": joints_dict})
                except Exception as e:
                    joint_states.append({"prim_path": path, "error": str(e)})
            result["joint_states"] = joint_states
            if skipped_paths:
                result["observe_skipped"] = [
                    {"prim_path": path, "reason": "art_observe_cap"}
                    for path in skipped_paths
                ]

        return result

    def get_simulation_state(self) -> Dict[str, Any]:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        is_playing = timeline.is_playing()
        is_stopped = timeline.is_stopped()

        if is_playing:
            state = "playing"
        elif is_stopped:
            state = "stopped"
        else:
            state = "paused"

        current_time = timeline.get_current_time()
        # Get physics dt from physics scene if available
        from pxr import UsdPhysics

        stage = self.get_stage()
        physics_dt = 1.0 / 60.0  # default
        for prim in stage.Traverse():
            # UsdPhysics.Scene is a typed (IsA) schema, not an applied-API
            # schema: HasAPI(Scene) is always False, so the old check never
            # found the scene and always returned the 1/60 default — making the
            # autospawn physics_dt==1/200 verification a lie. IsA is correct.
            if prim.IsA(UsdPhysics.Scene):
                time_step_attr = prim.GetAttribute("physxScene:timeStepsPerSecond")
                if time_step_attr and time_step_attr.Get():
                    steps_per_sec = time_step_attr.Get()
                    if steps_per_sec > 0:
                        physics_dt = 1.0 / steps_per_sec
                break

        return {
            "timeline_state": state,
            "current_time": current_time,
            "physics_dt": physics_dt,
        }

    def execute_script(self, code: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        import io
        import sys

        import carb
        import omni
        from pxr import Gf, Sdf, Usd, UsdGeom

        # Auto-add cwd to sys.path
        if cwd and cwd not in sys.path:
            sys.path.insert(0, cwd)

        local_ns = {
            "omni": omni,
            "carb": carb,
            "Usd": Usd,
            "UsdGeom": UsdGeom,
            "Sdf": Sdf,
            "Gf": Gf,
        }

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_out = io.StringIO()
        sys.stderr = captured_err = io.StringIO()
        try:
            self._ensure_physics_world()
            exec(code, local_ns)
            return {
                "status": "success",
                "message": "Script executed successfully",
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # Track exec() namespaces to clean up subscriptions on reload
    _exec_namespaces: Dict[str, dict] = {}

    def reload_script(
        self, file_path: str, module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        import importlib
        import io
        import os
        import sys

        # Auto-add parent directory to sys.path
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        # Clean up previous exec() namespace for this file (unsubscribe orphaned callbacks)
        abs_path = os.path.abspath(file_path)
        old_ns = self._exec_namespaces.get(abs_path)
        if old_ns:
            for key, val in old_ns.items():
                if hasattr(val, "unsubscribe"):
                    try:
                        val.unsubscribe()
                    except Exception:
                        pass

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_out = io.StringIO()
        sys.stderr = captured_err = io.StringIO()
        try:
            if module_name:
                # Reload existing module or import for first time
                if module_name in sys.modules:
                    _module = importlib.reload(sys.modules[module_name])
                    msg = f"Module '{module_name}' reloaded successfully"
                else:
                    _module = importlib.import_module(module_name)
                    msg = f"Module '{module_name}' imported successfully"
            else:
                # Execute file contents (hot-patch)
                if not os.path.isfile(file_path):
                    return {
                        "status": "error",
                        "message": f"File not found: {file_path}",
                    }
                with open(file_path, "r") as f:
                    code = f.read()
                import carb
                import omni
                from pxr import Gf, Sdf, Usd, UsdGeom

                local_ns = {
                    "omni": omni,
                    "carb": carb,
                    "Usd": Usd,
                    "UsdGeom": UsdGeom,
                    "Sdf": Sdf,
                    "Gf": Gf,
                    "__file__": file_path,
                }
                self._ensure_physics_world()
                exec(code, local_ns)
                # Track namespace so we can clean up subscriptions on next reload
                self._exec_namespaces[abs_path] = local_ns
                msg = f"Script '{os.path.basename(file_path)}' executed successfully"

            return {
                "status": "success",
                "message": msg,
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "stdout": captured_out.getvalue(),
                "stderr": captured_err.getvalue(),
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
