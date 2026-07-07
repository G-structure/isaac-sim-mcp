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

"""Scene management command handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..adapters.base import IsaacAdapterBase
from ._guards import guard_gravity_write, guard_pose_write

_discovered_envs: Optional[Dict[str, Dict[str, str]]] = None


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["scene.get_info"] = lambda **p: get_info(adapter, **p)
    registry["scene.create_physics"] = lambda **p: create_physics(adapter, **p)
    registry["scene.clear"] = lambda **p: clear(adapter, **p)
    registry["scene.list_prims"] = lambda **p: list_prims(adapter, **p)
    registry["scene.get_prim_info"] = lambda **p: get_prim_info(adapter, **p)
    registry["scene.list_environments"] = lambda **p: list_environments(adapter, **p)
    registry["scene.load_environment"] = lambda **p: load_environment(adapter, **p)


def get_info(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        stage = adapter.get_stage()
        assets_root = adapter.get_assets_root_path()
        prim_count = len(list(stage.TraverseAll()))
        stage_path = stage.GetRootLayer().realPath
        return {
            "status": "success",
            "message": "pong",
            "assets_root_path": assets_root,
            "stage_path": stage_path,
            "prim_count": prim_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_physics(
    adapter: IsaacAdapterBase, gravity: Optional[Sequence[float]] = None, scene_name: str = "PhysicsScene"
) -> Dict[str, Any]:
    try:
        # Physical-gravity gate: a session may not author a zero / tiny / sideways /
        # upward gravity (a fake-balance vector) into a new physics scene. OPERATOR
        # bypasses; gravity=None (engine default) is allowed.
        rejection = guard_gravity_write(adapter, gravity)
        if rejection is not None:
            return rejection
        # Sink guard: create_physics_scene issues a CreatePrim at the
        # caller-controlled /World/<scene_name>; a scene_name resolving onto (or
        # under) a robot articulation-root would author a PhysicsScene over it while
        # the sim runs. Guard the resolved path with the same fail-closed pose lock
        # (operator bypasses; a fresh/non-robot path or a stopped sim is allowed).
        scene_path = f"/World/{scene_name}"
        rejection = guard_pose_write(adapter, scene_path)
        if rejection is not None:
            return rejection
        scene_path = adapter.create_physics_scene(gravity=gravity, scene_name=scene_name)
        # Create ground plane with collision so objects don't fall through
        floor_path = "/World/groundPlane"
        adapter.create_prim(floor_path, "Plane")
        from pxr import UsdPhysics

        stage = adapter.get_stage()
        gp = stage.GetPrimAtPath(floor_path)
        if gp.IsValid() and not gp.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(gp)
        return {"status": "success", "message": f"Physics scene created at {scene_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear(adapter: IsaacAdapterBase, keep_physics: bool = False) -> Dict[str, Any]:
    try:
        stage = adapter.get_stage()
        # Prims to never delete (system prims)
        keep_paths = {
            "/OmniverseKit_Persp",
            "/OmniverseKit_Front",
            "/OmniverseKit_Top",
            "/OmniverseKit_Right",
            "/Render",
            "/Environment",
        }
        # Clear all root-level prims (robots created at root, etc.)
        root_prim = stage.GetPseudoRoot()
        to_delete = []
        for child in root_prim.GetChildren():
            path = str(child.GetPath())
            if path in keep_paths:
                continue
            if keep_physics and "Physics" in path:
                continue
            to_delete.append(path)
        # Sink guard: scene.clear is a bulk delete_prim over every root-level prim —
        # including a robot articulation-root — so it is a delete+recreate reposition
        # vector just like objects.delete. Fail closed and ATOMIC: if ANY target is
        # (or descends from) a robot root while the timeline is not stopped, reject
        # the whole clear BEFORE deleting anything (operator bypasses; non-robot
        # prims and a stopped sim are allowed).
        for path in to_delete:
            rejection = guard_pose_write(adapter, path)
            if rejection is not None:
                return rejection
        for path in to_delete:
            adapter.delete_prim(path)
        return {"status": "success", "message": "Scene cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_prims(adapter: IsaacAdapterBase, root_path: str = "/", prim_type: Optional[str] = None) -> Dict[str, Any]:
    try:
        prims = adapter.list_prims(root_path=root_path, prim_type=prim_type)
        return {"status": "success", "prims": prims}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_prim_info(adapter: IsaacAdapterBase, prim_path: str = "/") -> Dict[str, Any]:
    try:
        info = adapter.get_prim_info(prim_path)
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_env_library(adapter: IsaacAdapterBase) -> Dict[str, Dict[str, str]]:
    global _discovered_envs
    if _discovered_envs is not None:
        return _discovered_envs
    try:
        envs = adapter.discover_environments()
        if envs:
            _discovered_envs = envs
            print(f"Discovered {len(envs)} environments from asset server")
            return _discovered_envs
    except Exception as e:
        print(f"Environment discovery failed: {e}")
    _discovered_envs = {}
    return _discovered_envs


def list_environments(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    library = _get_env_library(adapter)
    return {"status": "success", "environment_count": len(library), "environments": library}


def load_environment(
    adapter: IsaacAdapterBase, environment: Optional[str] = None, prim_path: str = "/Environment"
) -> Dict[str, Any]:
    try:
        if not environment:
            return {
                "status": "error",
                "message": "environment is required. Use scene.list_environments to see options.",
            }

        library = _get_env_library(adapter)
        q = environment.lower().strip()

        # Exact match
        match = library.get(q)

        # Fuzzy match
        if not match:
            for key, info in library.items():
                if q in key or q in info.get("description", "").lower():
                    match = info
                    break

        if not match:
            available = list(library.keys())[:15]
            return {"status": "error", "message": f"Environment '{environment}' not found. Options: {available}"}

        # Sink guard: an add_reference_to_stage that drops an environment reference
        # onto (or over) a robot articulation-root while the timeline is not stopped is
        # a disguised teleport/overwrite. Route the caller-controlled prim_path through
        # the same fail-closed pose lock objects.transform/create use. OPERATOR bypasses.
        rejection = guard_pose_write(adapter, prim_path)
        if rejection is not None:
            return rejection

        assets_root = adapter.get_assets_root_path()
        full_path = assets_root + match["asset_path"]
        adapter.load_environment(full_path, prim_path)
        return {"status": "success", "message": f"Loaded environment: {match['description']}", "prim_path": prim_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}
