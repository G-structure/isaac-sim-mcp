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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Behavioural proof that the REMAINING authoring/delete sinks are guarded.

tests/test_pose_sink_guards.py + tests/test_rce_closure.py cover the pose-write
surface reachable via robots.create / assets.* / sensors.* / objects.create /
objects.clone. This file closes the sinks found in the comprehensive sweep that
were previously UNguarded and could teleport / overwrite / author-over / delete a
robot at a caller-controlled path while the sim is live:

  * ``lighting.create``   -> adapter.create_light (UsdLux.*.Define + set_prim_transform)
  * ``materials.create``  -> adapter.create_pbr_material (UsdShade.*.Define) /
                             adapter.create_physics_material (stage.DefinePrim)
  * ``materials.apply``   -> adapter.apply_material (UsdShade.MaterialBindingAPI.Bind)
  * ``objects.delete``    -> adapter.delete_prim (closes delete+recreate reposition)
  * ``scene.create_physics`` -> adapter.create_physics_scene CreatePrim at
                             /World/<scene_name> (caller-controlled)
  * ``scene.clear``       -> bulk adapter.delete_prim over every root-level prim
  * ``graphs.create_action_graph`` -> og.Controller.edit(CREATE_NODES) at graph_path

Each guarded sink must, in SESSION mode: REJECT a write over a registered robot
articulation-root (or a descendant, or when discovery fails) while the timeline is
PLAYING/PAUSED *and the sink never runs*, while ALLOWING a fresh non-robot path,
a stopped sim, and the operator/legacy path.

The stage here is modelled REALISTICALLY: the pseudo-root ``/`` holds real
TOP-LEVEL prims (``/World``, ``/Render``) and the robot is NESTED at ``/World/G1``
(not a direct pseudo-root child). This exposes the core-predicate hole where a
SESSION deletes / clears / transforms / authors-over an ANCESTOR of the robot
(``/World`` — or the pseudo-root ``/`` itself) to wipe or teleport the robot
through its parent: every such write while the sim is live must be REJECTED and
the sink must never run, exactly as a write onto the robot root itself is.

These tests drive the real handler bodies and the real
``MCPExtension._execute_command`` dispatch against a fake adapter that RECORDS
every sink call — they do NOT grep source. Isaac Sim is absent offline, so
``numpy`` / ``carb`` / ``omni.*`` / ``pxr`` are stubbed into ``sys.modules`` before
the real package is imported; the stubs only satisfy imports, every assertion is
against the real guard + handler dispatch logic.
TODO(probe:guard-live): re-run each rejection against a live articulation root.
"""

import contextlib
import os
import sys
import types
from unittest import mock

# ── stub the Isaac/omni import surface before importing the real package ──────

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension")
)

ROBOT = "/World/G1"


class _FakePrim:
    def __init__(self, path: str = "") -> None:
        self._path = path

    def GetPath(self):
        return self._path  # str(path) == path for a plain string

    def IsValid(self):
        return False

    def HasAPI(self, _api):
        return True


class _FakePseudoRoot:
    def __init__(self, children):
        self._children = children

    def GetChildren(self):
        return self._children


class _FakeStage:
    def __init__(self, children=()):
        self._children = [_FakePrim(p) for p in children]

    def GetPseudoRoot(self):
        return _FakePseudoRoot(self._children)

    def GetPrimAtPath(self, _path):
        return _FakePrim()

    def TraverseAll(self):
        return []


def _install_stubs() -> None:
    sys.modules.setdefault("numpy", mock.MagicMock())

    if "carb" not in sys.modules:
        carb = types.ModuleType("carb")
        carb.settings = types.SimpleNamespace(
            get_settings=lambda: types.SimpleNamespace(get=lambda *a, **k: None)
        )
        sys.modules["carb"] = carb

    omni = sys.modules.get("omni") or types.ModuleType("omni")
    sys.modules["omni"] = omni

    def _sub(parent_name: str, parent_mod, child: str):
        full = f"{parent_name}.{child}"
        mod = sys.modules.get(full) or types.ModuleType(full)
        sys.modules[full] = mod
        setattr(parent_mod, child, mod)
        return mod

    ext = _sub("omni", omni, "ext")
    if not hasattr(ext, "IExt"):
        class _IExt:  # `class MCPExtension(omni.ext.IExt)` needs a real base class
            def __init__(self, *a, **k):
                pass

        ext.IExt = _IExt

    kit = _sub("omni", omni, "kit")
    _sub("omni.kit", kit, "app")
    commands = _sub("omni.kit", kit, "commands")
    if not hasattr(commands, "execute"):
        commands.execute = lambda *a, **k: None
    _sub("omni", omni, "usd")

    # scene.create_physics does `from pxr import UsdPhysics` after its sink; a minimal
    # stub lets the allowed path finish cleanly. UsdGeom is only the up-axis fallback
    # (the gravity guard reads the axis off the adapter first).
    if "pxr" not in sys.modules:
        pxr = types.ModuleType("pxr")

        class _CollisionAPI:
            @staticmethod
            def Apply(_prim):
                return None

        pxr.UsdPhysics = types.SimpleNamespace(CollisionAPI=_CollisionAPI)
        pxr.UsdGeom = types.SimpleNamespace(GetStageUpAxis=lambda _stage: "Z")
        sys.modules["pxr"] = pxr

    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)


_install_stubs()

from isaac_sim_mcp_extension.extension import MCPExtension  # noqa: E402
from isaac_sim_mcp_extension.handlers import (  # noqa: E402
    _guards,
    graphs,
    lighting,
    materials,
    objects,
    scene,
)

REJECTED = _guards.REJECTED_BY  # "session_guard"


@contextlib.contextmanager
def _session_mode(value):
    """Force MCP_SESSION_MODE for the duration of a test, then restore it."""
    old = os.environ.get("MCP_SESSION_MODE")
    if value is None:
        os.environ.pop("MCP_SESSION_MODE", None)
    else:
        os.environ["MCP_SESSION_MODE"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MCP_SESSION_MODE", None)
        else:
            os.environ["MCP_SESSION_MODE"] = old


class FakeAdapter:
    """Records every authoring/delete sink so a test can assert it never ran."""

    def __init__(self, timeline="playing", gravity=9.81, dt=1.0 / 60.0, roots=(ROBOT,), children=()):
        self._timeline = timeline
        self._gravity = gravity
        self._dt = dt
        self._roots = roots
        self._children = tuple(children)
        self.create_light_calls = []
        self.create_pbr_calls = []
        self.create_physics_material_calls = []
        self.apply_material_calls = []
        self.delete_calls = []
        self.create_physics_scene_calls = []
        self.create_prim_calls = []
        self.set_prim_transform_calls = []

    # world-state hooks used by the guard
    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": self._dt}

    def list_articulation_roots(self):
        return self._roots

    def get_gravity_magnitude(self):
        return self._gravity

    def get_prim_transform(self, prim_path):
        return {"position": [0.0, 0.0, 1.0]}

    def get_stage_up_axis(self):
        return "Z"

    def get_stage(self):
        return _FakeStage(self._children)

    # lighting sink
    def create_light(self, light_type, prim_path, **kwargs):
        self.create_light_calls.append((light_type, prim_path))
        return object()

    # materials sinks
    def create_pbr_material(self, prim_path, **kwargs):
        self.create_pbr_calls.append(prim_path)
        return object()

    def create_physics_material(self, prim_path, **kwargs):
        self.create_physics_material_calls.append(prim_path)
        return object()

    def apply_material(self, material_path, target_prim_path):
        self.apply_material_calls.append((material_path, target_prim_path))
        return None

    # objects.transform sink (dispatch-level guard gates this one)
    def set_prim_transform(self, prim_path, **kwargs):
        self.set_prim_transform_calls.append((prim_path, kwargs))
        return None

    # objects.delete / scene.clear sink
    def delete_prim(self, prim_path):
        self.delete_calls.append(prim_path)
        return True

    # scene.create_physics sinks
    def create_physics_scene(self, gravity=None, scene_name="PhysicsScene"):
        self.create_physics_scene_calls.append(scene_name)
        return f"/World/{scene_name}"

    def create_prim(self, prim_path, prim_type="Xform", **kwargs):
        self.create_prim_calls.append(prim_path)
        return object()


def _is_reject(d):
    return d is not None and d.get("status") == "error" and d.get("rejected_by") == REJECTED


# ══ lighting.create — UsdLux define + set_prim_transform over a robot root ═════


def test_lighting_create_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = lighting.create(adapter, light_type="DistantLight", prim_path=ROBOT, position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_light_calls == [], "create_light sink must not run on reject"


def test_lighting_create_over_robot_descendant_blocked_while_paused():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="paused")
        d = lighting.create(adapter, prim_path=ROBOT + "/head", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_light_calls == []


def test_lighting_create_fresh_path_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = lighting.create(adapter, light_type="DistantLight", prim_path="/World/Light", position=[0, 0, 9])
    assert d.get("status") == "success", d
    assert adapter.create_light_calls == [("DistantLight", "/World/Light")]


def test_lighting_create_auto_name_resolved_before_guard_allowed():
    # No prim_path => handler resolves /World/<type>_<count> from the stage BEFORE
    # the guard; the resolved fresh path is non-robot, so the sink runs.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = lighting.create(adapter, light_type="DistantLight")
    assert d.get("status") == "success", d
    assert adapter.create_light_calls == [("DistantLight", "/World/DistantLight_0")]


def test_lighting_create_over_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        lighting.create(adapter, prim_path=ROBOT, position=[0, 0, 9])
    assert adapter.create_light_calls == [("DistantLight", ROBOT)], "authoring while stopped must be allowed"


def test_lighting_create_over_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        lighting.create(adapter, prim_path=ROBOT, position=[0, 0, 9])
    assert adapter.create_light_calls == [("DistantLight", ROBOT)], "operator mode must not apply the lock"


def test_lighting_create_fails_closed_when_discovery_fails():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = lighting.create(adapter, prim_path="/World/Light", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_light_calls == []


# ══ materials.create — UsdShade / DefinePrim over a robot root ════════════════


def test_materials_create_pbr_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.create(adapter, material_type="pbr", prim_path=ROBOT)
    assert _is_reject(d), d
    assert adapter.create_pbr_calls == [], "create_pbr_material sink must not run on reject"


def test_materials_create_physics_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.create(adapter, material_type="physics", prim_path=ROBOT)
    assert _is_reject(d), d
    assert adapter.create_physics_material_calls == [], "create_physics_material sink must not run"


def test_materials_create_fresh_path_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.create(adapter, material_type="pbr", prim_path="/World/Mat")
    assert d.get("status") == "success", d
    assert adapter.create_pbr_calls == ["/World/Mat"]


def test_materials_create_auto_name_resolved_before_guard_allowed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.create(adapter, material_type="pbr")
    assert d.get("status") == "success", d
    assert adapter.create_pbr_calls == ["/World/Material_0"]


def test_materials_create_over_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        materials.create(adapter, material_type="physics", prim_path=ROBOT)
    assert adapter.create_physics_material_calls == [ROBOT], "authoring while stopped must be allowed"


def test_materials_create_over_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        materials.create(adapter, material_type="pbr", prim_path=ROBOT)
    assert adapter.create_pbr_calls == [ROBOT], "operator mode must not apply the lock"


def test_materials_create_fails_closed_when_discovery_fails():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = materials.create(adapter, material_type="pbr", prim_path="/World/Mat")
    assert _is_reject(d), d
    assert adapter.create_pbr_calls == []


# ══ objects.delete — delete of a robot root is the delete+recreate cheat ══════


def test_objects_delete_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.delete(adapter, prim_path=ROBOT)
    assert _is_reject(d), d
    assert adapter.delete_calls == [], "delete_prim sink must not run on reject"


def test_objects_delete_robot_descendant_blocked_while_paused():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="paused")
        d = objects.delete(adapter, prim_path=ROBOT + "/pelvis")
    assert _is_reject(d), d
    assert adapter.delete_calls == []


def test_objects_delete_non_robot_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.delete(adapter, prim_path="/World/Cube")
    assert d.get("status") == "success", d
    assert adapter.delete_calls == ["/World/Cube"], "a non-robot delete must reach the sink"


def test_objects_delete_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        objects.delete(adapter, prim_path=ROBOT)
    assert adapter.delete_calls == [ROBOT], "deleting a robot while stopped (authoring) must be allowed"


def test_objects_delete_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        objects.delete(adapter, prim_path=ROBOT)
    assert adapter.delete_calls == [ROBOT], "operator mode must not apply the lock"


def test_objects_delete_fails_closed_when_discovery_fails():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = objects.delete(adapter, prim_path="/World/Cube")
    assert _is_reject(d), d
    assert adapter.delete_calls == []


# ══ scene.create_physics — CreatePrim at caller-controlled /World/<scene_name> ═


def test_create_physics_scene_name_over_robot_blocked_while_playing():
    # scene_name="G1" resolves to /World/G1 == the robot root.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = scene.create_physics(adapter, gravity=None, scene_name="G1")
    assert _is_reject(d), d
    assert adapter.create_physics_scene_calls == [], "create_physics_scene sink must not run on reject"


def test_create_physics_default_scene_name_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = scene.create_physics(adapter, gravity=None)
    assert d.get("status") == "success", d
    assert adapter.create_physics_scene_calls == ["PhysicsScene"]


def test_create_physics_scene_name_over_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        d = scene.create_physics(adapter, gravity=None, scene_name="G1")
    assert d.get("status") == "success", d
    assert adapter.create_physics_scene_calls == ["G1"], "authoring while stopped must be allowed"


def test_create_physics_scene_name_over_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        d = scene.create_physics(adapter, gravity=None, scene_name="G1")
    assert d.get("status") == "success", d
    assert adapter.create_physics_scene_calls == ["G1"], "operator mode must not apply the lock"


# ══ scene.clear — bulk delete_prim over the REAL top-level prims. The robot lives
# nested at /World/G1, so its parent /World is an ANCESTOR: clearing /World wipes
# the robot through its parent and must be refused while live. ═════════════════


def test_scene_clear_blocked_when_deleting_robot_ancestor_while_playing():
    # Realistic stage: pseudo-root children are /World and /Render; the robot is at
    # /World/G1. /Render is kept, so scene.clear targets /World — the robot ANCESTOR.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", children=("/World", "/Render"))
        d = scene.clear(adapter)
    assert _is_reject(d), d
    assert adapter.delete_calls == [], "fail-closed + atomic: /World is the robot's parent — delete nothing"


def test_scene_clear_without_robot_allowed_while_playing():
    # No robot present => /World is a plain prim; clearing it is fine even while live.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=(), children=("/World", "/Render"))
        d = scene.clear(adapter)
    assert d.get("status") == "success", d
    assert adapter.delete_calls == ["/World"], "with no robot present, /World is a plain prim to clear"


def test_scene_clear_deleting_robot_ancestor_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped", children=("/World", "/Render"))
        d = scene.clear(adapter)
    assert d.get("status") == "success", d
    assert adapter.delete_calls == ["/World"], "clearing the robot's parent while stopped must be allowed"


def test_scene_clear_deleting_robot_ancestor_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing", children=("/World", "/Render"))
        d = scene.clear(adapter)
    assert d.get("status") == "success", d
    assert adapter.delete_calls == ["/World"], "operator mode must not apply the lock"


# ══ graphs.create_action_graph — CREATE_NODES author-over at graph_path ═══════


def test_create_action_graph_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = graphs.create_action_graph(adapter, graph_path=ROBOT, nodes=[{"path": "t", "type": "omni.graph.action.OnPlaybackTick"}])
    assert _is_reject(d), d


def test_create_action_graph_fresh_path_passes_guard_while_playing():
    # Fresh non-robot graph_path clears the pose lock; it then fails on the absent
    # omni.graph.core import — proving the guard let it through (not rejected_by us).
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = graphs.create_action_graph(adapter, graph_path="/World/ActionGraph", nodes=[{"path": "t", "type": "omni.graph.action.OnPlaybackTick"}])
    assert d.get("rejected_by") != REJECTED, d


def test_create_action_graph_over_robot_passes_guard_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        d = graphs.create_action_graph(adapter, graph_path=ROBOT, nodes=[{"path": "t", "type": "omni.graph.action.OnPlaybackTick"}])
    assert d.get("rejected_by") != REJECTED, d


# ══ ANCESTOR bug — a SESSION write to a robot's PARENT (or the pseudo-root) wipes
# or teleports the robot THROUGH the ancestor, bypassing per-target sink guards.
# The robot is at /World/G1, so /World and / are ancestors and must be refused. ══


def test_objects_delete_robot_ancestor_world_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.delete(adapter, prim_path="/World")
    assert _is_reject(d), d
    assert adapter.delete_calls == [], "deleting the robot's parent must not reach the delete sink"


def test_objects_delete_pseudo_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.delete(adapter, prim_path="/")
    assert _is_reject(d), d
    assert adapter.delete_calls == [], "deleting the pseudo-root (ancestor of every robot) must be refused"


def test_objects_delete_fresh_sibling_of_robot_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = objects.delete(adapter, prim_path="/World/Cube")
    assert d.get("status") == "success", d
    assert adapter.delete_calls == ["/World/Cube"], "a sibling of the robot is not an ancestor — allowed"


def test_objects_delete_robot_ancestor_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        objects.delete(adapter, prim_path="/World")
    assert adapter.delete_calls == ["/World"], "deleting an ancestor while stopped (authoring) is allowed"


def test_objects_delete_robot_ancestor_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        objects.delete(adapter, prim_path="/World")
    assert adapter.delete_calls == ["/World"], "operator mode must not apply the lock"


def test_lighting_create_over_robot_ancestor_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = lighting.create(adapter, light_type="DistantLight", prim_path="/World", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_light_calls == [], "authoring a light over the robot's parent must not run the sink"


def test_lighting_create_over_pseudo_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = lighting.create(adapter, light_type="DistantLight", prim_path="/", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_light_calls == []


def test_lighting_create_over_robot_ancestor_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        lighting.create(adapter, prim_path="/World", position=[0, 0, 9])
    assert adapter.create_light_calls == [("DistantLight", "/World")], "authoring while stopped must be allowed"


# ══ materials.apply — MaterialBindingAPI.Bind authored onto a caller-controlled
# target_prim_path must not bind onto (or over an ancestor of) a live robot. ════


def test_materials_apply_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.apply_material(adapter, material_path="/World/Mat", target_prim_path=ROBOT)
    assert _is_reject(d), d
    assert adapter.apply_material_calls == [], "binding a material onto the robot root must not run the sink"


def test_materials_apply_over_robot_ancestor_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.apply_material(adapter, material_path="/World/Mat", target_prim_path="/World")
    assert _is_reject(d), d
    assert adapter.apply_material_calls == []


def test_materials_apply_fresh_target_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = materials.apply_material(adapter, material_path="/World/Mat", target_prim_path="/World/Cube")
    assert d.get("status") == "success", d
    assert adapter.apply_material_calls == [("/World/Mat", "/World/Cube")], "a fresh sibling target is allowed"


def test_materials_apply_over_robot_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        materials.apply_material(adapter, material_path="/World/Mat", target_prim_path=ROBOT)
    assert adapter.apply_material_calls == [("/World/Mat", ROBOT)], "authoring a binding while stopped is allowed"


def test_materials_apply_over_robot_allowed_in_operator_mode():
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        materials.apply_material(adapter, material_path="/World/Mat", target_prim_path=ROBOT)
    assert adapter.apply_material_calls == [("/World/Mat", ROBOT)], "operator mode must not apply the lock"


# ══ end-to-end dispatch: the real MCPExtension._execute_command path ══════════


def _make_extension(adapter):
    ext = MCPExtension()
    ext._adapter = adapter
    ext._session_mode = True
    reg = ext._registry
    lighting.register(reg, adapter)
    materials.register(reg, adapter)
    objects.register(reg, adapter)
    scene.register(reg, adapter)
    graphs.register(reg, adapter)
    ext._guard = _guards.SessionGuard(adapter, session_mode=True)
    return ext


def _dispatch(ext, cmd_type, params=None, capability="session"):
    return ext._execute_command({"type": cmd_type, "params": params or {}, "capability": capability})


def test_dispatch_lighting_create_over_robot_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "lighting.create", {"prim_path": ROBOT, "position": [0, 0, 9]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.create_light_calls == [], "the light-over-robot write must never touch the sink"


def test_dispatch_lighting_create_over_robot_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "lighting.create", {"prim_path": ROBOT, "position": [0, 0, 9]}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.create_light_calls == [("DistantLight", ROBOT)], "operator dispatch must reach the sink"


def test_dispatch_materials_create_over_robot_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "materials.create", {"material_type": "physics", "prim_path": ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.create_physics_material_calls == [], "material-over-robot write must never touch the sink"


def test_dispatch_materials_create_fresh_path_allowed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "materials.create", {"material_type": "pbr", "prim_path": "/World/Mat"})
    assert r["status"] == "success", r
    assert adapter.create_pbr_calls == ["/World/Mat"]


def test_dispatch_objects_delete_robot_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.delete", {"prim_path": ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.delete_calls == [], "delete-of-robot must never touch the sink"


def test_dispatch_objects_delete_robot_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.delete", {"prim_path": ROBOT}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.delete_calls == [ROBOT], "operator dispatch must reach the delete sink"


def test_dispatch_objects_delete_non_robot_allowed_for_session():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.delete", {"prim_path": "/World/Cube"})
    assert r["status"] == "success", r
    assert adapter.delete_calls == ["/World/Cube"]


def test_dispatch_scene_clear_over_robot_ancestor_rejected_and_not_executed():
    # /World is the robot's parent (robot at /World/G1); clearing it via the real
    # dispatch must be refused and delete nothing.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", children=("/World", "/Render"))
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.clear", {})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.delete_calls == [], "scene.clear must not delete anything when a robot is present"


def test_dispatch_objects_transform_robot_ancestor_rejected_and_not_executed():
    # objects.transform is gated at the DISPATCH level (POSE_WRITE_COMMANDS ->
    # evaluate_command/SessionGuard), not by a sink-body guard_pose_write. Prove the
    # single predicate fix propagates there too: transforming /World (robot parent)
    # is refused and set_prim_transform never runs.
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.transform", {"prim_path": "/World", "position": [9, 9, 9]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.set_prim_transform_calls == [], "teleporting the robot via its parent must not touch the sink"


def test_dispatch_objects_transform_pseudo_root_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.transform", {"prim_path": "/", "position": [9, 9, 9]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.set_prim_transform_calls == []


def test_dispatch_objects_transform_fresh_sibling_allowed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.transform", {"prim_path": "/World/Cube", "position": [1, 2, 3]})
    assert r["status"] == "success", r
    assert adapter.set_prim_transform_calls == [
        ("/World/Cube", {"position": [1, 2, 3], "rotation": None, "scale": None})
    ], "a fresh sibling of the robot must reach the transform sink"


def test_dispatch_objects_transform_robot_ancestor_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.transform", {"prim_path": "/World", "position": [9, 9, 9]}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.set_prim_transform_calls == [
        ("/World", {"position": [9, 9, 9], "rotation": None, "scale": None})
    ], "operator dispatch must reach the sink"


def test_dispatch_objects_transform_robot_ancestor_allowed_while_stopped():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "objects.transform", {"prim_path": "/World", "position": [9, 9, 9]})
    assert r["status"] == "success", r
    assert adapter.set_prim_transform_calls == [
        ("/World", {"position": [9, 9, 9], "rotation": None, "scale": None})
    ], "transforming an ancestor while stopped (authoring) must be allowed"


def test_dispatch_materials_apply_over_robot_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "materials.apply", {"material_path": "/World/Mat", "target_prim_path": ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.apply_material_calls == [], "material-bind-over-robot must never touch the sink"


def test_dispatch_materials_apply_over_robot_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(
            ext, "materials.apply", {"material_path": "/World/Mat", "target_prim_path": ROBOT}, capability="operator"
        )
    assert r["status"] == "success", r
    assert adapter.apply_material_calls == [("/World/Mat", ROBOT)], "operator dispatch must reach the bind sink"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
