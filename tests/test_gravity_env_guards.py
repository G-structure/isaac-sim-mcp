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

"""Behavioural proof that the last two SESSION cheating edges are closed:

  1. ``scene.load_environment`` is an ``add_reference_to_stage`` sink with a
     caller-controlled ``prim_path``. Dropping an environment reference onto (or
     over) a robot articulation-root while the timeline is live is a disguised
     teleport/overwrite; a SESSION call over a robot root must be REJECTED and the
     ``adapter.load_environment`` sink must NEVER run.
  2. ``simulation.set_physics`` and ``scene.create_physics`` both accept a gravity
     vector. A SESSION agent could set gravity=[0,0,0] (or a tiny / sideways /
     upward vector) to trivially "balance" a robot and then report honest-looking
     telemetry. A SESSION gravity write must be earth-like in magnitude AND point
     predominantly down along the stage up-axis, else it is REJECTED and the
     ``apply_physics_params`` / ``create_physics_scene`` sink must NEVER run.

The OPERATOR capability (trusted bootstrap) retains full control: it bypasses both
guards, so every gravity vector and every environment placement reaches the sink.

These tests do NOT grep source — they drive the real handler bodies (and the real
``MCPExtension._execute_command`` dispatch) against a fake adapter that RECORDS
every sink call, and assert the sink is / is not reached. Isaac Sim is absent
offline, so ``numpy`` / ``carb`` / ``omni.*`` / ``pxr`` are stubbed into
``sys.modules`` *before* the real handler package is imported; the stubs only
satisfy imports and every assertion is against the real guard + handler logic.
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
    def IsValid(self):
        return False

    def HasAPI(self, _api):
        return True


class _FakeStage:
    def GetPrimAtPath(self, _path):
        return _FakePrim()


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

    # pxr: scene.create_physics does `from pxr import UsdPhysics` after the sink; a
    # minimal stub lets the allowed path return success cleanly offline. The gravity
    # guard reads the up-axis from the adapter first, so UsdGeom is only a fallback.
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
from isaac_sim_mcp_extension.handlers import _guards, scene, simulation  # noqa: E402

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
    """Records the gravity + environment sinks so tests can assert they never ran."""

    def __init__(self, timeline="playing", up_axis="Z", roots=(ROBOT,)):
        self._timeline = timeline
        self._up_axis = up_axis
        self._roots = roots
        self.apply_physics_calls = []
        self.create_physics_scene_calls = []
        self.create_prim_calls = []
        self.load_environment_calls = []

    # world-state hooks used by the guard
    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": 1.0 / 60.0}

    def list_articulation_roots(self):
        return self._roots

    def get_gravity_magnitude(self):
        return 9.81

    def get_prim_transform(self, prim_path):
        return {"position": [0.0, 0.0, 1.0]}

    def get_stage_up_axis(self):
        return self._up_axis

    def get_stage(self):
        return _FakeStage()

    # gravity sinks — must NOT run on a rejected session write
    def apply_physics_params(self, gravity=None, **kwargs):
        self.apply_physics_calls.append(gravity)
        return {"gravity_effective": gravity}

    def create_physics_scene(self, gravity=None, scene_name="PhysicsScene"):
        self.create_physics_scene_calls.append(gravity)
        return f"/World/{scene_name}"

    def create_prim(self, prim_path, prim_type="Xform", **kwargs):
        self.create_prim_calls.append(prim_path)
        return object()

    # environment sinks — must NOT run on a rejected session load
    def discover_environments(self):
        return {
            "warehouse": {
                "asset_path": "/Environments/warehouse.usd",
                "description": "Warehouse",
            }
        }

    def get_assets_root_path(self):
        return ""

    def load_environment(self, env_path, prim_path="/Environment"):
        self.load_environment_calls.append((env_path, prim_path))


def _is_reject(d):
    return d is not None and d.get("status") == "error" and d.get("rejected_by") == REJECTED


def _fresh_env_cache():
    # scene._get_env_library caches discovery in a module global; clear it so a
    # per-test FakeAdapter's discover_environments repopulates the library.
    scene._discovered_envs = None


# Non-physical gravity vectors a session might use to fake a balance, on a Z-up stage.
_ZERO = [0.0, 0.0, 0.0]
_TINY = [0.0, 0.0, -0.5]
_SIDEWAYS = [9.81, 0.0, 0.0]
_UPWARD = [0.0, 0.0, 9.81]
_DIAGONAL = [7.0, 0.0, -7.0]  # magnitude ~9.9 (in band) but only ~45° from horizontal
_TOO_STRONG = [0.0, 0.0, -20.0]
_TOO_WEAK = [0.0, 0.0, -8.0]
_EARTH = [0.0, 0.0, -9.81]

_NON_PHYSICAL = {
    "zero": _ZERO,
    "tiny": _TINY,
    "sideways": _SIDEWAYS,
    "upward": _UPWARD,
    "diagonal": _DIAGONAL,
    "too_strong": _TOO_STRONG,
    "too_weak": _TOO_WEAK,
}


# ══ pure gravity decision — the envelope the sink guards enforce ═══════════════


def test_gravity_rejection_pure_rejects_non_physical_z_up():
    for name, g in _NON_PHYSICAL.items():
        assert _is_reject(_guards._gravity_rejection(g, "Z")), name


def test_gravity_rejection_pure_allows_earth_and_none_z_up():
    assert _guards._gravity_rejection(_EARTH, "Z") is None
    assert _guards._gravity_rejection([0.3, -0.2, -9.8], "Z") is None  # slight tilt ok
    assert _guards._gravity_rejection(None, "Z") is None


def test_gravity_rejection_pure_honours_y_up_axis():
    # On a Y-up stage, straight-down is -Y; a -Z earth vector is now "sideways".
    assert _guards._gravity_rejection([0.0, -9.81, 0.0], "Y") is None
    assert _is_reject(_guards._gravity_rejection(_EARTH, "Y"))


def test_gravity_rejection_pure_fails_closed_on_malformed():
    assert _is_reject(_guards._gravity_rejection([0.0, -9.81], "Z"))  # 2-vector
    assert _is_reject(_guards._gravity_rejection(["x", "y", "z"], "Z"))  # unparseable


# ══ simulation.set_physics — direct handler, SESSION default capability ═══════


def test_set_physics_session_rejects_non_physical_gravity_sink_never_runs():
    for name, g in _NON_PHYSICAL.items():
        with _session_mode("1"):
            adapter = FakeAdapter()
            d = simulation.set_physics(adapter, gravity=g)
        assert _is_reject(d), (name, d)
        assert adapter.apply_physics_calls == [], f"{name}: apply_physics_params must not run"


def test_set_physics_session_allows_earth_gravity():
    with _session_mode("1"):
        adapter = FakeAdapter()
        d = simulation.set_physics(adapter, gravity=_EARTH)
    assert d.get("status") == "success", d
    assert adapter.apply_physics_calls == [_EARTH], "earth gravity must reach the sink"


def test_set_physics_session_allows_gravity_none():
    with _session_mode("1"):
        adapter = FakeAdapter()
        d = simulation.set_physics(adapter, time_step=1.0 / 120.0)
    assert d.get("status") == "success", d
    assert adapter.apply_physics_calls == [None], "gravity=None (unchanged) must be allowed"


# ══ scene.create_physics — direct handler, SESSION default capability ═════════


def test_create_physics_session_rejects_non_physical_gravity_sink_never_runs():
    for name, g in _NON_PHYSICAL.items():
        _fresh_env_cache()
        with _session_mode("1"):
            adapter = FakeAdapter()
            d = scene.create_physics(adapter, gravity=g)
        assert _is_reject(d), (name, d)
        assert adapter.create_physics_scene_calls == [], f"{name}: create_physics_scene must not run"
        assert adapter.create_prim_calls == [], f"{name}: ground-plane create must not run"


def test_create_physics_session_allows_earth_gravity():
    with _session_mode("1"):
        adapter = FakeAdapter()
        d = scene.create_physics(adapter, gravity=_EARTH)
    assert d.get("status") == "success", d
    assert adapter.create_physics_scene_calls == [_EARTH], "earth gravity must reach the sink"


def test_create_physics_session_allows_gravity_none():
    with _session_mode("1"):
        adapter = FakeAdapter()
        d = scene.create_physics(adapter, gravity=None)
    assert d.get("status") == "success", d
    assert adapter.create_physics_scene_calls == [None]


# ══ scene.load_environment — reference-drop over a robot root is blocked ══════


def test_load_environment_over_robot_root_blocked_while_playing():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = scene.load_environment(adapter, environment="warehouse", prim_path=ROBOT)
    assert _is_reject(d), d
    assert adapter.load_environment_calls == [], "load_environment sink must not run on reject"


def test_load_environment_over_robot_descendant_blocked_while_paused():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="paused")
        d = scene.load_environment(adapter, environment="warehouse", prim_path=ROBOT + "/pelvis")
    assert _is_reject(d), d
    assert adapter.load_environment_calls == []


def test_load_environment_fresh_path_allowed_while_playing():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = scene.load_environment(adapter, environment="warehouse", prim_path="/Environment")
    assert d.get("status") == "success", d
    assert adapter.load_environment_calls == [("/Environments/warehouse.usd", "/Environment")]


def test_load_environment_over_robot_allowed_while_stopped():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        scene.load_environment(adapter, environment="warehouse", prim_path=ROBOT)
    assert adapter.load_environment_calls == [("/Environments/warehouse.usd", ROBOT)], (
        "authoring while stopped must be allowed"
    )


def test_load_environment_fails_closed_when_discovery_fails():
    # roots=None => cannot prove the target is not a robot => reject (fail-closed).
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = scene.load_environment(adapter, environment="warehouse", prim_path="/Environment")
    assert _is_reject(d), d
    assert adapter.load_environment_calls == []


# ══ end-to-end dispatch: SESSION rejected, OPERATOR retains full control ══════


def _make_extension(adapter):
    ext = MCPExtension()
    ext._adapter = adapter
    ext._session_mode = True
    reg = ext._registry
    scene.register(reg, adapter)
    simulation.register(reg, adapter)
    ext._guard = _guards.SessionGuard(adapter, session_mode=True)
    return ext


def _dispatch(ext, cmd_type, params=None, capability="session"):
    # capability mirrors the tag socket_server sets after authenticating the token.
    return ext._execute_command(
        {"type": cmd_type, "params": params or {}, "capability": capability}
    )


def test_dispatch_set_physics_zero_gravity_rejected_for_session():
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "simulation.set_physics", {"gravity": _ZERO})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.apply_physics_calls == [], "the fake-balance write must never touch the sink"


def test_dispatch_set_physics_non_physical_all_rejected_for_session():
    for name, g in _NON_PHYSICAL.items():
        with _session_mode("1"):
            adapter = FakeAdapter()
            ext = _make_extension(adapter)
            r = _dispatch(ext, "simulation.set_physics", {"gravity": g})
        assert r["status"] == "error" and REJECTED in r["message"], (name, r)
        assert adapter.apply_physics_calls == [], name


def test_dispatch_set_physics_earth_gravity_allowed_for_session():
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "simulation.set_physics", {"gravity": _EARTH})
    assert r["status"] == "success", r
    assert adapter.apply_physics_calls == [_EARTH]


def test_dispatch_set_physics_zero_gravity_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "simulation.set_physics", {"gravity": _ZERO}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.apply_physics_calls == [_ZERO], "operator must reach the sink (guard bypassed)"


def test_dispatch_create_physics_zero_gravity_rejected_for_session():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.create_physics", {"gravity": _ZERO})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.create_physics_scene_calls == []


def test_dispatch_create_physics_sideways_and_upward_rejected_for_session():
    for name, g in (("sideways", _SIDEWAYS), ("upward", _UPWARD), ("tiny", _TINY)):
        with _session_mode("1"):
            adapter = FakeAdapter()
            ext = _make_extension(adapter)
            r = _dispatch(ext, "scene.create_physics", {"gravity": g})
        assert r["status"] == "error" and REJECTED in r["message"], (name, r)
        assert adapter.create_physics_scene_calls == [], name


def test_dispatch_create_physics_earth_gravity_allowed_for_session():
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.create_physics", {"gravity": _EARTH})
    assert r["status"] == "success", r
    assert adapter.create_physics_scene_calls == [_EARTH]


def test_dispatch_create_physics_zero_gravity_allowed_for_operator():
    with _session_mode("1"):
        adapter = FakeAdapter()
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.create_physics", {"gravity": _ZERO}, capability="operator")
    assert r["status"] == "success", r
    assert adapter.create_physics_scene_calls == [_ZERO], "operator must reach the sink (bypassed)"


def test_dispatch_load_environment_over_robot_rejected_for_session():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.load_environment", {"environment": "warehouse", "prim_path": ROBOT})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.load_environment_calls == [], "the reference-drop teleport must never touch the sink"


def test_dispatch_load_environment_over_robot_allowed_for_operator():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(
            ext, "scene.load_environment", {"environment": "warehouse", "prim_path": ROBOT},
            capability="operator",
        )
    assert r["status"] == "success", r
    assert adapter.load_environment_calls == [("/Environments/warehouse.usd", ROBOT)], (
        "operator must reach the load sink (guard bypassed)"
    )


def test_dispatch_load_environment_fresh_path_allowed_for_session():
    _fresh_env_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "scene.load_environment", {"environment": "warehouse", "prim_path": "/Environment"})
    assert r["status"] == "success", r
    assert adapter.load_environment_calls == [("/Environments/warehouse.usd", "/Environment")]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
