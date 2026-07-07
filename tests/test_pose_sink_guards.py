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

"""Behavioural proof that EVERY remaining pose/location write sink is guarded.

objects.create / objects.clone are covered by tests/test_rce_closure.py. This
file closes the rest of the teleport surface — the handlers that reach a
set_world_pose / set_prim_transform / loader-location sink AT (or under) a robot
articulation-root:

  * ``robots.create`` (add_reference_to_stage + create_xform_prim.set_world_pose)
  * ``assets.import_urdf`` (adapter.set_prim_transform)
  * ``assets.load_usd`` / ``assets.search_usd`` (USDLoader.load_usd_from_url location)
  * ``sensors.create_camera`` / ``sensors.create_lidar`` (adapter.set_prim_transform)

These tests do NOT grep source. They drive the real handler bodies (and the real
``MCPExtension._execute_command`` dispatch that tests/test_rce_closure.py uses)
against a fake adapter that RECORDS every sink call, and assert that a write over
a registered robot root while the timeline is live is REJECTED *and* the sink is
never reached, while a fresh non-robot path is allowed and the sink DOES run.

Isaac Sim is absent offline, and the real ``isaac_sim_mcp_extension.usd`` module
uses ``X | None`` annotations that only parse on Python >= 3.10, so ``numpy`` /
``carb`` / ``omni.*`` are stubbed into ``sys.modules`` and the ``usd`` / ``gen3d``
leaf modules are replaced with recording fakes *before* the real handler package
is imported. The stubs only satisfy imports; every assertion is against the real
guard + handler dispatch logic.
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

    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)


_install_stubs()

# The real usd/gen3d leaf modules import carb/omni/requests/pxr/isaacsim and use
# PEP-604 unions that do not parse < 3.10. Replace them with recording fakes so
# the handler control-flow (guard + sink call) stays real but the sink is
# observable — this is exactly the mutation we must prove never runs on reject.


class _RecordingUSDLoader:
    """Stands in for isaac_sim_mcp_extension.usd.USDLoader; records the location sink."""

    instances = 0
    load_calls = []

    def __init__(self) -> None:
        _RecordingUSDLoader.instances += 1

    def load_usd_from_url(self, **kwargs):
        _RecordingUSDLoader.load_calls.append(kwargs)
        return {"loaded": True}


class _RecordingUSDSearch:
    """Stands in for isaac_sim_mcp_extension.usd.USDSearch3d; records the search sink."""

    instances = 0
    search_calls = []

    def __init__(self) -> None:
        _RecordingUSDSearch.instances += 1

    def search(self, *args, **kwargs):
        _RecordingUSDSearch.search_calls.append((args, kwargs))
        return "omniverse://fake/found.usd"


def _reset_usd_sinks() -> None:
    _RecordingUSDLoader.instances = 0
    _RecordingUSDLoader.load_calls = []
    _RecordingUSDSearch.instances = 0
    _RecordingUSDSearch.search_calls = []


def _install_leaf_fakes() -> None:
    import isaac_sim_mcp_extension as pkg

    usd_stub = types.ModuleType("isaac_sim_mcp_extension.usd")
    usd_stub.USDLoader = _RecordingUSDLoader
    usd_stub.USDSearch3d = _RecordingUSDSearch
    sys.modules["isaac_sim_mcp_extension.usd"] = usd_stub
    pkg.usd = usd_stub

    gen3d_stub = types.ModuleType("isaac_sim_mcp_extension.gen3d")

    class _Beaver3d:  # only referenced by assets.generate_3d, which is not tested here
        pass

    gen3d_stub.Beaver3d = _Beaver3d
    sys.modules["isaac_sim_mcp_extension.gen3d"] = gen3d_stub
    pkg.gen3d = gen3d_stub


_install_leaf_fakes()

from isaac_sim_mcp_extension.extension import MCPExtension  # noqa: E402
from isaac_sim_mcp_extension.handlers import _guards, assets, robots, sensors  # noqa: E402

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
    """Records every pose/location sink so tests can assert the write never ran."""

    def __init__(self, timeline="playing", gravity=9.81, dt=1.0 / 60.0, roots=(ROBOT,)):
        self._timeline = timeline
        self._gravity = gravity
        self._dt = dt
        self._roots = roots
        self.add_reference_calls = []
        self.set_world_pose_calls = []
        self.import_urdf_calls = []
        self.set_prim_transform_calls = []
        self.create_camera_calls = []
        self.create_lidar_calls = []

    # world-state hooks used by the guard
    def get_simulation_state(self):
        return {"timeline_state": self._timeline, "physics_dt": self._dt}

    def list_articulation_roots(self):
        return self._roots

    def get_gravity_magnitude(self):
        return self._gravity

    def get_prim_transform(self, prim_path):
        return {"position": [0.0, 0.0, 1.0]}

    # robots.create sinks
    def discover_robots(self):
        return {
            "g1": {
                "asset_path": "/g1.usd",
                "description": "Unitree G1",
                "manufacturer": "Unitree",
            }
        }

    def get_assets_root_path(self):
        return ""

    def add_reference_to_stage(self, asset_path, prim_path):
        self.add_reference_calls.append((asset_path, prim_path))

    def create_xform_prim(self, prim_path):
        outer = self

        class _Xform:
            def set_world_pose(self, position=None):
                outer.set_world_pose_calls.append(prim_path)

        return _Xform()

    def ensure_contact_reporting(self, prim_path):
        return None

    def get_robot_joint_info(self, prim_path):
        raise Exception("joint info unavailable offline")

    def get_joint_config(self, prim_path):
        raise Exception("joint config unavailable offline")

    # assets sinks
    def import_urdf(self, urdf_path, prim_path=None):
        self.import_urdf_calls.append((urdf_path, prim_path))
        return object()

    def set_prim_transform(self, prim_path, **kwargs):
        self.set_prim_transform_calls.append((prim_path, kwargs))

    # sensors sinks
    def create_camera(self, prim_path, resolution=None):
        self.create_camera_calls.append(prim_path)
        return object()

    def create_lidar(self, prim_path, config=None):
        self.create_lidar_calls.append(prim_path)

    def get_stage(self):  # pragma: no cover - no test hits an auto-name path
        raise AssertionError("get_stage should not be reached in these tests")


def _is_reject(d):
    return d is not None and d.get("status") == "error" and d.get("rejected_by") == REJECTED


def _fresh_robot_cache():
    # robots._get_robot_library caches discovery in a module global; clear it so a
    # per-test FakeAdapter's discover_robots is what populates the library.
    robots._discovered_robots = None


# ══ robots.create — spawn-then-pose over a robot root is blocked ══════════════


def test_robots_create_over_robot_root_blocked_while_playing():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = robots.create(adapter, robot_type="g1", prim_path=ROBOT, position=[0.0, 0.0, 5.0])
    assert _is_reject(d), d
    assert adapter.add_reference_calls == [], "add_reference_to_stage must not run on reject"
    assert adapter.set_world_pose_calls == [], "set_world_pose sink must not run on reject"


def test_robots_create_over_robot_descendant_blocked_while_paused():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="paused")
        d = robots.create(adapter, robot_type="g1", prim_path=ROBOT + "/pelvis", position=[0, 0, 5])
    assert _is_reject(d), d
    assert adapter.set_world_pose_calls == []


def test_robots_create_fresh_path_allowed_while_playing():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = robots.create(adapter, robot_type="g1", prim_path="/World/NewBot", position=[1, 2, 3])
    assert d.get("status") == "success", d
    assert adapter.add_reference_calls == [("/g1.usd", "/World/NewBot")]
    assert adapter.set_world_pose_calls == ["/World/NewBot"], "fresh path must reach the pose sink"


def test_robots_create_over_robot_allowed_while_stopped():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="stopped")
        robots.create(adapter, robot_type="g1", prim_path=ROBOT, position=[0, 0, 5])
    assert adapter.set_world_pose_calls == [ROBOT], "authoring while stopped must be allowed"


def test_robots_create_fails_closed_when_discovery_fails():
    # roots=None => cannot prove the target is not a robot => reject (fail-closed).
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing", roots=None)
        d = robots.create(adapter, robot_type="g1", prim_path="/World/NewBot", position=[1, 2, 3])
    assert _is_reject(d), d
    assert adapter.add_reference_calls == [] and adapter.set_world_pose_calls == []


def test_robots_create_over_robot_allowed_in_operator_mode():
    _fresh_robot_cache()
    with _session_mode("0"):
        adapter = FakeAdapter(timeline="playing")
        robots.create(adapter, robot_type="g1", prim_path=ROBOT, position=[0, 0, 5])
    assert adapter.set_world_pose_calls == [ROBOT], "operator mode must not apply the session lock"


# ══ assets.import_urdf — import-then-transform over a robot root is blocked ════


def test_import_urdf_over_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = assets.import_urdf(adapter, urdf_path="/x.urdf", prim_path=ROBOT, position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.import_urdf_calls == [], "import_urdf sink must not run on reject"
    assert adapter.set_prim_transform_calls == [], "set_prim_transform sink must not run on reject"


def test_import_urdf_fresh_path_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = assets.import_urdf(adapter, urdf_path="/x.urdf", prim_path="/World/fresh", position=[0, 0, 9])
    assert d.get("status") == "success", d
    assert adapter.import_urdf_calls == [("/x.urdf", "/World/fresh")]
    assert adapter.set_prim_transform_calls == [("/World/fresh", {"position": [0, 0, 9]})]


# ══ assets.load_usd — loader location write over a robot root is blocked ══════


def test_load_usd_over_robot_root_blocked_while_playing():
    _reset_usd_sinks()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = assets.load_usd(adapter, usd_url="omniverse://a.usd", prim_path=ROBOT, position=[0, 0, 9])
    assert _is_reject(d), d
    assert _RecordingUSDLoader.instances == 0, "USDLoader must not even be constructed on reject"
    assert _RecordingUSDLoader.load_calls == [], "load_usd_from_url location sink must not run"


def test_load_usd_fresh_path_allowed_while_playing():
    _reset_usd_sinks()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = assets.load_usd(adapter, usd_url="omniverse://a.usd", prim_path="/World/fresh", position=[0, 0, 9])
    assert d.get("status") == "success", d
    assert _RecordingUSDLoader.instances == 1
    assert len(_RecordingUSDLoader.load_calls) == 1
    assert _RecordingUSDLoader.load_calls[0]["target_path"] == "/World/fresh"


# ══ assets.search_usd — search-then-load over a robot root is blocked ═════════


def test_search_usd_over_robot_root_blocked_while_playing():
    _reset_usd_sinks()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = assets.search_usd(adapter, text_prompt="a chair", target_path=ROBOT, position=[0, 0, 9])
    assert _is_reject(d), d
    assert _RecordingUSDSearch.instances == 0, "USDSearch3d must not run on reject"
    assert _RecordingUSDLoader.load_calls == [], "loader location sink must not run on reject"


# ══ sensors — camera/lidar transform onto a robot subtree is blocked ══════════


def test_create_camera_under_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = sensors.create_camera(adapter, prim_path=ROBOT + "/head", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_camera_calls == [], "create_camera sink must not run on reject"
    assert adapter.set_prim_transform_calls == []


def test_create_camera_fresh_path_allowed_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = sensors.create_camera(adapter, prim_path="/World/Camera", position=[0, 0, 9])
    assert d.get("status") == "success", d
    assert adapter.create_camera_calls == ["/World/Camera"]
    assert adapter.set_prim_transform_calls == [("/World/Camera", {"position": [0, 0, 9], "rotation": None})]


def test_create_lidar_under_robot_root_blocked_while_playing():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        d = sensors.create_lidar(adapter, prim_path=ROBOT + "/lidar", position=[0, 0, 9])
    assert _is_reject(d), d
    assert adapter.create_lidar_calls == [], "create_lidar sink must not run on reject"
    assert adapter.set_prim_transform_calls == []


# ══ end-to-end dispatch: the real MCPExtension._execute_command path ══════════


def _make_extension(adapter):
    ext = MCPExtension()
    ext._adapter = adapter
    ext._session_mode = True
    reg = ext._registry
    robots.register(reg, adapter)
    assets.register(reg, adapter)
    sensors.register(reg, adapter)
    ext._guard = _guards.SessionGuard(adapter, session_mode=True)
    return ext


def _dispatch(ext, cmd_type, params=None):
    return ext._execute_command({"type": cmd_type, "params": params or {}})


def test_dispatch_robots_create_teleport_rejected_and_not_executed():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "robots.create", {"robot_type": "g1", "prim_path": ROBOT, "position": [0, 0, 5]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.set_world_pose_calls == [], "the teleport-via-create must never touch the sink"
    assert adapter.add_reference_calls == []


def test_dispatch_import_urdf_over_robot_rejected_and_not_executed():
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "assets.import_urdf", {"urdf_path": "/x.urdf", "prim_path": ROBOT, "position": [0, 0, 9]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert adapter.import_urdf_calls == [] and adapter.set_prim_transform_calls == []


def test_dispatch_load_usd_over_robot_rejected_and_not_executed():
    _reset_usd_sinks()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "assets.load_usd", {"usd_url": "omniverse://a.usd", "prim_path": ROBOT, "position": [0, 0, 9]})
    assert r["status"] == "error" and REJECTED in r["message"], r
    assert _RecordingUSDLoader.instances == 0 and _RecordingUSDLoader.load_calls == []


def test_dispatch_robots_create_fresh_path_allowed():
    _fresh_robot_cache()
    with _session_mode("1"):
        adapter = FakeAdapter(timeline="playing")
        ext = _make_extension(adapter)
        r = _dispatch(ext, "robots.create", {"robot_type": "g1", "prim_path": "/World/NewBot", "position": [1, 2, 3]})
    assert r["status"] == "success", r
    assert adapter.set_world_pose_calls == ["/World/NewBot"], "fresh non-robot path must reach the sink"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
