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

"""Validate that the adapter and handler structure is correct."""

import ast
import os

EXTENSION_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
)


def _parse_file(path):
    with open(path) as f:
        return ast.parse(f.read())


def test_adapter_base_has_all_abstract_methods():
    """Verify the base adapter defines all required abstract methods."""
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                    methods.add(node.name)
                elif (
                    isinstance(decorator, ast.Attribute)
                    and decorator.attr == "abstractmethod"
                ):
                    methods.add(node.name)
    expected = {
        "get_stage",
        "get_assets_root_path",
        "discover_environments",
        "load_environment",
        "create_prim",
        "delete_prim",
        "add_reference_to_stage",
        "set_prim_transform",
        "get_prim_transform",
        "list_prims",
        "get_prim_info",
        "create_xform_prim",
        "create_articulation",
        "discover_robots",
        "get_robot_joint_info",
        "set_joint_positions",
        "get_joint_positions",
        "create_world",
        "create_simulation_context",
        "create_physics_scene",
        "create_camera",
        "set_active_camera",
        "capture_camera_image",
        "create_lidar",
        "get_lidar_point_cloud",
        "create_pbr_material",
        "create_physics_material",
        "apply_material",
        "create_light",
        "modify_light",
        "clone_prim",
        "import_urdf",
        "play",
        "pause",
        "stop",
        "step",
        "execute_script",
        # Observability methods (issue #1)
        "get_simulation_state",
        "get_physics_state",
        "get_joint_config",
        "reload_script",
        # Dimensional data (issue #2)
        "get_prim_actual_size",
    }
    assert methods == expected, (
        f"Missing: {expected - methods}, Extra: {methods - expected}"
    )


def test_v5_adapter_implements_all_methods():
    """Verify v5 adapter implements every abstract method from base."""
    base_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    v5_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "v5.py"))

    base_methods = set()
    for node in ast.walk(base_tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
                ) or (
                    isinstance(decorator, ast.Attribute)
                    and decorator.attr == "abstractmethod"
                ):
                    base_methods.add(node.name)

    v5_methods = set()
    for node in ast.walk(v5_tree):
        if isinstance(node, ast.FunctionDef):
            v5_methods.add(node.name)

    missing = base_methods - v5_methods
    assert not missing, f"v5 adapter missing implementations: {missing}"


def test_v5_adapter_owns_initialized_camera_lifecycle():
    """Camera capture must reuse an initialized sensor without re-entering Kit."""
    path = os.path.join(EXTENSION_ROOT, "adapters", "v5.py")
    with open(path) as handle:
        source = handle.read()

    assert "self._camera_cache: Dict[str, Any] = {}" in source
    assert "camera.initialize()" in source
    assert "self._camera_cache[prim_path] = camera" in source
    assert "self._release_cached_cameras(prim_path)" in source
    assert "camera.destroy()" in source
    assert source.count("carb.log_warn(") >= 2
    capture_source = source.split("    def capture_camera_image", 1)[1].split(
        "    def create_lidar", 1
    )[0]
    assert "omni.kit.app.get_app().update()" not in capture_source
    assert "produced no rendered frame" in source
    assert "self._release_cached_cameras()" in source


def test_v5_camera_is_configured_before_runtime_initialization():
    path = os.path.join(EXTENSION_ROOT, "adapters", "v5.py")
    with open(path) as handle:
        source = handle.read()

    create_source = source.split("    def create_camera", 1)[1].split(
        "    def capture_camera_image", 1
    )[0]
    assert create_source.index("UsdGeom.Camera.Define") < create_source.index(
        "camera.initialize()"
    )
    assert create_source.index("GetClippingRangeAttr") < create_source.index(
        "camera.initialize()"
    )
    delete_source = source.split("    def delete_prim", 1)[1].split(
        "    def discover_environments", 1
    )[0]
    assert delete_source.index("self._release_cached_cameras") < delete_source.index(
        'omni.kit.commands.execute("DeletePrims"'
    )


def test_camera_capture_creates_artifact_parent_directory():
    """Artifact-first capture must work in a fresh workspace."""
    path = os.path.join(EXTENSION_ROOT, "handlers", "sensors.py")
    with open(path) as handle:
        source = handle.read()

    capture_source = source.split("def capture_image", 1)[1].split(
        "def create_lidar", 1
    )[0]
    assert (
        "Path(output_path).parent.mkdir(parents=True, exist_ok=True)" in capture_source
    )


def test_all_handler_modules_have_register():
    """Verify every handler module exposes a register(registry, adapter) function."""
    handlers_dir = os.path.join(EXTENSION_ROOT, "handlers")
    handler_files = [
        "scene.py",
        "objects.py",
        "lighting.py",
        "robots.py",
        "sensors.py",
        "materials.py",
        "assets.py",
        "gaussian_splats.py",
        "simulation.py",
    ]
    for filename in handler_files:
        filepath = os.path.join(handlers_dir, filename)
        assert os.path.exists(filepath), f"Handler file missing: {filename}"
        tree = _parse_file(filepath)
        func_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert "register" in func_names, f"{filename} missing register() function"
