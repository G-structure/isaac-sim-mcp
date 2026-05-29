"""Platform compatibility checks for the Cybernetic Physics session image."""

import ast
from pathlib import Path


EXTENSION_ROOT = (
    Path(__file__).resolve().parents[1] / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension"
)
EXTENSION_PATH = EXTENSION_ROOT / "extension.py"
HANDLERS_ROOT = EXTENSION_ROOT / "handlers"


def _extension_source() -> str:
    return EXTENSION_PATH.read_text()


def _registered_handler_commands() -> set[str]:
    commands: set[str] = set()
    for path in HANDLERS_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "registry":
                continue
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                commands.add(key.value)
    return commands


def test_legacy_gateway_commands_are_registered():
    source = _extension_source()
    for command in [
        "get_scene_info",
        "create_physics_scene",
        "create_robot",
        "transform",
        "execute_script",
        "generate_3d_from_text_or_image",
        "search_3d_usd_by_text",
        "omini_kit_command",
    ]:
        assert command in source


def test_legacy_alias_targets_map_to_registered_handlers():
    source = _extension_source()
    commands = _registered_handler_commands()
    expected_alias_targets = {
        "scene.get_info",
        "scene.create_physics",
        "robots.create",
        "objects.transform",
        "simulation.execute_script",
        "assets.generate_3d",
        "assets.search_usd",
    }
    assert expected_alias_targets <= commands
    assert "_legacy_omni_kit_command" in source
    assert "_legacy_create_physics_scene" in source
    assert "_legacy_create_robot" in source
    assert '"franka": "frankapanda"' in source
    assert '"carter": "carter_v1"' in source


def test_extension_reads_platform_socket_settings():
    source = _extension_source()
    assert "server.socket" in source
    assert "server.port" in source
    assert "8766" in source


def test_platform_autosave_hooks_are_present():
    source = _extension_source()
    assert "/data/workspace" in source
    assert "AUTOSAVE_INTERVAL" in source
    assert "scene.usd" in source

    tree = ast.parse(source)
    methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_start_autosave" in methods
    assert "_on_autosave_tick" in methods
    assert "_stop_autosave" in methods
