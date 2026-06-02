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


def test_extension_does_not_autosave_scene_usd():
    """Regression guard for the "two-writers-open-root" bug.

    scene.usd is the open ROOT layer and is persisted EXCLUSIVELY by
    warm_slot_agent.save_open_stage() (an in-place ctx.save_stage() of the open
    root, throttled and coordinated with the S3 sync daemon). The extension must
    NOT run its own autosave: a second, flattening, fire-and-forget writer
    (ctx.export_as_stage_async) on the same open root layer raced the agent and the
    sync daemon, dropped sublayers/references on every flatten, stacked overlapping
    exports, and swallowed coroutine-body errors. See issue "two-writers-open-root".
    """
    source = _extension_source()
    tree = ast.parse(source)
    methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    # The extension must not flatten/export the stage to disk on a timer.
    assert "export_as_stage_async" not in source
    assert "AUTOSAVE_INTERVAL" not in source
    assert "scene.usd" not in source

    # The old self-owned autosave hooks must stay deleted.
    assert "_start_autosave" not in methods
    assert "_on_autosave_tick" not in methods
    assert "_export_workspace_stage_async" not in methods
    assert "_stop_autosave" not in methods
