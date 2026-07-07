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

"""Isaac Sim MCP Extension — slim entry point.

Routes incoming socket commands to handler modules via a registry.
"""

from __future__ import annotations

import gc
import traceback
from typing import Any, Dict

import carb
import omni.kit.app
import omni.kit.commands
import omni.ext
import omni.usd

from .adapters import get_adapter
from .handlers import register_all_handlers
from .handlers._guards import (
    OPERATOR_CAPABILITY,
    SESSION_CAPABILITY,
    SessionGuard,
    operator_token,
    reset_current_capability,
    session_mode_enabled,
    session_token,
    set_current_capability,
)
from .socket_server import SocketServer

# Commands that run arbitrary Python, hot-reload it, execute an arbitrary Kit
# command, or mutate a ScriptNode's script — RCE-equivalent surfaces. They are
# permitted for the ``operator`` capability (trusted bootstrap: warm_slot_agent,
# g1_autospawn) and refused for ``session`` (the untrusted bridge->gateway agent)
# with "forbidden in session mode". create_action_graph is NOT listed here: it
# stays available to both, but a ScriptNode/script-input payload is rejected for
# session by the confirmed-sound guard in handlers.graphs.
_OPERATOR_ONLY_COMMANDS = frozenset(
    {
        "simulation.execute_script",
        "execute_script",
        "simulation.reload_script",
        "reload_script",
        "omini_kit_command",
        "graphs.edit_action_graph",
        # G1 driver install is TRUSTED setup: it subscribes a physics-step callback
        # and spawns a DDS side thread on the articulation, so it is refused for the
        # untrusted session capability here (before the handler) and permitted only
        # for operator (g1_autospawn). The agent-facing g1 actuation
        # (robots.g1.set_joint_command / set_gains / get_lowstate / set_control_mode /
        # reset) stays session-reachable — it is honest actuation the non-cheating
        # guards gate — and is NOT listed here.
        "robots.g1.spawn",
    }
)


class MCPExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        self.ext_id = None
        self._settings = carb.settings.get_settings()
        self._registry: Dict[str, Any] = {}
        self._adapter = None
        self._server: SocketServer | None = None
        self._session_mode: bool = True
        self._guard: SessionGuard | None = None

    def on_startup(self, ext_id: str) -> None:
        print("trigger  on_startup for: ", ext_id)
        self.ext_id = ext_id
        port = (
            self._settings.get(f"/exts/{ext_id}/server.port")
            or self._settings.get(f"/exts/{ext_id}/server.socket")
            or self._settings.get("/exts/isaac.sim.mcp/server.port")
            or self._settings.get("/exts/isaac.sim.mcp/server.socket")
            or 8766
        )
        host = (
            self._settings.get(f"/exts/{ext_id}/server.host")
            or self._settings.get("/exts/isaac.sim.mcp/server.host")
            or "localhost"
        )

        self._adapter = get_adapter()
        register_all_handlers(self._registry, self._adapter)
        self._register_legacy_handlers()

        # Session mode (default ON) installs the fail-closed non-cheating guards and
        # the framed+authenticated wire protocol. The registry keeps the FULL command
        # surface: the operator-only backdoors are no longer removed process-wide, so
        # trusted bootstrap presenting the operator token keeps execute_script etc.
        # The socket authenticates each command's token into an operator/session
        # capability and _execute_command enforces the split — anything reaching the
        # loopback :8766 socket bypasses gateway scopes, so the socket + dispatch are
        # the only place these invariants can be enforced.
        self._session_mode = session_mode_enabled()
        self._guard = SessionGuard(self._adapter, session_mode=self._session_mode)
        print(
            f"Registered {len(self._registry)} command handlers "
            f"(session_mode={self._session_mode})"
        )

        self._server = SocketServer(
            host,
            port,
            self._execute_command,
            framed=self._session_mode,
            operator_token=operator_token(),
            session_token=session_token(),
        )
        self._server.start()
        # Commands run Omniverse APIs that are main-thread only. The socket server
        # accepts connections on worker threads, so it needs the Kit main asyncio
        # loop to marshal calls onto. Capture it on the main thread (here) via a
        # one-shot coroutine: run_coroutine is safe to call from the main thread,
        # and get_running_loop() inside it returns the actual Kit loop.
        self._bind_main_loop()
        # NOTE: workspace stage persistence is owned EXCLUSIVELY by warm_slot_agent
        # (save_open_stage), which saves the open root layer IN PLACE and coordinates
        # with the S3 sync daemon. The extension intentionally does NOT autosave: a
        # second, flattening, fire-and-forget writer on the same open root layer
        # raced the agent and the sync daemon, dropped sublayers/references on every
        # flatten, stacked overlapping exports, and swallowed coroutine-body errors.
        # See issue "two-writers-open-root".

    def _bind_main_loop(self) -> None:
        import asyncio

        from omni.kit.async_engine import run_coroutine

        async def _bind() -> None:
            if self._server is not None:
                self._server.set_loop(asyncio.get_running_loop())

        run_coroutine(_bind())

    def on_shutdown(self) -> None:
        print("trigger  on_shutdown for: ", self.ext_id)
        if self._server:
            self._server.stop()
        self._registry.clear()
        gc.collect()

    # ── Per-token capability enforcement (replaces the process-global pop) ─────

    def _capability_denial(
        self, cmd_type: str, params: Dict[str, Any], capability: str
    ) -> Dict[str, Any] | None:
        """Refuse the operator-only exec/script surface, and a session ScriptNode
        action-graph, when the authenticated capability is ``session``. Returns a
        rejection dict or ``None`` to allow.

        The full registry is kept, so the operator capability (trusted bootstrap)
        reaches execute_script / reload_script / omini_kit_command /
        edit_action_graph and may build ScriptNode graphs; the session capability
        (the untrusted bridge->gateway agent) is refused here BEFORE the handler,
        which is the sole place the loopback socket's invariants hold.

        TODO(probe:guard-live): confirm on a live box that a session token is refused
        with "forbidden in session mode" while the operator token reaches the handler.
        """
        if capability == OPERATOR_CAPABILITY:
            return None
        if cmd_type in _OPERATOR_ONLY_COMMANDS:
            return {
                "status": "error",
                "message": f"{cmd_type!r} runs operator-only code and is forbidden in session mode",
                "rejected_by": "session_capability",
            }
        # create_action_graph stays available for pure OmniGraph wiring; a ScriptNode/
        # script-input payload (node type, inputs:script/scriptPath/usePath write or
        # connection, or the script_file shortcut) is refused for session capability
        # by the confirmed-sound guard, applied here so a session call cannot slip
        # past before reaching the handler.
        if cmd_type in ("graphs.create_action_graph", "create_action_graph"):
            from .handlers.graphs import _scriptnode_rce_rejection

            return _scriptnode_rce_rejection(
                nodes=params.get("nodes"),
                values=params.get("values"),
                connections=params.get("connections"),
                script_file=params.get("script_file"),
            )
        return None

    # ── Legacy command compatibility ──────────────────────────────────────────

    def _register_legacy_handlers(self) -> None:
        self._registry.update(
            {
                "get_scene_info": self._registry["scene.get_info"],
                "create_physics_scene": self._legacy_create_physics_scene,
                "create_robot": self._legacy_create_robot,
                "transform": self._registry["objects.transform"],
                "execute_script": self._registry["simulation.execute_script"],
                "generate_3d_from_text_or_image": self._registry["assets.generate_3d"],
                "search_3d_usd_by_text": self._registry["assets.search_usd"],
                "omini_kit_command": self._legacy_omni_kit_command,
            }
        )

    def _legacy_omni_kit_command(self, command: str = "CreatePrim", prim_type: str = "Sphere") -> Dict[str, Any]:
        omni.kit.commands.execute(command, prim_type=prim_type)
        return {"status": "success", "message": "command executed"}

    def _legacy_create_robot(
        self,
        robot_type: str = "g1",
        position: list[float] | None = None,
        name: str | None = None,
        prim_path: str | None = None,
    ) -> Dict[str, Any]:
        robot_aliases = {
            "franka": "frankapanda",
            "carter": "carter_v1",
        }
        return self._registry["robots.create"](
            robot_type=robot_aliases.get(robot_type.lower(), robot_type),
            position=position,
            name=name,
            prim_path=prim_path,
        )

    def _legacy_create_physics_scene(
        self,
        objects: list[dict[str, Any]] | None = None,
        floor: bool = True,
        gravity: list[float] | None = None,
        scene_name: str = "physics_scene",
    ) -> Dict[str, Any]:
        scene_result = self._registry["scene.create_physics"](
            gravity=gravity,
            scene_name=scene_name or "physics_scene",
        )
        if scene_result.get("status") != "success":
            return scene_result

        objects_created = 0
        if floor:
            floor_result = self._registry["objects.create"](
                object_type="Plane",
                prim_path="/World/ground",
                scale=[100, 100, 1],
            )
            if floor_result.get("status") == "success":
                objects_created += 1

        for index, obj in enumerate(objects or []):
            obj_type = obj.get("type", "Cube")
            prim_path = obj.get("path") or f"/World/{obj.get('name', f'object_{index}')}"
            rotation = obj.get("rotation")
            if isinstance(rotation, list) and len(rotation) == 4:
                rotation = None
            result = self._registry["objects.create"](
                object_type=obj_type,
                prim_path=prim_path,
                position=obj.get("position", [0, 0, 0]),
                rotation=rotation,
                scale=obj.get("scale", [1, 1, 1]),
                color=obj.get("color"),
                physics_enabled=obj.get("physics_enabled", True),
            )
            if result.get("status") != "success":
                return result
            objects_created += 1

        return {
            "status": "success",
            "message": f"Created physics scene with {objects_created} objects",
            "result": scene_name,
        }

    # ── Command routing ────────────────────────────────────────────────────────

    def _execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        cmd_type = command.get("type", "")
        params = command.get("params", {})
        # socket_server authenticates the token into this tag; trust only an exact
        # "operator", everything else is the least-privilege session (fail-safe).
        capability = (
            OPERATOR_CAPABILITY
            if command.get("capability") == OPERATOR_CAPABILITY
            else SESSION_CAPABILITY
        )

        # Publish the capability so the sink-level guards (objects.create/clone ->
        # guard_pose_write) and the ScriptNode gate honour operator/session without a
        # per-handler argument. Reset in finally so it never leaks to the next command.
        cap_token = set_current_capability(capability)
        try:
            denial = self._capability_denial(cmd_type, params, capability)
            if denial is not None:
                return denial

            # Fail-closed non-cheating guard runs before any handler; operator
            # capability bypasses it. A guard fault rejects (never falls open).
            if self._guard is not None:
                try:
                    denial = self._guard.guard(cmd_type, params, capability=capability)
                except Exception as e:
                    traceback.print_exc()
                    return {"status": "error", "message": f"session_guard error (fail-closed): {e}"}
                if denial is not None:
                    return denial

            handler = self._registry.get(cmd_type)
            if handler:
                try:
                    result = handler(**params)
                    if result and result.get("status") == "success":
                        return {"status": "success", "result": result}
                    else:
                        return {
                            "status": "error",
                            "message": result.get("message", "Unknown error") if result else "No result",
                        }
                except Exception as e:
                    traceback.print_exc()
                    return {"status": "error", "message": str(e)}
            return {"status": "error", "message": f"Unknown command: {cmd_type}"}
        finally:
            reset_current_capability(cap_token)
