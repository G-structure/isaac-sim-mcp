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

"""Stdlib regression tests for socket-to-Kit-main-thread dispatch."""

import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
    "socket_server.py",
)
_EXTENSION_PATH = os.path.join(os.path.dirname(_MODULE_PATH), "extension.py")


def _load_socket_server():
    module_name = "isaac_mcp_socket_server"
    spec = importlib.util.spec_from_file_location(module_name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tmp.bind(("localhost", 0))
    port = tmp.getsockname()[1]
    tmp.close()
    return port


def _roundtrip(port: int, command: dict, timeout: float = 5.0) -> dict:
    client = socket.create_connection(("localhost", port), timeout=timeout)
    try:
        client.sendall(json.dumps(command).encode("utf-8"))
        client.settimeout(timeout)
        data = client.recv(65536)
    finally:
        client.close()
    return json.loads(data.decode("utf-8"))


def _start_roundtrip(port: int, command: dict, timeout: float = 5.0):
    outcome = {}

    def run() -> None:
        try:
            outcome["response"] = _roundtrip(port, command, timeout=timeout)
        except BaseException as exc:  # Preserve worker failures for the test thread.
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


class SocketDispatchTests(unittest.TestCase):
    def _wait_for_queue_size(self, server, size: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while server._pending_commands.qsize() != size and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(server._pending_commands.qsize(), size)

    def _pump_until_complete(
        self, server, thread: threading.Thread, timeout: float = 2.0
    ) -> None:
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            server.drain_pending_commands()
            thread.join(timeout=0.01)
        self.assertFalse(thread.is_alive(), "socket roundtrip did not complete")

    def test_socket_roundtrip_runs_handler_on_drain_thread(self):
        mod = _load_socket_server()
        drain_thread_ident = threading.get_ident()

        def handler(command):
            return {
                "status": "success",
                "result": {"echo": command["type"], "thread": threading.get_ident()},
            }

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler)
        server.start()
        thread, outcome = _start_roundtrip(port, {"type": "simulation.ping"})
        try:
            self._pump_until_complete(server, thread)
        finally:
            server.stop()

        self.assertNotIn("error", outcome)
        response = outcome["response"]
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["result"]["echo"], "simulation.ping")
        self.assertEqual(response["result"]["thread"], drain_thread_ident)

    def test_command_before_update_pump_returns_diagnosable_bounded_error(self):
        mod = _load_socket_server()

        def handler(
            command,
        ):  # pragma: no cover - the queue is intentionally not pumped.
            return {"status": "success", "result": command}

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler, command_timeout=0.1)
        server.start()
        started = time.monotonic()
        try:
            response = _roundtrip(port, {"type": "simulation.ping"}, timeout=2.0)
            elapsed = time.monotonic() - started
        finally:
            server.stop()

        self.assertEqual(response["status"], "error")
        self.assertIn("update pump", response["message"])
        self.assertIn("0.1s", response["message"])
        self.assertLess(elapsed, 1.0)

    def test_queue_backpressure_rejects_unaccepted_command(self):
        mod = _load_socket_server()

        class TinyQueueServer(mod.SocketServer):
            _MAX_PENDING_COMMANDS = 1

        server = TinyQueueServer(
            "localhost",
            _free_port(),
            lambda command: {"status": "success", "result": command},
            command_timeout=2.0,
        )
        first_outcome = {}

        def enqueue_first() -> None:
            first_outcome["response"] = server._wait_for_main_thread({"id": 1})

        server.start()
        first_thread = threading.Thread(target=enqueue_first, daemon=True)
        first_thread.start()
        try:
            self._wait_for_queue_size(server, 1)
            started = time.monotonic()
            rejected = server._wait_for_main_thread({"id": 2})
            elapsed = time.monotonic() - started
        finally:
            server.stop()
            first_thread.join(timeout=2.0)

        self.assertEqual(rejected["status"], "error")
        self.assertIn("queue is full", rejected["message"])
        self.assertLess(elapsed, 0.5)
        self.assertFalse(first_thread.is_alive())
        self.assertIn("stopped", first_outcome["response"]["message"])
        self.assertFalse(server._outstanding_commands)

    def test_each_drain_processes_at_most_one_fixed_batch(self):
        mod = _load_socket_server()

        class BatchedServer(mod.SocketServer):
            _MAX_PENDING_COMMANDS = 4
            _MAX_COMMANDS_PER_DRAIN = 2

        handled = []
        server = BatchedServer(
            "localhost",
            _free_port(),
            lambda command: (
                handled.append(command["id"])
                or {"status": "success", "result": command}
            ),
        )
        outcomes = {}

        def enqueue(command_id: int) -> None:
            outcomes[command_id] = server._wait_for_main_thread({"id": command_id})

        server.start()
        workers = [
            threading.Thread(target=enqueue, args=(command_id,), daemon=True)
            for command_id in range(3)
        ]
        for worker in workers:
            worker.start()
        try:
            self._wait_for_queue_size(server, 3)
            self.assertEqual(server.drain_pending_commands(), 2)
            self.assertEqual(server._pending_commands.qsize(), 1)
            self.assertEqual(len(handled), 2)
            self.assertEqual(server.drain_pending_commands(), 1)
        finally:
            server.stop()
            for worker in workers:
                worker.join(timeout=2.0)

        self.assertEqual(sorted(handled), [0, 1, 2])
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(
            all(response["status"] == "success" for response in outcomes.values())
        )

    def test_drain_budget_counts_timed_out_tombstones(self):
        mod = _load_socket_server()

        class OneItemDrainServer(mod.SocketServer):
            _MAX_PENDING_COMMANDS = 3
            _MAX_COMMANDS_PER_DRAIN = 1

        server = OneItemDrainServer(
            "localhost",
            _free_port(),
            lambda command: {"status": "success", "result": command},
            command_timeout=0.2,
        )
        outcomes = {}

        def enqueue(command_id: int) -> None:
            outcomes[command_id] = server._wait_for_main_thread({"id": command_id})

        server.start()
        workers = [
            threading.Thread(target=enqueue, args=(command_id,), daemon=True)
            for command_id in range(3)
        ]
        for worker in workers:
            worker.start()
        try:
            self._wait_for_queue_size(server, 3)
            for worker in workers:
                worker.join(timeout=1.0)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertFalse(server._outstanding_commands)

            self.assertEqual(server.drain_pending_commands(), 0)
            self.assertEqual(server._pending_commands.qsize(), 2)
        finally:
            server.stop()

        self.assertTrue(
            all("update pump" in response["message"] for response in outcomes.values())
        )

    def test_handler_exception_becomes_error_response(self):
        mod = _load_socket_server()

        def handler(_command):
            raise RuntimeError("handler exploded")

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler)
        server.start()
        thread, outcome = _start_roundtrip(port, {"type": "simulation.ping"})
        try:
            self._pump_until_complete(server, thread)
        finally:
            server.stop()

        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["response"]["status"], "error")
        self.assertIn("handler exploded", outcome["response"]["message"])

    def test_stop_waits_for_started_command_and_preserves_its_result(self):
        mod = _load_socket_server()
        handler_entered = threading.Event()
        handler_exited = threading.Event()
        release_handler = threading.Event()

        def handler(_command):
            handler_entered.set()
            release_handler.wait(timeout=5.0)
            handler_exited.set()
            return {"status": "success", "result": {}}

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler, command_timeout=0.05)
        server.start()
        client_thread, outcome = _start_roundtrip(port, {"type": "simulation.ping"})
        pump_error = {}

        def pump() -> None:
            try:
                deadline = time.monotonic() + 2.0
                while not handler_entered.is_set() and time.monotonic() < deadline:
                    server.drain_pending_commands()
                    time.sleep(0.005)
                if not handler_entered.is_set():
                    raise AssertionError("handler was never dispatched")
            except BaseException as exc:
                pump_error["error"] = exc

        pump_thread = threading.Thread(target=pump, daemon=True)
        pump_thread.start()
        release_timer = threading.Timer(0.2, release_handler.set)
        try:
            self.assertTrue(handler_entered.wait(timeout=2.0))
            release_timer.start()
            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started
        finally:
            release_handler.set()
            release_timer.cancel()
            release_timer.join(timeout=1.0)
            if server.running:
                server.stop()
            client_thread.join(timeout=2.0)
            pump_thread.join(timeout=2.0)

        self.assertNotIn("error", pump_error)
        self.assertTrue(handler_exited.is_set())
        self.assertFalse(client_thread.is_alive())
        self.assertFalse(pump_thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["response"]["status"], "success")
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 3.0)

    def test_startup_bind_failure_is_raised_and_rolled_back(self):
        mod = _load_socket_server()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("localhost", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        server = mod.SocketServer(
            "localhost", port, lambda _command: {"status": "success"}
        )

        try:
            with self.assertRaises(OSError):
                server.start()
        finally:
            blocker.close()

        self.assertFalse(server.running)
        self.assertIsNone(server._socket)
        self.assertIsNone(server._server_thread)

    def test_extension_releases_pump_when_socket_startup_fails(self):
        package_name = "isaac_mcp_extension_startup_test"
        package = types.ModuleType(package_name)
        package.__path__ = [os.path.dirname(_EXTENSION_PATH)]

        class Settings:
            @staticmethod
            def get(_key):
                return None

        carb = types.ModuleType("carb")
        carb.settings = types.SimpleNamespace(get_settings=lambda: Settings())

        omni = types.ModuleType("omni")
        omni.__path__ = []
        omni_ext = types.ModuleType("omni.ext")
        omni_ext.IExt = object
        omni_kit = types.ModuleType("omni.kit")
        omni_kit.__path__ = []
        omni_kit_app = types.ModuleType("omni.kit.app")
        omni_kit_commands = types.ModuleType("omni.kit.commands")
        omni_usd = types.ModuleType("omni.usd")

        subscription = object()
        update_stream = types.SimpleNamespace(
            create_subscription_to_pop=lambda *_args, **_kwargs: subscription
        )
        omni_kit_app.get_app = lambda: types.SimpleNamespace(
            get_update_event_stream=lambda: update_stream
        )
        omni.ext = omni_ext
        omni.kit = omni_kit
        omni.usd = omni_usd
        omni_kit.app = omni_kit_app
        omni_kit.commands = omni_kit_commands

        adapters = types.ModuleType(f"{package_name}.adapters")
        adapters.get_adapter = lambda: object()
        handlers = types.ModuleType(f"{package_name}.handlers")
        handlers.register_all_handlers = lambda _registry, _adapter: None

        class FailingServer:
            instances = []

            def __init__(self, *_args):
                self.stopped = False
                self.instances.append(self)

            @staticmethod
            def start():
                raise OSError("bind failed")

            def stop(self):
                self.stopped = True

        socket_server = types.ModuleType(f"{package_name}.socket_server")
        socket_server.SocketServer = FailingServer
        modules = {
            package_name: package,
            f"{package_name}.adapters": adapters,
            f"{package_name}.handlers": handlers,
            f"{package_name}.socket_server": socket_server,
            "carb": carb,
            "omni": omni,
            "omni.ext": omni_ext,
            "omni.kit": omni_kit,
            "omni.kit.app": omni_kit_app,
            "omni.kit.commands": omni_kit_commands,
            "omni.usd": omni_usd,
        }

        with mock.patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.extension", _EXTENSION_PATH
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                extension = module.MCPExtension()
                extension._register_legacy_handlers = lambda: None
                with self.assertRaisesRegex(OSError, "bind failed"):
                    extension.on_startup("isaac.sim.mcp")
            finally:
                sys.modules.pop(spec.name, None)

        self.assertIsNone(extension._update_subscription)
        self.assertIsNone(extension._server)
        self.assertTrue(FailingServer.instances[0].stopped)

    def test_reentrant_drain_returns_without_deadlock(self):
        mod = _load_socket_server()
        nested_results = []
        server = None

        def handler(_command):
            nested_results.append(server.drain_pending_commands())
            return {"status": "success", "result": {}}

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler)
        server.start()
        thread, outcome = _start_roundtrip(port, {"type": "simulation.ping"})
        try:
            self._pump_until_complete(server, thread)
        finally:
            server.stop()

        self.assertNotIn("error", outcome)
        self.assertEqual(nested_results, [0])


if __name__ == "__main__":
    unittest.main()
