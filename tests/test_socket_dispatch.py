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
import unittest

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
    "socket_server.py",
)


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

    def test_stop_resolves_outstanding_command(self):
        mod = _load_socket_server()
        handler_entered = threading.Event()
        release_handler = threading.Event()

        def handler(_command):
            handler_entered.set()
            release_handler.wait(timeout=5.0)
            return {"status": "success", "result": {}}

        port = _free_port()
        server = mod.SocketServer("localhost", port, handler, command_timeout=5.0)
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
        try:
            self.assertTrue(handler_entered.wait(timeout=2.0))
            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started
        finally:
            release_handler.set()
            if server.running:
                server.stop()
            client_thread.join(timeout=2.0)
            pump_thread.join(timeout=2.0)

        self.assertNotIn("error", pump_error)
        self.assertFalse(client_thread.is_alive())
        self.assertFalse(pump_thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["response"]["status"], "error")
        self.assertIn("stopped", outcome["response"]["message"])
        self.assertLess(elapsed, 3.0)

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
