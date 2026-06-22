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

"""Regression test for the warm-node bridge wedge.

The socket server accepts connections on worker threads but command handlers
run Omniverse APIs that are main-thread only, so the handler must be marshalled
onto the Kit main asyncio loop in a *thread-safe* way. A prior implementation
called ``omni.kit.async_engine.run_coroutine`` directly from the worker thread,
which enqueued the coroutine without a thread-safe wakeup; it was never pumped,
so ``simulation.ping`` (and every other command) never got a response and the
bridge timed out waiting to register. These tests pin the thread-safe contract
without requiring Omniverse: ``socket_server`` is pure stdlib.
"""

import asyncio
import importlib.util
import json
import os
import socket
import threading
import time

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "isaac.sim.mcp_extension",
    "isaac_sim_mcp_extension",
    "socket_server.py",
)


def _load_socket_server():
    spec = importlib.util.spec_from_file_location("isaac_mcp_socket_server", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tmp.bind(("localhost", 0))
    port = tmp.getsockname()[1]
    tmp.close()
    return port


def _loop_in_thread():
    loop = asyncio.new_event_loop()
    ident_box: dict[str, int] = {}

    def run() -> None:
        asyncio.set_event_loop(loop)
        ident_box["ident"] = threading.get_ident()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    while "ident" not in ident_box:
        time.sleep(0.01)
    return loop, thread, ident_box["ident"]


def _roundtrip(port: int, command: dict) -> dict:
    client = socket.create_connection(("localhost", port), timeout=5)
    try:
        client.sendall(json.dumps(command).encode("utf-8"))
        client.settimeout(5)
        data = client.recv(65536)
    finally:
        client.close()
    return json.loads(data.decode("utf-8"))


def test_command_runs_on_bound_main_loop_from_worker_thread():
    mod = _load_socket_server()
    loop, _thread, loop_ident = _loop_in_thread()

    def handler(command):
        # Record which thread actually executed the handler.
        return {
            "status": "success",
            "result": {"echo": command["type"], "thread": threading.get_ident()},
        }

    port = _free_port()
    server = mod.SocketServer("localhost", port, handler)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping"})
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    assert resp["result"]["echo"] == "simulation.ping"
    # The handler must execute on the Kit main loop thread, not the socket worker.
    assert resp["result"]["thread"] == loop_ident


def test_command_before_loop_bound_returns_error_not_hang():
    mod = _load_socket_server()

    def handler(command):  # pragma: no cover - must not be reached
        return {"status": "success", "result": {}}

    port = _free_port()
    server = mod.SocketServer("localhost", port, handler)
    # Intentionally do NOT bind a loop (simulates the brief startup window).
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping"})
    finally:
        server.stop()

    assert resp["status"] == "error"
    assert "not ready" in resp["message"]
