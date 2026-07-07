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
import struct
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


def _recv_exact(client: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        data = client.recv(remaining)
        if not data:
            raise AssertionError("socket closed before full frame")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _roundtrip(port: int, command: dict) -> dict:
    """Framed round-trip: 4-byte big-endian length prefix + JSON, both ways.

    The default (session-mode) SocketServer speaks the framed protocol; token is
    unset here so only framing is exercised.
    """
    client = socket.create_connection(("localhost", port), timeout=5)
    try:
        body = json.dumps(command).encode("utf-8")
        client.sendall(struct.pack(">I", len(body)) + body)
        client.settimeout(5)
        (length,) = struct.unpack(">I", _recv_exact(client, 4))
        data = _recv_exact(client, length)
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


def _capability_server(mod, port, seen):
    """A framed SocketServer configured with both tokens; the handler records the
    command it was dispatched (so tests can read the authenticated ``capability``
    tag and confirm the token was stripped)."""
    def handler(command):
        seen["command"] = command
        return {"status": "success", "result": {"echo": command["type"]}}

    return mod.SocketServer(
        "localhost", port, handler, operator_token="op-secret", session_token="s3cret"
    )


def test_framed_operator_token_classified_operator_and_stripped():
    mod = _load_socket_server()
    loop, _thread, _ident = _loop_in_thread()
    seen = {}
    port = _free_port()
    server = _capability_server(mod, port, seen)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping", "token": "op-secret"})
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    # The operator token authenticates the command as full ``operator`` capability,
    # and is stripped before the handler sees it.
    assert seen["command"]["capability"] == "operator"
    assert "token" not in seen["command"]


def test_framed_session_token_classified_session_and_stripped():
    mod = _load_socket_server()
    loop, _thread, _ident = _loop_in_thread()
    seen = {}
    port = _free_port()
    server = _capability_server(mod, port, seen)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping", "token": "s3cret"})
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    assert seen["command"]["capability"] == "session"
    assert "token" not in seen["command"]


def test_framed_missing_token_defaults_to_session_capability():
    # Fail-safe: a missing token is served (not rejected) but at the least-privilege
    # ``session`` level — the guards + exec-command lock then keep it harmless.
    mod = _load_socket_server()
    loop, _thread, _ident = _loop_in_thread()
    seen = {}
    port = _free_port()
    server = _capability_server(mod, port, seen)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping"})  # no token supplied
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    assert seen["command"]["capability"] == "session"


def test_framed_unknown_token_defaults_to_session_capability():
    mod = _load_socket_server()
    loop, _thread, _ident = _loop_in_thread()
    seen = {}
    port = _free_port()
    server = _capability_server(mod, port, seen)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping", "token": "bogus"})
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    assert seen["command"]["capability"] == "session"


def test_framed_client_cannot_self_escalate_capability():
    # A client-supplied ``capability`` field must be overwritten by the authenticated
    # value: presenting no operator token stays ``session`` even if it claims operator.
    mod = _load_socket_server()
    loop, _thread, _ident = _loop_in_thread()
    seen = {}
    port = _free_port()
    server = _capability_server(mod, port, seen)
    server.set_loop(loop)
    server.start()
    try:
        resp = _roundtrip(port, {"type": "simulation.ping", "capability": "operator"})
    finally:
        server.stop()
        loop.call_soon_threadsafe(loop.stop)

    assert resp["status"] == "success"
    assert seen["command"]["capability"] == "session"
