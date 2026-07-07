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

"""Behavioral tests for IsaacConnection's capability-token attachment.

These drive the REAL socket_server (framed + two-token capability model) with the
REAL IsaacConnection so the whole wire contract is exercised end-to-end without
Omniverse:

  * an explicit operator token is attached to the framed request, authenticated by
    the server as the ``operator`` capability, and stripped before dispatch;
  * a wrong/missing token classifies fail-safe as ``session`` (least privilege),
    never operator;
  * with no explicit token the connection falls back to MCP_SESSION_TOKEN ->
    ``session``;
  * an explicit operator token WINS over a session token in the environment
    (trusted bootstrap escalates);
  * the default socket timeout covers the server's command budget.
"""

import asyncio
import importlib.util
import os
import socket
import sys
import threading
import time

import pytest

_HERE = os.path.dirname(__file__)
_SOCKET_SERVER = os.path.join(
    _HERE, "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension", "socket_server.py"
)
_CONNECTION = os.path.join(_HERE, "..", "isaac_mcp", "connection.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass ClassVar detection needs it registered
    spec.loader.exec_module(module)
    return module


socket_server = _load(_SOCKET_SERVER, "isaac_mcp_socket_server_tok")
connection = _load(_CONNECTION, "isaac_mcp_connection_tok")
SocketServer = socket_server.SocketServer
IsaacConnection = connection.IsaacConnection


def _free_port() -> int:
    tmp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tmp.bind(("localhost", 0))
    port = tmp.getsockname()[1]
    tmp.close()
    return port


def _loop_in_thread():
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    started.wait()
    return loop, thread


class _Server:
    """Real framed SocketServer bound to a live asyncio loop, recording commands.

    The handler echoes the ``capability`` the server tagged onto the command, so a
    test can assert exactly how the presented token was classified.
    """

    def __init__(self, operator_token=None, session_token=None):
        self.seen = []
        self.loop, self._thread = _loop_in_thread()
        self.port = _free_port()

        def handler(command):
            self.seen.append(command)
            return {
                "status": "success",
                "result": {"echo": command.get("type"), "capability": command.get("capability")},
            }

        self.server = SocketServer(
            "localhost",
            self.port,
            handler,
            operator_token=operator_token,
            session_token=session_token,
        )
        self.server.set_loop(self.loop)
        self.server.start()

    def stop(self):
        self.server.stop()
        self.loop.call_soon_threadsafe(self.loop.stop)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("MCP_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("MCP_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("MCP_SESSION_MODE", raising=False)
    return monkeypatch


def test_operator_token_authenticates_and_is_stripped(clean_env):
    srv = _Server(operator_token="OPERATOR-TOK", session_token="SESSION-TOK")
    try:
        conn = IsaacConnection(host="localhost", port=srv.port, token="OPERATOR-TOK")
        resp = conn.send_command_full("simulation.ping")
        conn.disconnect()
    finally:
        srv.stop()
    assert resp["status"] == "success"
    # The operator token is authenticated as the FULL capability.
    assert resp["result"]["capability"] == "operator"
    # Server strips the token before the handler sees it.
    assert srv.seen and "token" not in srv.seen[0]


def test_wrong_token_classifies_session_fail_safe(clean_env):
    srv = _Server(operator_token="OPERATOR-TOK", session_token="SESSION-TOK")
    try:
        conn = IsaacConnection(host="localhost", port=srv.port, token="WRONG")
        resp = conn.send_command_full("simulation.ping")
        conn.disconnect()
    finally:
        srv.stop()
    # Fail-safe: an unknown token is least-privilege session, never operator.
    assert resp["status"] == "success"
    assert resp["result"]["capability"] == "session"


def test_no_token_falls_back_to_session_env(clean_env):
    clean_env.setenv("MCP_SESSION_TOKEN", "SESSION-TOK")
    srv = _Server(operator_token="OPERATOR-TOK", session_token="SESSION-TOK")
    try:
        conn = IsaacConnection(host="localhost", port=srv.port)  # no explicit token
        assert conn.token == "SESSION-TOK"
        resp = conn.send_command_full("simulation.ping")
        conn.disconnect()
    finally:
        srv.stop()
    assert resp["status"] == "success"
    assert resp["result"]["capability"] == "session"


def test_explicit_operator_token_overrides_session_env(clean_env):
    # Even with a restricted session token in the environment, a trusted caller
    # that passes the operator token explicitly escalates to operator.
    clean_env.setenv("MCP_SESSION_TOKEN", "SESSION-TOK")
    srv = _Server(operator_token="OPERATOR-TOK", session_token="SESSION-TOK")
    try:
        conn = IsaacConnection(host="localhost", port=srv.port, token="OPERATOR-TOK")
        assert conn.token == "OPERATOR-TOK"
        resp = conn.send_command_full("simulation.ping")
        conn.disconnect()
    finally:
        srv.stop()
    assert resp["status"] == "success"
    assert resp["result"]["capability"] == "operator"


def test_default_timeout_covers_server_command_budget(clean_env):
    # Client timeout must be >= the server's command budget (290s) and the
    # 120s/300s bridge budgets so a late framed reply never desyncs the stream.
    assert IsaacConnection(host="localhost").timeout >= 290.0
