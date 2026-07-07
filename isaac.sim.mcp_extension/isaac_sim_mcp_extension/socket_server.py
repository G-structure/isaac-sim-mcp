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

"""TCP socket server for Isaac Sim MCP — connection and dispatch logic."""

from __future__ import annotations

import asyncio
import hmac
import json
import socket
import struct
import threading
import time
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, Optional

# Hard ceiling on a single framed request so a malicious/garbled length prefix
# cannot drive an unbounded allocation. 64 MiB is far above any real command.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class SocketServer:
    """Manages a TCP socket server that accepts JSON commands and returns responses.

    Two wire protocols are supported:

    * ``framed=True`` (default / session mode): each message is a 4-byte
      big-endian length prefix followed by that many UTF-8 JSON bytes, in both
      directions. Every request frame may carry a top-level ``"token"`` string,
      which is authenticated (constant-time compared) and classified into a
      capability level: a token equal to ``operator_token`` grants full ``operator``
      capability; anything else — the session token, an unknown token, or none —
      is the restricted ``session`` capability (fail-safe). The authenticated level
      is tagged onto the command as ``"capability"`` for dispatch, and the token is
      stripped. This is what lets the same loopback :8766 socket serve trusted
      bootstrap (operator) and the untrusted bridge->gateway agent (session) without
      the agent reaching the exec/script commands or the pose-lock bypass.
    * ``framed=False`` (operator/legacy, ``MCP_SESSION_MODE`` off): the historical
      recv-until-JSON-parses behaviour, kept for backward compatibility with old
      bridges; commands are tagged ``operator`` since this path is the deliberate
      fully-trusted escape hatch.

    Parameters
    ----------
    host:
        Hostname or IP address to bind to.
    port:
        Port number to listen on.
    command_handler:
        Callable invoked with the parsed command dict; must return a response dict.
    command_timeout:
        Per-command wall-clock budget on the Isaac main loop. Kept strictly below
        the bridge's 300s socket timeout (see ``isaac_mcp/connection.py``) so the
        server always emits a framed response BEFORE the client gives up — a late
        response would otherwise desync the length-prefixed stream.
    framed:
        Use the length-prefixed protocol (default) or the legacy raw-JSON one.
    operator_token:
        Full-capability shared secret (MCP_OPERATOR_TOKEN) for trusted bootstrap, or
        ``None`` to disable the operator level (every framed request then classifies
        as ``session``). TODO(probe:guard-live): the warm-node image must set
        MCP_OPERATOR_TOKEN on the extension and the trusted bootstrap clients, and
        MCP_SESSION_TOKEN on the bridge, for the capability split to engage live.
    session_token:
        Restricted shared secret (MCP_SESSION_TOKEN). Not required to dispatch — a
        missing or unknown token still classifies as ``session`` (fail-safe) and is
        served under the guards — it is documented here only to record the intended
        wiring; presence never grants more than the ``session`` level.
    """

    def __init__(
        self,
        host: str,
        port: int,
        command_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        command_timeout: float = 290.0,
        *,
        framed: bool = True,
        operator_token: Optional[str] = None,
        session_token: Optional[str] = None,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
    ) -> None:
        self.host = host
        self.port = port
        self._command_handler = command_handler
        self._command_timeout = command_timeout
        self._framed = framed
        self._operator_token = operator_token
        self._session_token = session_token
        self._max_message_bytes = max_message_bytes
        self.running: bool = False
        self._socket: socket.socket | None = None
        self._server_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Bind the Kit main asyncio loop used to run command handlers.

        Must be the loop that runs on Isaac's main thread (captured in
        ``on_startup``). Commands touch USD/stage/timeline APIs that are
        main-thread only, so the socket worker threads marshal onto this loop.
        """
        self._loop = loop

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind the socket and start the background accept loop."""
        if self.running:
            return
        self.running = True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            self._socket.listen(1)
            self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self._server_thread.start()
            print(f"Isaac Sim MCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {e}")
            self.stop()

    def stop(self) -> None:
        """Signal the server to stop and close the socket."""
        self.running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        self._server_thread = None
        print("Isaac Sim MCP server stopped")

    # ── Connection handling ────────────────────────────────────────────────────

    def _server_loop(self) -> None:
        self._socket.settimeout(1.0)
        while self.running:
            try:
                client, address = self._socket.accept()
                print(f"Connected to client: {address}")
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
                    time.sleep(0.5)

    def _handle_client(self, client: socket.socket) -> None:
        if self._framed:
            self._handle_client_framed(client)
        else:
            self._handle_client_legacy(client)

    def _handle_client_framed(self, client: socket.socket) -> None:
        client.settimeout(None)
        try:
            while self.running:
                header = self._recv_exact(client, 4)
                if header is None:
                    break
                (length,) = struct.unpack(">I", header)
                if length == 0 or length > self._max_message_bytes:
                    self._send_framed(client, {"status": "error", "message": f"invalid frame length {length}"})
                    break
                payload = self._recv_exact(client, length)
                if payload is None:
                    break
                try:
                    command = json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_framed(client, {"status": "error", "message": f"invalid JSON frame: {e}"})
                    continue
                command["capability"] = self._authenticate_capability(command)
                response = self._run_command_on_main_loop(command)
                self._send_framed(client, response)
        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            client.close()

    def _handle_client_legacy(self, client: socket.socket) -> None:
        client.settimeout(None)
        buffer = b""
        try:
            while self.running:
                data = client.recv(16384)
                if not data:
                    break
                buffer += data
                try:
                    command = json.loads(buffer.decode("utf-8"))
                    buffer = b""
                    self._dispatch_command(client, command)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            client.close()

    @staticmethod
    def _recv_exact(client: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Read exactly ``num_bytes`` from the socket, or ``None`` if it closes first."""
        chunks = []
        remaining = num_bytes
        while remaining > 0:
            data = client.recv(min(remaining, 65536))
            if not data:
                return None
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _send_framed(self, client: socket.socket, response: Dict[str, Any]) -> None:
        try:
            body = json.dumps(response).encode("utf-8")
            client.sendall(struct.pack(">I", len(body)) + body)
        except Exception:
            print("Failed to send response — client disconnected")

    def _authenticate_capability(self, command: Dict[str, Any]) -> str:
        """Authenticate the presented token and return its capability level.

        A ``token`` equal to the configured ``operator_token`` grants ``operator``
        (full) capability — trusted bootstrap such as warm_slot_agent and
        g1_autospawn. Anything else — the session token, an unknown token, or none —
        is ``session`` (restricted). Fail-safe: the default is the least-privilege
        ``session`` level, never operator.

        The presented ``token`` is stripped before dispatch, and any client-supplied
        ``capability`` field is dropped so a caller cannot self-escalate — only the
        value returned here (re-tagged by the caller) is trusted downstream.
        """
        supplied = command.pop("token", None)
        command.pop("capability", None)
        if (
            self._operator_token
            and isinstance(supplied, str)
            and hmac.compare_digest(supplied, self._operator_token)
        ):
            return "operator"
        return "session"

    def _dispatch_command(self, client: socket.socket, command: Dict[str, Any]) -> None:
        # This runs on a per-client worker thread. Omniverse USD/stage/timeline
        # APIs invoked by the handler are main-thread only, so the call is
        # marshalled onto the Kit asyncio loop with run_coroutine_threadsafe —
        # the *thread-safe* scheduler that also wakes the loop. (Calling
        # omni.kit.async_engine.run_coroutine() directly from this worker thread
        # enqueued the coroutine without a thread-safe wakeup, so it was never
        # pumped and every command — including simulation.ping — timed out,
        # which kept the bridge from ever registering.)
        # The legacy raw-JSON path is only reachable with framed=False (session mode
        # off): the deliberate fully-trusted escape hatch, so tag it ``operator``.
        command["capability"] = "operator"
        response = self._run_command_on_main_loop(command)
        try:
            client.sendall(json.dumps(response).encode("utf-8"))
        except Exception:
            print("Failed to send response — client disconnected")

    def _run_command_on_main_loop(self, command: Dict[str, Any]) -> Dict[str, Any]:
        loop = self._loop
        if loop is None or not loop.is_running():
            # Loop not captured yet (briefly, right after on_startup) or shutting
            # down. Return an error so the socket stays responsive and the client
            # retries, rather than blocking this worker thread indefinitely.
            return {"status": "error", "message": "Isaac main loop not ready"}

        async def _call() -> Dict[str, Any]:
            return self._command_handler(command)

        try:
            future = asyncio.run_coroutine_threadsafe(_call(), loop)
            return future.result(timeout=self._command_timeout)
        except FutureTimeoutError:
            return {
                "status": "error",
                "message": f"Command timed out after {self._command_timeout}s on Isaac main loop",
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
