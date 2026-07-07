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

"""Socket connection to the Isaac Sim extension server."""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("IsaacMCPServer")

DEFAULT_PORT = 8766

# Hard ceiling on a single framed response, mirrors the extension's socket_server.
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _session_mode_enabled() -> bool:
    """Match the extension: session-safe (framed+token) unless MCP_SESSION_MODE is falsey."""
    raw = os.environ.get("MCP_SESSION_MODE")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _session_token() -> Optional[str]:
    token = os.environ.get("MCP_SESSION_TOKEN")
    return token or None


def _operator_token() -> Optional[str]:
    """The full-capability OPERATOR token from the environment, or ``None``.

    Held only by trusted bootstrap callers (g1_autospawn, warm_slot_agent); the
    untrusted agent path never sets it and stays on the restricted session token.
    """
    token = os.environ.get("MCP_OPERATOR_TOKEN")
    return token or None


@dataclass
class IsaacConnection:
    """Manages a persistent TCP socket connection to the Isaac Sim extension."""

    host: str = "localhost"
    port: int = 0
    # Per-operation socket timeout (seconds). MUST stay >= the extension's
    # SocketServer.command_timeout so the client never gives up before the server
    # emits its framed response — a late response would desync the length-prefixed
    # stream. Default 300s covers the server's 290s ceiling and the 120s/300s
    # bridge budgets (all clients now share this one floor).
    timeout: float = 300.0
    # Capability token attached verbatim to every framed request. Trusted
    # bootstrap callers (g1_autospawn, warm_slot_agent) pass the OPERATOR token so
    # their execute_script / scene-setup keeps working under the per-token model;
    # leaving it None falls back to MCP_SESSION_TOKEN so the untrusted agent path
    # (bridge -> gateway) stays restricted. The extension maps the presented token
    # to a capability (operator|session) and a missing/unknown token fails safe to
    # the restricted session level.
    token: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.port == 0:
            self.port = int(os.environ.get("ISAAC_MCP_PORT", DEFAULT_PORT))
        # Wire protocol is chosen symmetrically with the extension server. Default
        # is the length-framed + token protocol; the legacy raw-JSON path is only
        # used when MCP_SESSION_MODE is explicitly disabled.
        self.framed = _session_mode_enabled()
        # An explicit token (e.g. the operator token) wins; otherwise fall back to
        # the per-session token from the environment.
        if self.token is None:
            self.token = _session_token()

    sock: Optional[socket.socket] = field(default=None, repr=False)
    framed: bool = field(default=True, repr=False)

    def connect(self) -> bool:
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Isaac at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Isaac: {e}")
            self.sock = None
            return False

    def disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.sock = None

    def receive_full_response(self, sock: socket.socket, buffer_size: int = 16384) -> bytes:
        chunks = []
        sock.settimeout(self.timeout)
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise Exception("Connection closed before receiving any data")
                        break
                    chunks.append(chunk)
                    try:
                        data = b"".join(chunks)
                        json.loads(data.decode("utf-8"))
                        return data
                    except json.JSONDecodeError:
                        continue
                except socket.timeout:
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError):
                    raise
        except socket.timeout:
            pass

        if chunks:
            data = b"".join(chunks)
            try:
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                raise Exception("Incomplete JSON response received")
        raise Exception("No data received")

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        chunks = []
        remaining = num_bytes
        while remaining > 0:
            data = sock.recv(min(remaining, 65536))
            if not data:
                return None
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _recv_framed(self, sock: socket.socket) -> Dict[str, Any]:
        header = self._recv_exact(sock, 4)
        if header is None:
            raise Exception("Connection closed before frame header")
        (length,) = struct.unpack(">I", header)
        if length == 0 or length > _MAX_MESSAGE_BYTES:
            raise Exception(f"Invalid frame length {length}")
        payload = self._recv_exact(sock, length)
        if payload is None:
            raise Exception("Connection closed mid-frame")
        return json.loads(payload.decode("utf-8"))

    def send_command_full(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a command and return the FULL response envelope (``status`` +
        ``result`` + any top-level fields), WITHOUT extracting ``result`` or
        raising on an error status.

        Uses the same wire protocol as :meth:`send_command` — length-framed +
        per-session token by default, legacy raw-JSON when ``MCP_SESSION_MODE`` is
        off — so transport wrappers that need the raw status envelope (e.g. the
        session bridge, which forwards the server's status to the control plane)
        reuse this ONE framing implementation instead of re-deriving it.
        """
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Isaac")

        command = {"type": command_type, "params": params or {}}
        try:
            self.sock.settimeout(self.timeout)
            if self.framed:
                if self.token:
                    command["token"] = self.token
                body = json.dumps(command).encode("utf-8")
                self.sock.sendall(struct.pack(">I", len(body)) + body)
                return self._recv_framed(self.sock)
            self.sock.sendall(json.dumps(command).encode("utf-8"))
            response_data = self.receive_full_response(self.sock)
            return json.loads(response_data.decode("utf-8"))
        except socket.timeout:
            self.sock = None
            raise Exception("Timeout waiting for Isaac response")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            self.sock = None
            raise Exception(f"Connection to Isaac lost: {e}")
        except json.JSONDecodeError as e:
            self.sock = None
            raise Exception(f"Invalid response from Isaac: {e}")
        except Exception as e:
            self.sock = None
            raise Exception(f"Communication error with Isaac: {e}")

    def send_command(self, command_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.send_command_full(command_type, params)
        if response.get("status") == "error":
            raise Exception(response.get("message", "Unknown error from Isaac"))
        return response.get("result", {})


_isaac_connection: Optional[IsaacConnection] = None


def get_isaac_connection() -> IsaacConnection:
    """Get or create a persistent Isaac connection singleton."""
    global _isaac_connection
    if _isaac_connection is not None:
        return _isaac_connection
    _isaac_connection = IsaacConnection(host="localhost")
    if not _isaac_connection.connect():
        _isaac_connection = None
        raise Exception("Could not connect to Isaac. Make sure the Isaac addon is running.")
    return _isaac_connection


def reset_isaac_connection() -> None:
    """Disconnect and clear the global connection (used during shutdown)."""
    global _isaac_connection
    if _isaac_connection:
        _isaac_connection.disconnect()
        _isaac_connection = None
