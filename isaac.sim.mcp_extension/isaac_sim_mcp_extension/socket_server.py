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
import json
import socket
import threading
import time
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict


class SocketServer:
    """Manages a TCP socket server that accepts JSON commands and returns responses.

    Parameters
    ----------
    host:
        Hostname or IP address to bind to.
    port:
        Port number to listen on.
    command_handler:
        Callable invoked with the parsed command dict; must return a response dict.
    """

    def __init__(
        self,
        host: str,
        port: int,
        command_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        command_timeout: float = 120.0,
    ) -> None:
        self.host = host
        self.port = port
        self._command_handler = command_handler
        self._command_timeout = command_timeout
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

    def _dispatch_command(self, client: socket.socket, command: Dict[str, Any]) -> None:
        # This runs on a per-client worker thread. Omniverse USD/stage/timeline
        # APIs invoked by the handler are main-thread only, so the call is
        # marshalled onto the Kit asyncio loop with run_coroutine_threadsafe —
        # the *thread-safe* scheduler that also wakes the loop. (Calling
        # omni.kit.async_engine.run_coroutine() directly from this worker thread
        # enqueued the coroutine without a thread-safe wakeup, so it was never
        # pumped and every command — including simulation.ping — timed out,
        # which kept the bridge from ever registering.)
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
