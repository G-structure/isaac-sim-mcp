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

import json
import queue
import socket
import threading
import time
import traceback
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(eq=False)
class _PendingCommand:
    command: Dict[str, Any]
    future: Future[Dict[str, Any]]
    started: bool = False


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

    _MAX_PENDING_COMMANDS = 64
    _MAX_COMMANDS_PER_DRAIN = 8

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
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending_commands: queue.Queue[_PendingCommand] = queue.Queue(
            maxsize=self._MAX_PENDING_COMMANDS
        )
        self._outstanding_commands: set[_PendingCommand] = set()
        self._drain_lock = threading.Lock()
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._client_threads: set[threading.Thread] = set()
        self._dispatch_condition = threading.Condition()
        self._active_dispatches = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind the socket and start the background accept loop."""
        with self._lifecycle_lock:
            if self._is_running():
                return

            server_socket: socket.socket | None = None
            try:
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.host, self.port))
                server_socket.listen(1)
                server_socket.settimeout(0.1)

                with self._state_lock:
                    self.running = True
                    self._socket = server_socket

                self._server_thread = threading.Thread(
                    target=self._server_loop,
                    args=(server_socket,),
                    daemon=True,
                )
                self._server_thread.start()
                print(f"Isaac Sim MCP server started on {self.host}:{self.port}")
            except (OSError, RuntimeError) as exc:
                if server_socket is not None:
                    self._close_socket(server_socket)
                with self._state_lock:
                    self.running = False
                    self._socket = None
                self._server_thread = None
                print(f"Failed to start server: {exc}")
                raise

    def stop(self) -> None:
        """Stop accepting work, resolve pending commands, and join workers."""
        with self._lifecycle_lock:
            with self._state_lock:
                self.running = False
                server_socket = self._socket
                self._socket = None

            if server_socket is not None:
                self._close_socket(server_socket)

            started_commands = self._resolve_waiting_commands(
                {
                    "status": "error",
                    "message": "Isaac MCP server stopped before command completed",
                }
            )
            self._discard_queued_commands()

            server_thread = self._server_thread
            if (
                server_thread is not None
                and server_thread is not threading.current_thread()
            ):
                server_thread.join(timeout=1.0)
            self._server_thread = None

            self._wait_for_started_commands(started_commands)
            self._wait_for_active_dispatches(timeout=1.0)
            with self._clients_lock:
                clients = list(self._clients)
                client_threads = list(self._client_threads)

            for client in clients:
                self._close_socket(client)

            deadline = time.monotonic() + 1.0
            for thread in client_threads:
                if thread is threading.current_thread():
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                thread.join(timeout=remaining)

            print("Isaac Sim MCP server stopped")

    def drain_pending_commands(self) -> int:
        """Execute queued commands on the calling thread.

        The retained Kit update-event callback is the production caller. A
        non-blocking guard prevents a handler-triggered nested update from
        recursively draining the queue.
        """
        if not self._drain_lock.acquire(blocking=False):
            return 0

        dequeued = 0
        executed = 0
        try:
            while dequeued < self._MAX_COMMANDS_PER_DRAIN:
                try:
                    pending = self._pending_commands.get_nowait()
                except queue.Empty:
                    break
                dequeued += 1

                with self._state_lock:
                    if pending not in self._outstanding_commands or not self.running:
                        continue
                    pending.started = True

                try:
                    response = self._command_handler(pending.command)
                except Exception as e:
                    traceback.print_exc()
                    response = {"status": "error", "message": str(e)}

                with self._state_lock:
                    if pending in self._outstanding_commands:
                        self._outstanding_commands.remove(pending)
                        pending.future.set_result(response)
                executed += 1
        finally:
            self._drain_lock.release()

        return executed

    # ── Connection handling ────────────────────────────────────────────────────

    def _server_loop(self, server_socket: socket.socket) -> None:
        while self._is_running():
            try:
                client, address = server_socket.accept()
                print(f"Connected to client: {address}")
                self._start_client_thread(client)
            except socket.timeout:
                continue
            except Exception as e:
                if self._is_running():
                    print(f"Error accepting connection: {e}")
                    time.sleep(0.5)

    def _start_client_thread(self, client: socket.socket) -> None:
        thread = threading.Thread(
            target=self._handle_client, args=(client,), daemon=True
        )
        with self._state_lock:
            if not self.running:
                self._close_socket(client)
                return
            with self._clients_lock:
                self._clients.add(client)
                self._client_threads.add(thread)
                thread.start()

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
            if self._is_running():
                print(f"Error in client handler: {e}")
        finally:
            self._close_socket(client)
            with self._clients_lock:
                self._clients.discard(client)
                self._client_threads.discard(threading.current_thread())

    def _dispatch_command(self, client: socket.socket, command: Dict[str, Any]) -> None:
        if not self._begin_dispatch():
            return
        try:
            response = self._wait_for_main_thread(command)
            try:
                client.sendall(json.dumps(response).encode("utf-8"))
            except Exception:
                print("Failed to send response - client disconnected")
        finally:
            self._finish_dispatch()

    def _wait_for_main_thread(self, command: Dict[str, Any]) -> Dict[str, Any]:
        pending = _PendingCommand(command=command, future=Future())
        with self._state_lock:
            if not self.running:
                return {"status": "error", "message": "Isaac MCP server is stopping"}
            self._outstanding_commands.add(pending)
            try:
                self._pending_commands.put_nowait(pending)
            except queue.Full:
                self._outstanding_commands.remove(pending)
                return {
                    "status": "error",
                    "message": "Isaac MCP command queue is full; retry later",
                }

        try:
            return pending.future.result(timeout=self._command_timeout)
        except FutureTimeoutError:
            with self._state_lock:
                if pending not in self._outstanding_commands:
                    return pending.future.result()
                if pending.started:
                    wait_for_completion = True
                else:
                    self._outstanding_commands.remove(pending)
                    pending.future.cancel()
                    wait_for_completion = False

            if wait_for_completion:
                return pending.future.result()
            return {
                "status": "error",
                "message": f"Command timed out after {self._command_timeout:g}s waiting for the Isaac main-thread update pump",
            }

    def _resolve_waiting_commands(
        self, response: Dict[str, Any]
    ) -> list[_PendingCommand]:
        with self._state_lock:
            started_commands = []
            for pending in list(self._outstanding_commands):
                if pending.started:
                    started_commands.append(pending)
                    continue
                self._outstanding_commands.remove(pending)
                if not pending.future.done():
                    pending.future.set_result(dict(response))
            return started_commands

    @staticmethod
    def _wait_for_started_commands(
        started_commands: list[_PendingCommand],
    ) -> None:
        for pending in started_commands:
            pending.future.result()

    def _discard_queued_commands(self) -> None:
        while True:
            try:
                self._pending_commands.get_nowait()
            except queue.Empty:
                return

    def _begin_dispatch(self) -> bool:
        with self._state_lock:
            if not self.running:
                return False
            with self._dispatch_condition:
                self._active_dispatches += 1
            return True

    def _finish_dispatch(self) -> None:
        with self._dispatch_condition:
            self._active_dispatches -= 1
            self._dispatch_condition.notify_all()

    def _wait_for_active_dispatches(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._dispatch_condition:
            while self._active_dispatches:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._dispatch_condition.wait(timeout=remaining)

    def _is_running(self) -> bool:
        with self._state_lock:
            return self.running

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
