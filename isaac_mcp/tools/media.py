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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Viewport media MCP tools."""

import json
from typing import TYPE_CHECKING, Callable, Optional

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]") -> None:

    @mcp.tool("capture_video")
    def capture_video(
        output_path: Optional[str] = None,
        duration_seconds: float = 5.0,
        frame_count: Optional[int] = None,
        fps: int = 24,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_prim_path: Optional[str] = None,
    ) -> str:
        """Capture the active viewport to an MP4 under /data/workspace/media."""
        try:
            conn = get_connection()
            result = conn.send_command(
                "media.capture_video",
                {
                    "output_path": output_path,
                    "duration_seconds": duration_seconds,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "camera_prim_path": camera_prim_path,
                },
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("download_artifact")
    def download_artifact(path: str, max_bytes: int = 25 * 1024 * 1024) -> str:
        """Download a workspace artifact as base64 data."""
        try:
            conn = get_connection()
            result = conn.send_command("media.download_artifact", {"path": path, "max_bytes": max_bytes})
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
