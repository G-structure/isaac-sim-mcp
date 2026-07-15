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

"""Gaussian-splat asset MCP tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, List, Optional

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(
    mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]"
) -> None:
    @mcp.tool("load_gaussian_splat")
    def load_gaussian_splat(
        usd_path: str,
        prim_path: str = "/World/GaussianSplat",
        position: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        frame_camera: bool = True,
        wait_timeout_seconds: float = 20.0,
        warmup_frames: int = 8,
        sample_frames: int = 30,
    ) -> str:
        """Load and prove a Gaussian-splat USD/USDZ asset.

        A successful RPC is not sufficient evidence by itself. Callers must
        require ``ready=true`` for the render path, ``fidelity_ready=true`` for
        authored source color/opacity, and inspect the returned schema,
        stage-loading, renderer, frame-time, and GPU evidence before claiming
        the splat works.

        Args:
            usd_path: Absolute local path or http(s), file, or omniverse USD URL.
            prim_path: New reference root, which must be below /World.
            position: Optional [x, y, z] translation.
            scale: Optional positive [sx, sy, sz] scale.
            frame_camera: Frame the referenced subtree in the active viewport.
            wait_timeout_seconds: Bounded wait for stage assets, at most 60 seconds.
            warmup_frames: Kit update frames excluded from timing, at most 120.
            sample_frames: Kit update frames measured for evidence, at most 240.
        """
        params = {
            "usd_path": usd_path,
            "prim_path": prim_path,
            "frame_camera": frame_camera,
            "wait_timeout_seconds": wait_timeout_seconds,
            "warmup_frames": warmup_frames,
            "sample_frames": sample_frames,
        }
        if position is not None:
            params["position"] = position
        if scale is not None:
            params["scale"] = scale
        return _send(get_connection, "assets.load_gaussian_splat", params)

    @mcp.tool("inspect_gaussian_splat")
    def inspect_gaussian_splat(
        prim_path: str = "/World/GaussianSplat",
        frame_camera: bool = False,
        wait_timeout_seconds: float = 10.0,
        warmup_frames: int = 4,
        sample_frames: int = 30,
    ) -> str:
        """Re-run render-readiness proof for an existing Gaussian subtree.

        Require ``ready=true`` before treating schema composition and renderer
        availability as proven, and ``fidelity_ready=true`` before claiming
        authored source color/opacity was validated. Visual correctness still
        requires a viewport or CUA image review, as stated in the response.
        """
        return _send(
            get_connection,
            "assets.inspect_gaussian_splat",
            {
                "prim_path": prim_path,
                "frame_camera": frame_camera,
                "wait_timeout_seconds": wait_timeout_seconds,
                "warmup_frames": warmup_frames,
                "sample_frames": sample_frames,
            },
        )


def _send(
    get_connection: "Callable[[], IsaacConnection]",
    command: str,
    params: dict,
) -> str:
    try:
        result = get_connection().send_command(command, params)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})
