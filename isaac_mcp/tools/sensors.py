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

"""Sensor MCP tools."""

import json
from typing import TYPE_CHECKING, Callable, List, Optional

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from isaac_mcp.connection import IsaacConnection


def register_tools(
    mcp: FastMCP, get_connection: "Callable[[], IsaacConnection]"
) -> None:

    @mcp.tool("create_camera")
    def create_camera(
        prim_path: str = "/World/Camera",
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        orientation: Optional[List[float]] = None,
        resolution: Optional[List[int]] = None,
        focal_length: Optional[float] = None,
        focus_distance: Optional[float] = None,
        horizontal_aperture: Optional[float] = None,
        vertical_aperture: Optional[float] = None,
        clipping_range: Optional[List[float]] = None,
    ) -> str:
        """Add a camera sensor to the scene.

        Args:
            prim_path: Prim path for the camera.
            position: [x, y, z] world position.
            rotation: [rx, ry, rz] rotation in degrees.
            orientation: [w, x, y, z] quaternion. Mutually exclusive with rotation.
            resolution: [width, height] image resolution. Default 1280x720.
            focal_length: Perspective focal length in stage camera units.
            focus_distance: Distance to the focus plane.
            horizontal_aperture: Horizontal sensor aperture.
            vertical_aperture: Vertical sensor aperture.
            clipping_range: [near, far] clipping distances in stage units.
        """
        try:
            conn = get_connection()
            params = {"prim_path": prim_path}
            if position is not None:
                params["position"] = position
            if rotation is not None:
                params["rotation"] = rotation
            if orientation is not None:
                params["orientation"] = orientation
            if resolution is not None:
                params["resolution"] = resolution
            if focal_length is not None:
                params["focal_length"] = focal_length
            if focus_distance is not None:
                params["focus_distance"] = focus_distance
            if horizontal_aperture is not None:
                params["horizontal_aperture"] = horizontal_aperture
            if vertical_aperture is not None:
                params["vertical_aperture"] = vertical_aperture
            if clipping_range is not None:
                params["clipping_range"] = clipping_range
            result = conn.send_command("sensors.create_camera", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("capture_image")
    def capture_image(
        prim_path: str = "/World/Camera", output_path: Optional[str] = None
    ) -> str:
        """Capture an RGB image from a camera sensor.

        Args:
            prim_path: Prim path of the camera.
            output_path: File path to save the image. Returns metadata only if not set.
        """
        try:
            conn = get_connection()
            params = {"prim_path": prim_path}
            if output_path:
                params["output_path"] = output_path
            result = conn.send_command("sensors.capture_image", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("set_active_camera")
    def set_active_camera(prim_path: str = "/World/Camera") -> str:
        """Select a camera for the active interactive viewport.

        This changes the streamed viewer framing without changing any sensor's
        pose, optics, or captured observations.
        """
        try:
            conn = get_connection()
            result = conn.send_command(
                "sensors.set_active_camera", {"prim_path": prim_path}
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("create_lidar")
    def create_lidar(
        prim_path: str = "/World/Lidar",
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        config: Optional[str] = None,
    ) -> str:
        """Add a lidar sensor to the scene.

        Args:
            prim_path: Prim path for the lidar.
            position: [x, y, z] world position.
            rotation: [rx, ry, rz] rotation in degrees.
            config: Lidar configuration name (e.g. "Example_Rotary").
        """
        try:
            conn = get_connection()
            params = {"prim_path": prim_path}
            if position:
                params["position"] = position
            if rotation:
                params["rotation"] = rotation
            if config:
                params["config"] = config
            result = conn.send_command("sensors.create_lidar", params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool("get_lidar_point_cloud")
    def get_lidar_point_cloud(prim_path: str = "/World/Lidar") -> str:
        """Get point cloud data from a lidar sensor.

        Args:
            prim_path: Prim path of the lidar sensor.
        """
        try:
            conn = get_connection()
            result = conn.send_command(
                "sensors.get_point_cloud", {"prim_path": prim_path}
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
