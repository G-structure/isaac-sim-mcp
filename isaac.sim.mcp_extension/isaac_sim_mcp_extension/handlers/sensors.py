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

"""Sensor creation and data capture command handlers."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from ..adapters.base import IsaacAdapterBase


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["sensors.create_camera"] = lambda **p: create_camera(adapter, **p)
    registry["sensors.set_active_camera"] = lambda **p: set_active_camera(adapter, **p)
    registry["sensors.capture_image"] = lambda **p: capture_image(adapter, **p)
    registry["sensors.create_lidar"] = lambda **p: create_lidar(adapter, **p)
    registry["sensors.get_point_cloud"] = lambda **p: get_point_cloud(adapter, **p)


def create_camera(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/Camera",
    position: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    resolution: Optional[Sequence[int]] = None,
    focal_length: Optional[float] = None,
    focus_distance: Optional[float] = None,
    horizontal_aperture: Optional[float] = None,
    vertical_aperture: Optional[float] = None,
    clipping_range: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    try:
        if rotation is not None and orientation is not None:
            raise ValueError("rotation and orientation are mutually exclusive")
        position_value = _float_vector("position", position, 3)
        rotation_value = _float_vector("rotation", rotation, 3)
        orientation_value = _normalized_quaternion(orientation)
        resolution_value = _resolution(resolution)
        clipping_value = _clipping_range(clipping_range)
        focal_value = _positive_optional("focal_length", focal_length)
        focus_value = _positive_optional("focus_distance", focus_distance)
        horizontal_value = _positive_optional(
            "horizontal_aperture", horizontal_aperture
        )
        vertical_value = _positive_optional("vertical_aperture", vertical_aperture)
        adapter.create_camera(
            prim_path,
            resolution=resolution_value,
            position=position_value,
            rotation=rotation_value,
            orientation=orientation_value,
            focal_length=focal_value,
            focus_distance=focus_value,
            horizontal_aperture=horizontal_value,
            vertical_aperture=vertical_value,
            clipping_range=clipping_value,
        )
        return {
            "status": "success",
            "message": f"Camera created at {prim_path}",
            "prim_path": prim_path,
            "resolution": list(resolution_value),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def set_active_camera(
    adapter: IsaacAdapterBase, prim_path: str = "/World/Camera"
) -> Dict[str, Any]:
    try:
        return {
            "status": "success",
            "message": f"Active viewport camera set to {prim_path}",
            **adapter.set_active_camera(prim_path),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _float_vector(
    name: str, value: Optional[Sequence[float]], size: int
) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    if len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _normalized_quaternion(
    value: Optional[Sequence[float]],
) -> Optional[tuple[float, float, float, float]]:
    converted = _float_vector("orientation", value, 4)
    if converted is None:
        return None
    norm = math.sqrt(sum(item * item for item in converted))
    if norm <= 1e-12:
        raise ValueError("orientation quaternion must not be zero")
    return tuple(item / norm for item in converted)


def _resolution(value: Optional[Sequence[int]]) -> tuple[int, int]:
    if value is None:
        return (1280, 720)
    if len(value) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value
    ):
        raise ValueError("resolution must contain two positive integers")
    return (value[0], value[1])


def _positive_optional(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _clipping_range(
    value: Optional[Sequence[float]],
) -> Optional[tuple[float, float]]:
    converted = _float_vector("clipping_range", value, 2)
    if converted is None:
        return None
    near, far = converted
    if near <= 0 or far <= near:
        raise ValueError("clipping_range must satisfy 0 < near < far")
    return (near, far)


def capture_image(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/Camera",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        image_data = adapter.capture_camera_image(prim_path)
        if output_path:
            from pathlib import Path

            from PIL import Image

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(image_data)
            img.save(output_path)
            return {
                "status": "success",
                "message": f"Image saved to {output_path}",
                "output_path": output_path,
            }
        return {
            "status": "success",
            "message": "Image captured",
            "shape": list(image_data.shape) if hasattr(image_data, "shape") else None,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_lidar(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/Lidar",
    position: Optional[Sequence[float]] = None,
    rotation: Optional[Sequence[float]] = None,
    config: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        adapter.create_lidar(prim_path, config=config)
        if position or rotation:
            adapter.set_prim_transform(prim_path, position=position, rotation=rotation)
        return {
            "status": "success",
            "message": f"Lidar created at {prim_path}",
            "prim_path": prim_path,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_point_cloud(
    adapter: IsaacAdapterBase, prim_path: str = "/World/Lidar"
) -> Dict[str, Any]:
    try:
        pc = adapter.get_lidar_point_cloud(prim_path)
        point_count = len(pc) if pc is not None else 0
        return {
            "status": "success",
            "message": f"Got {point_count} points",
            "point_count": point_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
