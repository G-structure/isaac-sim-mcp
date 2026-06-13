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

"""Viewport media capture and artifact download handlers."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


WORKSPACE_ROOT = Path("/data/workspace").resolve()
MEDIA_ROOT = WORKSPACE_ROOT / "media"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


def register(registry: Dict[str, Any], adapter: Any) -> None:
    registry["media.capture_video"] = lambda **p: capture_video(**p)
    registry["media.download_artifact"] = lambda **p: download_artifact(**p)


def _workspace_path(path: str | None, default_name: str) -> Path:
    raw = path or str(MEDIA_ROOT / default_name)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = MEDIA_ROOT / candidate
    resolved = candidate.resolve()
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        raise ValueError(f"path must be under {WORKSPACE_ROOT}")
    return resolved


def capture_video(
    output_path: Optional[str] = None,
    duration_seconds: float = 5.0,
    frame_count: Optional[int] = None,
    fps: int = 24,
    width: Optional[int] = None,
    height: Optional[int] = None,
    camera_prim_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture the active viewport into an MP4 in /data/workspace/media."""
    try:
        import cv2
        import omni.kit.app
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        fps = max(1, min(int(fps), 60))
        total_frames = int(frame_count) if frame_count is not None else int(max(0.1, duration_seconds) * fps)
        total_frames = max(1, min(total_frames, 1800))
        default_name = f"viewport_{int(time.time())}.mp4"
        destination = _workspace_path(output_path, default_name)
        if destination.suffix.lower() != ".mp4":
            destination = destination.with_suffix(".mp4")
        destination.parent.mkdir(parents=True, exist_ok=True)

        frame_dir = destination.parent / f".{destination.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport")
        if camera_prim_path:
            try:
                viewport.camera_path = camera_prim_path
            except Exception:
                pass
        if width and height:
            try:
                viewport.resolution = (int(width), int(height))
            except Exception:
                pass

        app = omni.kit.app.get_app()
        frame_paths: list[Path] = []
        for index in range(total_frames):
            app.update()
            frame_path = frame_dir / f"frame_{index:05d}.png"
            capture_viewport_to_file(viewport, file_path=str(frame_path))
            deadline = time.time() + 5
            while time.time() < deadline:
                app.update()
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    break
                time.sleep(0.01)
            if not frame_path.exists() or frame_path.stat().st_size <= 0:
                raise RuntimeError(f"viewport capture did not produce frame {index}")
            frame_paths.append(frame_path)

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError("captured frame could not be read by OpenCV")
        actual_height, actual_width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (actual_width, actual_height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter failed to open")
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                if frame.shape[:2] != (actual_height, actual_width):
                    frame = cv2.resize(frame, (actual_width, actual_height))
                writer.write(frame)
        finally:
            writer.release()
            for frame_path in frame_paths:
                try:
                    frame_path.unlink()
                except Exception:
                    pass
            try:
                frame_dir.rmdir()
            except Exception:
                pass

        return {
            "status": "success",
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "fps": fps,
            "frames": len(frame_paths),
            "width": actual_width,
            "height": actual_height,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def download_artifact(path: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> Dict[str, Any]:
    """Return a workspace artifact as base64 bytes."""
    try:
        artifact = _workspace_path(path, "")
        if not artifact.is_file():
            return {"status": "error", "message": f"artifact not found: {artifact}"}
        size = artifact.stat().st_size
        effective_max = min(max(1, int(max_bytes)), MAX_DOWNLOAD_BYTES)
        if size > effective_max:
            return {
                "status": "error",
                "message": f"artifact is {size} bytes, above max_bytes={effective_max}",
                "path": str(artifact),
                "bytes": size,
            }
        payload = artifact.read_bytes()
        mime_type = mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
        return {
            "status": "success",
            "path": str(artifact),
            "bytes": size,
            "mime_type": mime_type,
            "encoding": "base64",
            "data": base64.b64encode(payload).decode("ascii"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
