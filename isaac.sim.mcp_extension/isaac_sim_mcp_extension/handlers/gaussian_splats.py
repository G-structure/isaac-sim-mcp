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

"""Bounded runtime proof for Gaussian-splat USD assets.

This module owns one invariant: a caller only receives ``ready=True`` after the
referenced subtree contains a recognized Gaussian representation, its runtime
arrays agree and bounded numeric validation passes, stage loading has settled,
and the required RTX renderer components are available. The checks are
deliberately generic USD inspection so legacy NuRec assets remain diagnosable
after the standard ``ParticleField3DGaussianSplat`` schema becomes the default.
"""

from __future__ import annotations

import math
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlsplit

from ..adapters.base import IsaacAdapterBase
from ..gaussian_splat_validation import (
    PARTICLE_FIELD_TYPE as _STANDARD_TYPE,
    has_spg,
    inspect_root as _inspect_root,
    redacted_source as _display_source,
)


_SUPPORTED_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_MAX_WAIT_SECONDS = 60.0
_MAX_WARMUP_FRAMES = 120
_MAX_SAMPLE_FRAMES = 240


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["assets.load_gaussian_splat"] = lambda **p: load_gaussian_splat(
        adapter, **p
    )
    registry["assets.inspect_gaussian_splat"] = lambda **p: inspect_gaussian_splat(
        adapter, **p
    )


def load_gaussian_splat(
    adapter: IsaacAdapterBase,
    usd_path: str,
    prim_path: str = "/World/GaussianSplat",
    position: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    frame_camera: bool = True,
    wait_timeout_seconds: float = 20.0,
    warmup_frames: int = 8,
    sample_frames: int = 30,
) -> Dict[str, Any]:
    """Reference and prove a Gaussian-splat USD/USDZ asset in the open stage.

    The new reference is rolled back when it composes no supported Gaussian
    representation. A recognized but not-yet-ready asset is retained so its
    structured diagnostics can guide a renderer or asset repair.
    """
    created_prim = False
    stage = None
    safe_source = _display_source(usd_path)
    try:
        source = _validated_source(usd_path)
        target_path = _validated_prim_path(prim_path)
        position_value = _finite_vector("position", position, 3)
        scale_value = _positive_vector("scale", scale, 3)
        wait_seconds = _bounded_float(
            "wait_timeout_seconds",
            wait_timeout_seconds,
            minimum=0.1,
            maximum=_MAX_WAIT_SECONDS,
        )
        warmup_count = _bounded_int(
            "warmup_frames", warmup_frames, minimum=0, maximum=_MAX_WARMUP_FRAMES
        )
        sample_count = _bounded_int(
            "sample_frames", sample_frames, minimum=0, maximum=_MAX_SAMPLE_FRAMES
        )
        frame_camera_value = _validated_bool("frame_camera", frame_camera)

        from pxr import Sdf

        stage = adapter.get_stage()
        existing = stage.GetPrimAtPath(target_path)
        if existing.IsValid():
            return {
                "status": "error",
                "message": (
                    f"Prim already exists at {target_path}; choose a new prim_path "
                    "or remove the existing prim explicitly"
                ),
            }

        _author_asset_reference(stage, target_path, source)
        created_prim = True
        if position_value is not None or scale_value is not None:
            adapter.set_prim_transform(
                target_path,
                position=position_value,
                scale=scale_value,
            )
        stage.Load(Sdf.Path(target_path))

        evidence = _runtime_evidence(
            stage=stage,
            root_path=target_path,
            configure_renderer=True,
            frame_camera=frame_camera_value,
            wait_timeout_seconds=wait_seconds,
            warmup_frames=warmup_count,
            sample_frames=sample_count,
        )
        if not evidence["asset"]["representations"]:
            stage.RemovePrim(target_path)
            created_prim = False
            return {
                "status": "error",
                "message": (
                    f"Referenced asset {safe_source} contained no "
                    "ParticleField3DGaussianSplat or legacy NuRec Volume; "
                    f"the reference at {target_path} was rolled back"
                ),
            }

        ready = bool(evidence["render_path_ready"])
        fidelity_ready = bool(evidence["fidelity_ready"])
        if fidelity_ready:
            message = (
                f"Gaussian asset render path and authored source-fidelity checks "
                f"are ready at {target_path}"
            )
        elif ready:
            message = (
                f"Gaussian asset render path is ready at {target_path}, but "
                "authored source fidelity is not proven"
            )
        else:
            message = f"Gaussian asset loaded at {target_path} but is not render-ready"
        return {
            "status": "success",
            "message": message,
            "ready": ready,
            "source": safe_source,
            "prim_path": target_path,
            **evidence,
        }
    except Exception as exc:
        if created_prim and stage is not None:
            try:
                stage.RemovePrim(prim_path)
            except Exception:
                pass
        error = _redacted_error(exc, str(usd_path), safe_source)
        return {
            "status": "error",
            "message": f"Could not load Gaussian asset {safe_source}: {error}",
        }


def inspect_gaussian_splat(
    adapter: IsaacAdapterBase,
    prim_path: str = "/World/GaussianSplat",
    frame_camera: bool = False,
    wait_timeout_seconds: float = 10.0,
    warmup_frames: int = 4,
    sample_frames: int = 30,
) -> Dict[str, Any]:
    """Re-run bounded render-readiness proof for an existing stage subtree."""
    try:
        target_path = _validated_prim_path(prim_path)
        wait_seconds = _bounded_float(
            "wait_timeout_seconds",
            wait_timeout_seconds,
            minimum=0.1,
            maximum=_MAX_WAIT_SECONDS,
        )
        warmup_count = _bounded_int(
            "warmup_frames", warmup_frames, minimum=0, maximum=_MAX_WARMUP_FRAMES
        )
        sample_count = _bounded_int(
            "sample_frames", sample_frames, minimum=0, maximum=_MAX_SAMPLE_FRAMES
        )
        frame_camera_value = _validated_bool("frame_camera", frame_camera)
        stage = adapter.get_stage()
        root = stage.GetPrimAtPath(target_path)
        if not root.IsValid():
            return {"status": "error", "message": f"Prim not found: {target_path}"}

        evidence = _runtime_evidence(
            stage=stage,
            root_path=target_path,
            configure_renderer=False,
            frame_camera=frame_camera_value,
            wait_timeout_seconds=wait_seconds,
            warmup_frames=warmup_count,
            sample_frames=sample_count,
        )
        ready = bool(evidence["render_path_ready"])
        fidelity_ready = bool(evidence["fidelity_ready"])
        if fidelity_ready:
            message = (
                f"Gaussian asset render path and authored source-fidelity checks "
                f"are ready at {target_path}"
            )
        elif ready:
            message = (
                f"Gaussian asset render path is ready at {target_path}, but "
                "authored source fidelity is not proven"
            )
        else:
            message = f"Gaussian asset at {target_path} is not render-ready"
        return {
            "status": "success",
            "message": message,
            "ready": ready,
            "prim_path": target_path,
            **evidence,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Could not inspect Gaussian asset at {prim_path}: {exc}",
        }


def _author_asset_reference(stage: Any, target_path: str, source: str) -> Any:
    """Compose a referenced default prim without overriding its schema type."""
    root = stage.OverridePrim(target_path)
    if not root.GetReferences().AddReference(source):
        stage.RemovePrim(target_path)
        raise RuntimeError("USD reference could not be authored")
    return root


def _runtime_evidence(
    *,
    stage: Any,
    root_path: str,
    configure_renderer: bool,
    frame_camera: bool,
    wait_timeout_seconds: float,
    warmup_frames: int,
    sample_frames: int,
) -> Dict[str, Any]:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    context = omni.usd.get_context()
    root = stage.GetPrimAtPath(root_path)
    configuration = (
        _configure_renderer_for_asset(app, stage, root)
        if configure_renderer
        else {"attempted": False, "reason": "inspection is read-only"}
    )
    # NuRec/SPG settings are sampled at the first Hydra sync for the stage.
    # Keep this wait, and therefore its first app.update(), after setup.
    stage_loading = _wait_for_stage(
        app,
        context,
        timeout_seconds=wait_timeout_seconds,
    )
    root = stage.GetPrimAtPath(root_path)
    renderer = _renderer_evidence(app)
    renderer["configuration"] = configuration
    asset = _inspect_root(root, renderer=renderer)
    errors = list(asset["errors"])
    warnings = list(asset["warnings"])
    errors.extend(configuration.get("errors", []))
    if not stage_loading["complete"]:
        errors.append(
            "Stage assets did not finish loading before the bounded timeout; retry inspection"
        )

    camera = _frame_viewport(root_path) if frame_camera else {"requested": False}
    if frame_camera and not camera.get("framed", False):
        warnings.append(
            f"Viewport framing failed: {camera.get('error', 'unknown error')}"
        )

    performance = _measure_update_loop(
        app,
        warmup_frames=warmup_frames,
        sample_frames=sample_frames,
    )
    gpu = _gpu_evidence()
    readiness = _readiness_flags(asset, runtime_errors=errors)
    return {
        # ``ready`` is retained for the existing MCP convention, but its scope
        # is explicit: only a CUA/viewport comparison can prove final pixels.
        "ready": readiness["render_path_ready"],
        **readiness,
        "readiness_scope": "usd_schema_stage_loading_and_renderer_components",
        "fidelity_scope": "authored_particle_field_radiance_and_opacity",
        "pixel_proof": {
            "attempted": False,
            "passed": None,
            "reason": (
                "Schema inspection and Kit timing cannot prove visible pixels; "
                "compare a viewport/CUA frame with the reported render prims visible and hidden"
            ),
            "render_prim_paths": asset["render_prim_paths"],
        },
        "stage_loading": stage_loading,
        "asset": {**asset, "errors": errors, "warnings": warnings},
        "renderer": renderer,
        "camera": camera,
        "performance": performance,
        "gpu": gpu,
    }


def _readiness_flags(
    asset: Dict[str, Any], *, runtime_errors: Sequence[str]
) -> Dict[str, bool]:
    """Separate standards renderability from authored source-fidelity proof."""
    schema_ready = not asset["schema_errors"]
    render_path_ready = not runtime_errors
    fidelity_ready = bool(
        render_path_ready
        and not asset["fidelity_errors"]
        and not asset["fidelity_limitations"]
    )
    return {
        "schema_ready": schema_ready,
        "render_path_ready": render_path_ready,
        "fidelity_ready": fidelity_ready,
    }


def _renderer_evidence(app: Any) -> Dict[str, Any]:
    extension_names = (
        "omni.hydra.rtx",
        "omni.rtx.spg",
        "omni.usd.schema.usd_particle_field",
        "omni.kit.converter.gsplat",
        "isaacsim.replicator.nurec_utils",
    )
    extensions = {}
    try:
        extension_manager = app.get_extension_manager()
    except Exception:
        extension_manager = None
    for name in extension_names:
        if extension_manager is None:
            extensions[name] = None
            continue
        try:
            extensions[name] = bool(extension_manager.is_extension_enabled(name))
        except Exception:
            extensions[name] = None

    settings_paths = (
        "/app/useFabricSceneDelegate",
        "/renderer/multiGpu/enabled",
        "/rtx/rtpt/gaussian/skipTonemapping/enabled",
        "/rtx/spg/enabled",
        "/omni/rtx/nre/compositing/disableNuRecPostProcessings",
    )
    try:
        import carb

        settings_interface = carb.settings.get_settings()
        settings = {path: settings_interface.get(path) for path in settings_paths}
    except Exception:
        settings = {path: None for path in settings_paths}

    schema_evidence = _particle_field_schema_evidence()
    version_path = Path("/isaac-sim/VERSION")
    version = None
    try:
        if version_path.is_file():
            version = version_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return {
        "isaac_sim_version": version,
        **schema_evidence,
        "extensions": extensions,
        "settings": settings,
    }


def _particle_field_schema_evidence() -> Dict[str, bool]:
    """Probe the codeless schema registry independently of Python bindings.

    Isaac Sim 6.0.1 ships a pre-26.03 OpenUSD Python module and supplies the
    ParticleField schema through ``omni.usd.schema.usd_particle_field``. The
    registered concrete prim definition is therefore the render-path signal;
    ``pxr.UsdVol`` class generation is only diagnostic.
    """
    try:
        from pxr import Usd, UsdVol

        definition = Usd.SchemaRegistry().FindConcretePrimDefinition(_STANDARD_TYPE)
        schema_available = definition is not None
        python_binding_available = hasattr(UsdVol, _STANDARD_TYPE)
    except Exception:
        schema_available = False
        python_binding_available = False
    return {
        "particle_field_schema_available": schema_available,
        "particle_field_python_binding_available": python_binding_available,
    }


def _configure_renderer_for_asset(app: Any, stage: Any, root: Any) -> Dict[str, Any]:
    """Apply only settings required by representations present in ``root``.

    Every ParticleField/NuRec representation goes through the installed NuRec
    setup helper. Plain ParticleFields keep the engine tonemapping default;
    SPG/PPISP assets additionally require the launch-time SPG extension and its
    pre-Hydra overrides. Call this before the first ``app.update()`` after the
    asset is composed.
    """
    spg_present = has_spg(root)
    asset_mode = "spg_ppisp" if spg_present else "plain_gaussian"
    actions: list[str] = []
    errors: list[str] = []
    if spg_present:
        try:
            extension_manager = app.get_extension_manager()
            if not extension_manager.is_extension_enabled("omni.rtx.spg"):
                errors.append(
                    "omni.rtx.spg is not enabled; SPG must be enabled at process launch"
                )
        except Exception as exc:
            errors.append(f"could not inspect omni.rtx.spg launch state: {exc}")
    if errors:
        return {
            "attempted": True,
            "asset_mode": asset_mode,
            "actions": actions,
            "errors": errors,
        }
    try:
        from isaacsim.replicator.nurec_utils.rendering_setup import (
            setup_for_rendering,
        )

        success, nurec, utility_spg, problems = setup_for_rendering(stage)
        if not success:
            errors.extend(str(problem) for problem in problems)
        elif not nurec:
            errors.append(
                "NuRec utility did not classify the composed asset as a NuRec representation"
            )
        elif bool(utility_spg) != spg_present:
            errors.append("NuRec utility SPG classification disagrees with the asset")
        else:
            actions.append(
                "applied isaacsim.replicator.nurec_utils pre-Hydra renderer setup"
            )
    except Exception as exc:
        errors.append(f"could not apply NuRec pre-Hydra renderer setup: {exc}")
    return {
        "attempted": True,
        "asset_mode": asset_mode,
        "actions": actions,
        "errors": errors,
    }


def _wait_for_stage(
    app: Any,
    context: Any,
    *,
    timeout_seconds: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    updates = 0
    consecutive_complete = 0
    status = ("", 0, 0)
    streaming: Optional[bool] = None
    while time.perf_counter() - started < timeout_seconds:
        app.update()
        updates += 1
        try:
            status = context.get_stage_loading_status()
        except Exception:
            status = ("unavailable", 0, 0)
        try:
            streaming = bool(context.get_stage_streaming_status())
        except Exception:
            streaming = None
        _message, loaded, total = status
        complete = (total <= 0 or loaded >= total) and streaming is not True
        consecutive_complete = consecutive_complete + 1 if complete else 0
        if consecutive_complete >= 2:
            break
        time.sleep(0.005)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    message, loaded, total = status
    return {
        "complete": consecutive_complete >= 2,
        "message": str(message),
        "loaded_files": int(loaded),
        "total_files": int(total),
        "streaming": streaming,
        "updates": updates,
        "elapsed_ms": round(elapsed_ms, 3),
    }


def _frame_viewport(prim_path: str) -> Dict[str, Any]:
    try:
        import omni.kit.app
        from omni.kit.viewport.utility import (
            frame_viewport_prims,
            get_active_viewport,
        )

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport")
        framed = bool(frame_viewport_prims(viewport, prims=[prim_path]))
        omni.kit.app.get_app().update()
        return {
            "requested": True,
            "framed": framed,
            "camera_path": str(viewport.camera_path),
            "resolution": list(viewport.resolution),
        }
    except Exception as exc:
        return {"requested": True, "framed": False, "error": str(exc)}


def _measure_update_loop(
    app: Any,
    *,
    warmup_frames: int,
    sample_frames: int,
) -> Dict[str, Any]:
    for _ in range(warmup_frames):
        app.update()
    timings_ms = []
    for _ in range(sample_frames):
        started = time.perf_counter()
        app.update()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    if not timings_ms:
        return {
            "measurement": "kit_update_loop",
            "warmup_frames": warmup_frames,
            "sample_frames": 0,
            "measured": False,
        }
    ordered = sorted(timings_ms)
    average_ms = statistics.fmean(timings_ms)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    p99_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1)
    return {
        "measurement": "kit_update_loop",
        "note": "CPU-observed Kit update duration; pair with viewport/CUA evidence for visual correctness",
        "warmup_frames": warmup_frames,
        "sample_frames": sample_frames,
        "measured": True,
        "average_ms": round(average_ms, 3),
        "median_ms": round(statistics.median(timings_ms), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "p99_ms": round(ordered[p99_index], 3),
        "maximum_ms": round(max(timings_ms), 3),
        "estimated_updates_per_second": (
            round(1000.0 / average_ms, 3) if average_ms > 0 else None
        ),
    }


def _gpu_evidence() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        fields = [part.strip() for part in completed.stdout.strip().split(",")]
        if len(fields) != 5:
            raise RuntimeError("unexpected nvidia-smi result shape")
        memory_used_mib = int(fields[1])
        memory_free_mib = int(fields[2])
        memory_total_mib = int(fields[3])
        if memory_total_mib <= 0:
            raise RuntimeError("nvidia-smi reported non-positive total memory")
        return {
            "available": True,
            "name": fields[0],
            "memory_used_mib": memory_used_mib,
            "memory_free_mib": memory_free_mib,
            "memory_total_mib": memory_total_mib,
            "memory_used_ratio": round(memory_used_mib / memory_total_mib, 4),
            "utilization_percent": int(fields[4]),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _validated_source(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("usd_path is required")
    source = value.strip()
    parsed = urlsplit(source)
    if parsed.scheme not in ("", "file", "https", "http", "omniverse"):
        raise ValueError(
            "usd_path must be an absolute local path or an http(s), file, or omniverse URL"
        )
    if parsed.scheme == "" and not Path(source).is_absolute():
        raise ValueError("local usd_path must be absolute")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in _SUPPORTED_USD_SUFFIXES:
        raise ValueError("usd_path must end in .usd, .usda, .usdc, or .usdz")
    return source


def _validated_prim_path(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/World/"):
        raise ValueError("prim_path must be an absolute child of /World")
    if "//" in value or value.endswith("/"):
        raise ValueError("prim_path is not a valid USD prim path")
    return value


def _finite_vector(
    name: str, value: Optional[Sequence[float]], size: int
) -> Optional[tuple[float, ...]]:
    if value is None:
        return None
    if len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    converted = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _positive_vector(
    name: str, value: Optional[Sequence[float]], size: int
) -> Optional[tuple[float, ...]]:
    converted = _finite_vector(name, value, size)
    if converted is not None and any(component <= 0 for component in converted):
        raise ValueError(f"{name} must contain only positive values")
    return converted


def _bounded_float(name: str, value: float, *, minimum: float, maximum: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not minimum <= converted <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return converted


def _bounded_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validated_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _redacted_error(exc: Exception, sensitive_value: str, replacement: str) -> str:
    message = str(exc).replace(sensitive_value, replacement)

    def redact_url(match: re.Match[str]) -> str:
        return _display_source(match.group(0))

    return re.sub(r"(?:https?|omniverse)://[^\s'\"<>]+", redact_url, message)
