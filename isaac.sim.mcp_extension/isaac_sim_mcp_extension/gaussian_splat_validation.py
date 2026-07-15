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

"""Representation-neutral validation for Gaussian-splat USD subtrees.

The public ``inspect_root`` boundary turns a composed USD subtree and renderer
snapshot into deterministic, JSON-safe schema evidence. It intentionally does
not import Kit or mutate renderer state, so the same validation can be tested
without an Isaac process and replaced independently of MCP orchestration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit


PARTICLE_FIELD_TYPE = "ParticleField3DGaussianSplat"
_LEGACY_VOLUME_TYPE = "Volume"
_LEGACY_FIELD_TYPE = "OmniNuRecFieldAsset"
_MAX_VALUE_SAMPLES = 2048
_PARTICLE_ATTRIBUTES = {
    "positions": ("positions", "positionsh"),
    "orientations": ("orientations", "orientationsh"),
    "scales": ("scales", "scalesh"),
    "opacities": ("opacities", "opacitiesh"),
    "sh_coefficients": (
        "radiance:sphericalHarmonicsCoefficients",
        "radiance:sphericalHarmonicsCoefficientsh",
    ),
}


def inspect_root(root: Any, *, renderer: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect standard ParticleField and legacy NuRec representations."""
    prims = list(walk_prims(root))
    particle_fields = [
        _validate_particle_field(prim)
        for prim in prims
        if str(prim.GetTypeName()) == PARTICLE_FIELD_TYPE
    ]
    legacy_volumes = [
        _validate_legacy_volume(prim) for prim in prims if _is_legacy_nurec_volume(prim)
    ]
    spg = _inspect_spg(prims, renderer)

    representations = []
    if particle_fields:
        representations.append("particle_field_3d_gaussian_splat")
    if legacy_volumes:
        representations.append("legacy_nurec_volume")

    errors: list[str] = []
    fidelity_errors: list[str] = []
    fidelity_limitations: list[str] = []
    warnings: list[str] = []
    for report in [*particle_fields, *legacy_volumes]:
        errors.extend(report["errors"])
        fidelity_errors.extend(report.get("fidelity_errors", []))
        fidelity_limitations.extend(report.get("fidelity_limitations", []))
        warnings.extend(report["warnings"])
    errors.extend(spg["errors"])
    warnings.extend(spg["warnings"])

    if not representations:
        errors.append(
            "No ParticleField3DGaussianSplat or legacy NuRec Volume exists under the target prim"
        )
    schema_errors = list(errors)
    errors.extend(spg["runtime_errors"])
    extensions = renderer.get("extensions", {})
    if representations and extensions.get("omni.hydra.rtx") is not True:
        errors.append(
            "omni.hydra.rtx is not confirmed enabled; RTX Gaussian rendering is unavailable"
        )
    if particle_fields and renderer.get("particle_field_schema_available") is False:
        errors.append(
            "The runtime OpenUSD build does not expose ParticleField3DGaussianSplat"
        )

    settings = renderer.get("settings", {})
    if particle_fields and settings.get("/app/useFabricSceneDelegate") is not True:
        warnings.append(
            "Fabric Scene Delegate is not explicitly enabled; Kit 110.1 is the supported ParticleField path"
        )
    if representations and settings.get("/renderer/multiGpu/enabled") is True:
        errors.append(
            "Multi-GPU rendering is enabled; Gaussian rendering requires /renderer/multiGpu/enabled=false"
        )
    elif representations and settings.get("/renderer/multiGpu/enabled") is not False:
        warnings.append(
            "Multi-GPU rendering is not explicitly disabled; single-GPU Gaussian sessions are the tested path"
        )
    if (
        spg["present"]
        and settings.get("/rtx/rtpt/gaussian/skipTonemapping/enabled") is not False
    ):
        warnings.append(
            "SPG/PPISP is present but Gaussian skip-tonemapping is not explicitly false; output color may diverge from the asset"
        )
    if (
        spg["present"]
        and settings.get("/omni/rtx/nre/compositing/disableNuRecPostProcessings")
        is not True
    ):
        errors.append(
            "SPG/PPISP is present but NuRec post-processing is not disabled; PPISP output would be processed twice"
        )

    render_prim_paths = [
        report["prim_path"] for report in [*particle_fields, *legacy_volumes]
    ]
    render_prims = [
        {
            "prim_path": report["prim_path"],
            "authored_visibility": report["authored_visibility"],
        }
        for report in [*particle_fields, *legacy_volumes]
    ]
    return {
        "representations": representations,
        "render_prim_paths": render_prim_paths,
        "render_prims": render_prims,
        "particle_fields": particle_fields,
        "legacy_volumes": legacy_volumes,
        "spg": spg,
        "prim_count": len(prims),
        "schema_errors": _unique(schema_errors),
        "fidelity_errors": _unique(fidelity_errors),
        "fidelity_limitations": _unique(fidelity_limitations),
        "errors": _unique(errors),
        "warnings": _unique(warnings),
    }


def has_spg(root: Any) -> bool:
    """Return whether a subtree authors an SPG source asset."""
    return any(
        attribute_value(prim, "info:spg:sourceAsset") is not None
        for prim in walk_prims(root)
    )


def redacted_source(value: str) -> str:
    """Remove query strings and fragments before returning an asset identifier."""
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme:
            safe_netloc = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
    except Exception:
        pass
    return str(value).split("?", 1)[0].split("#", 1)[0]


def _validate_particle_field(prim: Any) -> Dict[str, Any]:
    path = str(prim.GetPath())
    errors: list[str] = []
    fidelity_errors: list[str] = []
    warnings: list[str] = []
    values: Dict[str, Any] = {}
    attributes: Dict[str, Dict[str, Any]] = {}

    for logical_name, aliases in _PARTICLE_ATTRIBUTES.items():
        attr, value = _first_nonempty_authored_attribute(prim, aliases)
        if attr is None or value is None:
            if logical_name == "positions":
                errors.append(
                    f"{path}: required Gaussian attribute {logical_name} is not authored"
                )
            elif logical_name == "opacities":
                fidelity_errors.append(
                    f"{path}: opacities are not authored; source opacity cannot be proven"
                )
                warnings.append(
                    f"{path}: opacities are not authored; OpenUSD uses fully opaque particles"
                )
            attributes[logical_name] = {"name": None, "count": None}
            continue
        try:
            count = len(value)
        except TypeError:
            count = None
            errors.append(f"{path}: {logical_name} is not an array attribute")
        attributes[logical_name] = {
            "name": str(attr.GetName()),
            "type": str(attr.GetTypeName()),
            "count": count,
        }
        if count is not None:
            values[logical_name] = value

    positions = values.get("positions")
    particle_count = len(positions) if positions is not None else 0
    if particle_count <= 0:
        errors.append(f"{path}: positions must contain at least one Gaussian")
    for name in ("orientations", "scales", "opacities"):
        value = values.get(name)
        if value is not None and len(value) != particle_count:
            errors.append(
                f"{path}: {name} count {len(value)} does not match positions count {particle_count}"
            )

    degree_attr, degree_value = _first_authored_attribute(
        prim, ("radiance:sphericalHarmonicsDegree",)
    )
    degree: Optional[int] = 3
    degree_source = "schema_fallback"
    if degree_attr is None or degree_value is None:
        if values.get("sh_coefficients") is not None:
            warnings.append(
                f"{path}: spherical harmonics degree is not authored; schema fallback 3 was used"
            )
    else:
        degree_source = "authored"
        try:
            degree = int(degree_value)
            if degree not in (0, 1, 2, 3):
                errors.append(
                    f"{path}: spherical harmonics degree {degree} is outside 0..3"
                )
        except (TypeError, ValueError):
            errors.append(f"{path}: spherical harmonics degree is not an integer")

    coefficient_attr, coefficients = _first_nonempty_authored_attribute(
        prim, _PARTICLE_ATTRIBUTES["sh_coefficients"]
    )
    element_size = None
    interpolation = None
    if coefficient_attr is not None:
        element_size = coefficient_attr.GetMetadata("elementSize")
        interpolation = coefficient_attr.GetMetadata("interpolation")
    if coefficients is None:
        fidelity_errors.append(
            f"{path}: SH coefficients are not authored; source radiance cannot be proven"
        )
        warnings.append(
            f"{path}: SH coefficients are not authored; OpenUSD uses a degree-0 DC color of (0.5, 0.5, 0.5)"
        )
    elif degree is not None:
        expected_element_size = (degree + 1) ** 2
        expected_coefficients = particle_count * expected_element_size
        if len(coefficients) != expected_coefficients:
            errors.append(
                f"{path}: SH coefficient count {len(coefficients)} does not match "
                f"{particle_count} particles x {expected_element_size}"
            )
        if element_size is not None and element_size != expected_element_size:
            errors.append(
                f"{path}: SH elementSize {element_size!r} does not match degree {degree}"
            )
        elif element_size is None:
            warnings.append(
                f"{path}: SH elementSize metadata is not explicit; coefficient count was validated from degree"
            )
        if interpolation is not None and str(interpolation) != "vertex":
            errors.append(
                f"{path}: SH coefficient interpolation must be vertex, got {interpolation}"
            )
        elif interpolation is None:
            warnings.append(
                f"{path}: SH coefficient interpolation metadata is not explicit"
            )

    _validate_numeric_samples(
        path,
        positions=values.get("positions"),
        orientations=values.get("orientations"),
        scales=values.get("scales"),
        opacities=values.get("opacities"),
        coefficients=values.get("sh_coefficients"),
        errors=errors,
    )
    extent = _extent_report(prim, path, errors)
    return {
        "prim_path": path,
        "particle_count": particle_count,
        "sh_degree": degree,
        "sh_degree_source": degree_source,
        "sh_element_size": element_size,
        "sh_interpolation": str(interpolation) if interpolation is not None else None,
        "attributes": attributes,
        "extent": extent,
        "projection_mode_hint": attribute_value(prim, "projectionModeHint"),
        "sorting_mode_hint": attribute_value(prim, "sortingModeHint"),
        "authored_visibility": _authored_visibility(prim),
        "sampled_values_per_attribute": min(particle_count, _MAX_VALUE_SAMPLES),
        "errors": _unique(errors),
        "fidelity_errors": _unique(fidelity_errors),
        "fidelity_limitations": [],
        "warnings": _unique(warnings),
    }


def _validate_legacy_volume(prim: Any) -> Dict[str, Any]:
    path = str(prim.GetPath())
    errors: list[str] = []
    warnings: list[str] = []
    fields = []
    roles = set()
    for child in walk_prims(prim):
        if child is prim or str(child.GetTypeName()) != _LEGACY_FIELD_TYPE:
            continue
        raw_path = attribute_value(child, "filePath")
        asset_path, resolved_path = _asset_paths(raw_path)
        role = attribute_value(child, "fieldRole")
        if role is not None:
            roles.add(str(role))
        if not asset_path:
            errors.append(f"{child.GetPath()}: NuRec field has no filePath")
        elif not resolved_path:
            warnings.append(
                f"{child.GetPath()}: NuRec sidecar {asset_path} has no resolved path yet"
            )
        fields.append(
            {
                "prim_path": str(child.GetPath()),
                "field_role": str(role) if role is not None else None,
                "asset_path": redacted_source(asset_path) if asset_path else None,
                "resolved": bool(resolved_path),
            }
        )
    for required_role in ("density", "emissiveColor"):
        if required_role not in roles:
            errors.append(f"{path}: NuRec Volume has no {required_role} field asset")
    extent = _extent_report(prim, path, errors)
    return {
        "prim_path": path,
        "particle_count": None,
        "particle_count_reason": "legacy .nurec payload is opaque to generic USD inspection",
        "field_assets": fields,
        "extent": extent,
        "authored_visibility": _authored_visibility(prim),
        "errors": _unique(errors),
        "fidelity_errors": [],
        "fidelity_limitations": [
            f"{path}: legacy .nurec payload content is opaque to generic USD inspection"
        ],
        "warnings": _unique(warnings),
    }


def _inspect_spg(prims: Iterable[Any], renderer: Dict[str, Any]) -> Dict[str, Any]:
    sources = []
    errors: list[str] = []
    warnings: list[str] = []
    for prim in prims:
        value = attribute_value(prim, "info:spg:sourceAsset")
        if value is None:
            continue
        asset_path, resolved_path = _asset_paths(value)
        sub_identifier = attribute_value(prim, "info:spg:sourceAsset:subIdentifier")
        if not asset_path:
            errors.append(f"{prim.GetPath()}: SPG shader has no source asset")
        elif not resolved_path:
            warnings.append(
                f"{prim.GetPath()}: SPG source {asset_path} has no resolved path yet"
            )
        sources.append(
            {
                "prim_path": str(prim.GetPath()),
                "asset_path": redacted_source(asset_path) if asset_path else None,
                "sub_identifier": (
                    str(sub_identifier) if sub_identifier is not None else None
                ),
                "resolved": bool(resolved_path),
            }
        )
    enabled = renderer.get("extensions", {}).get("omni.rtx.spg")
    runtime_errors = []
    if sources and enabled is not True:
        runtime_errors.append("SPG shaders are present but omni.rtx.spg is not enabled")
    return {
        "present": bool(sources),
        "extension_enabled": enabled,
        "sources": sources,
        "errors": _unique(errors),
        "runtime_errors": runtime_errors,
        "warnings": _unique(warnings),
    }


def _validate_numeric_samples(
    path: str,
    *,
    positions: Any,
    orientations: Any,
    scales: Any,
    opacities: Any,
    coefficients: Any,
    errors: list[str],
) -> None:
    for value in _sample_values(positions):
        components = _components(value)
        if len(components) != 3 or not _all_finite(components):
            errors.append(f"{path}: sampled positions must be finite 3D vectors")
            break
    for value in _sample_values(scales):
        components = _components(value)
        if (
            len(components) != 3
            or not _all_finite(components)
            or any(component <= 0 for component in components)
        ):
            errors.append(f"{path}: sampled scales must be finite positive 3D vectors")
            break
    for value in _sample_values(opacities):
        components = _components(value)
        if (
            len(components) != 1
            or not _all_finite(components)
            or any(component < 0 or component > 1 for component in components)
        ):
            errors.append(f"{path}: sampled opacities must be finite and within [0, 1]")
            break
    for value in _sample_values(orientations):
        components = _quaternion_components(value)
        norm = math.sqrt(sum(component * component for component in components))
        if (
            len(components) != 4
            or not _all_finite(components)
            or not math.isclose(norm, 1.0, rel_tol=0.02, abs_tol=0.02)
        ):
            errors.append(
                f"{path}: sampled orientations must be finite unit quaternions"
            )
            break
    for value in _sample_values(coefficients):
        components = _components(value)
        if len(components) != 3 or not _all_finite(components):
            errors.append(f"{path}: sampled SH coefficients must be finite 3D vectors")
            break


def _extent_report(prim: Any, path: str, errors: list[str]) -> Optional[Dict[str, Any]]:
    extent = attribute_value(prim, "extent")
    try:
        extent_count = len(extent) if extent is not None else 0
    except TypeError:
        extent_count = 0
    if extent_count != 2:
        errors.append(f"{path}: extent must contain minimum and maximum bounds")
        return None
    minimum = _components(extent[0])
    maximum = _components(extent[1])
    if (
        len(minimum) != 3
        or len(maximum) != 3
        or not _all_finite([*minimum, *maximum])
        or any(low > high for low, high in zip(minimum, maximum))
    ):
        errors.append(f"{path}: extent bounds are invalid")
        return None
    return {"minimum": minimum, "maximum": maximum}


def _is_legacy_nurec_volume(prim: Any) -> bool:
    if str(prim.GetTypeName()) != _LEGACY_VOLUME_TYPE:
        return False
    return attribute_value(prim, "omni:nurec:isNuRecVolume") is True


def walk_prims(root: Any) -> Iterable[Any]:
    pending = [root]
    while pending:
        prim = pending.pop(0)
        yield prim
        pending.extend(list(prim.GetChildren()))


def _first_authored_attribute(prim: Any, names: Sequence[str]) -> tuple[Any, Any]:
    for name in names:
        attr = prim.GetAttribute(name)
        if not attr:
            continue
        has_authored = getattr(attr, "HasAuthoredValueOpinion", None)
        if callable(has_authored) and not has_authored():
            continue
        value = attr.Get()
        if value is not None:
            return attr, value
    return None, None


def _first_nonempty_authored_attribute(
    prim: Any, names: Sequence[str]
) -> tuple[Any, Any]:
    """Prefer an authored non-empty float array, then its half alias.

    OpenUSD's ParticleField APIs select float storage only when it contains at
    least one value. Preserve the first authored empty array as a diagnostic
    fallback when no alias contains data.
    """
    fallback: tuple[Any, Any] = (None, None)
    for name in names:
        attr = prim.GetAttribute(name)
        if not attr:
            continue
        has_authored = getattr(attr, "HasAuthoredValueOpinion", None)
        if callable(has_authored) and not has_authored():
            continue
        value = attr.Get()
        if value is None:
            continue
        if fallback[0] is None:
            fallback = (attr, value)
        try:
            if len(value) > 0:
                return attr, value
        except TypeError:
            return attr, value
    return fallback


def attribute_value(prim: Any, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr:
        return None
    return attr.Get()


def _authored_visibility(prim: Any) -> str:
    value = attribute_value(prim, "visibility")
    return str(value) if value is not None else "inherited"


def _asset_paths(value: Any) -> tuple[str, str]:
    if value is None:
        return "", ""
    raw = getattr(value, "path", None)
    resolved = getattr(value, "resolvedPath", None)
    return str(raw if raw is not None else value), str(resolved or "")


def _sample_values(values: Any) -> Iterable[Any]:
    if values is None:
        return ()
    count = len(values)
    if count <= _MAX_VALUE_SAMPLES:
        return values
    step = max(1, math.ceil(count / _MAX_VALUE_SAMPLES))
    return (values[index] for index in range(0, count, step))


def _components(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return [math.nan]


def _quaternion_components(value: Any) -> list[float]:
    get_real = getattr(value, "GetReal", None)
    get_imaginary = getattr(value, "GetImaginary", None)
    if callable(get_real) and callable(get_imaginary):
        return [float(get_real()), *_components(get_imaginary())]
    return _components(value)


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
