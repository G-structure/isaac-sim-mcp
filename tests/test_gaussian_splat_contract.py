"""Contract tests for bounded Gaussian-splat runtime evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = REPO_ROOT / "isaac.sim.mcp_extension" / "isaac_sim_mcp_extension"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package, path in (
    ("isaac_sim_mcp_extension", EXTENSION_ROOT),
    ("isaac_sim_mcp_extension.adapters", EXTENSION_ROOT / "adapters"),
    ("isaac_sim_mcp_extension.handlers", EXTENSION_ROOT / "handlers"),
):
    module = types.ModuleType(package)
    module.__path__ = [str(path)]
    sys.modules.setdefault(package, module)

base_module = types.ModuleType("isaac_sim_mcp_extension.adapters.base")
base_module.IsaacAdapterBase = object
sys.modules.setdefault("isaac_sim_mcp_extension.adapters.base", base_module)
gaussian_splats = _load_module(
    "isaac_sim_mcp_extension.handlers.gaussian_splats_contract_test",
    EXTENSION_ROOT / "handlers" / "gaussian_splats.py",
)


class FakeAttribute:
    def __init__(
        self,
        name: str,
        value: Any,
        *,
        type_name: str = "unknown",
        metadata: dict[str, Any] | None = None,
        authored: bool = True,
    ) -> None:
        self.name = name
        self.value = value
        self.type_name = type_name
        self.metadata = metadata or {}
        self.authored = authored

    def Get(self) -> Any:
        return self.value

    def GetName(self) -> str:
        return self.name

    def GetTypeName(self) -> str:
        return self.type_name

    def GetMetadata(self, name: str) -> Any:
        return self.metadata.get(name)

    def HasAuthoredValueOpinion(self) -> bool:
        return self.authored


class FakePrim:
    def __init__(
        self,
        path: str,
        type_name: str,
        *,
        attributes: list[FakeAttribute] | None = None,
        children: list["FakePrim"] | None = None,
    ) -> None:
        self.path = path
        self.type_name = type_name
        self.attributes = {attr.name: attr for attr in attributes or []}
        self.children = children or []

    def GetPath(self) -> str:
        return self.path

    def GetTypeName(self) -> str:
        return self.type_name

    def GetAttribute(self, name: str) -> FakeAttribute | None:
        return self.attributes.get(name)

    def GetChildren(self) -> list["FakePrim"]:
        return self.children


class FakeAssetPath:
    def __init__(self, path: str, resolved_path: str = "") -> None:
        self.path = path
        self.resolvedPath = resolved_path


def _renderer(*, spg: bool = False, skip_tonemapping: bool = False) -> dict[str, Any]:
    return {
        "particle_field_schema_available": True,
        "extensions": {"omni.hydra.rtx": True, "omni.rtx.spg": spg},
        "settings": {
            "/app/useFabricSceneDelegate": True,
            "/renderer/multiGpu/enabled": False,
            "/rtx/rtpt/gaussian/skipTonemapping/enabled": skip_tonemapping,
        },
    }


def _valid_particle_field(*, scale_count: int = 2) -> FakePrim:
    attributes = [
        FakeAttribute(
            "positions", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], type_name="point3f[]"
        ),
        FakeAttribute(
            "orientations",
            [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            type_name="quatf[]",
        ),
        FakeAttribute(
            "scales",
            [(1.0, 1.0, 1.0)] * scale_count,
            type_name="float3[]",
        ),
        FakeAttribute("opacities", [0.5, 0.75], type_name="float[]"),
        FakeAttribute("radiance:sphericalHarmonicsDegree", 0, type_name="int"),
        FakeAttribute(
            "radiance:sphericalHarmonicsCoefficients",
            [(0.2, 0.3, 0.4), (0.5, 0.6, 0.7)],
            type_name="float3[]",
            metadata={"elementSize": 1, "interpolation": "vertex"},
        ),
        FakeAttribute(
            "extent", [(-0.5, -0.5, -0.5), (1.5, 0.5, 0.5)], type_name="float3[]"
        ),
    ]
    return FakePrim(
        "/World/GaussianSplat/Gaussians",
        "ParticleField3DGaussianSplat",
        attributes=attributes,
    )


def test_standard_particle_field_proves_counts_ranges_and_renderer() -> None:
    field = _valid_particle_field()
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["representations"] == ["particle_field_3d_gaussian_splat"]
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["particle_fields"][0]["particle_count"] == 2
    assert report["particle_fields"][0]["sh_element_size"] == 1
    assert report["particle_fields"][0]["extent"] == {
        "minimum": [-0.5, -0.5, -0.5],
        "maximum": [1.5, 0.5, 0.5],
    }
    assert gaussian_splats._readiness_flags(
        report, runtime_errors=report["errors"]
    ) == {
        "schema_ready": True,
        "render_path_ready": True,
        "fidelity_ready": True,
    }
    assert report["render_prims"] == [
        {
            "prim_path": "/World/GaussianSplat/Gaussians",
            "authored_visibility": "inherited",
        }
    ]


def test_standard_particle_field_fails_closed_on_array_mismatch() -> None:
    field = _valid_particle_field(scale_count=1)
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert any("scales count 1" in error for error in report["errors"])
    assert report["particle_fields"][0]["particle_count"] == 2
    assert (
        gaussian_splats._readiness_flags(report, runtime_errors=report["errors"])[
            "schema_ready"
        ]
        is False
    )


def test_standard_particle_field_metadata_is_diagnostic_not_required() -> None:
    field = _valid_particle_field()
    coefficients = field.attributes["radiance:sphericalHarmonicsCoefficients"]
    coefficients.metadata = {}
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["schema_errors"] == []
    assert report["errors"] == []
    assert report["fidelity_errors"] == []
    assert any("elementSize metadata" in warning for warning in report["warnings"])
    assert any("interpolation metadata" in warning for warning in report["warnings"])
    assert report["render_prim_paths"] == ["/World/GaussianSplat/Gaussians"]


def test_optional_schema_fallbacks_do_not_claim_source_fidelity() -> None:
    field = _valid_particle_field()
    for name in (
        "opacities",
        "radiance:sphericalHarmonicsDegree",
        "radiance:sphericalHarmonicsCoefficients",
    ):
        field.attributes.pop(name)
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())
    readiness = gaussian_splats._readiness_flags(
        report, runtime_errors=report["errors"]
    )

    assert report["schema_errors"] == []
    assert report["errors"] == []
    assert any("source opacity" in error for error in report["fidelity_errors"])
    assert any("source radiance" in error for error in report["fidelity_errors"])
    assert any("fully opaque" in warning for warning in report["warnings"])
    assert any("degree-0 DC color" in warning for warning in report["warnings"])
    assert readiness == {
        "schema_ready": True,
        "render_path_ready": True,
        "fidelity_ready": False,
    }


def test_legacy_nurec_volume_reports_opaque_payload_without_guessing_count() -> None:
    density = FakePrim(
        "/World/GaussianSplat/gauss/density_field",
        "OmniNuRecFieldAsset",
        attributes=[
            FakeAttribute(
                "filePath",
                FakeAssetPath("./model.nurec", "/archive/model.nurec"),
            ),
            FakeAttribute("fieldRole", "density"),
        ],
    )
    color = FakePrim(
        "/World/GaussianSplat/gauss/emissive_color_field",
        "OmniNuRecFieldAsset",
        attributes=[
            FakeAttribute(
                "filePath",
                FakeAssetPath("./model.nurec", "/archive/model.nurec"),
            ),
            FakeAttribute("fieldRole", "emissiveColor"),
        ],
    )
    volume = FakePrim(
        "/World/GaussianSplat/gauss",
        "Volume",
        attributes=[
            FakeAttribute("omni:nurec:isNuRecVolume", True),
            FakeAttribute("extent", [(-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)]),
        ],
        children=[density, color],
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[volume])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["representations"] == ["legacy_nurec_volume"]
    assert report["errors"] == []
    assert report["legacy_volumes"][0]["particle_count"] is None
    assert "opaque" in report["legacy_volumes"][0]["particle_count_reason"]
    assert len(report["legacy_volumes"][0]["field_assets"]) == 2
    assert report["fidelity_errors"] == []
    assert any("opaque" in item for item in report["fidelity_limitations"])
    assert (
        gaussian_splats._readiness_flags(report, runtime_errors=report["errors"])[
            "fidelity_ready"
        ]
        is False
    )


def test_spg_sidecars_require_the_runtime_extension() -> None:
    field = _valid_particle_field()
    shader = FakePrim(
        "/World/GaussianSplat/Render/PPISP",
        "Shader",
        attributes=[
            FakeAttribute(
                "info:spg:sourceAsset",
                FakeAssetPath("./ppisp.cu", "/archive/ppisp.cu"),
            ),
            FakeAttribute("info:spg:sourceAsset:subIdentifier", "ppispProcess"),
        ],
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field, shader])

    disabled = gaussian_splats._inspect_root(root, renderer=_renderer(spg=False))
    enabled = gaussian_splats._inspect_root(root, renderer=_renderer(spg=True))

    assert disabled["spg"]["present"] is True
    assert any("omni.rtx.spg" in error for error in disabled["errors"])
    assert enabled["spg"]["errors"] == []


def test_plain_particle_field_keeps_renderer_tonemapping_policy() -> None:
    field = _valid_particle_field()
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    configuration = gaussian_splats._configure_renderer_for_asset(object(), root)
    report = gaussian_splats._inspect_root(
        root,
        renderer=_renderer(skip_tonemapping=True),
    )

    assert configuration == {
        "attempted": True,
        "asset_mode": "plain_gaussian",
        "actions": [],
        "errors": [],
    }
    assert not any("tonemapping" in warning for warning in report["warnings"])


def test_presigned_source_is_never_echoed_with_query_or_fragment() -> None:
    source = "https://assets.example/scan.usdz?X-Amz-Signature=secret#fragment"

    assert gaussian_splats._validated_source(source) == source
    assert gaussian_splats._display_source(source) == "https://assets.example/scan.usdz"
    assert (
        gaussian_splats._display_source(
            "https://user:password@assets.example/scan.usdz?token=secret"
        )
        == "https://assets.example/scan.usdz"
    )
    error = gaussian_splats._redacted_error(
        RuntimeError(f"resolver rejected {source}"),
        source,
        "https://assets.example/scan.usdz",
    )
    assert "secret" not in error
    assert "fragment" not in error


def test_update_measurement_is_bounded_and_labeled_as_cpu_observed() -> None:
    class App:
        updates = 0

        def update(self) -> None:
            self.updates += 1

    app = App()

    report = gaussian_splats._measure_update_loop(app, warmup_frames=2, sample_frames=3)

    assert app.updates == 5
    assert report["measurement"] == "kit_update_loop"
    assert report["sample_frames"] == 3
    assert report["measured"] is True
    assert report["p99_ms"] >= 0
    assert "CUA" in report["note"]


def test_gpu_evidence_reports_free_memory_and_utilization_ratio(monkeypatch) -> None:
    monkeypatch.setattr(
        gaussian_splats.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="NVIDIA GeForce RTX 4090, 8192, 16384, 24576, 77\n"
        ),
    )

    assert gaussian_splats._gpu_evidence() == {
        "available": True,
        "name": "NVIDIA GeForce RTX 4090",
        "memory_used_mib": 8192,
        "memory_free_mib": 16384,
        "memory_total_mib": 24576,
        "memory_used_ratio": 0.3333,
        "utilization_percent": 77,
    }


class RecordingMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, name: str):
        def register(function):
            self.tools[name] = function
            return function

        return register


class RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def send_command(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((command, params))
        return {"status": "success", "ready": True}


def test_mcp_tool_forwards_the_bounded_proof_contract(monkeypatch) -> None:
    mcp_module = types.ModuleType("mcp")
    mcp_module.__path__ = []
    server_module = types.ModuleType("mcp.server")
    server_module.__path__ = []
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = RecordingMCP
    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    tool_module = _load_module(
        "isaac_mcp_gaussian_splats_contract_test",
        REPO_ROOT / "isaac_mcp" / "tools" / "gaussian_splats.py",
    )
    mcp = RecordingMCP()
    connection = RecordingConnection()
    tool_module.register_tools(mcp, lambda: connection)

    result = json.loads(
        mcp.tools["load_gaussian_splat"](
            usd_path="/data/workspace/scan.usdz",
            prim_path="/World/Scan",
            sample_frames=60,
        )
    )

    assert result == {"status": "success", "ready": True}
    assert connection.calls == [
        (
            "assets.load_gaussian_splat",
            {
                "usd_path": "/data/workspace/scan.usdz",
                "prim_path": "/World/Scan",
                "frame_camera": True,
                "wait_timeout_seconds": 20.0,
                "warmup_frames": 8,
                "sample_frames": 60,
            },
        )
    ]


def test_handler_registration_keeps_the_asset_namespace() -> None:
    registry: dict[str, Any] = {}

    gaussian_splats.register(registry, object())

    assert set(registry) == {
        "assets.load_gaussian_splat",
        "assets.inspect_gaussian_splat",
    }
