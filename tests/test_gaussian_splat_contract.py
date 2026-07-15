"""Contract tests for bounded Gaussian-splat runtime evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


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


def _install_nurec_setup(monkeypatch, setup_for_rendering) -> None:
    rendering_setup = types.ModuleType(
        "isaacsim.replicator.nurec_utils.rendering_setup"
    )
    rendering_setup.setup_for_rendering = setup_for_rendering
    for name in (
        "isaacsim",
        "isaacsim.replicator",
        "isaacsim.replicator.nurec_utils",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.replicator.nurec_utils.rendering_setup",
        rendering_setup,
    )


def _renderer(
    *,
    spg: bool = False,
    skip_tonemapping: bool = False,
    multi_gpu: bool = False,
    disable_nurec_post: bool = True,
    spg_setting: bool | None = None,
) -> dict[str, Any]:
    return {
        "particle_field_schema_available": True,
        "extensions": {"omni.hydra.rtx": True, "omni.rtx.spg": spg},
        "settings": {
            "/app/useFabricSceneDelegate": True,
            "/renderer/multiGpu/enabled": multi_gpu,
            "/rtx/rtpt/gaussian/skipTonemapping/enabled": skip_tonemapping,
            "/omni/rtx/nre/compositing/disableNuRecPostProcessings": disable_nurec_post,
            "/rtx/spg/enabled": spg if spg_setting is None else spg_setting,
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


def test_short_optional_array_uses_default_without_blocking_render() -> None:
    field = _valid_particle_field(scale_count=1)
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["errors"] == []
    assert any("scales count 1" in warning for warning in report["warnings"])
    assert any("scales is too short" in item for item in report["fidelity_limitations"])
    assert report["particle_fields"][0]["particle_count"] == 2
    assert gaussian_splats._readiness_flags(
        report, runtime_errors=report["errors"]
    ) == {
        "schema_ready": True,
        "render_path_ready": True,
        "fidelity_ready": False,
    }


def test_long_optional_array_is_truncated_and_ignored_extra_is_not_sampled() -> None:
    field = _valid_particle_field()
    field.attributes["scales"].value.append((float("nan"), 1.0, 1.0))
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["errors"] == []
    assert any("scales count 3" in warning for warning in report["warnings"])
    assert any("truncated" in item for item in report["fidelity_limitations"])


@pytest.mark.parametrize("coefficient_count", [1, 3])
def test_sh_length_fallback_or_truncation_does_not_block_render(
    coefficient_count: int,
) -> None:
    field = _valid_particle_field()
    coefficients = field.attributes["radiance:sphericalHarmonicsCoefficients"]
    coefficients.value = [(0.2, 0.3, 0.4)] * coefficient_count
    if coefficient_count > 2:
        coefficients.value[-1] = (float("nan"), 0.0, 0.0)
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["errors"] == []
    assert report["schema_errors"] == []
    if coefficient_count < 2:
        assert any("source radiance" in item for item in report["fidelity_errors"])
        assert any("degree-0 gray fallback" in item for item in report["warnings"])
    else:
        assert any("SH coefficient count 3" in item for item in report["warnings"])
        assert any("SH coefficients" in item for item in report["fidelity_limitations"])


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


def test_optional_orientations_and_scales_use_schema_defaults() -> None:
    field = _valid_particle_field()
    field.attributes.pop("orientations")
    field.attributes.pop("scales")
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["schema_errors"] == []
    assert report["errors"] == []
    assert report["particle_fields"][0]["attributes"]["orientations"] == {
        "name": None,
        "count": None,
    }
    assert report["particle_fields"][0]["attributes"]["scales"] == {
        "name": None,
        "count": None,
    }


def test_empty_optional_arrays_use_schema_defaults() -> None:
    field = _valid_particle_field()
    for name in (
        "orientations",
        "scales",
        "opacities",
        "radiance:sphericalHarmonicsCoefficients",
    ):
        field.attributes[name].value = []
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["schema_errors"] == []
    assert report["errors"] == []
    assert any("source opacity" in error for error in report["fidelity_errors"])
    assert any("source radiance" in error for error in report["fidelity_errors"])


def test_nonempty_half_array_wins_over_empty_float_array() -> None:
    field = _valid_particle_field()
    positions = field.attributes["positions"].value
    field.attributes["positions"].value = []
    field.attributes["positionsh"] = FakeAttribute(
        "positionsh", positions, type_name="point3h[]"
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer())

    assert report["errors"] == []
    assert report["particle_fields"][0]["particle_count"] == 2
    assert report["particle_fields"][0]["attributes"]["positions"]["name"] == (
        "positionsh"
    )


def test_enabled_multi_gpu_fails_render_readiness() -> None:
    field = _valid_particle_field()
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])

    report = gaussian_splats._inspect_root(root, renderer=_renderer(multi_gpu=True))

    assert any("Multi-GPU rendering is enabled" in error for error in report["errors"])
    assert (
        gaussian_splats._readiness_flags(report, runtime_errors=report["errors"])[
            "render_path_ready"
        ]
        is False
    )


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


def test_spg_requires_nurec_post_processing_to_be_disabled() -> None:
    field = _valid_particle_field()
    shader = FakePrim(
        "/World/GaussianSplat/Render/PPISP",
        "Shader",
        attributes=[
            FakeAttribute(
                "info:spg:sourceAsset",
                FakeAssetPath("./ppisp.cu", "/archive/ppisp.cu"),
            )
        ],
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field, shader])

    report = gaussian_splats._inspect_root(
        root,
        renderer=_renderer(spg=True, disable_nurec_post=False),
    )

    assert any("processed twice" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("renderer", "message"),
    [
        ({"spg_setting": False}, "/rtx/spg/enabled=true"),
        ({"skip_tonemapping": True}, "skipTonemapping/enabled=false"),
    ],
)
def test_spg_required_settings_gate_render_readiness(
    renderer: dict[str, Any], message: str
) -> None:
    field = _valid_particle_field()
    shader = FakePrim(
        "/World/GaussianSplat/Render/PPISP",
        "Shader",
        attributes=[
            FakeAttribute(
                "info:spg:sourceAsset",
                FakeAssetPath("./ppisp.cu", "/archive/ppisp.cu"),
            )
        ],
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field, shader])

    report = gaussian_splats._inspect_root(
        root,
        renderer=_renderer(spg=True, **renderer),
    )

    assert any(message in error for error in report["errors"])
    assert (
        gaussian_splats._readiness_flags(report, runtime_errors=report["errors"])[
            "render_path_ready"
        ]
        is False
    )


def test_plain_particle_field_uses_nurec_setup_without_forcing_tonemapping(
    monkeypatch,
) -> None:
    field = _valid_particle_field()
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field])
    stage = object()
    calls: list[Any] = []

    def setup_for_rendering(received_stage: Any):
        calls.append(received_stage)
        return True, True, False, []

    _install_nurec_setup(monkeypatch, setup_for_rendering)

    configuration = gaussian_splats._configure_renderer_for_asset(object(), stage, root)
    report = gaussian_splats._inspect_root(
        root,
        renderer=_renderer(skip_tonemapping=True),
    )

    assert configuration == {
        "attempted": True,
        "asset_mode": "plain_gaussian",
        "actions": ["applied isaacsim.replicator.nurec_utils pre-Hydra renderer setup"],
        "errors": [],
    }
    assert calls == [stage]
    assert not any("tonemapping" in warning for warning in report["warnings"])


def test_spg_setup_uses_installed_nurec_utility(monkeypatch) -> None:
    field = _valid_particle_field()
    shader = FakePrim(
        "/World/GaussianSplat/Render/PPISP",
        "Shader",
        attributes=[
            FakeAttribute(
                "info:spg:sourceAsset",
                FakeAssetPath("./ppisp.cu", "/archive/ppisp.cu"),
            )
        ],
    )
    root = FakePrim("/World/GaussianSplat", "Xform", children=[field, shader])
    stage = object()
    calls: list[Any] = []

    class Extensions:
        def is_extension_enabled(self, name: str) -> bool:
            calls.append(("extension", name))
            return True

    class App:
        def get_extension_manager(self) -> Extensions:
            return Extensions()

    def setup_for_rendering(received_stage: Any):
        calls.append(("setup", received_stage))
        return True, True, True, []

    _install_nurec_setup(monkeypatch, setup_for_rendering)

    configuration = gaussian_splats._configure_renderer_for_asset(App(), stage, root)

    assert configuration["errors"] == []
    assert "pre-Hydra" in configuration["actions"][0]
    assert calls == [("extension", "omni.rtx.spg"), ("setup", stage)]


def test_runtime_configures_before_waiting_for_stage(monkeypatch) -> None:
    app = object()
    context = object()
    root = object()
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: root)
    events: list[str] = []

    app_module = types.ModuleType("omni.kit.app")
    app_module.get_app = lambda: app
    usd_module = types.ModuleType("omni.usd")
    usd_module.get_context = lambda: context
    kit_module = types.ModuleType("omni.kit")
    kit_module.__path__ = []
    kit_module.app = app_module
    omni_module = types.ModuleType("omni")
    omni_module.__path__ = []
    omni_module.kit = kit_module
    omni_module.usd = usd_module
    for name, module in (
        ("omni", omni_module),
        ("omni.kit", kit_module),
        ("omni.kit.app", app_module),
        ("omni.usd", usd_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        gaussian_splats,
        "_configure_renderer_for_asset",
        lambda *_args: events.append("configure") or {"attempted": True, "errors": []},
    )
    monkeypatch.setattr(
        gaussian_splats,
        "_wait_for_stage",
        lambda *_args, **_kwargs: events.append("wait") or {"complete": True},
    )
    monkeypatch.setattr(
        gaussian_splats,
        "_renderer_evidence",
        lambda _app: {"extensions": {}, "settings": {}},
    )
    monkeypatch.setattr(
        gaussian_splats,
        "_inspect_root",
        lambda *_args, **_kwargs: {
            "errors": [],
            "warnings": [],
            "schema_errors": [],
            "fidelity_errors": [],
            "fidelity_limitations": [],
            "render_prim_paths": [],
        },
    )
    monkeypatch.setattr(
        gaussian_splats,
        "_measure_update_loop",
        lambda *_args, **_kwargs: {"measured": False},
    )
    monkeypatch.setattr(gaussian_splats, "_gpu_evidence", lambda: {})

    gaussian_splats._runtime_evidence(
        stage=stage,
        root_path="/World/Scan",
        configure_renderer=True,
        frame_camera=False,
        wait_timeout_seconds=1.0,
        warmup_frames=0,
        sample_frames=0,
    )

    assert events == ["configure", "wait"]


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


def test_codeless_particle_field_schema_does_not_require_python_binding(
    monkeypatch,
) -> None:
    class Registry:
        def FindConcretePrimDefinition(self, type_name: str) -> object | None:
            return object() if type_name == "ParticleField3DGaussianSplat" else None

    pxr_module = types.ModuleType("pxr")
    pxr_module.Usd = SimpleNamespace(SchemaRegistry=lambda: Registry())
    pxr_module.UsdVol = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    assert gaussian_splats._particle_field_schema_evidence() == {
        "particle_field_schema_available": True,
        "particle_field_python_binding_available": False,
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


def test_renderer_evidence_reports_the_dlss_frame_generation_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        def get(self, path: str) -> object:
            return path == "/app/useFabricSceneDelegate"

    carb = types.ModuleType("carb")
    carb.settings = types.SimpleNamespace(get_settings=lambda: Settings())
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setattr(
        gaussian_splats,
        "_particle_field_schema_evidence",
        lambda: {"particle_field_schema_available": True},
    )

    class ExtensionManager:
        def is_extension_enabled(self, _name: str) -> bool:
            return True

    app = types.SimpleNamespace(get_extension_manager=lambda: ExtensionManager())

    evidence = gaussian_splats._renderer_evidence(app)

    assert evidence["settings"]["/rtx-transient/dlssg/enabled"] is False


def test_asset_reference_uses_an_untyped_override_without_pxr() -> None:
    class References:
        sources: list[str] = []

        def AddReference(self, source: str) -> bool:
            self.sources.append(source)
            return True

    class Prim:
        references = References()

        def GetReferences(self) -> References:
            return self.references

    class Stage:
        override_paths: list[str] = []

        def OverridePrim(self, path: str) -> Prim:
            self.override_paths.append(path)
            return Prim()

        def DefinePrim(self, *_args, **_kwargs):
            raise AssertionError(
                "a local type opinion would mask the referenced schema"
            )

    stage = Stage()
    root = gaussian_splats._author_asset_reference(
        stage,
        "/World/Scan",
        "/data/workspace/scan.usdz",
    )

    assert isinstance(root, Prim)
    assert stage.override_paths == ["/World/Scan"]
    assert root.references.sources == ["/data/workspace/scan.usdz"]


def test_untyped_reference_preserves_converter_default_prim_schema(tmp_path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    source_path = tmp_path / "converter-shaped.usdc"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_prim = source_stage.DefinePrim(
        "/GaussianSplat", "ParticleField3DGaussianSplat"
    )
    source_stage.SetDefaultPrim(source_prim)
    source_stage.GetRootLayer().Save()

    target_stage = Usd.Stage.CreateInMemory()
    target_stage.DefinePrim("/World", "Xform")
    gaussian_splats._author_asset_reference(
        target_stage,
        "/World/Scan",
        str(source_path),
    )

    composed = target_stage.GetPrimAtPath("/World/Scan")
    assert composed.GetTypeName() == "ParticleField3DGaussianSplat"
