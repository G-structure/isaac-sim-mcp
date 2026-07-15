"""Contract tests for bounded contact-integrity evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac.sim.mcp_extension"
    / "isaac_sim_mcp_extension"
    / "contact_integrity.py"
)
SPEC = importlib.util.spec_from_file_location("contact_integrity_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
contact_integrity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contact_integrity
SPEC.loader.exec_module(contact_integrity)


class FakeContactView:
    def __init__(
        self,
        *,
        separation: float = -0.003,
        normal_impulse: float = 2.0,
        count: int = 1,
    ) -> None:
        self.separation = separation
        self.normal_impulse = normal_impulse
        self.count = count
        self.destroyed = False

    def get_contact_force_data(self, dt: float):
        assert dt == 1.0
        return (
            np.full((self.count, 1), self.normal_impulse),
            np.zeros((self.count, 3)),
            np.tile(np.array([[0.0, 0.0, 1.0]]), (self.count, 1)),
            np.full((self.count, 1), self.separation),
            np.array([[self.count]]),
            np.array([[0]]),
        )

    def get_friction_data(self, dt: float):
        assert dt == 1.0
        return (
            np.array([[0.1, 0.0, 0.0]]),
            np.zeros((1, 3)),
            np.array([[1]]),
            np.array([[0]]),
        )

    def destroy(self) -> None:
        self.destroyed = True


def _sampler(
    *,
    max_contacts: int = 64,
    view: FakeContactView | None = None,
    limits: dict[str, float] | None = None,
):
    sampler = contact_integrity.ContactIntegritySampler(
        None,
        {
            "pairs": [
                {
                    "label": "left-cube",
                    "sensor_path": "/World/left_finger",
                    "filter_path": "/World/cube",
                }
            ],
            "max_contacts_per_pair": max_contacts,
            "limits": limits or {},
        },
    )
    pair = sampler.config.pairs[0]
    sampler._views = [(pair, view or FakeContactView())]
    return sampler


def test_contact_trace_preserves_penetration_impulse_and_friction() -> None:
    sampler = _sampler()

    sampler.sample(update_index=0, physics_dt_seconds=1.0 / 120.0)
    result = sampler.result(requested_updates=1, physics_dt_seconds=1.0 / 120.0)

    assert result["complete"] is True
    assert result["summary"] == {
        "updates_with_contact": 1,
        "maximum_penetration_m": pytest.approx(0.003),
        "maximum_normal_impulse_ns": pytest.approx(2.0),
    }
    pair = result["samples"][0]["pairs"][0]
    assert pair["maximum_penetration_m"] == pytest.approx(0.003)
    assert pair["maximum_normal_impulse_ns"] == pytest.approx(2.0)
    assert pair["total_normal_impulse_ns"] == pytest.approx(2.0)
    assert pair["contacts"][0] == {
        "point_m": [0.0, 0.0, 0.0],
        "normal_filter_to_sensor": [0.0, 0.0, 1.0],
        "signed_separation_m": pytest.approx(-0.003),
        "penetration_m": pytest.approx(0.003),
        "normal_impulse_ns": pytest.approx(2.0),
        "normal_force_n": pytest.approx(240.0),
    }
    assert pair["friction_contacts"][0]["impulse_magnitude_ns"] == pytest.approx(0.1)


def test_configured_limits_report_machine_readable_violations() -> None:
    sampler = _sampler(
        limits={
            "maximum_penetration_m": 0.002,
            "maximum_normal_impulse_ns": 1.0,
        }
    )

    sampler.sample(update_index=7, physics_dt_seconds=1.0 / 120.0)
    result = sampler.result(requested_updates=1, physics_dt_seconds=1.0 / 120.0)

    assert result["complete"] is True
    assert result["within_configured_limits"] is False
    assert result["limits"] == {
        "maximum_penetration_m": pytest.approx(0.002),
        "maximum_normal_impulse_ns": pytest.approx(1.0),
    }
    assert result["violations"] == [
        {
            "update_index": 7,
            "pair_label": "left-cube",
            "metric": "maximum_penetration_m",
            "observed": pytest.approx(0.003),
            "limit": pytest.approx(0.002),
        },
        {
            "update_index": 7,
            "pair_label": "left-cube",
            "metric": "maximum_normal_impulse_ns",
            "observed": pytest.approx(2.0),
            "limit": pytest.approx(1.0),
        },
    ]


def test_negative_normal_impulse_fails_closed() -> None:
    sampler = _sampler(view=FakeContactView(normal_impulse=-0.1))

    sampler.sample(update_index=0, physics_dt_seconds=1.0 / 120.0)
    result = sampler.result(requested_updates=1, physics_dt_seconds=1.0 / 120.0)

    assert result["complete"] is False
    assert "negative normal impulse" in result["errors"][0]


def test_full_contact_buffer_invalidates_trace() -> None:
    sampler = _sampler(max_contacts=1)

    sampler.sample(update_index=0, physics_dt_seconds=1.0 / 120.0)
    result = sampler.result(requested_updates=1, physics_dt_seconds=1.0 / 120.0)

    assert result["complete"] is False
    assert result["saturated_pairs"] == ["left-cube"]
    assert result["samples"][0]["pairs"][0]["buffer_saturated"] is True


def test_contact_trace_rejects_unbounded_client_requests() -> None:
    sampler = _sampler(max_contacts=256)

    with pytest.raises(ValueError, match="at least one update"):
        sampler.validate_request_size(0)
    with pytest.raises(ValueError, match="bounded response budget"):
        sampler.validate_request_size(33)

    sampler.validate_request_size(32)


def test_nonfinite_contact_data_fails_closed() -> None:
    sampler = _sampler(view=FakeContactView(separation=float("nan")))

    sampler.sample(update_index=0, physics_dt_seconds=1.0 / 120.0)
    result = sampler.result(requested_updates=1, physics_dt_seconds=1.0 / 120.0)

    assert result["complete"] is False
    assert "non-finite signed separation" in result["errors"][0]


def test_contact_configuration_rejects_ambiguous_paths_and_labels() -> None:
    with pytest.raises(ValueError, match="absolute USD paths"):
        contact_integrity.ContactIntegrityConfig.parse(
            {"pairs": [{"a": "World/a", "b": "/World/b"}]}
        )
    with pytest.raises(ValueError, match="must differ"):
        contact_integrity.ContactIntegrityConfig.parse(
            {"pairs": [{"a": "/World/a", "b": "/World/a"}]}
        )
    with pytest.raises(ValueError, match="duplicate"):
        contact_integrity.ContactIntegrityConfig.parse(
            {
                "pairs": [
                    {"label": "same", "a": "/World/a", "b": "/World/b"},
                    {"label": "same", "a": "/World/c", "b": "/World/d"},
                ]
            }
        )
    with pytest.raises(ValueError, match="unsupported contact integrity limits"):
        contact_integrity.ContactIntegrityConfig.parse(
            {
                "pairs": [{"a": "/World/a", "b": "/World/b"}],
                "limits": {"maximum_penetraton_m": 0.001},
            }
        )
