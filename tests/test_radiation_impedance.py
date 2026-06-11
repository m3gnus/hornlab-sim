from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_sim.methods import _bem_backend
from hornlab_sim.methods import radiation_impedance


class _FakeVelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


@dataclass
class _FakeConfig:
    mesh_scale: float = 1.0
    velocity_mode: str = _FakeVelocityMode.ACCELERATION
    velocity_sources: dict[int, complex] = field(default_factory=dict)


def _patch_bem_backend(monkeypatch, solve_frequencies):
    api = SimpleNamespace(
        name="bempp",
        load_mesh=lambda path, scale=1.0: path,
        solve_frequencies=solve_frequencies,
        VelocityMode=_FakeVelocityMode,
        default_config=lambda formulation: _FakeConfig(),
    )
    monkeypatch.setattr(_bem_backend, "resolve_backend", lambda config=None: "bempp")
    monkeypatch.setattr(_bem_backend, "backend_api", lambda backend: api)
    return api


def _fake_three_tag_mesh():
    """Three equal 0.5 m^2 triangles with tags 2, 3, and 4."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    elements = np.array(
        [
            [0, 1, 2],
            [1, 3, 4],
            [3, 5, 6],
        ],
        dtype=np.int32,
    )
    grid = SimpleNamespace(vertices=vertices, elements=elements)
    physical_tags = np.array([2, 3, 4], dtype=np.int32)
    return SimpleNamespace(grid=grid, physical_tags=physical_tags)


def test_aperture_matrix_uses_one_basis_per_source(monkeypatch):
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0])
    captured_sources = []

    def fake_solve_frequencies(loaded, frequencies, cfg):
        captured_sources.append(dict(cfg.velocity_sources))
        active = [tag for tag, value in cfg.velocity_sources.items() if value != 0]
        assert len(active) == 1
        source_tag = active[0]
        source_factor = 10 if source_tag == 2 else 20
        return _fake_result(
            frequencies,
            {
                2: np.array([source_factor + 1, source_factor + 2], dtype=complex),
                3: np.array([source_factor + 3, source_factor + 4], dtype=complex),
            },
        )

    _patch_bem_backend(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"driver": [2], "port": [3]},
        freqs,
        normal_velocity=2.0,
    )

    assert result.aperture_names == ["driver", "port"]
    assert len(captured_sources) == 2
    assert captured_sources[0] == {2: 2.0 + 0.0j, 3: 0.0 + 0.0j}
    assert captured_sources[1] == {2: 0.0 + 0.0j, 3: 2.0 + 0.0j}

    # Each single-tag aperture has A=0.5 m^2. With v=2 m/s, Q=1 m^3/s,
    # so Z equals the mocked average pressure.
    np.testing.assert_allclose(result.impedance_matrix[:, 0, 0], [11, 12])
    np.testing.assert_allclose(result.impedance_matrix[:, 1, 0], [13, 14])
    np.testing.assert_allclose(result.impedance_matrix[:, 0, 1], [21, 22])
    np.testing.assert_allclose(result.impedance_matrix[:, 1, 1], [23, 24])


def test_velocity_mode_matrix_normalizes_by_volume_velocity_v_times_area(monkeypatch):
    mesh = _fake_three_tag_mesh()

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {
                2: np.array([12.0 + 0.0j]),
            },
        )

    _patch_bem_backend(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"driver": [2]},
        np.array([100.0]),
        normal_velocity=3.0,
    )

    # Tag 2 area is 0.5 m^2, so U = v*A = 1.5 m^3/s.
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(8.0 + 0.0j)


def test_multi_tag_receiver_pressure_is_area_weighted(monkeypatch):
    mesh = _fake_three_tag_mesh()

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {
                2: np.array([10.0 + 0.0j]),
                3: np.array([20.0 + 0.0j]),
                4: np.array([40.0 + 0.0j]),
            },
        )

    _patch_bem_backend(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"combined": [2, 3], "single": [4]},
        np.array([100.0]),
        normal_velocity=1.0,
    )

    # combined receiver average is (10*0.5 + 20*0.5) / 1.0 = 15 Pa.
    # combined source Q is 1.0 m^3/s, so Z=15.
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(15.0)


def test_termination_load_conjugates_solver_matrix():
    solver_matrix = np.array(
        [[[100.0 - 25.0j]]],
        dtype=np.complex128,
    )

    load = radiation_impedance.termination_load_from_solver_matrix(
        solver_matrix,
        receiver_index=0,
    )

    np.testing.assert_allclose(load, [100.0 + 25.0j])


def test_termination_load_reduces_in_phase_lr_pair():
    solver_matrix = np.array(
        [
            [
                [10.0 - 2.0j, 3.0 - 5.0j],
                [3.0 - 5.0j, 10.0 - 2.0j],
            ]
        ],
        dtype=np.complex128,
    )

    left_load = radiation_impedance.termination_load_from_solver_matrix(
        solver_matrix,
        receiver_index=0,
        source_indices=[0, 1],
    )

    np.testing.assert_allclose(left_load, [13.0 + 7.0j])


def test_terminated_chamber_port_branch_matches_lumped_network():
    freqs = np.array([100.0, 200.0])
    load = np.array([20.0 + 3.0j, 30.0 + 4.0j])

    result = radiation_impedance.terminated_chamber_port_branch(
        freqs,
        load,
        chamber_volume_m3=2.0e-5,
        port_area_m2=5.0e-4,
        port_length_m=0.02,
        interior_end_correction_length_m=0.001,
        rho=1.2,
        c=340.0,
    )

    omega = 2.0 * np.pi * freqs
    y_chamber = 1j * omega * (2.0e-5 / (1.2 * 340.0 * 340.0))
    z_port = 1j * omega * 1.2 * 0.021 / 5.0e-4
    y_series = 1.0 / (z_port + load)
    total_y = y_chamber + y_series
    np.testing.assert_allclose(result.input_impedance, 1.0 / total_y)
    np.testing.assert_allclose(
        result.exit_to_input_volume_velocity_ratio,
        y_series / total_y,
    )


def test_collapse_aperture_matrix_area_weights_subtags():
    freqs = np.array([100.0])
    matrix = np.zeros((1, 3, 3), dtype=np.complex128)
    # Names: left_a, left_b, right. Left subtags have 25/75% of the area.
    matrix[0] = np.array(
        [
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 20.0],
            [30.0, 40.0, 5.0],
        ],
        dtype=np.complex128,
    )
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=freqs,
        aperture_names=["left_a", "left_b", "right"],
        aperture_area_m2={"left_a": 0.25, "left_b": 0.75, "right": 1.0},
        impedance_matrix=matrix,
        solver_logs=[],
    )

    collapsed = radiation_impedance.collapse_aperture_matrix(
        result,
        {"left": ["left_a", "left_b"], "right": ["right"]},
    )

    # Uniform velocity over the left aggregate means source weights are
    # proportional to subtag area, and the receiver pressure average is also
    # area-weighted.
    expected_left_left = (
        0.25 * 0.25 * 1.0
        + 0.25 * 0.75 * 2.0
        + 0.75 * 0.25 * 3.0
        + 0.75 * 0.75 * 4.0
    )
    expected_left_right = 0.25 * 10.0 + 0.75 * 20.0
    expected_right_left = 0.25 * 30.0 + 0.75 * 40.0

    assert collapsed.aperture_names == ["left", "right"]
    assert collapsed.aperture_area_m2 == {"left": 1.0, "right": 1.0}
    assert collapsed.impedance_matrix[0, 0, 0] == pytest.approx(expected_left_left)
    assert collapsed.impedance_matrix[0, 0, 1] == pytest.approx(expected_left_right)
    assert collapsed.impedance_matrix[0, 1, 0] == pytest.approx(expected_right_left)
    assert collapsed.impedance_matrix[0, 1, 1] == pytest.approx(5.0)


def test_matrix_diagnostics_reports_reciprocity_and_passivity():
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=np.array([100.0, 200.0]),
        aperture_names=["a", "b"],
        aperture_area_m2={"a": 1.0, "b": 1.0},
        impedance_matrix=np.array(
            [
                [[2.0 + 1.0j, 0.5], [0.5, 1.0 + 0.2j]],
                [[-1.0 + 0.0j, 0.0], [0.25, 1.0 + 0.0j]],
            ],
            dtype=np.complex128,
        ),
        solver_logs=[],
    )

    diagnostics = radiation_impedance.matrix_diagnostics(result)

    assert diagnostics.reciprocity_max_abs[0] == pytest.approx(0.0)
    assert diagnostics.passivity_ok[0]
    assert diagnostics.reciprocity_max_abs[1] == pytest.approx(0.25)
    assert diagnostics.passivity_min_eig[1] < 0.0
    assert not diagnostics.passivity_ok[1]


def test_low_ka_baffled_piston_reference_scaling():
    radius_m = 0.05
    freqs = np.array([50.0, 100.0])

    z = radiation_impedance.low_ka_baffled_piston_radiation_impedance(
        radius_m,
        freqs,
    )

    assert z[1].real / z[0].real == pytest.approx(4.0)
    assert z[1].imag / z[0].imag == pytest.approx(2.0)


def test_matrix_diagnostics_low_ka_self_impedance_matches_reference():
    radius_m = 0.04
    freqs = np.array([80.0, 160.0])
    expected = radiation_impedance.low_ka_baffled_piston_radiation_impedance(
        radius_m,
        freqs,
    )
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=freqs,
        aperture_names=["piston"],
        aperture_area_m2={"piston": np.pi * radius_m * radius_m},
        impedance_matrix=expected.reshape(2, 1, 1),
        solver_logs=[],
    )

    diagnostics = radiation_impedance.matrix_diagnostics(
        result,
        piston_radius_m_by_aperture={"piston": radius_m},
    )

    np.testing.assert_allclose(
        diagnostics.low_ka_self_impedance_rel_error["piston"],
        [0.0, 0.0],
    )


def test_empty_frequencies_raise():
    with pytest.raises(ValueError, match="frequencies_hz is empty"):
        radiation_impedance.solve_aperture_matrix(
            _fake_three_tag_mesh(),
            {"driver": [2]},
            np.array([]),
        )


def test_zero_normal_velocity_raises():
    with pytest.raises(ValueError, match="normal_velocity must be nonzero"):
        radiation_impedance.solve_aperture_matrix(
            _fake_three_tag_mesh(),
            {"driver": [2]},
            np.array([100.0]),
            normal_velocity=0.0,
        )


def _fake_result(freqs, surface_pressure_avg):
    return SimpleNamespace(
        surface_pressure_avg=surface_pressure_avg,
        solver_log=[],
    )
