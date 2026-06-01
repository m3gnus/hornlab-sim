from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_sim.methods import radiation_impedance


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

    monkeypatch.setattr(
        "hornlab_solver.solve_frequencies",
        fake_solve_frequencies,
    )

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

    monkeypatch.setattr(
        "hornlab_solver.solve_frequencies",
        fake_solve_frequencies,
    )

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"combined": [2, 3], "single": [4]},
        np.array([100.0]),
        normal_velocity=1.0,
    )

    # combined receiver average is (10*0.5 + 20*0.5) / 1.0 = 15 Pa.
    # combined source Q is 1.0 m^3/s, so Z=15.
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(15.0)


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
