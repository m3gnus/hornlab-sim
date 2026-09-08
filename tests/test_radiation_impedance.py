from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_sim.methods import radiation_impedance


class _FakeVelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


@dataclass
class _FakeConfig:
    mesh_scale: float = 1.0
    mesh_validate: bool = True
    mesh_merge_tol: float = 1e-9
    mesh_repair_normals: bool = False
    native_symmetry_plane: str | None = None
    aperture_tag: int | None = None
    velocity_mode: str = _FakeVelocityMode.ACCELERATION
    velocity_sources: dict[int, complex] = field(default_factory=dict)


def _patch_metal_api(monkeypatch, solve_frequencies):
    api = SimpleNamespace(
        name="metal",
        load_mesh=lambda path, **kwargs: path,
        solve_frequencies=solve_frequencies,
        VelocityMode=_FakeVelocityMode,
        default_config=lambda formulation: _FakeConfig(),
    )
    monkeypatch.setattr(radiation_impedance, "_metal_api", lambda config=None: api)
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


def _fake_three_tag_mesh_columns():
    """The same three triangles in the canonical Bempp column layout."""
    mesh = _fake_three_tag_mesh()
    mesh.grid.vertices = mesh.grid.vertices.T
    mesh.grid.elements = mesh.grid.elements.T
    return mesh


def _fake_ambiguous_three_by_three_mesh():
    """Three vertices and three elements: both arrays are (3, 3)."""
    grid = SimpleNamespace(
        vertices=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
            dtype=np.float64,
        ),
        elements=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32),
    )
    return SimpleNamespace(
        grid=grid, physical_tags=np.array([2, 3, 4], dtype=np.int32)
    )


# ---------------------------------------------------------------------------
# Shared triangle-area helper: layout must never be guessed at N == 3
# ---------------------------------------------------------------------------


def test_column_layout_three_triangle_mesh_normalizes_by_the_real_area(monkeypatch):
    """A three-element Bempp-layout mesh used to yield zero-area apertures.

    Each triangle is 0.5 m^2, so a unit normal velocity gives Q = 0.5 m^3/s
    and Z = p/Q = 2*p.
    """
    mesh = _fake_three_tag_mesh_columns()

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(frequencies, {2: np.array([6.0 - 8.0j])})

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh, {"port": [2]}, np.array([100.0]), normal_velocity=1.0
    )

    assert result.aperture_area_m2["port"] == pytest.approx(0.5)
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(12.0 - 16.0j)


def test_ambiguous_mesh_is_rejected_before_solving(monkeypatch):
    mesh = _fake_ambiguous_three_by_three_mesh()

    def fail_solve_frequencies(loaded, frequencies, cfg):
        raise AssertionError("must not solve with an unresolved mesh layout")

    _patch_metal_api(monkeypatch, fail_solve_frequencies)

    with pytest.raises(ValueError, match="Cannot determine the mesh array layout"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, np.array([100.0]), normal_velocity=1.0
        )


def test_explicit_layout_reaches_both_area_helpers(monkeypatch):
    """``mesh_array_layout`` must reach the per-tag areas too.

    The receiver aggregation is tag-area weighted, so a layout that only
    reached ``_aperture_face_areas`` would still fail here.
    """
    mesh = _fake_ambiguous_three_by_three_mesh()
    row_area = 1.5 * np.sqrt(2.0)

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {2: np.array([4.0 + 0.0j]), 3: np.array([8.0 + 0.0j])},
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"combined": [2, 3]},
        np.array([100.0]),
        normal_velocity=1.0,
        mesh_array_layout="rows",
    )

    # Both tags have the same area, so the weighted average is (4+8)/2 = 6,
    # and Q = 1.0 * (2 * row_area).
    assert result.aperture_area_m2["combined"] == pytest.approx(2.0 * row_area)
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(6.0 / (2.0 * row_area))


def test_aperture_matrix_preload_forwards_native_symmetry_plane(monkeypatch):
    mesh = _fake_three_tag_mesh()
    captured_load = {}

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {2: np.array([12.0 + 0.0j])},
        )

    api = _patch_metal_api(monkeypatch, fake_solve_frequencies)

    def capture_load_mesh(path, **kwargs):
        captured_load["path"] = path
        captured_load["kwargs"] = kwargs
        return mesh

    api.load_mesh = capture_load_mesh
    config = _FakeConfig(native_symmetry_plane="yz")

    radiation_impedance.solve_aperture_matrix(
        "reduced-domain.msh",
        {"driver": [2]},
        np.array([100.0]),
        config=config,
    )

    assert captured_load["path"] == "reduced-domain.msh"
    assert captured_load["kwargs"]["native_symmetry_plane"] == "yz"


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

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

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


def test_aperture_matrix_prefers_multi_source_backend(monkeypatch):
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0])
    multi_calls = []

    def fail_solve_frequencies(loaded, frequencies, cfg):
        raise AssertionError(
            "per-source solve_frequencies must not run when the backend "
            "exposes solve_multi_source"
        )

    api = _patch_metal_api(monkeypatch, fail_solve_frequencies)

    def fake_solve_multi_source(loaded, frequencies, source_dicts, cfg):
        multi_calls.append([dict(sources) for sources in source_dicts])
        results = []
        for sources in source_dicts:
            active = [tag for tag, value in sources.items() if value != 0]
            assert len(active) == 1
            source_factor = 10 if active[0] == 2 else 20
            results.append(
                _fake_result(
                    frequencies,
                    {
                        2: np.array(
                            [source_factor + 1, source_factor + 2], dtype=complex
                        ),
                        3: np.array(
                            [source_factor + 3, source_factor + 4], dtype=complex
                        ),
                    },
                )
            )
        return results

    api.solve_multi_source = fake_solve_multi_source

    result = radiation_impedance.solve_aperture_matrix(
        mesh,
        {"driver": [2], "port": [3]},
        freqs,
        normal_velocity=2.0,
    )

    # ONE multi-RHS call carrying every basis column, in aperture order.
    assert len(multi_calls) == 1
    assert multi_calls[0] == [
        {2: 2.0 + 0.0j, 3: 0.0 + 0.0j},
        {2: 0.0 + 0.0j, 3: 2.0 + 0.0j},
    ]
    # Same matrix as test_aperture_matrix_uses_one_basis_per_source.
    np.testing.assert_allclose(result.impedance_matrix[:, 0, 0], [11, 12])
    np.testing.assert_allclose(result.impedance_matrix[:, 1, 0], [13, 14])
    np.testing.assert_allclose(result.impedance_matrix[:, 0, 1], [21, 22])
    np.testing.assert_allclose(result.impedance_matrix[:, 1, 1], [23, 24])


# ---------------------------------------------------------------------------
# Incomplete / malformed basis sweeps must be rejected, never broadcast
# ---------------------------------------------------------------------------


def test_single_row_early_stop_is_rejected(monkeypatch):
    """The NUM-2 acceptance case: one solved row must not fill three.

    ``SolveConfig(on_frequency_result=...)`` returning ``False`` after the
    first frequency is a supported Metal outcome. The truncated result used
    to broadcast into every requested frequency and be labelled as solved.
    """
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0, 400.0])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {2: np.array([3.0 - 4.0j])},
            frequencies_hz=np.asarray(frequencies)[:1],
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    with pytest.raises(RuntimeError, match="returned 1 frequency"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, freqs, normal_velocity=1.0
        )


def test_multi_row_partial_sweep_is_rejected(monkeypatch):
    """A partial sweep that is longer than one row is rejected too."""
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0, 400.0])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {2: np.array([1.0 + 0.0j, 2.0 + 0.0j])},
            frequencies_hz=np.asarray(frequencies)[:2],
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    with pytest.raises(RuntimeError, match="returned 2 frequency"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, freqs, normal_velocity=1.0
        )


def test_reordered_frequency_axis_is_rejected(monkeypatch):
    """A complete but reordered axis would silently mislabel every row."""
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0, 400.0])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {2: np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j])},
            frequencies_hz=np.array([400.0, 100.0, 200.0]),
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    with pytest.raises(RuntimeError, match="expected the 3 requested"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, freqs, normal_velocity=1.0
        )


def test_missing_basis_column_is_rejected(monkeypatch):
    """Fewer basis results than source apertures must not truncate silently."""
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0])

    def fail_solve_frequencies(loaded, frequencies, cfg):
        raise AssertionError("multi-source path must be used here")

    api = _patch_metal_api(monkeypatch, fail_solve_frequencies)

    def short_multi_source(loaded, frequencies, source_dicts, cfg):
        return [
            _fake_result(
                frequencies,
                {
                    2: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
                    3: np.array([3.0 + 0.0j, 4.0 + 0.0j]),
                },
            )
        ]

    api.solve_multi_source = short_multi_source

    with pytest.raises(RuntimeError, match="returned 1 basis result"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"driver": [2], "port": [3]}, freqs, normal_velocity=1.0
        )


def test_short_pressure_vector_with_complete_axis_is_rejected(monkeypatch):
    """A truthful frequency axis does not excuse a broadcastable pressure row."""
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0, 400.0])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(frequencies, {2: np.array([3.0 - 4.0j])})

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    with pytest.raises(RuntimeError, match=r"has shape \(1,\), expected \(3,\)"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, freqs, normal_velocity=1.0
        )


def test_missing_frequency_axis_is_rejected(monkeypatch):
    """A result that cannot confirm its axis cannot be assembled."""
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return SimpleNamespace(
            surface_pressure_avg={2: np.array([1.0 + 0.0j, 2.0 + 0.0j])},
            solver_log=[],
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    with pytest.raises(RuntimeError, match="did not report frequencies_hz"):
        radiation_impedance.solve_aperture_matrix(
            mesh, {"port": [2]}, freqs, normal_velocity=1.0
        )


def test_complete_sweep_still_assembles_exact_values(monkeypatch):
    """The guard must not reject a well-formed sweep.

    Tag 2 has area 0.5 m^2 and unit normal velocity, so Q = 0.5 m^3/s and
    Z = p/Q is exactly twice the reported average pressure.
    """
    mesh = _fake_three_tag_mesh()
    freqs = np.array([100.0, 200.0, 400.0])
    pressures = np.array([3.0 - 4.0j, 1.0 + 1.0j, -2.0 + 0.5j])

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(frequencies, {2: pressures})

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    result = radiation_impedance.solve_aperture_matrix(
        mesh, {"port": [2]}, freqs, normal_velocity=1.0
    )

    np.testing.assert_array_equal(result.frequencies_hz, freqs)
    np.testing.assert_allclose(result.impedance_matrix[:, 0, 0], pressures / 0.5)


def test_velocity_mode_matrix_normalizes_by_volume_velocity_v_times_area(monkeypatch):
    mesh = _fake_three_tag_mesh()

    def fake_solve_frequencies(loaded, frequencies, cfg):
        return _fake_result(
            frequencies,
            {
                2: np.array([12.0 + 0.0j]),
            },
        )

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

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

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

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
    series_resistance = 123.0

    result = radiation_impedance.terminated_chamber_port_branch(
        freqs,
        load,
        chamber_volume_m3=2.0e-5,
        port_area_m2=5.0e-4,
        port_length_m=0.02,
        interior_end_correction_length_m=0.001,
        series_resistance_pa_s_m3=series_resistance,
        rho=1.2,
        c=340.0,
    )

    omega = 2.0 * np.pi * freqs
    y_chamber = 1j * omega * (2.0e-5 / (1.2 * 340.0 * 340.0))
    z_port = 1j * omega * 1.2 * 0.021 / 5.0e-4
    y_series = 1.0 / (series_resistance + z_port + load)
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


def test_matrix_diagnostics_passivity_uses_full_hermitian_part():
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=np.array([100.0]),
        aperture_names=["a", "b"],
        aperture_area_m2={"a": 1.0, "b": 1.0},
        impedance_matrix=np.array(
            [[[1.0, 2.0j], [-2.0j, 1.0]]],
            dtype=np.complex128,
        ),
        solver_logs=[],
    )

    diagnostics = radiation_impedance.matrix_diagnostics(result)

    assert diagnostics.passivity_min_eig[0] == pytest.approx(-1.0)
    assert diagnostics.passivity_min_eig_reciprocal[0] == pytest.approx(1.0)
    # The raw failure is caused entirely by non-reciprocity. The projected
    # matrix remains passive, so the two failure modes are reported separately.
    assert diagnostics.passivity_ok[0]


def test_matrix_diagnostics_separates_subtag_scale_nonreciprocity():
    matrix = np.eye(9, dtype=np.complex128) * (1.0 + 2.0e5j)
    matrix[0, 1] += 400.0j
    matrix[1, 0] -= 400.0j
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=np.array([500.0]),
        aperture_names=[f"subtag_{idx}" for idx in range(9)],
        aperture_area_m2={f"subtag_{idx}": 1.0 for idx in range(9)},
        impedance_matrix=matrix[None, :, :],
        solver_logs=[],
    )

    diagnostics = radiation_impedance.matrix_diagnostics(result)

    assert diagnostics.reciprocity_max_rel[0] == pytest.approx(0.004)
    assert diagnostics.passivity_min_eig[0] == pytest.approx(-399.0)
    assert diagnostics.passivity_min_eig_reciprocal[0] == pytest.approx(1.0)
    assert diagnostics.passivity_ok[0]


def test_matrix_diagnostics_reciprocal_loss_remains_nonpassive():
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=np.array([100.0]),
        aperture_names=["a", "b"],
        aperture_area_m2={"a": 1.0, "b": 1.0},
        impedance_matrix=np.array(
            [[[-1.0 + 10.0j, 0.5j], [0.5j, 1.0 + 10.0j]]],
            dtype=np.complex128,
        ),
        solver_logs=[],
    )

    diagnostics = radiation_impedance.matrix_diagnostics(result)

    assert diagnostics.reciprocity_max_abs[0] == pytest.approx(0.0)
    assert diagnostics.passivity_min_eig[0] == pytest.approx(-1.0)
    assert diagnostics.passivity_min_eig_reciprocal[0] == pytest.approx(-1.0)
    assert not diagnostics.passivity_ok[0]


def test_matrix_diagnostics_relative_passivity_tolerance_scales_projection():
    result = radiation_impedance.RadiationImpedanceResult(
        frequencies_hz=np.array([100.0]),
        aperture_names=["a", "b"],
        aperture_area_m2={"a": 1.0, "b": 1.0},
        impedance_matrix=np.array(
            [[[-1.0e-6, 0.0], [0.0, 100.0]]],
            dtype=np.complex128,
        ),
        solver_logs=[],
    )

    without_relative = radiation_impedance.matrix_diagnostics(result)
    with_relative = radiation_impedance.matrix_diagnostics(
        result,
        passivity_rtol=1.0e-8,
    )

    assert not without_relative.passivity_ok[0]
    assert with_relative.passivity_ok[0]


def test_low_ka_baffled_piston_reference_scaling():
    radius_m = 0.05
    freqs = np.array([50.0, 100.0])

    z = radiation_impedance.low_ka_baffled_piston_radiation_impedance(
        radius_m,
        freqs,
    )

    assert z[1].real / z[0].real == pytest.approx(4.0)
    assert z[1].imag / z[0].imag == pytest.approx(2.0)


def test_matrix_diagnostics_low_ka_converts_solver_convention():
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
        impedance_matrix=np.conjugate(expected).reshape(2, 1, 1),
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


def _fake_result(freqs, surface_pressure_avg, *, frequencies_hz=None):
    """Stand-in for a Metal SolveResult.

    ``frequencies_hz`` defaults to the requested axis, which is what a
    complete sweep echoes back. Pass a shorter axis to fake an early stop.
    """
    if frequencies_hz is None:
        frequencies_hz = freqs
    return SimpleNamespace(
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64).reshape(-1),
        surface_pressure_avg=surface_pressure_avg,
        solver_log=[],
    )
