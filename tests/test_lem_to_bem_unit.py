"""Unit tests for the LEM->BEM coupling layer.

These tests do not require a real BEM solve — they exercise the input
validation, aperture-area computation, and per-frequency dispatching with
the solver mocked. A separate integration test (test_lem_to_bem_smoke.py)
runs an actual BEM solve end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_sim.methods import lem_to_bem
from hornlab_sim.methods.lem_to_bem import _aperture_face_areas, _concat_results


# ---------------------------------------------------------------------------
# Test fixtures: a fake mesh and a fake SolveResult so we can avoid Metal.
# ---------------------------------------------------------------------------


class _FakeVelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


@dataclass
class _FakeConfig:
    mesh_scale: float = 1.0
    velocity_mode: str = _FakeVelocityMode.ACCELERATION
    velocity_sources: dict[int, complex] = field(default_factory=dict)


@dataclass
class _FakeSolveResult:
    frequencies_hz: np.ndarray
    pressure_complex: np.ndarray
    directivity_db: np.ndarray
    impedance: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)
    solver_log: list[dict] = field(default_factory=list)
    surface_pressure_avg: dict[int, np.ndarray] | None = None

    @property
    def spl_db(self):
        return self.directivity_db


def _patch_metal_api(monkeypatch, solve_frequencies):
    api = SimpleNamespace(
        name="metal",
        load_mesh=lambda path, scale=1.0: path,
        solve_frequencies=solve_frequencies,
        VelocityMode=_FakeVelocityMode,
        default_config=lambda formulation: _FakeConfig(),
    )
    monkeypatch.setattr(lem_to_bem, "_metal_api", lambda config=None: api)
    return api


def _fake_unit_square_mesh(extra_tags: dict[int, list[tuple[float, float, float]]] | None = None):
    """Build a fake LoadedMesh with two triangles forming a 1x1 unit square.

    Triangle 0: (0,0,0)-(1,0,0)-(0,1,0)  -> physical tag 2 -> 0.5 m^2
    Triangle 1: (1,0,0)-(1,1,0)-(0,1,0)  -> physical tag 3 -> 0.5 m^2
    """
    # grid object only needs .vertices and .elements as numpy arrays
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )  # (4, 3)
    elements = np.array(
        [
            [0, 1, 2],  # tag 2
            [1, 3, 2],  # tag 3
        ],
        dtype=np.int32,
    )  # (2, 3)

    grid = SimpleNamespace(vertices=vertices, elements=elements)
    physical_tags = np.array([2, 3], dtype=np.int32)
    info = SimpleNamespace(n_triangles=2)
    return SimpleNamespace(grid=grid, physical_tags=physical_tags, info=info)


# ---------------------------------------------------------------------------
# Area computation
# ---------------------------------------------------------------------------


def test_aperture_face_areas_single_tag():
    mesh = _fake_unit_square_mesh()
    areas = _aperture_face_areas(mesh, {"tri_a": [2]})
    assert areas["tri_a"] == pytest.approx(0.5)


def test_aperture_face_areas_multi_tag_sums():
    """Multi-tag aperture: total area is the sum across all listed tags."""
    mesh = _fake_unit_square_mesh()
    areas = _aperture_face_areas(mesh, {"whole_square": [2, 3]})
    assert areas["whole_square"] == pytest.approx(1.0)


def test_aperture_face_areas_unknown_tag_raises():
    mesh = _fake_unit_square_mesh()
    with pytest.raises(ValueError, match=r"tags \[99\] not present"):
        _aperture_face_areas(mesh, {"missing": [99]})


def test_aperture_face_areas_empty_tag_list_raises():
    mesh = _fake_unit_square_mesh()
    with pytest.raises(ValueError, match="empty tag list"):
        _aperture_face_areas(mesh, {"empty": []})


# ---------------------------------------------------------------------------
# Input validation through solve()
# ---------------------------------------------------------------------------


def test_aperture_name_mismatch_raises():
    freqs = np.array([100.0, 200.0])
    U = {"throat": np.ones(2, dtype=complex)}
    tags = {"slot": [3]}
    with pytest.raises(ValueError, match="Aperture name mismatch"):
        lem_to_bem.solve(
            mesh="dummy.msh", lem_velocities=U, aperture_tags=tags, frequencies_hz=freqs
        )


def test_velocity_shape_mismatch_raises():
    freqs = np.array([100.0, 200.0, 300.0])
    U = {"throat": np.ones(2, dtype=complex)}  # wrong: should be 3
    tags = {"throat": [2]}
    with pytest.raises(ValueError, match=r"velocity shape .* expected"):
        lem_to_bem.solve(
            mesh="dummy.msh", lem_velocities=U, aperture_tags=tags, frequencies_hz=freqs
        )


def test_empty_frequencies_raises():
    U = {"throat": np.array([], dtype=complex)}
    tags = {"throat": [2]}
    with pytest.raises(ValueError, match="frequencies_hz is empty"):
        lem_to_bem.solve(
            mesh="dummy.msh", lem_velocities=U, aperture_tags=tags,
            frequencies_hz=np.array([]),
        )


# ---------------------------------------------------------------------------
# Area-mismatch warning (does not raise)
# ---------------------------------------------------------------------------


def test_area_mismatch_warns_but_does_not_throw(monkeypatch):
    """Two apertures with very different face areas -> warning emitted."""
    # Build a mesh where tag 2 is half the unit square (0.5 m^2)
    # and tag 3 is the other half (0.5 m^2). Then assign a fake third
    # aperture by remapping. Simpler: extend the fake mesh to have
    # three different-area faces.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    elements = np.array(
        [
            [0, 1, 2],  # small (0.5 m^2) tag 2
            [1, 3, 4],  # large area tag 3
            [1, 4, 2],  # tag 3 again
        ],
        dtype=np.int32,
    )
    grid = SimpleNamespace(vertices=vertices, elements=elements)
    physical_tags = np.array([2, 3, 3], dtype=np.int32)
    info = SimpleNamespace(n_triangles=3)
    mesh = SimpleNamespace(grid=grid, physical_tags=physical_tags, info=info)

    # Mock solve_frequencies to avoid actually solving
    fake_result = _fake_solve_result(freqs=[100.0])

    def fake_solve_frequencies(loaded, freqs_list, cfg):
        return fake_result

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    freqs = np.array([100.0])
    U = {
        "tiny": np.ones(1, dtype=complex),
        "huge": np.ones(1, dtype=complex),
    }
    tags = {"tiny": [2], "huge": [3]}

    with pytest.warns(UserWarning, match="differs from mean"):
        lem_to_bem.solve(
            mesh=mesh,
            lem_velocities=U,
            aperture_tags=tags,
            frequencies_hz=freqs,
            area_tolerance=0.05,
        )


# ---------------------------------------------------------------------------
# Velocity-source dict construction (the hot path of the per-frequency loop)
# ---------------------------------------------------------------------------


def test_velocity_sources_dict_captures_complex_u_over_area(monkeypatch):
    """For one frequency with one aperture, v_n = U / A and is complex."""
    mesh = _fake_unit_square_mesh()
    captured = []

    def fake_solve_frequencies(loaded, freqs, cfg):
        captured.append({"freqs": list(freqs), "sources": dict(cfg.velocity_sources)})
        return _fake_solve_result(freqs=freqs)

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    freqs = np.array([200.0, 500.0])
    U = np.array([3.0 + 4.0j, 1.0 - 1.0j])  # m^3/s
    lem_to_bem.solve(
        mesh=mesh,
        lem_velocities={"tri_a": U},
        aperture_tags={"tri_a": [2]},  # tag 2 has area 0.5 m^2
        frequencies_hz=freqs,
    )

    # Two single-freq calls
    assert len(captured) == 2
    # v_n at f=200: (3+4j) / 0.5 = 6+8j
    assert captured[0]["sources"][2] == pytest.approx(6.0 + 8.0j)
    # v_n at f=500: (1-1j) / 0.5 = 2-2j
    assert captured[1]["sources"][2] == pytest.approx(2.0 - 2.0j)


def test_multi_tag_aperture_applies_same_vn_to_each_tag(monkeypatch):
    """A multi-tag aperture writes the same v_n to every listed physical group."""
    mesh = _fake_unit_square_mesh()
    captured = []

    def fake_solve_frequencies(loaded, freqs, cfg):
        captured.append(dict(cfg.velocity_sources))
        return _fake_solve_result(freqs=freqs)

    _patch_metal_api(monkeypatch, fake_solve_frequencies)

    freqs = np.array([100.0])
    U = np.array([2.0 + 0.0j])
    # aperture covers tags 2 and 3, total area 1.0 m^2 (0.5 + 0.5)
    lem_to_bem.solve(
        mesh=mesh,
        lem_velocities={"whole_face": U},
        aperture_tags={"whole_face": [2, 3]},
        frequencies_hz=freqs,
    )

    # v_n = 2.0 / 1.0 = 2+0j, applied to both tag 2 and tag 3
    assert captured[0][2] == pytest.approx(2.0 + 0.0j)
    assert captured[0][3] == pytest.approx(2.0 + 0.0j)


# ---------------------------------------------------------------------------
# Concatenation helper
# ---------------------------------------------------------------------------


def test_concat_results_stacks_along_freq_axis():
    r1 = _fake_solve_result(freqs=[100.0])
    r2 = _fake_solve_result(freqs=[200.0])
    out = _concat_results([r1, r2], frequencies_hz=[100.0, 200.0])
    assert out.pressure_complex.shape[0] == 2
    assert out.spl_db.shape[0] == 2
    assert out.impedance.shape[0] == 2
    assert list(out.frequencies_hz) == [100.0, 200.0]


# ---------------------------------------------------------------------------
# Fake SolveResult helper (avoids importing solver internals)
# ---------------------------------------------------------------------------


def _fake_solve_result(freqs):
    """Build a minimal stand-in for hornlab_metal_bem.SolveResult."""
    n_freq = len(freqs)
    n_planes = 2
    n_angles = 5

    return _FakeSolveResult(
        frequencies_hz=np.asarray(freqs, dtype=np.float64),
        pressure_complex=np.zeros((n_freq, n_planes, n_angles), dtype=np.complex128),
        directivity_db=np.zeros((n_freq, n_planes, n_angles), dtype=np.float64),
        impedance=np.zeros(n_freq, dtype=np.complex128),
        timings={},
        solver_log=[],
        surface_pressure_avg=None,
    )
