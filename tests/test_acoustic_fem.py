from __future__ import annotations

import numpy as np
import pytest

from hornlab_sim.methods import acoustic_fem


def _cube_mesh(length_m: float = 0.1) -> acoustic_fem.AcousticFEMMesh:
    points = length_m * np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ],
        dtype=np.int64,
    )
    faces: dict[tuple[int, int, int], int] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for tet in tetrahedra:
        for face in (
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ):
            key = tuple(sorted(int(v) for v in face))
            counts[key] = counts.get(key, 0) + 1
            faces[key] = face
    triangles = np.asarray(
        [faces[key] for key, count in counts.items() if count == 1],
        dtype=np.int64,
    )
    centroids = np.mean(points[triangles], axis=1)
    tags = np.full(triangles.shape[0], 1, dtype=np.int64)
    tags[np.isclose(centroids[:, 0], 0.0)] = 2
    tags[np.isclose(centroids[:, 0], length_m)] = 3
    return acoustic_fem.AcousticFEMMesh(
        points_m=points,
        tetrahedra=tetrahedra,
        boundary_triangles=triangles,
        boundary_tags=tags,
        boundary_name_to_tag={"rigid": 1, "DRIVER": 2, "ENTRY": 3},
    )


def test_low_frequency_cube_matches_lumped_compliance_and_reciprocity():
    system = acoustic_fem.assemble_system(_cube_mesh(), ["DRIVER", "ENTRY"])
    frequency = 20.0
    result = acoustic_fem.solve_multiport(
        system,
        np.asarray([frequency]),
        loss_factor=0.0,
    )
    expected = 1j * acoustic_fem.RHO_AIR * acoustic_fem.C_AIR**2 / (
        2.0 * np.pi * frequency * system.volume_m3
    )
    assert result.impedance_matrix[0, 0, 0] == pytest.approx(expected, rel=0.025)
    assert result.impedance_matrix[0, 0, 1] == pytest.approx(
        result.impedance_matrix[0, 1, 0], rel=1.0e-10
    )
    assert system.boundary_areas_m2["DRIVER"] == pytest.approx(0.01)
    assert system.boundary_areas_m2["ENTRY"] == pytest.approx(0.01)


def test_zero_exterior_load_transfers_driver_flow_to_open_entry():
    system = acoustic_fem.assemble_system(_cube_mesh(), ["DRIVER", "ENTRY"])
    result = acoustic_fem.solve_multiport(
        system,
        np.asarray([10.0, 20.0]),
        loss_factor=0.0,
    )
    coupled = acoustic_fem.couple_exterior_impedance(
        result,
        np.zeros((2, 1, 1), dtype=np.complex128),
        driver_boundary="DRIVER",
        entry_boundaries=["ENTRY"],
    )
    assert coupled.entry_to_driver_volume_velocity[:, 0] == pytest.approx(
        np.ones(2), rel=0.015
    )
    assert np.abs(coupled.driver_acoustic_load[0]) < 0.02 * abs(
        result.impedance_matrix[0, 0, 0]
    )


def test_mesh_validation_rejects_missing_boundary_group():
    with pytest.raises(ValueError, match="missing boundary groups"):
        acoustic_fem.assemble_system(_cube_mesh(), ["NOT_PRESENT"])
