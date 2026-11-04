from __future__ import annotations

import numpy as np
import pytest

from hornlab_sim.methods import transfer_matrix


def test_area_discontinuity_is_identity_in_pressure_volume_velocity_domain():
    frequencies = np.array([100.0, 250.0, 1000.0])

    junction = transfer_matrix.area_discontinuity_matrix(
        2.0e-3,
        5.0e-4,
        frequencies,
    )

    expected = np.broadcast_to(np.eye(2, dtype=complex), junction.shape)
    np.testing.assert_array_equal(junction, expected)


def test_segmented_duct_cascade_needs_no_area_transformer():
    frequencies = np.array([137.0, 389.0, 911.0])
    areas = [2.0e-3, 5.0e-4]
    lengths = [0.12, 0.07]
    perimeters = [0.18, 0.09]
    matrices = [
        transfer_matrix.uniform_tube_matrix(
            length,
            area,
            frequencies,
            perimeter=perimeter,
            losses=False,
        )
        for area, length, perimeter in zip(areas, lengths, perimeters)
    ]
    expected = transfer_matrix.input_impedance(
        transfer_matrix.cascade_matrices(matrices),
        None,
    )

    actual = transfer_matrix.duct_input_impedance(
        areas,
        perimeters,
        lengths,
        frequencies,
        losses=False,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_tmm_load_rejects_different_frequency_grid_with_same_length():
    omega = 2.0 * np.pi * np.array([100.0, 200.0, 400.0])
    load = transfer_matrix.TMMLoad(
        _Z_tmm=np.array([1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j]),
        _omega_ref=omega,
    )

    with pytest.raises(ValueError, match="frequency grid does not match"):
        load.load_impedance(2.0 * np.pi * np.array([125.0, 250.0, 500.0]))


def test_tmm_load_accepts_matching_frequency_grid_within_tolerance():
    omega = 2.0 * np.pi * np.array([100.0, 200.0, 400.0])
    impedance = np.array([1.0 + 2.0j, 2.0 + 3.0j, 3.0 + 4.0j])
    load = transfer_matrix.TMMLoad(_Z_tmm=impedance, _omega_ref=omega)

    actual = load.load_impedance(omega * (1.0 + 5.0e-10))

    np.testing.assert_allclose(actual, impedance)
