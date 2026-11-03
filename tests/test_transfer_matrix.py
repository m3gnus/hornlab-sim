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


@pytest.mark.parametrize("environment", [{"rho": 2.42}, {"c": 350.0}])
def test_precomputed_tmm_load_rejects_runtime_air_mismatch(environment):
    omega = 2 * np.pi * np.array([100.0, 200.0])
    load = transfer_matrix.TMMLoad(np.ones(2, dtype=complex), omega)
    with pytest.raises(ValueError, match="air properties.*rebuild"):
        load.load_impedance(omega, **environment)


@pytest.mark.parametrize("termination", ["rigid", "baffled", "unbaffled"])
def test_duct_construction_uses_requested_density_and_sound_speed(termination):
    freqs = np.array([100.0, 300.0])
    kwargs = dict(areas=[0.002], perimeters=[0.18], lengths=[0.1],
                  freq=freqs, termination=termination, losses=False)
    default = transfer_matrix.duct_input_impedance(**kwargs)
    doubled = transfer_matrix.duct_input_impedance(**kwargs, rho=2 * transfer_matrix.RHO)
    np.testing.assert_allclose(doubled, 2 * default, rtol=1e-14)
    # Doubling both frequency and sound speed keeps kL and radiation ka fixed.
    shifted = transfer_matrix.duct_input_impedance(
        **{**kwargs, "freq": 2 * freqs}, c=2 * transfer_matrix.C_SOUND,
    )
    np.testing.assert_allclose(shifted, 2 * default, rtol=1e-14)


def test_slot_tmm_load_retains_air_provenance_including_port():
    from hornlab_sim.methods.bandpass import Port
    freqs = np.array([100.0, 200.0])
    port = Port(area=0.002, length=0.02)
    geometry = dict(slot_opening_W_mm=20.0, slot_height_mm=100.0,
                    apex_depth_mm=100.0, apex_width_mm=10.0, freq=freqs)
    rho, c = 1.3, 350.0
    load = transfer_matrix.make_slot_tmm_load(**geometry, exit_port=port, rho=rho, c=c)
    cavity = transfer_matrix.slot_pocket_impedance(**geometry, rho=rho, c=c)
    omega = 2 * np.pi * freqs
    expected = 1 / (1 / cavity + 1 / port.impedance(omega, rho=rho, c=c))
    np.testing.assert_array_equal(load.load_impedance(omega, rho=rho, c=c), expected)
    with pytest.raises(ValueError, match="air properties.*rebuild"):
        load.load_impedance(omega)


def test_viscothermal_wavenumber_uses_construction_density():
    omega = 2 * np.pi * np.array([100.0, 200.0])
    default = transfer_matrix._complex_wavenumber(omega, 0.002, 0.18, True)
    denser = transfer_matrix._complex_wavenumber(
        omega, 0.002, 0.18, True, rho=4 * transfer_matrix.RHO,
    )
    lossless = omega / transfer_matrix.C_SOUND
    np.testing.assert_allclose(denser - lossless, (default - lossless) / 2, rtol=1e-12)


def test_simulate_rejects_tmm_load_from_a_different_environment():
    from hornlab_sim.methods.bandpass import Chamber, Driver, simulate
    freqs = np.array([100.0, 200.0])
    load = transfer_matrix.make_slot_tmm_load(20.0, 100.0, 100.0, 10.0, freqs)
    driver = Driver(Sd=0.013, Bl=5.8, Re=5.6, Mms=0.018, Cms=0.00085)
    with pytest.raises(ValueError, match="air properties.*rebuild"):
        simulate(driver, load, Chamber(0.005), freqs, rho=2 * transfer_matrix.RHO)
