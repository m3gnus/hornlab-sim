from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_sim.methods.bandpass import Chamber, Driver, Port
from hornlab_sim.methods.helmholtz import (
    bigmeh_mid_chamber_helmholtz,
    helmholtz,
)
from hornlab_sim.methods.port_acoustics import (
    frustum_port_acoustic_mass,
    viscothermal_port_q,
)


def test_ingard_interior_correction_matches_cafmeh_volume_shift_regression():
    """CAFMEH clay-volume regression from the archived thread.

    The measurement gives 4x dummy chambers changing from 130 mL to 64 mL,
    with the notch moving from 577 Hz to 713 Hz (+23.6%). The same post does
    not state every dummy-port dimension, so this pins explicit assumptions:
    one equivalent CAFMEH-style throat per chamber, 8.5 cm2 minimum throat
    area, 10 mm effective wall depth, and a 21.2 cm2 chamber-side flare as
    documented in design-insights §17a for the conical CAFMEH port family.
    The legacy volume-only sensitivity is independent of those dimensions.
    """
    geometry = dict(
        port_count=1,
        entry_area_cm2=8.5,
        chamber_area_cm2=21.2,
        tube_depth_mm=10.0,
        target_fc_hz=600.0,
    )
    large_legacy = bigmeh_mid_chamber_helmholtz(
        chamber_volume_cc=130.0,
        **geometry,
    )
    small_legacy = bigmeh_mid_chamber_helmholtz(
        chamber_volume_cc=64.0,
        **geometry,
    )
    large_ingard = bigmeh_mid_chamber_helmholtz(
        chamber_volume_cc=130.0,
        interior_end_correction="ingard",
        **geometry,
    )
    small_ingard = bigmeh_mid_chamber_helmholtz(
        chamber_volume_cc=64.0,
        interior_end_correction="ingard",
        **geometry,
    )

    legacy_shift = small_legacy["f_Hz"] / large_legacy["f_Hz"] - 1.0
    ingard_shift = small_ingard["f_Hz"] / large_ingard["f_Hz"] - 1.0
    measured_shift = 713.0 / 577.0 - 1.0

    assert legacy_shift == pytest.approx(math.sqrt(130.0 / 64.0) - 1.0)
    assert legacy_shift == pytest.approx(0.4252, rel=5e-4)
    assert 0.18 <= ingard_shift <= 0.30
    assert abs(ingard_shift - measured_shift) < abs(legacy_shift - measured_shift)
    assert small_ingard["interior_delta_L_m"] > large_ingard["interior_delta_L_m"]


def test_frustum_port_mass_reduces_to_uniform_tube_mass():
    rho = 1.21
    radius = 0.018
    length = 0.030
    expected = rho * length / (math.pi * radius * radius)
    assert frustum_port_acoustic_mass(radius, radius, length, rho) == pytest.approx(
        expected
    )


def test_frustum_helmholtz_frequency_rises_as_exit_flares():
    common = dict(
        chamber_volume_cc=100.0,
        port_count=1,
        entry_radius_m=0.010,
        tube_depth_mm=20.0,
        target_fc_hz=1000.0,
        port_model="frustum",
    )
    straight = bigmeh_mid_chamber_helmholtz(exit_radius_m=0.010, **common)
    flared = bigmeh_mid_chamber_helmholtz(exit_radius_m=0.020, **common)

    assert flared["f_Hz"] > straight["f_Hz"]
    assert flared["L_eff_m"] < straight["L_eff_m"]


def test_geometry_derived_port_q_is_sane_and_monotonic():
    q_small = viscothermal_port_q(1000.0, 6.0e-4, hydraulic_radius_m=0.003)
    q_large = viscothermal_port_q(1000.0, 12.0e-4, hydraulic_radius_m=0.004)

    assert 10.0 <= q_small <= 40.0
    assert 10.0 <= q_large <= 40.0
    assert q_large > q_small

    port = Port(
        area=6.0e-4,
        length=0.024,
        Q_port=None,
        Q_port_eval_hz=1000.0,
    )
    assert port.derived_Q_port(1000.0) == pytest.approx(q_small)


def test_numeric_q_port_keeps_legacy_series_loss():
    omega = np.array([2.0 * math.pi * 1000.0])
    numeric = Port(area=6.0e-4, length=0.024, Q_port=50.0)
    derived = Port(
        area=6.0e-4,
        length=0.024,
        Q_port=None,
        Q_port_eval_hz=1000.0,
    )

    assert np.real(numeric.impedance(omega))[0] != pytest.approx(
        np.real(derived.impedance(omega))[0]
    )
    assert np.real(numeric.impedance(omega))[0] == pytest.approx(
        omega[0] * numeric.Mport / 50.0
        + 1.21 * omega[0] ** 2 / (2.0 * math.pi * 343.0)
    )


def test_chamber_thermal_compliance_correction_is_opt_in_and_clamped():
    omega = np.array([2.0 * math.pi * 1000.0])
    cold_omega = np.array([2.0 * math.pi * 1.0e-3])
    adiabatic = Chamber(volume=1.0e-4)
    thermal = Chamber(volume=1.0e-4, thermal_surface_area_m2=0.10)

    assert thermal.Cab == pytest.approx(adiabatic.Cab)
    assert thermal.Cab_for(omega)[0] > adiabatic.Cab
    assert thermal.Cab_for(cold_omega)[0] == pytest.approx(1.4 * adiabatic.Cab)


def test_lr2_blocked_electrical_impedance_matches_hand_calculation():
    driver = Driver(
        Sd=0.01,
        Bl=5.0,
        Re=6.0,
        Le=0.5e-3,
        le2_h=2.0e-3,
        re2_ohm=20.0,
        Mms=0.020,
        Cms=1.0e-3,
        Rms=1.0,
    )
    omega = np.array([2.0 * math.pi * 100.0, 2.0 * math.pi * 10000.0])
    s = 1j * omega
    expected = 6.0 + s * 0.5e-3 + (s * 2.0e-3 * 20.0) / (
        s * 2.0e-3 + 20.0
    )

    assert driver.blocked_electrical_impedance(omega) == pytest.approx(expected)
    assert np.angle(expected[1]) > np.angle(expected[0])
    assert abs(expected[1]) > abs(expected[0])


def test_air_property_override_scales_helmholtz_frequency_with_sound_speed():
    f_default = helmholtz(1.0e-3, 5.0e-4, L_geom_m=0.020)
    f_warm = helmholtz(1.0e-3, 5.0e-4, L_geom_m=0.020, c=350.0, rho=1.15)

    assert f_warm / f_default == pytest.approx(350.0 / 343.0)
