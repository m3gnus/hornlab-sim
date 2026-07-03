"""Regression tests for the end-correction suppression rule.

The AGENTS.md rule for LEM->BEM coupling states: when an aperture is
being radiated by BEM, the LEM call for that aperture must pass
``end_corr="none"`` to avoid double-counting the radiation reactance
(once in the LEM's end correction, once in the BEM-computed radiation
impedance).

These tests verify that:

1. The rule has a measurable effect — i.e., for a typical slot-pocket
   geometry, suppressing the end correction changes the Helmholtz
   resonance by enough to matter (>1%). If the rule had no measurable
   effect, it wouldn't be worth enforcing.

2. The ``end_corr="none"`` path is wired correctly through the
   Helmholtz module and produces lower frequencies (because removing
   the end correction reduces L_eff, which raises f. Wait: f =
   c/(2pi) * sqrt(A/(V*L_eff)). Smaller L_eff means LARGER f.
   So end_corr="none" produces a HIGHER resonance frequency than
   end_corr="flanged_free".)

These are unit-level tests on the Helmholtz formula — they do not run
a BEM solve. They lock in the rule's behavior so future API changes
to the Helmholtz module don't silently break the LEM->BEM contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_sim.methods.bandpass import Port
from hornlab_sim.methods.helmholtz import helmholtz, slot_helmholtz


# ---------------------------------------------------------------------------
# Direct Helmholtz formula: end_corr suppression raises the resonance
# ---------------------------------------------------------------------------


def test_end_corr_none_raises_helmholtz_frequency():
    """Removing the end correction raises f (smaller L_eff -> higher f)."""
    # 130 mm x 376 mm rectangular opening, 12 L cavity, 50 mm geom length.
    # Numbers are physically representative of a full-size slot port.
    V_m3 = 0.012
    A_m2 = 0.130 * 0.376
    L_geom_m = 0.050  # 50 mm of port tube so L_eff > 0 even with no end corr

    f_with = helmholtz(V_m3, A_m2, L_geom_m=L_geom_m, end_corr="flanged_free")
    f_none = helmholtz(V_m3, A_m2, L_geom_m=L_geom_m, end_corr="none")

    # Smaller L_eff -> higher f
    assert f_none > f_with, (
        f"end_corr='none' should give a HIGHER Helmholtz frequency "
        f"(smaller L_eff). Got f_none={f_none:.1f} <= f_with={f_with:.1f}."
    )


def test_end_corr_effect_is_measurable():
    """For a typical slot geometry the end correction shifts f by >1%."""
    V_m3 = 0.012
    A_m2 = 0.130 * 0.376
    L_geom_m = 0.050

    f_with = helmholtz(V_m3, A_m2, L_geom_m=L_geom_m, end_corr="flanged_free")
    f_none = helmholtz(V_m3, A_m2, L_geom_m=L_geom_m, end_corr="none")

    rel_shift = (f_none - f_with) / f_with
    assert rel_shift > 0.01, (
        f"End-correction suppression should shift f by >1% to be worth "
        f"enforcing. Got {rel_shift*100:.2f}%."
    )


def test_end_corr_effect_dominates_at_short_geom_length():
    """As L_geom shrinks toward 0, the end correction dominates L_eff."""
    V_m3 = 0.012
    A_m2 = 0.130 * 0.376

    f_with_short = helmholtz(V_m3, A_m2, L_geom_m=0.005, end_corr="flanged_free")
    f_with_long = helmholtz(V_m3, A_m2, L_geom_m=0.100, end_corr="flanged_free")
    # Shorter L_geom means end correction is a larger fraction of L_eff,
    # so removing it has more impact. At L_geom=0.005 the end correction
    # IS most of L_eff -- changing end_corr shifts f a lot.
    f_none_short = helmholtz(V_m3, A_m2, L_geom_m=0.005, end_corr="none")
    f_none_long = helmholtz(V_m3, A_m2, L_geom_m=0.100, end_corr="none")

    shift_short = (f_none_short - f_with_short) / f_with_short
    shift_long = (f_none_long - f_with_long) / f_with_long
    assert shift_short > shift_long, (
        f"End-correction relative effect should grow as L_geom shrinks. "
        f"shift_short={shift_short*100:.1f}%, shift_long={shift_long*100:.1f}%."
    )


def test_helmholtz_rejects_zero_l_eff():
    """When end_corr='none' and L_geom_m=0, L_eff=0 and the formula errors.

    This is the degenerate case that catches the canonical workflow
    mistake of computing a slot Helmholtz with no neck AND no end
    correction. For LEM->BEM coupling there is no meaningful Helmholtz
    resonance in this case -- the BEM handles the slot's radiation
    fully. The right LEM model here is not the Helmholtz function at
    all but rather a bandpass model (bp4_sealed_rear) that doesn't
    require L_eff > 0 to be well-defined.
    """
    V_m3 = 0.012
    A_m2 = 0.130 * 0.376
    with pytest.raises(ValueError, match="L_eff must be positive"):
        helmholtz(V_m3, A_m2, L_geom_m=0.0, end_corr="none")


# ---------------------------------------------------------------------------
# slot_helmholtz: same suppression semantics on the slot-pocket wrapper
# ---------------------------------------------------------------------------


def test_slot_helmholtz_suppresses_end_corr_when_requested():
    """slot_helmholtz with end_corr suppression gives a higher
    Helmholtz frequency when the interpretation has a non-zero geometric
    port length.

    The default "slot_pocket" interpretation sets L_geom=0 (slot exit =
    baffled hole) so passing end_corr="none" degenerates to L_eff=0 and
    errors. That degeneracy is itself a signal: the canonical
    interpretation says BEM does the radiation, so the lumped resonance
    is not defined -- you need the bandpass model instead.

    The "back_cavity_long_port" interpretation uses slot_depth as L_geom
    (treating the slot as a long port into a back cavity). With
    L_geom_m > 0 the end_corr suppression is well-defined and shifts f
    upward as expected.
    """
    kwargs = dict(
        opening_W_mm=130.0,
        slot_depth_mm=500.0,
        slot_height_mm=376.0,
        interpretation="back_cavity_long_port",
    )

    r_default = slot_helmholtz(end_corr="flanged_free", **kwargs)
    r_none = slot_helmholtz(end_corr="none", **kwargs)

    f_default = float(r_default["f_Hz"])
    f_none = float(r_none["f_Hz"])

    assert f_none > f_default, (
        f"slot_helmholtz with end_corr='none' should give a "
        f"higher resonance frequency. Got f_none={f_none:.1f} <= "
        f"f_default={f_default:.1f}."
    )


def test_slot_helmholtz_slot_pocket_interp_degenerates_with_no_end_corr():
    """In the canonical slot_pocket interpretation, end_corr='none' is
    degenerate (L_eff=0). That degeneracy IS the LEM->BEM signal: when
    BEM handles the slot radiation, there is no meaningful lumped
    Helmholtz frequency and the right LEM model is bp4_sealed_rear,
    not slot_helmholtz.
    """
    with pytest.raises(ValueError, match="L_eff must be positive"):
        slot_helmholtz(
            opening_W_mm=130.0,
            slot_depth_mm=500.0,
            slot_height_mm=376.0,
            interpretation="slot_pocket",
            end_corr="none",
        )


# ---------------------------------------------------------------------------
# bandpass.Port: suppress BEM-radiated external aperture loading
# ---------------------------------------------------------------------------


def test_bandpass_port_can_suppress_external_end_correction():
    """radiation_external=False keeps the chamber-side correction only."""
    port = Port(
        area=0.130 * 0.376,
        length=0.05,
        flanged_inside=True,
        flanged_outside=True,
        radiation_external=False,
    )

    expected = port.length + port.end_correction(True)
    assert port.L_eff == pytest.approx(expected)


def test_bandpass_port_external_radiation_default_is_unchanged():
    """Default Port behavior still includes both end corrections."""
    port = Port(
        area=0.130 * 0.376,
        length=0.05,
        flanged_inside=True,
        flanged_outside=True,
    )

    expected = port.length + port.end_correction(True) + port.end_correction(True)
    assert port.L_eff == pytest.approx(expected)


def test_bandpass_port_can_suppress_external_radiation_resistance():
    """BEM-coupled ports should not add LEM-side radiation resistance."""
    omega = np.array([2 * math.pi * 200.0])
    port = Port(
        area=0.130 * 0.376,
        length=0.05,
        flanged_inside=True,
        flanged_outside=True,
        radiation_external=False,
        Q_port=math.inf,
    )

    assert np.real(port.impedance(omega))[0] == pytest.approx(0.0)
