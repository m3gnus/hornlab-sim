"""Forward LEM → BEM coupling.

Drive a BEM solve with per-aperture, per-frequency complex volume velocities
computed by a lumped (LEM) network. The LEM-emitted volume velocity ``U`` at
each aperture is converted to normal velocity ``v_n = U / A_face`` and
imposed as a Neumann boundary condition on the matching mesh physical
groups.

This is the one-way ("forward") coupling. BEM-side radiation impedance is
not fed back into the LEM solve. That is sufficient for directivity
heatmaps in the BEM-dominated range (above the chamber/slot Helmholtz
features) but will systematically miss back-loading at slot resonances and
sharp BEM-side impedance features. Use the canonical ABEC / AKABAK
formulation (Pawera 2012, Acoustics Nantes) if/when that matters.

Acoustic engineering rules — these are load-bearing, see
`hornlab-sim/AGENTS.md` for the full statement:

1. **End-correction suppression at BEM-coupled apertures.** The LEM call
   that produces the aperture velocity ``U`` MUST pass ``end_corr="none"``
   for any aperture whose radiation impedance is being computed by BEM.
   Otherwise the end correction is double-counted (once by the LEM's
   radiation reactance, once by the BEM-computed radiation impedance). The
   coupling layer cannot detect or fix this misuse; it is the caller's
   responsibility.

2. **Aperture → list of physical group IDs (multi-tag).** Each aperture
   name maps to a list of physical group IDs, not 1:1. BIGMEH meshes
   typically produce more than one tag per slot (exit + walls). The
   coupling layer applies ``v_n`` to every tag in the list.

3. **Area mismatch is a warning, not an error.** Mesh-side total face area
   for an aperture may differ from the LEM-assumed throat area S by up to
   a few percent due to mesh discretization (chamfers, curvature). Warned
   above ``area_tolerance``, not raised.

4. **Velocity mode.** This module uses ``VelocityMode.VELOCITY`` internally
   (overriding the canonical ``ACCELERATION`` default), since LEM provides
   actual velocity directly. Physically equivalent to passing
   ``j·omega·v`` in ``ACCELERATION`` mode.

5. **Phase reference.** All LEM aperture velocities share a common
   excitation reference (driver terminal voltage). The ``+i·omega·rho·v_n``
   Neumann data convention here matches the canonical ``hornlab-bempp-bem``
   sign convention.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

import numpy as np
from numpy.typing import NDArray

from . import _bem_backend

if TYPE_CHECKING:
    from hornlab_bempp_bem import SolveConfig, SolveResult
    from hornlab_bempp_bem.mesh import LoadedMesh


MeshLike = Union[str, Path, "LoadedMesh", Any]


def solve(
    mesh: MeshLike,
    lem_velocities: Mapping[str, NDArray[np.complex128]],
    aperture_tags: Mapping[str, list[int]],
    frequencies_hz: NDArray[np.float64],
    config: Any | None = None,
    *,
    area_tolerance: float = 0.05,
) -> Any:
    """Solve BEM with LEM-derived complex velocity sources per aperture.

    Parameters
    ----------
    mesh : path or ``LoadedMesh``
        Mesh with physical groups matching every value in ``aperture_tags``.
    lem_velocities : dict[str, complex array]
        Per-aperture complex volume velocity ``U(f)`` in m^3/s. Each value
        must be a 1-D array of length ``len(frequencies_hz)``.
    aperture_tags : dict[str, list[int]]
        Per-aperture list of physical group IDs for the BEM mesh faces that
        belong to that aperture. Multi-tag aware (one slot may produce
        several tags).
    frequencies_hz : array of float
        Frequencies to solve at, in Hz.
    config : SolveConfig, optional
        Solver configuration from ``hornlab_metal_bem`` or
        ``hornlab_bempp_bem``. The config type selects the backend. If
        omitted, ``HORNLAB_SIM_BEM_BACKEND=metal|bempp`` selects a backend;
        unset or ``auto`` prefers available native Metal and falls back to
        bempp. The default formulation remains ``COMPLEX_K``. The coupling
        layer always overrides ``velocity_mode`` and ``velocity_sources``
        per frequency.
    area_tolerance : float, default 0.05
        Per-aperture area-spread threshold for the diagnostic warning. The
        spread is measured between apertures; useful for catching a
        miscoded tag list. No threshold check is performed against any
        LEM-assumed S (the LEM passes its U directly).

    Returns
    -------
    SolveResult
        Selected backend's SolveResult with complex pressure of shape
        ``(n_freq, n_planes, n_angles)``. The per-aperture v_n applied at
        each frequency is recorded in ``result.solver_log``.

    Raises
    ------
    ValueError
        Aperture name mismatch between ``lem_velocities`` and
        ``aperture_tags``, velocity array shape mismatch with frequencies,
        or zero-area aperture.

    Notes
    -----
    Implementation: this calls the solver once per frequency with mutated
    ``config.velocity_sources``. That is correct but does not amortize the
    BEM matrix factorization across velocity-source perturbations. A
    future optimization would refactor the solver to share factorization
    across multiple right-hand sides (the ABEC superposition trick); for
    now the layer is correct but not asymptotically optimal.
    """
    # ----- Validate input shapes / names ----------------------------------

    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    n_freq = freqs.size
    if n_freq == 0:
        raise ValueError("frequencies_hz is empty")

    lem_keys = set(lem_velocities)
    tag_keys = set(aperture_tags)
    only_lem = lem_keys - tag_keys
    only_tags = tag_keys - lem_keys
    if only_lem or only_tags:
        raise ValueError(
            "Aperture name mismatch between lem_velocities and aperture_tags. "
            f"Only in lem_velocities: {sorted(only_lem) or '[]'}. "
            f"Only in aperture_tags: {sorted(only_tags) or '[]'}."
        )

    for name, U in lem_velocities.items():
        U_arr = np.asarray(U)
        if U_arr.shape != (n_freq,):
            raise ValueError(
                f"Aperture {name!r} velocity shape {U_arr.shape}, "
                f"expected ({n_freq},)"
            )

    # ----- Resolve config -------------------------------------------------

    backend = _bem_backend.resolve_backend(config)
    api = _bem_backend.backend_api(backend)
    if config is None:
        config = api.default_config("complex_k")

    # ----- Load mesh once -------------------------------------------------

    if isinstance(mesh, (str, Path)):
        loaded = api.load_mesh(mesh, scale=config.mesh_scale)
    else:
        loaded = mesh

    # ----- Compute aperture face areas -----------------------------------

    aperture_area_m2 = _aperture_face_areas(loaded, aperture_tags)

    # Diagnostic: warn if apertures disagree wildly on area. This catches
    # most miscoded tag lists (e.g. wall vs exit confusion) without forcing
    # a hard equality check against a user-supplied S.
    if len(aperture_area_m2) >= 2:
        areas = np.array(list(aperture_area_m2.values()))
        mean_a = areas.mean()
        for name, a in aperture_area_m2.items():
            rel = abs(a - mean_a) / mean_a if mean_a > 0 else 0.0
            if rel > area_tolerance:
                warnings.warn(
                    f"Aperture {name!r} face area {a*1e4:.2f} cm^2 differs "
                    f"from mean {mean_a*1e4:.2f} cm^2 by "
                    f"{rel*100:.1f}% (> area_tolerance={area_tolerance*100:.0f}%). "
                    "Verify aperture_tags lists the right physical groups.",
                    stacklevel=2,
                )

    # ----- Per-frequency solve -------------------------------------------

    # The coupling layer always uses VELOCITY mode regardless of caller's
    # config, because LEM emits volume velocity directly.
    base_config = replace(config, velocity_mode=api.VelocityMode.VELOCITY)

    per_freq_results = []
    per_freq_log = []
    for i, f in enumerate(freqs):
        # Build the per-frequency velocity_sources dict.
        # v_n[tag] = U_aperture(f) / A_aperture
        sources: dict[int, complex] = {}
        log_entry: dict[str, complex] = {}
        for name, tags in aperture_tags.items():
            U_i = complex(lem_velocities[name][i])
            v_n = U_i / aperture_area_m2[name]
            log_entry[name] = v_n
            for tag in tags:
                # If multiple apertures share a tag, last-wins. Document
                # as a misuse case; we currently do not detect it.
                sources[int(tag)] = v_n

        freq_config = replace(base_config, velocity_sources=sources)
        single = api.solve_frequencies(loaded, [float(f)], freq_config)
        per_freq_results.append(single)
        per_freq_log.append({"frequency_hz": float(f), "v_n_per_aperture": log_entry})

    aggregated = _concat_results(per_freq_results, freqs)
    aggregated.solver_log.append({"lem_to_bem": per_freq_log})
    return aggregated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aperture_face_areas(
    loaded: "LoadedMesh",
    aperture_tags: Mapping[str, list[int]],
) -> dict[str, float]:
    """Return total face area in m^2 per aperture name."""
    grid = loaded.grid

    # Bempp Grid stores vertices as (3, n_vertices) and elements as
    # (3, n_elements) with vertex indices. Defensive transpose.
    vertices = np.asarray(grid.vertices)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    elements = np.asarray(grid.elements)
    if elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    tags = loaded.physical_tags

    result: dict[str, float] = {}
    for name, tag_list in aperture_tags.items():
        if not tag_list:
            raise ValueError(f"Aperture {name!r} has empty tag list")
        mask = np.isin(tags, list(tag_list))
        if not mask.any():
            raise ValueError(
                f"Aperture {name!r} tags {list(tag_list)} not present in mesh "
                f"(available physical tags: {sorted(set(int(t) for t in tags))})"
            )
        area = float(tri_areas[mask].sum())
        if area <= 0.0:
            raise ValueError(
                f"Aperture {name!r} computed zero total face area (tags {tag_list})"
            )
        result[name] = area
    return result


def _concat_results(per_freq_results, frequencies_hz):
    """Concatenate single-frequency SolveResults along the frequency axis."""
    from dataclasses import replace as _replace

    if not per_freq_results:
        raise ValueError("no per-frequency results to concatenate")

    first = per_freq_results[0]

    pressure_complex = np.concatenate(
        [r.pressure_complex for r in per_freq_results], axis=0
    )
    directivity_db = np.concatenate(
        [r.directivity_db for r in per_freq_results], axis=0
    )
    impedance = np.concatenate([r.impedance for r in per_freq_results], axis=0)
    spl_field = _bem_backend.normalized_spl_field_name(first)

    # Per-tag surface pressure (if populated)
    surface_pressure_avg = None
    if first.surface_pressure_avg is not None:
        surface_pressure_avg = {}
        for tag in first.surface_pressure_avg:
            surface_pressure_avg[tag] = np.concatenate(
                [r.surface_pressure_avg[tag] for r in per_freq_results], axis=0
            )

    return _replace(
        first,
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
        pressure_complex=pressure_complex,
        **{spl_field: directivity_db},
        impedance=impedance,
        surface_pressure_avg=surface_pressure_avg,
        solver_log=[entry for r in per_freq_results for entry in r.solver_log],
        timings={
            "lem_to_bem_total_s": sum(
                r.timings.get("total_s", 0.0) for r in per_freq_results
            )
        },
    )
