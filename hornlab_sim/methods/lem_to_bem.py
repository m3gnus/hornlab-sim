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
   name maps to a list of physical group IDs, not 1:1. Parametric cabinet
   meshes typically produce more than one tag per slot (exit + walls). The
   coupling layer applies ``v_n`` to every tag in the list. A physical group
   may belong to only one aperture; overlapping tag lists are rejected.

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
   Neumann data convention here matches the canonical Metal sign convention.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from hornlab_metal_bem.mesh import LoadedMesh


MeshLike = Union[str, Path, "LoadedMesh", Any]
METAL_EXTRA_INSTALL_HINT = 'pip install "hornlab-sim[metal]"'


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
        Solver configuration from ``hornlab_metal_bem``. If omitted, a
        native Metal ``COMPLEX_K`` config is used. The coupling
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
        Metal SolveResult with complex pressure of shape
        ``(n_freq, n_planes, n_angles)``. The per-aperture v_n applied at
        each frequency is recorded in ``result.solver_log``.

    Raises
    ------
    ValueError
        Aperture name mismatch between ``lem_velocities`` and
        ``aperture_tags``, velocity array shape mismatch with frequencies,
        overlapping physical tags, or zero-area aperture.

    Notes
    -----
    On Metal versions with ``solve_multi_source``, the coupling layer solves
    one unit-velocity basis per aperture over the full frequency array, then
    combines those complex fields with ``U_a(f) / A_a``. This shares each
    frequency's assembly and factorization across aperture right-hand sides.
    Older Metal versions, and configs with callbacks whose observable behavior
    must be preserved, use the original per-frequency solve loop.
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

    tag_owners: dict[int, str] = {}
    for name, tags in aperture_tags.items():
        for raw_tag in tags:
            tag = int(raw_tag)
            owner = tag_owners.get(tag)
            if owner is not None and owner != name:
                raise ValueError(
                    f"Physical tag {tag} is assigned to multiple apertures: "
                    f"{owner!r} and {name!r}"
                )
            tag_owners[tag] = name

    for name, U in lem_velocities.items():
        U_arr = np.asarray(U)
        if U_arr.shape != (n_freq,):
            raise ValueError(
                f"Aperture {name!r} velocity shape {U_arr.shape}, "
                f"expected ({n_freq},)"
            )

    # ----- Resolve config -------------------------------------------------

    api = _metal_api(config)
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

    # ----- Solve and combine ---------------------------------------------

    # The coupling layer always uses VELOCITY mode regardless of caller's
    # config, because LEM emits volume velocity directly.
    base_config = replace(config, velocity_mode=api.VelocityMode.VELOCITY)

    aperture_names = list(aperture_tags)
    aperture_indices = {
        name: index for index, name in enumerate(aperture_names)
    }
    velocity_weights = np.column_stack(
        [
            np.asarray(lem_velocities[name], dtype=np.complex128)
            / aperture_area_m2[name]
            for name in aperture_names
        ]
    )
    per_freq_log = [
        {
            "frequency_hz": float(frequency),
            "v_n_per_aperture": {
                name: complex(velocity_weights[index, aperture_index])
                for aperture_index, name in enumerate(aperture_names)
            },
        }
        for index, frequency in enumerate(freqs)
    ]

    solve_multi = getattr(api, "solve_multi_source", None)
    callback_names = (
        "on_frequency_result",
        "progress_callback",
        "velocity_source_callback",
    )
    has_callbacks = any(
        getattr(base_config, name, None) is not None for name in callback_names
    )
    if solve_multi is not None and not has_callbacks:
        all_tags = sorted(tag_owners)
        basis_sources: list[dict[int, complex]] = []
        for name in aperture_names:
            sources = {tag: 0.0 + 0.0j for tag in all_tags}
            for tag in aperture_tags[name]:
                sources[int(tag)] = 1.0 + 0.0j
            basis_sources.append(sources)

        basis_results = solve_multi(loaded, freqs, basis_sources, base_config)
        first_sources = {
            tag: complex(velocity_weights[0, aperture_indices[owner]])
            for tag, owner in tag_owners.items()
        }
        combined_config = replace(base_config, velocity_sources=first_sources)
        aggregated = _combine_basis_results(
            basis_results,
            velocity_weights,
            freqs,
            combined_config=combined_config,
        )
        aggregated.solver_log.append({"lem_to_bem": per_freq_log})
        return aggregated

    per_freq_results = []
    for i, f in enumerate(freqs):
        # Build the per-frequency velocity_sources dict.
        # v_n[tag] = U_aperture(f) / A_aperture
        sources: dict[int, complex] = {}
        for name, tags in aperture_tags.items():
            U_i = complex(lem_velocities[name][i])
            v_n = U_i / aperture_area_m2[name]
            for tag in tags:
                sources[int(tag)] = v_n

        freq_config = replace(base_config, velocity_sources=sources)
        single = api.solve_frequencies(loaded, [float(f)], freq_config)
        per_freq_results.append(single)

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
    tri_areas = _triangle_face_areas(loaded)
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


def _triangle_face_areas(loaded: "LoadedMesh") -> NDArray[np.float64]:
    """Return the area of every triangular mesh face in m^2."""
    grid = loaded.grid

    # Some mesh loaders expose vertices/elements transposed. Normalize to
    # row-major arrays before computing triangle areas.
    vertices = np.asarray(grid.vertices)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    elements = np.asarray(grid.elements)
    if elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _concat_results(per_freq_results, frequencies_hz):
    """Concatenate single-frequency SolveResults along the frequency axis."""
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

    # Per-tag surface pressure (if populated)
    surface_pressure_avg = None
    if first.surface_pressure_avg is not None:
        surface_pressure_avg = {}
        for tag in first.surface_pressure_avg:
            surface_pressure_avg[tag] = np.concatenate(
                [r.surface_pressure_avg[tag] for r in per_freq_results], axis=0
            )

    surface_pressure_complex = None
    surface_fields = [
        getattr(result, "surface_pressure_complex", None)
        for result in per_freq_results
    ]
    if all(field is not None for field in surface_fields):
        surface_pressure_complex = np.concatenate(surface_fields, axis=0)
    elif any(field is not None for field in surface_fields):
        raise ValueError("per-frequency surface-pressure fields differ")

    native_diagnostics = [
        entry
        for result in per_freq_results
        for entry in getattr(result, "native_diagnostics", [])
    ]

    optional_fields = {}
    if hasattr(first, "surface_pressure_complex"):
        optional_fields["surface_pressure_complex"] = surface_pressure_complex
    if hasattr(first, "native_diagnostics"):
        optional_fields["native_diagnostics"] = native_diagnostics
    return replace(
        first,
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
        pressure_complex=pressure_complex,
        directivity_db=directivity_db,
        impedance=impedance,
        surface_pressure_avg=surface_pressure_avg,
        solver_log=[entry for r in per_freq_results for entry in r.solver_log],
        timings={
            "lem_to_bem_total_s": sum(
                r.timings.get("total_s", 0.0) for r in per_freq_results
            )
        },
        **optional_fields,
    )


def _combine_basis_results(
    basis_results,
    velocity_weights,
    frequencies_hz,
    *,
    combined_config,
):
    """Linearly combine unit-aperture SolveResults at each frequency."""
    if not basis_results:
        raise ValueError("no aperture basis results to combine")
    weights = np.asarray(velocity_weights, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    expected = (frequencies.size, len(basis_results))
    if weights.shape != expected:
        raise ValueError(
            f"velocity_weights shape {weights.shape}, expected {expected}"
        )

    def weighted_result_field(name: str):
        values = np.stack(
            [np.asarray(getattr(result, name)) for result in basis_results],
            axis=0,
        )
        if values.shape[1] != frequencies.size:
            raise ValueError(
                f"basis result {name} frequency count {values.shape[1]}, "
                f"expected {frequencies.size}"
            )
        return np.einsum("fs,sf...->f...", weights, values)

    pressure_complex = weighted_result_field("pressure_complex")
    impedance = weighted_result_field("impedance")
    first = basis_results[0]
    angles = np.asarray(first.observation_angles_deg, dtype=np.float64)
    on_axis_index = int(np.argmin(np.abs(angles)))
    floor_amplitude = 20.0e-6 * 10.0 ** (-120.0 / 20.0)
    amplitudes = np.maximum(np.abs(pressure_complex), floor_amplitude)
    spl_raw = 20.0 * np.log10(amplitudes / 20.0e-6)
    directivity_db = spl_raw - spl_raw[..., on_axis_index][..., None]

    surface_pressure_avg = None
    if first.surface_pressure_avg is not None:
        tags = tuple(first.surface_pressure_avg)
        for result in basis_results[1:]:
            if result.surface_pressure_avg is None or set(
                result.surface_pressure_avg
            ) != set(tags):
                raise ValueError("aperture basis surface-pressure tags differ")
        surface_pressure_avg = {}
        for tag in tags:
            values = np.stack(
                [
                    np.asarray(result.surface_pressure_avg[tag])
                    for result in basis_results
                ],
                axis=0,
            )
            surface_pressure_avg[tag] = np.einsum("fs,sf->f", weights, values)

    surface_fields = [
        getattr(result, "surface_pressure_complex", None)
        for result in basis_results
    ]
    surface_pressure_complex = None
    if all(field is not None for field in surface_fields):
        values = np.stack(
            [np.asarray(field) for field in surface_fields],
            axis=0,
        )
        surface_pressure_complex = np.einsum(
            "fs,sf...->f...", weights, values
        )
    elif any(field is not None for field in surface_fields):
        raise ValueError("aperture basis surface-pressure fields differ")

    solver_log = _combine_basis_solver_logs(
        basis_results,
        weights,
        frequencies,
    )
    optional_fields = {}
    if hasattr(first, "surface_pressure_complex"):
        optional_fields["surface_pressure_complex"] = surface_pressure_complex
    return replace(
        first,
        frequencies_hz=np.array(frequencies, copy=True),
        pressure_complex=np.asarray(pressure_complex, dtype=np.complex128),
        directivity_db=np.asarray(directivity_db, dtype=np.float64),
        impedance=np.asarray(impedance, dtype=np.complex128),
        config=combined_config,
        timings={
            "lem_to_bem_total_s": sum(
                result.timings.get("total_s", 0.0) for result in basis_results
            )
        },
        solver_log=solver_log,
        surface_pressure_avg=surface_pressure_avg,
        **optional_fields,
    )


def _combine_basis_solver_logs(basis_results, weights, frequencies):
    """Reconstruct one combined-source solver-log entry per frequency."""
    frequency_count = len(frequencies)
    basis_logs: list[list[dict]] = []
    for result in basis_results:
        logs = [
            entry
            for entry in result.solver_log
            if isinstance(entry, dict) and "frequency_hz" in entry
        ]
        if len(logs) != frequency_count:
            raise ValueError(
                "aperture basis solver-log frequency count differs from result"
            )
        basis_logs.append(logs)

    combined_logs: list[dict] = []
    for frequency_index, frequency in enumerate(frequencies):
        source_entries = [logs[frequency_index] for logs in basis_logs]
        entry = dict(source_entries[0])
        entry["frequency_hz"] = float(frequency)

        if all(source.get("impedance") is not None for source in source_entries):
            entry["impedance"] = sum(
                weights[frequency_index, source_index] * source["impedance"]
                for source_index, source in enumerate(source_entries)
            )

        sphere_fields = [
            source.get("observation_sphere_pressure_complex")
            for source in source_entries
        ]
        if all(field is not None for field in sphere_fields):
            entry["observation_sphere_pressure_complex"] = sum(
                weights[frequency_index, source_index] * np.asarray(field)
                for source_index, field in enumerate(sphere_fields)
            )
        elif any(field is not None for field in sphere_fields):
            raise ValueError("aperture basis sphere-pressure fields differ")

        entry["timing_s"] = sum(
            float(source.get("timing_s", 0.0)) for source in source_entries
        )
        if all("field_s" in source for source in source_entries):
            entry["field_s"] = sum(
                float(source["field_s"]) for source in source_entries
            )
        combined_logs.append(entry)
    return combined_logs


def _metal_api(config: Any | None = None):
    """Return the Metal BEM API used by the coupling layer."""
    if config is not None:
        module = type(config).__module__
        if not module.startswith("hornlab_metal_bem"):
            raise ValueError(
                f"hornlab-sim BEM coupling requires a hornlab_metal_bem "
                f"SolveConfig; got {type(config)!r}"
            )

    try:
        import hornlab_metal_bem as metal
        from hornlab_metal_bem.config import VelocityMode
    except ModuleNotFoundError as exc:
        if exc.name == "hornlab_metal_bem" or str(exc.name).startswith(
            "hornlab_metal_bem."
        ):
            raise ImportError(
                "hornlab-sim BEM coupling requires the optional "
                "hornlab_metal_bem dependency. Install it with: "
                f"{METAL_EXTRA_INSTALL_HINT}"
            ) from exc
        raise

    def default_config(formulation: str | None):
        if formulation is None:
            return metal.native_config()
        return metal.native_config(formulation=formulation)

    solve_multi_source = None
    if hasattr(metal, "solve_multi_source"):

        def solve_multi_source(mesh, freqs, sources, config):
            return metal.solve_multi_source(
                mesh,
                sources,
                config,
                frequencies_hz=freqs,
            )

    return SimpleNamespace(
        name="metal",
        load_mesh=metal.load_mesh,
        solve_frequencies=metal.solve_frequencies,
        solve_multi_source=solve_multi_source,
        VelocityMode=VelocityMode,
        default_config=default_config,
    )
