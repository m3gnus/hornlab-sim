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

5. **Phase reference and time convention.** All LEM aperture velocities share
   a common excitation reference (driver terminal voltage), but a shared
   reference is *not* a shared time convention. This package's LEM/TMM
   methods are engineering ``e^{+j*omega*t}`` (``s = +j*omega``), while the
   Metal solver is physics ``e^{-i*omega*t}`` with an outgoing
   ``exp(+i*k*r)`` kernel. The two phasor conventions are related by complex
   conjugation, so an engineering volume velocity MUST be conjugated once
   before it becomes solver-convention Neumann data. ``solve`` does that
   conversion itself, controlled by ``velocity_convention``; the reverse
   boundary is ``radiation_impedance.termination_load_from_solver_matrix``.

Cross-repository counterpart (GIT-WORKFLOW.md section 5)
--------------------------------------------------------
``velocity_convention="engineering"`` is the default because the documented
input of this API is LEM/TMM volume velocity. Any consumer that already
converted its source phasors to solver convention before calling ``solve``
must now pass ``velocity_convention="solver"``, otherwise the conversion is
applied twice and the complex source phase is reversed. The counterpart
obligation is recorded in this package's ``AGENTS.md`` and ``README.md``.
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

#: Accepted values for ``solve(velocity_convention=...)``.
#:
#: ``"engineering"``  ``e^{+j*omega*t}`` phasors, as emitted by this
#:                    package's LEM/TMM methods. Conjugated once before use.
#: ``"solver"``       ``e^{-i*omega*t}`` phasors, already in the Metal
#:                    solver's convention. Applied unchanged.
VELOCITY_CONVENTIONS = ("engineering", "solver")

#: Accepted values for ``mesh_array_layout``.
#:
#: ``"columns"``  Bempp/``PureGrid`` layout: ``vertices`` is
#:                ``(3, n_vertices)`` and ``elements`` is ``(3, n_elements)``.
#: ``"rows"``     ``(n_vertices, 3)`` and ``(n_elements, 3)``.
MESH_ARRAY_LAYOUTS = ("rows", "columns")


def solve(
    mesh: MeshLike,
    lem_velocities: Mapping[str, NDArray[np.complex128]],
    aperture_tags: Mapping[str, list[int]],
    frequencies_hz: NDArray[np.float64],
    config: Any | None = None,
    *,
    area_tolerance: float = 0.05,
    velocity_convention: str = "engineering",
    mesh_array_layout: str | None = None,
) -> Any:
    """Solve BEM with LEM-derived complex velocity sources per aperture.

    Parameters
    ----------
    mesh : path or ``LoadedMesh``
        Mesh with physical groups matching every value in ``aperture_tags``.
    lem_velocities : dict[str, complex array]
        Per-aperture complex volume velocity ``U(f)`` in m^3/s. Each value
        must be a 1-D array of length ``len(frequencies_hz)``. Interpreted
        in the time convention named by ``velocity_convention``.
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
    velocity_convention : {"engineering", "solver"}, default "engineering"
        Time convention of ``lem_velocities``.

        ``"engineering"`` means ``e^{+j*omega*t}`` phasors, which is what
        every LEM/TMM method in this package emits. They are converted once
        to the solver's ``e^{-i*omega*t}`` convention by complex conjugation
        before either execution path builds a Neumann source, so a caller
        that follows the documented API needs no conversion of its own.

        ``"solver"`` means the caller already supplies ``e^{-i*omega*t}``
        phasors; they are applied unchanged. Use it only when the values
        were produced by, or already converted for, the BEM solver — passing
        engineering phasors here reverses the complex source phase and turns
        destructive multi-aperture interference into constructive.

        Real-valued velocities are identical in both conventions.
    mesh_array_layout : {"rows", "columns"}, optional
        Layout of ``mesh.grid.vertices`` / ``mesh.grid.elements``.
        ``"columns"`` is the canonical Bempp/``PureGrid`` layout,
        ``(3, n_vertices)`` and ``(3, n_elements)``; ``"rows"`` is
        ``(n_vertices, 3)`` and ``(n_elements, 3)``. Leave it ``None`` for
        ordinary meshes, where the layout is settled by the array shapes or
        by the per-element areas the grid carries. It is required only for a
        grid whose arrays are exactly ``(3, 3)``, which is ambiguous.

    Returns
    -------
    SolveResult
        Metal SolveResult with complex pressure of shape
        ``(n_freq, n_planes, n_angles)``. The per-aperture v_n applied at
        each frequency is recorded in ``result.solver_log``, in the solver
        convention actually imposed on the mesh, alongside the
        ``velocity_convention`` the caller declared for its inputs.

    Raises
    ------
    ValueError
        Aperture name mismatch between ``lem_velocities`` and
        ``aperture_tags``, velocity array shape mismatch with frequencies,
        overlapping physical tags, zero-area aperture, or an unknown
        ``velocity_convention``.

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

    if velocity_convention not in VELOCITY_CONVENTIONS:
        raise ValueError(
            f"velocity_convention must be one of {list(VELOCITY_CONVENTIONS)}, "
            f"got {velocity_convention!r}"
        )

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

    aperture_area_m2 = _aperture_face_areas(
        loaded, aperture_tags, mesh_array_layout=mesh_array_layout
    )

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
    # One conversion point for both execution paths: U/A in the caller's
    # declared convention, then a single conversion into the solver's
    # e^{-i*omega*t} convention. Conjugating here rather than in each branch
    # is what keeps the basis and sequential paths bitwise consistent.
    input_weights = np.column_stack(
        [
            np.asarray(lem_velocities[name], dtype=np.complex128)
            / aperture_area_m2[name]
            for name in aperture_names
        ]
    )
    velocity_weights = _to_solver_convention(input_weights, velocity_convention)
    per_freq_log = [
        {
            "frequency_hz": float(frequency),
            "velocity_convention": velocity_convention,
            "v_n_per_aperture": {
                name: complex(weight)
                for name, weight in zip(aperture_names, frequency_weights)
            },
        }
        for frequency, frequency_weights in zip(freqs, velocity_weights)
    ]

    solve_multi = getattr(api, "solve_multi_source", None)
    has_callbacks = any(
        getattr(base_config, name, None) is not None
        for name in (
            "on_frequency_result",
            "progress_callback",
            "velocity_source_callback",
        )
    )
    if solve_multi is not None and not has_callbacks:
        basis_sources = [
            {
                tag: 1.0 + 0.0j if owner == name else 0.0 + 0.0j
                for tag, owner in sorted(tag_owners.items())
            }
            for name in aperture_names
        ]

        basis_results = solve_multi(loaded, freqs, basis_sources, base_config)
        combined_config = replace(
            base_config,
            velocity_sources={
                tag: complex(velocity_weights[0, aperture_indices[owner]])
                for tag, owner in tag_owners.items()
            },
        )
        aggregated = _combine_basis_results(
            basis_results,
            velocity_weights,
            freqs,
            combined_config=combined_config,
            aperture_indices=aperture_indices,
            tag_owners=tag_owners,
        )
    else:
        per_freq_results = []
        for index, frequency in enumerate(freqs):
            sources: dict[int, complex] = {}
            for name, tags in aperture_tags.items():
                velocity = complex(
                    velocity_weights[index, aperture_indices[name]]
                )
                for tag in tags:
                    sources[int(tag)] = velocity
            freq_config = replace(base_config, velocity_sources=sources)
            single = api.solve_frequencies(
                loaded, [float(frequency)], freq_config
            )
            per_freq_results.append(single)
        aggregated = _concat_results(per_freq_results, freqs)

    aggregated.solver_log.append({"lem_to_bem": per_freq_log})
    return aggregated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_solver_convention(
    weights: NDArray[np.complex128],
    velocity_convention: str,
) -> NDArray[np.complex128]:
    """Return ``weights`` as ``e^{-i*omega*t}`` solver-convention phasors.

    An engineering phasor ``X`` stands for ``Re{X e^{+j*omega*t}}``, which is
    the same real signal as ``Re{conj(X) e^{-i*omega*t}}``. The conversion
    between the two conventions is therefore exactly complex conjugation, and
    it is an involution: ``termination_load_from_solver_matrix`` applies the
    same operation on the return boundary.
    """
    array = np.asarray(weights, dtype=np.complex128)
    if velocity_convention == "engineering":
        return np.conjugate(array)
    if velocity_convention == "solver":
        return array
    raise ValueError(
        f"velocity_convention must be one of {list(VELOCITY_CONVENTIONS)}, "
        f"got {velocity_convention!r}"
    )


def _aperture_face_areas(
    loaded: "LoadedMesh",
    aperture_tags: Mapping[str, list[int]],
    *,
    mesh_array_layout: str | None = None,
) -> dict[str, float]:
    """Return total face area in m^2 per aperture name."""
    tri_areas = _triangle_face_areas(loaded, mesh_array_layout=mesh_array_layout)
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


def _triangle_face_areas(
    loaded: "LoadedMesh",
    *,
    mesh_array_layout: str | None = None,
) -> NDArray[np.float64]:
    """Return the area of every triangular mesh face in m^2.

    Two array layouts are in use. ``hornlab_metal_bem.mesh.PureGrid`` is
    Bempp-shaped: ``vertices`` is ``(3, n_vertices)`` and ``elements`` is
    ``(3, n_elements)``, triangles in columns. Convenience/test grids are
    often row-major, ``(n_vertices, 3)`` and ``(n_elements, 3)``.

    A ``(3, 3)`` array satisfies both, so the layout is not recoverable from
    the shape for a three-element or three-vertex mesh. This helper therefore
    never guesses at that shape. It resolves the layout, in order, from an
    explicit ``mesh_array_layout``, from a layout the grid declares itself,
    from authoritative per-element areas the grid already carries
    (``PureGrid.volumes``), and only then from an unambiguous shape. If none
    of those apply it raises rather than picking one.
    """
    grid = loaded.grid
    layout = _resolve_mesh_array_layout(mesh_array_layout, grid)

    if layout is None:
        areas = _declared_element_areas(loaded)
        if areas is not None:
            return areas
        raise ValueError(
            "Cannot determine the mesh array layout: grid.vertices has shape "
            f"{np.asarray(grid.vertices).shape} and grid.elements has shape "
            f"{np.asarray(grid.elements).shape}, which is valid both as "
            "(3, n) with triangles in columns (the canonical Bempp/PureGrid "
            "layout) and as (n, 3) with triangles in rows. The grid carries "
            "no per-element areas to settle it. Pass "
            "mesh_array_layout='columns' or 'rows', or supply a grid that "
            "declares its layout."
        )

    vertices = np.asarray(grid.vertices, dtype=np.float64)
    elements = np.asarray(grid.elements)
    if layout == "columns":
        vertices = vertices.T
        elements = elements.T
    _check_triangle_arrays(vertices, elements, layout)

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _resolve_mesh_array_layout(mesh_array_layout: str | None, grid: Any) -> str | None:
    """Return ``"rows"``, ``"columns"``, or ``None`` when still ambiguous."""
    if mesh_array_layout is not None:
        if mesh_array_layout not in MESH_ARRAY_LAYOUTS:
            raise ValueError(
                "mesh_array_layout must be one of "
                f"{list(MESH_ARRAY_LAYOUTS)} or None, got {mesh_array_layout!r}"
            )
        return mesh_array_layout

    declared = getattr(grid, "array_layout", None)
    if declared is not None:
        if declared not in MESH_ARRAY_LAYOUTS:
            raise ValueError(
                f"grid declares array_layout={declared!r}; expected one of "
                f"{list(MESH_ARRAY_LAYOUTS)}"
            )
        return declared

    vertex_layout = _layout_from_shape(np.asarray(grid.vertices).shape)
    element_layout = _layout_from_shape(np.asarray(grid.elements).shape)
    if vertex_layout is not None and element_layout is not None:
        if vertex_layout != element_layout:
            raise ValueError(
                f"grid.vertices implies a {vertex_layout!r} layout while "
                f"grid.elements implies {element_layout!r}; pass an explicit "
                "mesh_array_layout"
            )
        return vertex_layout
    # One array may still settle both: the two are always stored alike.
    return vertex_layout if vertex_layout is not None else element_layout


def _layout_from_shape(shape: tuple[int, ...]) -> str | None:
    if len(shape) != 2:
        raise ValueError(
            f"expected a 2-D vertex/element array, got shape {shape}"
        )
    rows, columns = shape
    if columns == 3 and rows != 3:
        return "rows"
    if rows == 3 and columns != 3:
        return "columns"
    return None


def _declared_element_areas(loaded: "LoadedMesh") -> NDArray[np.float64] | None:
    """Return authoritative per-element areas the grid already carries.

    ``PureGrid.volumes`` is the triangle-area array the mesh loader computed
    from the same scaled vertices, in element order, so it matches
    ``loaded.physical_tags`` element for element. Anything that does not line
    up with the tag count is ignored rather than trusted.
    """
    volumes = getattr(loaded.grid, "volumes", None)
    if volumes is None:
        return None
    areas = np.asarray(volumes, dtype=np.float64).reshape(-1)
    n_elements = int(np.asarray(loaded.physical_tags).reshape(-1).size)
    if areas.size != n_elements:
        return None
    return areas


def _check_triangle_arrays(
    vertices: NDArray[np.float64],
    elements: NDArray[np.int_],
    layout: str,
) -> None:
    """Reject an index that the resolved layout cannot address."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"with mesh_array_layout={layout!r} the vertex array resolves to "
            f"shape {vertices.shape}, expected (n_vertices, 3)"
        )
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError(
            f"with mesh_array_layout={layout!r} the element array resolves to "
            f"shape {elements.shape}, expected (n_elements, 3)"
        )
    if elements.size and (
        int(elements.min()) < 0 or int(elements.max()) >= vertices.shape[0]
    ):
        raise ValueError(
            f"with mesh_array_layout={layout!r} triangle indices span "
            f"[{int(elements.min())}, {int(elements.max())}] but the mesh "
            f"resolves to {vertices.shape[0]} vertices; the layout is wrong"
        )


def _check_shared_sphere_geometry(results, *, context: str) -> None:
    """Require frequency/source-independent sphere sampling geometry."""
    first = results[0]
    for name in ("sphere_points", "sphere_theta_deg", "sphere_phi_deg"):
        reference = getattr(first, name, None)
        for result in results[1:]:
            candidate = getattr(result, name, None)
            if reference is None or candidate is None:
                if reference is not None or candidate is not None:
                    raise ValueError(f"{context} sphere geometry fields differ")
            elif not np.array_equal(
                np.asarray(reference),
                np.asarray(candidate),
            ):
                raise ValueError(f"{context} sphere geometry fields differ")


def _concat_results(per_freq_results, frequencies_hz):
    """Concatenate single-frequency SolveResults along the frequency axis."""
    if not per_freq_results:
        raise ValueError("no per-frequency results to concatenate")

    first = per_freq_results[0]
    _check_shared_sphere_geometry(
        per_freq_results,
        context="per-frequency",
    )

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

    sphere_pressure_complex = None
    sphere_fields = [
        getattr(result, "sphere_pressure_complex", None)
        for result in per_freq_results
    ]
    if all(field is not None for field in sphere_fields):
        sphere_pressure_complex = np.concatenate(sphere_fields, axis=0)
    elif any(field is not None for field in sphere_fields):
        raise ValueError("per-frequency sphere-pressure fields differ")

    native_diagnostics = [
        entry
        for result in per_freq_results
        for entry in getattr(result, "native_diagnostics", [])
    ]

    optional_fields = {}
    if hasattr(first, "surface_pressure_complex"):
        optional_fields["surface_pressure_complex"] = surface_pressure_complex
    if hasattr(first, "sphere_pressure_complex"):
        optional_fields["sphere_pressure_complex"] = sphere_pressure_complex
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
    aperture_indices,
    tag_owners,
):
    """Linearly combine unit-aperture results, including impedance per row."""
    if not basis_results:
        raise ValueError("no aperture basis results to combine")
    weights = np.asarray(velocity_weights, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    expected = (frequencies.size, len(basis_results))
    if weights.shape != expected:
        raise ValueError(
            f"velocity_weights shape {weights.shape}, expected {expected}"
        )
    _check_shared_sphere_geometry(
        basis_results,
        context="aperture basis",
    )

    def weighted_result_field(name: str, *, optional: bool = False):
        fields = [
            getattr(result, name, None) if optional else getattr(result, name)
            for result in basis_results
        ]
        if optional and any(field is None for field in fields):
            if not all(field is None for field in fields):
                raise ValueError(f"aperture basis {name} fields differ")
            return None
        values = np.stack(
            [np.asarray(field) for field in fields],
            axis=0,
        )
        if values.shape[1] != frequencies.size:
            raise ValueError(
                f"basis result {name} frequency count {values.shape[1]}, "
                f"expected {frequencies.size}"
            )
        return np.einsum("fs,sf...->f...", weights, values)

    pressure_complex = weighted_result_field("pressure_complex")
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

    impedance = None
    if surface_pressure_avg is not None:
        impedance_rows = []
        for frequency_index, frequency_weights in enumerate(weights):
            source_tag = _lowest_driven_tag(
                frequency_weights,
                aperture_indices,
                tag_owners,
            )
            pressure = surface_pressure_avg.get(source_tag)
            if pressure is None or np.asarray(pressure).shape != frequencies.shape:
                impedance_rows = []
                break
            impedance_rows.append(np.asarray(pressure)[frequency_index])
        if len(impedance_rows) == frequencies.size:
            impedance = np.asarray(impedance_rows, dtype=np.complex128)

    surface_pressure_complex = weighted_result_field(
        "surface_pressure_complex", optional=True
    )
    sphere_pressure_complex = weighted_result_field(
        "sphere_pressure_complex", optional=True
    )

    solver_log = _combine_basis_solver_logs(
        basis_results,
        weights,
        frequencies,
        impedance,
        aperture_indices,
        tag_owners,
    )
    if impedance is None and solver_log and all(
        entry.get("impedance") is not None for entry in solver_log
    ):
        impedance = np.asarray(
            [entry["impedance"] for entry in solver_log],
            dtype=np.complex128,
        )
    optional_fields = {}
    if hasattr(first, "surface_pressure_complex"):
        optional_fields["surface_pressure_complex"] = surface_pressure_complex
    if hasattr(first, "sphere_pressure_complex"):
        optional_fields["sphere_pressure_complex"] = sphere_pressure_complex
    return replace(
        first,
        frequencies_hz=np.array(frequencies, copy=True),
        pressure_complex=np.asarray(pressure_complex, dtype=np.complex128),
        directivity_db=np.asarray(directivity_db, dtype=np.float64),
        impedance=(
            None if impedance is None else np.asarray(impedance, dtype=np.complex128)
        ),
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


def _combine_basis_solver_logs(
    basis_results,
    weights,
    frequencies,
    combined_impedance,
    aperture_indices,
    tag_owners,
):
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
    for frequency_index, (frequency, frequency_weights, source_entries) in enumerate(
        zip(frequencies, weights, zip(*basis_logs))
    ):
        entry = dict(source_entries[0])
        entry["frequency_hz"] = float(frequency)

        surface_pressure_logs = [
            source.get("surface_pressure_avg") for source in source_entries
        ]
        combined_log_surface_pressure = None
        if any(value is not None for value in surface_pressure_logs):
            if all(isinstance(value, Mapping) for value in surface_pressure_logs):
                log_tags = set.intersection(
                    *(set(value) for value in surface_pressure_logs)
                )
                combined_log_surface_pressure = {}
                for tag in log_tags:
                    pressures = [
                        _log_pressure_at_frequency(
                            value[tag], frequency_index, frequency_count
                        )
                        for value in surface_pressure_logs
                    ]
                    if all(value is not None for value in pressures):
                        combined_log_surface_pressure[tag] = sum(
                            weight * pressure
                            for weight, pressure in zip(frequency_weights, pressures)
                        )
                entry["surface_pressure_avg"] = combined_log_surface_pressure
            else:
                entry["surface_pressure_avg"] = None

        if combined_impedance is not None:
            entry["impedance"] = complex(combined_impedance[frequency_index])
        else:
            source_tag = _lowest_driven_tag(
                frequency_weights,
                aperture_indices,
                tag_owners,
            )
            pressure = (
                None
                if combined_log_surface_pressure is None
                else combined_log_surface_pressure.get(source_tag)
            )
            entry["impedance"] = (
                None if pressure is None else complex(pressure)
            )

        sphere_fields = [
            source.get("observation_sphere_pressure_complex")
            for source in source_entries
        ]
        if all(field is not None for field in sphere_fields):
            entry["observation_sphere_pressure_complex"] = sum(
                weight * np.asarray(field)
                for weight, field in zip(frequency_weights, sphere_fields)
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


def _lowest_driven_tag(frequency_weights, aperture_indices, tag_owners):
    """Select the combined impedance tag independently at each frequency."""
    driven_tags = [
        int(tag)
        for tag, aperture in tag_owners.items()
        if frequency_weights[aperture_indices[aperture]] != 0
    ]
    return min(driven_tags, default=min(tag_owners, default=2))


def _log_pressure_at_frequency(value, frequency_index, frequency_count):
    """Return a scalar or per-frequency logged surface pressure, if valid."""
    try:
        pressure = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError):
        return None
    if pressure.ndim == 0:
        return complex(pressure)
    if pressure.shape == (frequency_count,):
        return complex(pressure[frequency_index])
    return None


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
