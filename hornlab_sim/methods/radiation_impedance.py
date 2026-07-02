"""BEM-derived aperture radiation impedance matrices.

This module is the first reduced-coupling step toward FEM-interior /
BEM-exterior workflows.  It computes the small dense matrix that maps
aperture volume velocities to average aperture pressures:

    p_i(f) = sum_j Z_ij(f) Q_j(f)

The implementation runs the canonical Metal BEM path with one unit-velocity
basis solve per source aperture.  It is not a full
trace-space FEM-BEM coupling; it is the aperture-basis approximation intended
for MEH cavities, ports, and throat/mouth interfaces where a small number of
patch-averaged unknowns is a useful first model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, Union

import numpy as np
from numpy.typing import NDArray

from .lem_to_bem import _aperture_face_areas

if TYPE_CHECKING:
    from hornlab_metal_bem import SolveConfig, SolveResult
    from hornlab_metal_bem.mesh import LoadedMesh


MeshLike = Union[str, Path, "LoadedMesh", Any]
RHO_AIR = 1.2041
C_AIR = 343.0


@dataclass(frozen=True)
class RadiationImpedanceResult:
    """Aperture radiation impedance matrix result.

    Attributes
    ----------
    frequencies_hz
        Frequencies solved, shape ``(F,)``.
    aperture_names
        Matrix aperture order.  ``impedance_matrix[:, i, j]`` maps source
        aperture ``aperture_names[j]`` volume velocity to receiver aperture
        ``aperture_names[i]`` average pressure.
    aperture_area_m2
        Total face area per aperture name.
    impedance_matrix
        Complex matrix with shape ``(F, N, N)`` and units Pa*s/m^3.
    solver_logs
        One log entry per source-aperture basis solve.
    """

    frequencies_hz: NDArray[np.float64]
    aperture_names: list[str]
    aperture_area_m2: dict[str, float]
    impedance_matrix: NDArray[np.complex128]
    solver_logs: list[dict]


@dataclass(frozen=True)
class RadiationMatrixDiagnostics:
    """Numerical and physical checks for an aperture impedance matrix."""

    reciprocity_max_abs: NDArray[np.float64]
    reciprocity_max_rel: NDArray[np.float64]
    passivity_min_eig: NDArray[np.float64]
    passivity_ok: NDArray[np.bool_]
    low_ka_self_impedance: dict[str, NDArray[np.complex128]]
    low_ka_self_impedance_rel_error: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class TerminatedBranchResult:
    """Chamber-port branch response with an external BEM termination load."""

    frequencies_hz: NDArray[np.float64]
    termination_load: NDArray[np.complex128]
    input_impedance: NDArray[np.complex128]
    exit_to_input_volume_velocity_ratio: NDArray[np.complex128]


def solve_aperture_matrix(
    mesh: MeshLike,
    aperture_tags: Mapping[str, list[int]],
    frequencies_hz: NDArray[np.float64],
    config: Any | None = None,
    *,
    normal_velocity: complex = 1.0 + 0.0j,
) -> RadiationImpedanceResult:
    """Compute a BEM radiation impedance matrix for aperture patches.

    Parameters
    ----------
    mesh
        Path to a surface ``.msh`` or a preloaded ``hornlab_metal_bem.LoadedMesh``.
    aperture_tags
        Mapping from aperture name to one or more physical group IDs.
    frequencies_hz
        Positive frequencies to solve.
    config
        Optional BEM solve config from ``hornlab_metal_bem``. If omitted, a
        native Metal config is used. The function overrides ``velocity_mode`` and
        ``velocity_sources`` for each source basis.
    normal_velocity
        Unit normal velocity imposed on all faces of the active source
        aperture.  Must be nonzero.

    Returns
    -------
    RadiationImpedanceResult
        Dense aperture matrix ``Z[f, receiver, source]``.
    """
    freqs = _validate_frequencies(frequencies_hz)
    aperture_names = _validate_aperture_tags(aperture_tags)
    drive_velocity = complex(normal_velocity)
    if abs(drive_velocity) <= 0.0:
        raise ValueError("normal_velocity must be nonzero")

    api = _metal_api(config)
    if config is None:
        config = api.default_config(None)

    if isinstance(mesh, (str, Path)):
        loaded = api.load_mesh(mesh, scale=config.mesh_scale)
    else:
        loaded = mesh

    aperture_area_m2 = _aperture_face_areas(loaded, aperture_tags)
    unique_tags = sorted({int(tag) for tags in aperture_tags.values() for tag in tags})
    tag_area_m2 = _tag_face_areas(loaded, unique_tags)

    base_config = replace(config, velocity_mode=api.VelocityMode.VELOCITY)
    matrix = np.zeros(
        (freqs.size, len(aperture_names), len(aperture_names)),
        dtype=np.complex128,
    )
    solver_logs: list[dict] = []

    for source_idx, source_name in enumerate(aperture_names):
        sources = {tag: 0.0 + 0.0j for tag in unique_tags}
        for tag in aperture_tags[source_name]:
            sources[int(tag)] = drive_velocity

        source_config = replace(base_config, velocity_sources=sources)
        result = api.solve_frequencies(loaded, freqs, source_config)
        if result.surface_pressure_avg is None:
            raise RuntimeError(
                f"{api.name} result did not include surface_pressure_avg; "
                "cannot assemble aperture radiation matrix"
            )

        volume_velocity = drive_velocity * aperture_area_m2[source_name]
        for recv_idx, recv_name in enumerate(aperture_names):
            p_avg = _aggregate_aperture_pressure(
                result.surface_pressure_avg,
                aperture_tags[recv_name],
                tag_area_m2,
            )
            matrix[:, recv_idx, source_idx] = p_avg / volume_velocity

        solver_logs.append(
            {
                "source_aperture": source_name,
                "source_tags": [int(tag) for tag in aperture_tags[source_name]],
                "volume_velocity_m3_s": volume_velocity,
                "solver_log": result.solver_log,
            }
        )

    return RadiationImpedanceResult(
        frequencies_hz=freqs,
        aperture_names=aperture_names,
        aperture_area_m2=aperture_area_m2,
        impedance_matrix=matrix,
        solver_logs=solver_logs,
    )


def _metal_api(config: Any | None = None):
    """Return the Metal BEM API used by the radiation matrix solver."""
    if config is not None:
        module = type(config).__module__
        if not module.startswith("hornlab_metal_bem"):
            raise ValueError(
                f"radiation_impedance requires a hornlab_metal_bem "
                f"SolveConfig; got {type(config)!r}"
            )

    import hornlab_metal_bem as metal
    from hornlab_metal_bem.config import VelocityMode

    def default_config(formulation: str | None):
        if formulation is None:
            return metal.native_config()
        return metal.native_config(formulation=formulation)

    return SimpleNamespace(
        name="metal",
        load_mesh=metal.load_mesh,
        solve_frequencies=metal.solve_frequencies,
        VelocityMode=VelocityMode,
        default_config=default_config,
    )


def termination_load_from_solver_matrix(
    solver_matrix: NDArray[np.complex128],
    *,
    receiver_index: int | None = None,
    source_indices: list[int] | tuple[int, ...] | NDArray[np.int_] | None = None,
    source_weights: NDArray[np.complex128] | list[complex] | tuple[complex, ...] | None = None,
) -> NDArray[np.complex128]:
    """Convert solver-convention aperture impedance into an engineering load.

    ``hornlab_sim.methods.radiation_impedance`` returns the complex conjugate
    of the e^{+jwt} engineering-convention impedance. The 260611 BIGMEH
    termination-attribution validation locked this against the exact
    pulsating-sphere solution, so the LEM/TMM insertion convention is encoded
    here once as ``conj(Z_solver)``.

    If ``receiver_index`` is omitted, the full matrix is conjugated. If it is
    provided, the helper returns the reduced load seen by that receiver:

    ``Z_load_i = sum_j conj(Z_solver[i, j]) * (Q_j / Q_i)``.

    The in-phase L/R port reduction is therefore ``source_indices=[left,
    right]`` with the default unit ``source_weights``.
    """
    matrix = np.asarray(solver_matrix, dtype=np.complex128)
    engineering = np.conjugate(matrix)
    if receiver_index is None:
        if source_indices is not None or source_weights is not None:
            raise ValueError(
                "source_indices/source_weights require receiver_index"
            )
        return np.array(engineering, dtype=np.complex128, copy=True)

    if matrix.ndim != 3:
        raise ValueError(
            "receiver reduction expects solver_matrix shape (n_freq, n, n), "
            f"got {matrix.shape}"
        )
    n_apertures = matrix.shape[1]
    if matrix.shape[2] != n_apertures:
        raise ValueError(
            "solver_matrix must be square in its last two dimensions, "
            f"got {matrix.shape}"
        )
    receiver = int(receiver_index)
    if receiver < 0 or receiver >= n_apertures:
        raise IndexError(
            f"receiver_index {receiver} out of range for {n_apertures} apertures"
        )

    if source_indices is None:
        sources = np.array([receiver], dtype=np.int64)
    else:
        sources = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if sources.size == 0:
        raise ValueError("source_indices must not be empty")
    bad = sources[(sources < 0) | (sources >= n_apertures)]
    if bad.size:
        raise IndexError(
            f"source index {int(bad[0])} out of range for {n_apertures} apertures"
        )

    if source_weights is None:
        weights = np.ones(sources.size, dtype=np.complex128)
    else:
        weights = np.asarray(source_weights, dtype=np.complex128).reshape(-1)
        if weights.shape != (sources.size,):
            raise ValueError(
                f"source_weights shape {weights.shape}, expected ({sources.size},)"
            )
    if not np.all(np.isfinite(weights.real) & np.isfinite(weights.imag)):
        raise ValueError("source_weights must be finite complex values")

    return np.sum(engineering[:, receiver, sources] * weights[None, :], axis=1)


def terminated_chamber_port_branch(
    frequencies_hz: NDArray[np.float64],
    termination_load: NDArray[np.complex128],
    *,
    chamber_volume_m3: float,
    port_area_m2: float,
    port_length_m: float,
    interior_end_correction_length_m: float = 0.0,
    series_resistance_pa_s_m3: float = 0.0,
    rho: float = RHO_AIR,
    c: float = C_AIR,
) -> TerminatedBranchResult:
    """Return input impedance and ``U_exit/U_in`` for a terminated port branch.

    Network topology matches the 260601 d070 port-exit comparison helper:
    chamber compliance in shunt at the branch input, in parallel with a series
    path made from port acoustic mass plus the external termination load. The
    BEM-terminated aperture must not also include LEM-side external radiation
    loading upstream; use ``end_corr="none"`` or ``radiation_external=False``
    for that aperture before applying this helper.
    """
    freqs = _validate_frequencies(frequencies_hz)
    load = _validate_load_array(termination_load, freqs)
    volume = _positive_finite("chamber_volume_m3", chamber_volume_m3)
    area = _positive_finite("port_area_m2", port_area_m2)
    length = _nonnegative_finite("port_length_m", port_length_m)
    interior = _nonnegative_finite(
        "interior_end_correction_length_m",
        interior_end_correction_length_m,
    )
    series_resistance = _nonnegative_finite(
        "series_resistance_pa_s_m3",
        series_resistance_pa_s_m3,
    )
    rho_f = _positive_finite("rho", rho)
    c_f = _positive_finite("c", c)

    omega = 2.0 * np.pi * freqs
    compliance = volume / (rho_f * c_f * c_f)
    y_chamber = 1j * omega * compliance
    z_port = 1j * omega * rho_f * (length + interior) / area
    z_series = series_resistance + z_port + load
    y_series = 1.0 / z_series
    total_admittance = y_chamber + y_series
    zin = 1.0 / total_admittance
    transfer_ratio = y_series / total_admittance

    return TerminatedBranchResult(
        frequencies_hz=freqs,
        termination_load=load,
        input_impedance=np.asarray(zin, dtype=np.complex128),
        exit_to_input_volume_velocity_ratio=np.asarray(
            transfer_ratio,
            dtype=np.complex128,
        ),
    )


def collapse_aperture_matrix(
    result: RadiationImpedanceResult,
    aperture_groups: Mapping[str, list[str]],
) -> RadiationImpedanceResult:
    """Collapse a sub-aperture matrix into aggregate uniformly-driven apertures.

    This is the bookkeeping half of the Phase-1 sub-tag refinement check:
    solving each sub-tag independently and collapsing the matrix should match
    a direct one-tag aperture solve when all sub-tags are driven with the same
    normal velocity.
    """
    group_names = list(aperture_groups)
    if not group_names:
        raise ValueError("aperture_groups is empty")
    name_to_index = {name: i for i, name in enumerate(result.aperture_names)}
    matrix = _validate_matrix_shape(result)
    collapsed = np.zeros(
        (result.frequencies_hz.size, len(group_names), len(group_names)),
        dtype=np.complex128,
    )
    collapsed_areas: dict[str, float] = {}

    for group_name, members in aperture_groups.items():
        if not members:
            raise ValueError(f"aperture group {group_name!r} is empty")
        missing = [name for name in members if name not in name_to_index]
        if missing:
            raise ValueError(
                f"aperture group {group_name!r} references unknown aperture(s): "
                f"{missing}"
            )
        area = sum(float(result.aperture_area_m2[name]) for name in members)
        if area <= 0.0:
            raise ValueError(f"aperture group {group_name!r} has zero area")
        collapsed_areas[group_name] = area

    for recv_group_idx, recv_group_name in enumerate(group_names):
        recv_members = aperture_groups[recv_group_name]
        recv_area = collapsed_areas[recv_group_name]
        recv_weights = [
            float(result.aperture_area_m2[name]) / recv_area
            for name in recv_members
        ]
        recv_indices = [name_to_index[name] for name in recv_members]
        for source_group_idx, source_group_name in enumerate(group_names):
            source_members = aperture_groups[source_group_name]
            source_area = collapsed_areas[source_group_name]
            source_weights = [
                float(result.aperture_area_m2[name]) / source_area
                for name in source_members
            ]
            source_indices = [name_to_index[name] for name in source_members]
            value = np.zeros(result.frequencies_hz.size, dtype=np.complex128)
            for recv_weight, recv_idx in zip(recv_weights, recv_indices):
                for source_weight, source_idx in zip(source_weights, source_indices):
                    value += recv_weight * source_weight * matrix[:, recv_idx, source_idx]
            collapsed[:, recv_group_idx, source_group_idx] = value

    return RadiationImpedanceResult(
        frequencies_hz=np.array(result.frequencies_hz, dtype=np.float64, copy=True),
        aperture_names=group_names,
        aperture_area_m2=collapsed_areas,
        impedance_matrix=collapsed,
        solver_logs=list(result.solver_logs),
    )


def low_ka_baffled_piston_radiation_impedance(
    radius_m: float,
    frequencies_hz: NDArray[np.float64],
    *,
    rho: float = RHO_AIR,
    c: float = C_AIR,
) -> NDArray[np.complex128]:
    """Low-``ka`` baffled circular-piston reference, in ``p_avg / Q`` units.

    The normalized baffled-piston radiation impedance is approximated as
    ``R + jX = (ka)^2/2 + j*8*ka/(3*pi)``. This is only intended for the
    low-frequency attribution gate, where ``ka`` is small enough that the
    leading terms are the invariant being tested.
    """
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius_m must be positive and finite, got {radius_m!r}")
    freqs = _validate_frequencies(frequencies_hz)
    area = math.pi * radius * radius
    ka = (2.0 * math.pi * freqs / float(c)) * radius
    normalized = 0.5 * ka * ka + 1j * (8.0 / (3.0 * math.pi)) * ka
    return np.asarray((float(rho) * float(c) / area) * normalized, dtype=np.complex128)


def matrix_diagnostics(
    result: RadiationImpedanceResult,
    *,
    piston_radius_m_by_aperture: Mapping[str, float] | None = None,
    rho: float = RHO_AIR,
    c: float = C_AIR,
    low_ka_max: float = 0.35,
    passivity_tol: float = 1e-9,
) -> RadiationMatrixDiagnostics:
    """Return reciprocity, passivity, and optional low-``ka`` diagnostics."""
    matrix = _validate_matrix_shape(result)
    transposed = np.swapaxes(matrix, 1, 2)
    diff = matrix - transposed
    reciprocity_max_abs = np.max(np.abs(diff), axis=(1, 2))
    denom = np.maximum(
        np.max(np.maximum(np.abs(matrix), np.abs(transposed)), axis=(1, 2)),
        1.0,
    )
    reciprocity_max_rel = reciprocity_max_abs / denom

    passivity_min_eig = np.zeros(result.frequencies_hz.size, dtype=np.float64)
    for idx, z in enumerate(matrix):
        real_symmetric = 0.5 * (np.real(z) + np.real(z).T)
        passivity_min_eig[idx] = float(np.min(np.linalg.eigvalsh(real_symmetric)))
    passivity_ok = passivity_min_eig >= -float(passivity_tol)

    low_ka_self_impedance: dict[str, NDArray[np.complex128]] = {}
    low_ka_self_impedance_rel_error: dict[str, NDArray[np.float64]] = {}
    if piston_radius_m_by_aperture:
        name_to_index = {name: i for i, name in enumerate(result.aperture_names)}
        for name, radius in piston_radius_m_by_aperture.items():
            if name not in name_to_index:
                raise ValueError(f"unknown aperture {name!r} for low-ka check")
            expected = low_ka_baffled_piston_radiation_impedance(
                radius,
                result.frequencies_hz,
                rho=rho,
                c=c,
            )
            actual = matrix[:, name_to_index[name], name_to_index[name]]
            ka = (2.0 * math.pi * result.frequencies_hz / float(c)) * float(radius)
            rel = np.full(result.frequencies_hz.size, np.nan, dtype=np.float64)
            mask = ka <= float(low_ka_max)
            rel[mask] = (
                np.abs(actual[mask] - expected[mask])
                / np.maximum(np.abs(expected[mask]), np.finfo(np.float64).tiny)
            )
            low_ka_self_impedance[name] = expected
            low_ka_self_impedance_rel_error[name] = rel

    return RadiationMatrixDiagnostics(
        reciprocity_max_abs=reciprocity_max_abs,
        reciprocity_max_rel=reciprocity_max_rel,
        passivity_min_eig=passivity_min_eig,
        passivity_ok=passivity_ok,
        low_ka_self_impedance=low_ka_self_impedance,
        low_ka_self_impedance_rel_error=low_ka_self_impedance_rel_error,
    )


def _validate_frequencies(values: NDArray[np.float64]) -> NDArray[np.float64]:
    freqs = np.asarray(values, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies_hz is empty")
    bad = freqs[~np.isfinite(freqs) | (freqs <= 0.0)]
    if bad.size:
        raise ValueError(f"frequencies_hz must be positive finite values: {bad[:5]}")
    return freqs


def _validate_load_array(
    values: NDArray[np.complex128],
    freqs: NDArray[np.float64],
) -> NDArray[np.complex128]:
    load = np.asarray(values, dtype=np.complex128).reshape(-1)
    if load.size == 1 and freqs.size != 1:
        load = np.full(freqs.shape, load[0], dtype=np.complex128)
    if load.shape != freqs.shape:
        raise ValueError(
            f"termination_load shape {load.shape}, expected {freqs.shape}"
        )
    if not np.all(np.isfinite(load.real) & np.isfinite(load.imag)):
        raise ValueError("termination_load must contain finite complex values")
    return load


def _positive_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value_f


def _nonnegative_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not math.isfinite(value_f) or value_f < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return value_f


def _validate_matrix_shape(
    result: RadiationImpedanceResult,
) -> NDArray[np.complex128]:
    matrix = np.asarray(result.impedance_matrix, dtype=np.complex128)
    expected = (
        np.asarray(result.frequencies_hz).reshape(-1).size,
        len(result.aperture_names),
        len(result.aperture_names),
    )
    if matrix.shape != expected:
        raise ValueError(
            f"impedance_matrix shape {matrix.shape} does not match expected {expected}"
        )
    return matrix


def _validate_aperture_tags(aperture_tags: Mapping[str, list[int]]) -> list[str]:
    names = list(aperture_tags)
    if not names:
        raise ValueError("aperture_tags is empty")
    for name, tags in aperture_tags.items():
        if not tags:
            raise ValueError(f"Aperture {name!r} has empty tag list")
    return names


def _tag_face_areas(loaded: "LoadedMesh", tags: list[int]) -> dict[int, float]:
    vertices, elements = _mesh_vertices_elements(loaded)
    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    physical_tags = np.asarray(loaded.physical_tags)
    result: dict[int, float] = {}
    for tag in tags:
        mask = physical_tags == int(tag)
        if not mask.any():
            raise ValueError(
                f"physical tag {tag} not present in mesh "
                f"(available physical tags: {sorted(set(int(t) for t in physical_tags))})"
            )
        result[int(tag)] = float(tri_areas[mask].sum())
    return result


def _mesh_vertices_elements(
    loaded: "LoadedMesh",
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    vertices = np.asarray(loaded.grid.vertices)
    if vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    elements = np.asarray(loaded.grid.elements)
    if elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T
    return vertices, elements


def _aggregate_aperture_pressure(
    surface_pressure_avg: Mapping[int, NDArray[np.complex128]],
    tags: list[int],
    tag_area_m2: Mapping[int, float],
) -> NDArray[np.complex128]:
    total_area = sum(tag_area_m2[int(tag)] for tag in tags)
    if total_area <= 0.0:
        raise ValueError(f"aperture tags {tags} have zero total area")

    weighted = None
    for tag in tags:
        tag_i = int(tag)
        if tag_i not in surface_pressure_avg:
            raise RuntimeError(
                f"surface_pressure_avg is missing tag {tag_i}; "
                "make sure all receiver tags are included in velocity_sources"
            )
        arr = np.asarray(surface_pressure_avg[tag_i], dtype=np.complex128)
        contrib = arr * tag_area_m2[tag_i]
        weighted = contrib if weighted is None else weighted + contrib

    if weighted is None:
        raise ValueError("cannot aggregate an empty aperture")
    return weighted / total_area
