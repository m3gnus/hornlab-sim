"""BEM-derived aperture radiation impedance matrices.

This module is the first reduced-coupling step toward FEM-interior /
BEM-exterior workflows.  It computes the small dense matrix that maps
aperture volume velocities to average aperture pressures:

    p_i(f) = sum_j Z_ij(f) Q_j(f)

The implementation reuses the canonical ``hornlab_bempp_bem`` BEM path by
running one unit-velocity basis solve per source aperture.  It is not a
full trace-space FEM-BEM coupling; it is the aperture-basis approximation
intended for MEH cavities, ports, and throat/mouth interfaces where a small
number of patch-averaged unknowns is a useful first model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Union

import numpy as np
from numpy.typing import NDArray

from .lem_to_bem import _aperture_face_areas

if TYPE_CHECKING:
    from hornlab_bempp_bem import SolveConfig, SolveResult
    from hornlab_bempp_bem.mesh import LoadedMesh


MeshLike = Union[str, Path, "LoadedMesh"]


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


def solve_aperture_matrix(
    mesh: MeshLike,
    aperture_tags: Mapping[str, list[int]],
    frequencies_hz: NDArray[np.float64],
    config: "SolveConfig | None" = None,
    *,
    normal_velocity: complex = 1.0 + 0.0j,
) -> RadiationImpedanceResult:
    """Compute a BEM radiation impedance matrix for aperture patches.

    Parameters
    ----------
    mesh
        Path to a surface ``.msh`` or a preloaded ``hornlab_bempp_bem.LoadedMesh``.
    aperture_tags
        Mapping from aperture name to one or more physical group IDs.
    frequencies_hz
        Positive frequencies to solve.
    config
        Optional BEM solve config.  The function overrides
        ``velocity_mode`` and ``velocity_sources`` for each source basis.
    normal_velocity
        Unit normal velocity imposed on all faces of the active source
        aperture.  Must be nonzero.

    Returns
    -------
    RadiationImpedanceResult
        Dense aperture matrix ``Z[f, receiver, source]``.
    """
    from hornlab_bempp_bem import SolveConfig as _SC
    from hornlab_bempp_bem import solve_frequencies
    from hornlab_bempp_bem.config import VelocityMode
    from hornlab_bempp_bem.mesh import load_mesh

    if config is None:
        config = _SC()

    freqs = _validate_frequencies(frequencies_hz)
    aperture_names = _validate_aperture_tags(aperture_tags)
    drive_velocity = complex(normal_velocity)
    if abs(drive_velocity) <= 0.0:
        raise ValueError("normal_velocity must be nonzero")

    if isinstance(mesh, (str, Path)):
        loaded = load_mesh(mesh, scale=config.mesh_scale)
    else:
        loaded = mesh

    aperture_area_m2 = _aperture_face_areas(loaded, aperture_tags)
    unique_tags = sorted({int(tag) for tags in aperture_tags.values() for tag in tags})
    tag_area_m2 = _tag_face_areas(loaded, unique_tags)

    base_config = replace(config, velocity_mode=VelocityMode.VELOCITY)
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
        result = solve_frequencies(loaded, freqs, source_config)
        if result.surface_pressure_avg is None:
            raise RuntimeError(
                "hornlab_bempp_bem result did not include surface_pressure_avg; "
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


def _validate_frequencies(values: NDArray[np.float64]) -> NDArray[np.float64]:
    freqs = np.asarray(values, dtype=np.float64).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies_hz is empty")
    bad = freqs[~np.isfinite(freqs) | (freqs <= 0.0)]
    if bad.size:
        raise ValueError(f"frequencies_hz must be positive finite values: {bad[:5]}")
    return freqs


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
