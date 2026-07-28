"""Tetrahedral pressure-acoustics FEM and reduced multiport coupling.

The solver uses first-order tetrahedra for the interior Helmholtz problem
with rigid walls and uniform-volume-velocity bases on named boundary patches.
It returns the frequency-dependent matrix

``p_average = Z_interior @ q_outward``

where every entry of ``q_outward`` is positive out of the FEM air volume and
phasors use the engineering ``exp(+j omega t)`` convention.  This deliberately
small interface is intended for chamber/port reduction before coupling to an
exterior BEM radiation matrix; it is not a full trace-space FEM-BEM solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FEM_EXTRA_INSTALL_HINT = 'pip install "hornlab-sim[fem]"'
RHO_AIR = 1.2041
C_AIR = 343.0


@dataclass(frozen=True)
class AcousticFEMMesh:
    """Linear tetrahedral volume mesh with tagged boundary triangles."""

    points_m: NDArray[np.float64]
    tetrahedra: NDArray[np.int64]
    boundary_triangles: NDArray[np.int64]
    boundary_tags: NDArray[np.int64]
    boundary_name_to_tag: Mapping[str, int]


@dataclass(frozen=True)
class AcousticFEMSystem:
    """Frequency-independent assembled FEM matrices and port projectors."""

    mesh: AcousticFEMMesh
    stiffness: object
    mass: object
    boundary_names: tuple[str, ...]
    boundary_areas_m2: Mapping[str, float]
    boundary_average: object
    volume_m3: float


@dataclass(frozen=True)
class AcousticFEMResult:
    """Interior multiport impedance matrix in engineering convention."""

    frequencies_hz: NDArray[np.float64]
    boundary_names: tuple[str, ...]
    boundary_areas_m2: Mapping[str, float]
    impedance_matrix: NDArray[np.complex128]
    volume_m3: float
    loss_factor: float


@dataclass(frozen=True)
class FEMBEMCouplingResult:
    """Reduced coupling of one FEM driver boundary to exterior BEM entries."""

    frequencies_hz: NDArray[np.float64]
    driver_boundary: str
    entry_boundaries: tuple[str, ...]
    entry_to_driver_volume_velocity: NDArray[np.complex128]
    driver_acoustic_load: NDArray[np.complex128]


def _scipy_sparse():
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        if exc.name == "scipy" or str(exc.name).startswith("scipy."):
            raise ImportError(
                "acoustic FEM requires the optional SciPy dependency. Install "
                f"it with: {FEM_EXTRA_INSTALL_HINT}"
            ) from exc
        raise
    return sparse, sparse_linalg


def load_gmsh_mesh(
    path: str | Path,
    *,
    scale_to_m: float = 1.0,
) -> AcousticFEMMesh:
    """Load tetrahedra and tagged boundary triangles from a Gmsh mesh."""
    try:
        import meshio
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        if exc.name == "meshio" or str(exc.name).startswith("meshio."):
            raise ImportError(
                "loading an acoustic FEM mesh requires meshio. Install it with: "
                f"{FEM_EXTRA_INSTALL_HINT}"
            ) from exc
        raise

    scale = float(scale_to_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_to_m must be positive and finite")
    raw = meshio.read(Path(path))
    if "tetra" not in raw.cells_dict:
        raise ValueError(f"FEM mesh has no first-order tetrahedra: {path}")
    if "triangle" not in raw.cells_dict:
        raise ValueError(f"FEM mesh has no boundary triangles: {path}")
    try:
        triangle_tags = raw.cell_data_dict["gmsh:physical"]["triangle"]
    except KeyError as exc:
        raise ValueError("FEM boundary triangles have no gmsh:physical tags") from exc

    names: dict[str, int] = {}
    for name, value in raw.field_data.items():
        data = np.asarray(value).reshape(-1)
        if data.size >= 2 and int(data[1]) == 2:
            names[str(name)] = int(data[0])
    if not names:
        raise ValueError("FEM mesh has no named physical surface groups")
    return validate_mesh(
        AcousticFEMMesh(
            points_m=np.asarray(raw.points[:, :3], dtype=np.float64) * scale,
            tetrahedra=np.asarray(raw.cells_dict["tetra"], dtype=np.int64),
            boundary_triangles=np.asarray(
                raw.cells_dict["triangle"], dtype=np.int64
            ),
            boundary_tags=np.asarray(triangle_tags, dtype=np.int64).reshape(-1),
            boundary_name_to_tag=names,
        )
    )


def validate_mesh(mesh: AcousticFEMMesh) -> AcousticFEMMesh:
    points = np.asarray(mesh.points_m, dtype=np.float64)
    tetrahedra = np.asarray(mesh.tetrahedra, dtype=np.int64)
    triangles = np.asarray(mesh.boundary_triangles, dtype=np.int64)
    tags = np.asarray(mesh.boundary_tags, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("points_m must have shape (n>=4, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_m contains non-finite coordinates")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or tetrahedra.size == 0:
        raise ValueError("tetrahedra must have shape (n>=1, 4)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.size == 0:
        raise ValueError("boundary_triangles must have shape (n>=1, 3)")
    if tags.shape != (triangles.shape[0],):
        raise ValueError("boundary_tags length must match boundary_triangles")
    for label, cells in (("tetrahedra", tetrahedra), ("boundary_triangles", triangles)):
        if int(np.min(cells)) < 0 or int(np.max(cells)) >= points.shape[0]:
            raise ValueError(f"{label} contains an out-of-range point index")
    if len(set(mesh.boundary_name_to_tag.values())) != len(mesh.boundary_name_to_tag):
        raise ValueError("boundary physical-group tags must be unique")
    return AcousticFEMMesh(
        points_m=points,
        tetrahedra=tetrahedra,
        boundary_triangles=triangles,
        boundary_tags=tags,
        boundary_name_to_tag={
            str(name): int(tag) for name, tag in mesh.boundary_name_to_tag.items()
        },
    )


def assemble_system(
    mesh: AcousticFEMMesh,
    boundary_names: Sequence[str],
) -> AcousticFEMSystem:
    """Assemble linear-tetrahedron stiffness/mass and boundary averages."""
    sparse, _ = _scipy_sparse()
    checked = validate_mesh(mesh)
    names = tuple(str(name) for name in boundary_names)
    if not names:
        raise ValueError("at least one boundary name is required")
    if len(set(names)) != len(names):
        raise ValueError("boundary_names must be unique")
    missing = [name for name in names if name not in checked.boundary_name_to_tag]
    if missing:
        raise ValueError(f"FEM mesh is missing boundary groups: {', '.join(missing)}")

    coords = checked.points_m[checked.tetrahedra]
    affine = np.concatenate(
        [np.ones((coords.shape[0], 4, 1), dtype=np.float64), coords], axis=2
    )
    try:
        inverse = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("FEM mesh contains a degenerate tetrahedron") from exc
    jac = coords[:, 1:, :] - coords[:, :1, :]
    volumes = np.abs(np.linalg.det(jac)) / 6.0
    if np.any(volumes <= 1.0e-18):
        raise ValueError("FEM mesh contains a zero-volume tetrahedron")
    gradients = inverse[:, 1:, :]
    stiffness_local = volumes[:, None, None] * np.einsum(
        "mki,mkj->mij", gradients, gradients
    )
    mass_template = np.ones((4, 4), dtype=np.float64) + np.eye(4)
    mass_local = volumes[:, None, None] * mass_template[None, :, :] / 20.0

    cells = checked.tetrahedra
    rows = np.repeat(cells, 4, axis=1).reshape(-1)
    cols = np.tile(cells, (1, 4)).reshape(-1)
    shape = (checked.points_m.shape[0], checked.points_m.shape[0])
    stiffness = sparse.coo_matrix(
        (stiffness_local.reshape(-1), (rows, cols)), shape=shape
    ).tocsr()
    mass = sparse.coo_matrix(
        (mass_local.reshape(-1), (rows, cols)), shape=shape
    ).tocsr()

    boundary_average = np.zeros((checked.points_m.shape[0], len(names)), dtype=np.float64)
    areas: dict[str, float] = {}
    tri_coords = checked.points_m[checked.boundary_triangles]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(tri_coords[:, 1] - tri_coords[:, 0], tri_coords[:, 2] - tri_coords[:, 0]),
        axis=1,
    )
    for column, name in enumerate(names):
        tag = checked.boundary_name_to_tag[name]
        selected = np.flatnonzero(checked.boundary_tags == tag)
        if selected.size == 0:
            raise ValueError(f"FEM boundary group {name!r} has no triangles")
        area = float(np.sum(tri_area[selected]))
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError(f"FEM boundary group {name!r} has zero area")
        local_weight = tri_area[selected] / (3.0 * area)
        for local_vertex in range(3):
            np.add.at(
                boundary_average[:, column],
                checked.boundary_triangles[selected, local_vertex],
                local_weight,
            )
        areas[name] = area

    return AcousticFEMSystem(
        mesh=checked,
        stiffness=stiffness,
        mass=mass,
        boundary_names=names,
        boundary_areas_m2=areas,
        boundary_average=sparse.csc_matrix(boundary_average),
        volume_m3=float(np.sum(volumes)),
    )


def solve_multiport(
    system: AcousticFEMSystem,
    frequencies_hz: NDArray[np.float64] | Sequence[float],
    *,
    rho: float = RHO_AIR,
    c: float = C_AIR,
    loss_factor: float = 0.002,
) -> AcousticFEMResult:
    """Solve every uniform boundary-volume-velocity basis per frequency."""
    _, sparse_linalg = _scipy_sparse()
    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if freqs.size == 0 or not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("frequencies_hz must contain positive finite values")
    rho_f = float(rho)
    c_f = float(c)
    loss = float(loss_factor)
    if not np.isfinite(rho_f) or rho_f <= 0.0:
        raise ValueError("rho must be positive and finite")
    if not np.isfinite(c_f) or c_f <= 0.0:
        raise ValueError("c must be positive and finite")
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("loss_factor must be non-negative and finite")

    projector = system.boundary_average.toarray()
    n_ports = len(system.boundary_names)
    impedance = np.empty((freqs.size, n_ports, n_ports), dtype=np.complex128)
    for index, frequency in enumerate(freqs):
        omega = 2.0 * np.pi * float(frequency)
        wavenumber = (omega / c_f) * (1.0 - 1j * loss)
        matrix = (system.stiffness - (wavenumber * wavenumber) * system.mass).tocsc()
        rhs = (-1j * omega * rho_f) * projector
        try:
            pressure = sparse_linalg.spsolve(matrix, rhs)
        except RuntimeError as exc:
            raise RuntimeError(
                f"acoustic FEM factorization failed at {frequency:g} Hz"
            ) from exc
        if not np.all(np.isfinite(pressure)):
            raise RuntimeError(
                f"acoustic FEM solve produced non-finite pressure at "
                f"{frequency:g} Hz"
            )
        if pressure.ndim == 1:
            pressure = pressure[:, None]
        impedance[index] = np.asarray(projector.T @ pressure, dtype=np.complex128)

    return AcousticFEMResult(
        frequencies_hz=np.array(freqs, dtype=np.float64, copy=True),
        boundary_names=system.boundary_names,
        boundary_areas_m2=dict(system.boundary_areas_m2),
        impedance_matrix=impedance,
        volume_m3=float(system.volume_m3),
        loss_factor=loss,
    )


def couple_exterior_impedance(
    interior: AcousticFEMResult,
    exterior_impedance: NDArray[np.complex128],
    *,
    driver_boundary: str,
    entry_boundaries: Sequence[str],
    exterior_convention: Literal["engineering", "solver"] = "engineering",
) -> FEMBEMCouplingResult:
    """Condense FEM entries against an exterior BEM radiation matrix.

    ``exterior_impedance[f, receiver, source]`` must use the same entry order
    as ``entry_boundaries`` and map outward entry volume velocity to interface
    pressure. The default ``exterior_convention="engineering"`` preserves the
    engineering ``exp(+j omega t)`` contract. Matrices returned by
    :func:`hornlab_sim.methods.radiation_impedance.solve_aperture_matrix` use
    the conjugated solver convention; pass ``exterior_convention="solver"``
    to convert them here, or pre-convert with
    :func:`hornlab_sim.methods.radiation_impedance.termination_load_from_solver_matrix`.
    The returned ratios use positive driver volume velocity *into* the chamber
    and positive entry flow out.
    """
    entries = tuple(str(name) for name in entry_boundaries)
    driver = str(driver_boundary)
    if not entries:
        raise ValueError("at least one entry boundary is required")
    if len(set(entries)) != len(entries):
        raise ValueError("entry_boundaries must be unique")
    if driver in entries:
        raise ValueError("driver_boundary must not appear in entry_boundaries")
    if exterior_convention not in ("engineering", "solver"):
        raise ValueError(
            "exterior_convention must be 'engineering' or 'solver'"
        )
    names = interior.boundary_names
    try:
        driver_index = names.index(driver)
        entry_indices = [names.index(name) for name in entries]
    except ValueError as exc:
        raise ValueError("driver or entry boundary is absent from the FEM result") from exc
    exterior = np.asarray(exterior_impedance, dtype=np.complex128)
    if exterior_convention == "solver":
        exterior = np.conjugate(exterior)
    expected = (interior.frequencies_hz.size, len(entries), len(entries))
    if exterior.shape != expected:
        raise ValueError(
            f"exterior_impedance shape {exterior.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(exterior.real) & np.isfinite(exterior.imag)):
        raise ValueError("exterior_impedance contains non-finite values")

    entry_index = np.asarray(entry_indices)
    reduced_interface = interior.impedance_matrix[
        :, entry_index[:, None], entry_index
    ]
    reduced_interface -= exterior
    entry_drive = interior.impedance_matrix[:, entry_index, driver_index]
    try:
        ratios = np.linalg.solve(
            reduced_interface,
            entry_drive[..., None],
        )[..., 0]
    except np.linalg.LinAlgError as exc:
        failed_index = 0
        for f_index, (matrix, drive) in enumerate(
            zip(reduced_interface, entry_drive)
        ):
            try:
                np.linalg.solve(matrix, drive)
            except np.linalg.LinAlgError:
                failed_index = f_index
                break
        raise RuntimeError(
            "FEM-BEM reduced interface solve failed at "
            f"{interior.frequencies_hz[failed_index]:g} Hz"
        ) from exc
    driver_load = np.fromiter(
        (
            -matrix[driver_index, driver_index]
            + matrix[driver_index, entry_index] @ ratio
            for matrix, ratio in zip(interior.impedance_matrix, ratios)
        ),
        dtype=np.complex128,
        count=interior.frequencies_hz.size,
    )
    return FEMBEMCouplingResult(
        frequencies_hz=np.array(interior.frequencies_hz, copy=True),
        driver_boundary=driver,
        entry_boundaries=entries,
        entry_to_driver_volume_velocity=ratios,
        driver_acoustic_load=driver_load,
    )


__all__ = [
    "AcousticFEMMesh",
    "AcousticFEMResult",
    "AcousticFEMSystem",
    "FEMBEMCouplingResult",
    "assemble_system",
    "couple_exterior_impedance",
    "load_gmsh_mesh",
    "solve_multiport",
    "validate_mesh",
]
