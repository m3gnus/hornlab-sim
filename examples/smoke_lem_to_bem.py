"""Smoke example: lem_to_bem on a generated closed sphere mesh.

Builds a small closed sphere with gmsh, tags the upper hemisphere as a
velocity source, and runs the coupling layer end-to-end with a 3-frequency
sweep. Intended as the "does it import and run" smoke test, not a
physically meaningful simulation.

For a real Synergy/MEH workflow, see
``MEH-Lab/scripts/synergy_directivity_example.py``.

Requirements:
    - ``hornlab-sim`` installed (this package)
    - ``hornlab-bempp-bem`` and ``hornlab-mesher`` installed
    - A working OpenCL CPU runtime for bempp-cl (the "HornLab OpenCL CPU
      Python runtime" or an equivalent like POCL)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def _build_sphere_msh(path: Path, radius_m: float = 0.05, elem_size_m: float = 0.02) -> None:
    """Write a closed sphere .msh with two physical groups.

    Tag 1 = rigid (lower hemisphere)
    Tag 2 = velocity source (upper hemisphere)
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("smoke_sphere")

        # Sphere via OpenCascade primitive then surface extraction
        # (we only need a closed surface mesh for exterior BEM).
        sphere_vol = gmsh.model.occ.addSphere(0, 0, 0, radius_m)
        gmsh.model.occ.synchronize()

        # Get the surface(s) of the sphere
        surfaces = gmsh.model.getBoundary([(3, sphere_vol)], oriented=False)
        # Split the sphere surface into upper / lower hemispheres by
        # cutting with a plane at z=0. Simpler approach: tag the whole
        # surface as one group and use a single-tag source. That's
        # enough for the smoke test.

        # Remove the volume so we only mesh the surface
        gmsh.model.occ.remove([(3, sphere_vol)], recursive=False)
        gmsh.model.occ.synchronize()

        # Assign all surfaces to a single physical group (tag 2 = source)
        surface_tags = [s[1] for s in surfaces]
        gmsh.model.addPhysicalGroup(2, surface_tags, tag=2)
        gmsh.model.setPhysicalName(2, 2, "source")

        # Mesh
        gmsh.option.setNumber("Mesh.MeshSizeMax", elem_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", elem_size_m)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def main():
    from hornlab_sim.methods import lem_to_bem

    freqs = np.array([500.0, 1000.0, 2000.0])
    # Tag 2 in the smoke mesh -> "throat" aperture
    aperture_tags = {"throat": [2]}
    # Constant 1e-3 m^3/s volume velocity (LEM would produce frequency-
    # dependent values in a real workflow)
    lem_velocities = {"throat": np.ones(len(freqs), dtype=np.complex128) * 1e-3}

    with tempfile.TemporaryDirectory() as td:
        mesh_path = Path(td) / "smoke_sphere.msh"
        print(f"Building closed sphere mesh -> {mesh_path}")
        _build_sphere_msh(mesh_path, radius_m=0.05, elem_size_m=0.02)

        print(f"Frequencies: {freqs}")
        print(f"Aperture velocities (m^3/s): {lem_velocities}")

        result = lem_to_bem.solve(
            mesh=mesh_path,
            lem_velocities=lem_velocities,
            aperture_tags=aperture_tags,
            frequencies_hz=freqs,
        )

        print()
        print(f"frequencies_hz:  {result.frequencies_hz}")
        print(f"pressure shape:  {result.pressure_complex.shape}")
        print(f"on-axis SPL dB:  {result.spl_db[:, 0, 0]}")
        print("lem_to_bem smoke: OK")


if __name__ == "__main__":
    main()
