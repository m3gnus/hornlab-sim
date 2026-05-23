# hornlab-sim

Canonical lumped and LEM-BEM coupling simulators for HornLab.

This package holds the small/fast acoustic models that complement the BEM
pipeline in `hornlab-mesher` + `hornlab-solver`.

## Modules

- `hornlab_sim.methods.bandpass` — BP4/BP6S lumped enclosure simulation, validated within ~2 dB vs Hornresp
- `hornlab_sim.methods.transfer_matrix` — segmented TMM with Kirchhoff-Benade viscothermal losses (Beranek & Mellow 2012)
- `hornlab_sim.methods.helmholtz` — slot-pocket and mid-chamber Helmholtz resonance (BIGMEH canonical interpretation: pocket = cavity, exit = baffled hole, L=0, end correction only)
- `hornlab_sim.methods.lem_to_bem` — forward LEM → BEM coupling (LEM-computed aperture velocities drive a BEM Neumann boundary condition)

## Install

```bash
pip install -e .              # core (LEM/TMM/Helmholtz only)
pip install -e .[bem]         # + hornlab-solver and hornlab-mesher for LEM-BEM coupling
pip install -e .[dev]         # + pytest
```

## Agent / user guidance

See `AGENTS.md` for the decision tree: which method to reach for given the
design question being asked.

For LEM→BEM coupling, suppress LEM-side external radiation loading on any
BEM-radiated aperture. Use `end_corr="none"` with Helmholtz helpers and
`Port(..., radiation_external=False)` with bandpass ports.
