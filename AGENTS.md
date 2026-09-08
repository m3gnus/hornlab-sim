# hornlab-sim — Agent Instructions

This package is the canonical home for lumped (LEM), transfer-matrix (TMM),
Helmholtz, and LEM↔BEM coupling simulators. There is **no global project
orchestrator**, **no project registry**, **no MEH intent table** in this
package. Read this file and pick the right reusable method.

Project-specific orchestration belongs in the consuming project. Keep this
package limited to reusable physics methods.

## Decision tree

| Design question | Method | Module |
|---|---|---|
| Box tuning, Fb, Qtc, BP4/BP6S alignment | LEM bandpass | `hornlab_sim.methods.bandpass` (`bp4_sealed_rear`, `bp6s_dual_ported`) |
| Hornresp text import/export or validation overlay | Hornresp interchange | `hornlab_sim.hornresp` |
| Slot-pocket Helmholtz tuning (slot-pocket = front cavity + baffled hole) | LEM Helmholtz | `hornlab_sim.methods.helmholtz` (`slot_helmholtz`, `mid_chamber_helmholtz_from_params`) |
| Mid-chamber resonance for a midport pocket | LEM Helmholtz | `hornlab_sim.methods.helmholtz` (`mid_chamber_helmholtz_from_params`) |
| Axial duct impedance with viscothermal losses (segmented horn/duct) | TMM | `hornlab_sim.methods.transfer_matrix` (`duct_input_impedance`, `uniform_tube_matrix`, `make_slot_tmm_load`) or `hornlab-tmm` |
| Xmax/amp-voltage output ceiling | Max SPL | `hornlab_sim.methods.max_spl` |
| Vented box seed and short-port feasibility screen | Bass-reflex screening | `hornlab_sim.methods.bass_reflex` |
| Full 3D directivity from prescribed velocity sources (no LEM) | BEM | `hornlab_metal_bem` directly |
| Full 3D directivity with realistic LEM/TMM-derived source velocities at apertures | Source-basis BEM coupling | `hornlab_sim.methods.lem_to_bem` for generic/custom meshes; consuming project adapters own observation frames, caching, and result schemas |
| Reduced FEM/BEM-style aperture back-loading | BEM radiation impedance matrix | `hornlab_sim.methods.radiation_impedance` |
| Meshed 3D chamber modes and unequal multi-entry flow | Pressure FEM multiport matrix | `hornlab_sim.methods.acoustic_fem` (optional `fem` extra) |
| Voltage-driven coupled cone+port velocities for a BEM-terminated cardioid branch | LEM driver/BEM termination coupling | `hornlab_sim.methods.driver_coupling` |

## Canonical interpretations (load-bearing — do not change without explicit user sign-off)

- **Slot pocket = front cavity. Slot exit = baffled hole, L=0, end correction only.** This is the canonical interpretation for this package. Helmholtz code in this package implements it by default. Memory: `feedback_helmholtz_slot_pocket.md`.
- **Kirchhoff-Benade viscothermal losses** are built into `transfer_matrix.py`. Beranek & Mellow 2012 formulation. Don't replace with a different loss model without rerunning the Hornresp parity tests.
- **Bandpass `Port(Q_port=None)` derives viscothermal Q from geometry** using the same Kirchhoff-Benade boundary-layer scaling. Pass an explicit numeric `Q_port` for fixed-Q legacy behavior.
- **BP4/BP6S** are validated within ~2 dB vs Hornresp across the LF passband.
- **Reduced FEM/BEM uses 1.2041 kg/m^3 on both sides.** `acoustic_fem` matches
  the canonical Metal BEM density. The validated legacy LEM/TMM/Helmholtz
  methods intentionally retain their pinned 1.21 kg/m^3 defaults.
- **The forward LEM→BEM boundary has an explicit convention seam.**
  `lem_to_bem.solve(velocity_convention=...)` declares the time convention of
  the supplied volume velocities. The default `"engineering"` conjugates
  `U/A` once into the solver's `e^{-iωt}` convention; `"solver"` applies it
  unchanged. See LEM↔BEM coupling rule 7 for the consumer obligation.
- **Reduced exterior matrices have an explicit convention seam.**
  `radiation_impedance.solve_aperture_matrix` returns the conjugated Metal
  solver convention. Before passing that matrix to
  `acoustic_fem.couple_exterior_impedance`, either set
  `exterior_convention="solver"` or convert it once with
  `radiation_impedance.termination_load_from_solver_matrix`. The coupling
  function's default remains engineering `exp(+j omega t)` for compatibility.

## LEM↔BEM coupling rules

When using `hornlab_sim.methods.lem_to_bem`:

1. **Suppress LEM-side radiation loading at any aperture that BEM is going to radiate.** For Helmholtz helpers, pass `end_corr="none"` for that aperture. For bandpass `Port`, pass `radiation_external=False` so both the outside end correction and radiation resistance are suppressed. Otherwise the radiation loading is double-counted (once by LEM, once by BEM-computed radiation impedance) and you get systematic dB errors at the LF/MF crossover.
2. **Aperture name → list of physical group IDs.** Parametric cabinet meshes already produce multiple tags per slot (exit + walls). The coupling layer expects `dict[str, list[int]]`, not 1:1.
   Each physical group ID must belong to exactly one aperture; overlapping tag lists are rejected before the solver runs.
3. **Area mismatch is a warning, not an error.** Sum of BEM face areas vs LEM-assumed S can disagree up to ~5% from mesh discretization. Warn, log to result metadata, continue.
4. **Velocity vs acceleration mode.** Canonical solver default is `velocity_mode=ACCELERATION`. LEM emits volume velocity U. The coupling layer converts U → v_n and lets the solver apply jω. Don't pass acceleration directly.
5. **Phase reference.** All LEM aperture velocities share an excitation reference (driver terminal voltage). Mesh-side BCs must use the same complex sign convention. The +iωρv convention matches `hornlab_metal_bem` canonical settings.
6. **Mesh array layout is never guessed.** `hornlab_metal_bem.mesh.PureGrid` is Bempp-shaped: `vertices` is `(3, n_vertices)` and `elements` is `(3, n_elements)`, triangles in columns. Convenience grids are often row-major. A `(3, 3)` array is valid as either, so the aperture-area helper shared by `lem_to_bem` and `radiation_impedance` resolves the layout from an explicit `mesh_array_layout=`, a `grid.array_layout` attribute, the authoritative per-element areas the grid carries (`PureGrid.volumes`), or an unambiguous shape — and raises when none of those apply. Do not restore shape-based inference: a three-element or three-vertex mesh silently produced zero-area apertures under it.
7. **Time convention at the forward boundary.** A shared excitation reference is not a shared *time* convention. This package's LEM/TMM methods are engineering `e^{+jωt}` (`s = +jω`); the Metal solver is `e^{-iωt}` with an outgoing `exp(+ikr)` kernel, so an engineering source phasor must be conjugated once on the way in. `lem_to_bem.solve` does that itself: `velocity_convention="engineering"` (the default) conjugates `U/A` before either the basis or the sequential path builds a Neumann source, and `velocity_convention="solver"` applies the values unchanged. **Consumer obligation:** a caller that already converted its source phasors to solver convention must pass `velocity_convention="solver"`, or the conversion is applied twice and destructive multi-aperture interference becomes constructive. Real-valued velocities are identical in both conventions. This is the forward counterpart of the return-boundary seam below (`termination_load_from_solver_matrix`), and both sides are the same `conj` involution.

For project validation plots, prefer a project adapter over calling this
generic solver-level coupling directly when the project owns a specific
observation frame, basis cache, or result schema.

## Solver settings (BEM side)

- BM=off everywhere
- COMPLEX_K formulation; complex_k_shift=0.005 when enclosed
- LU solver for narrow slots, AUTO otherwise
- DP0/P1, q=4 regular quadrature
- +iωρv sign convention

These defaults live in `hornlab_metal_bem.SolveConfig`. Don't override without explicit reason.

## What lives where

| Concept | Location |
|---|---|
| LEM/TMM/Helmholtz core math | `hornlab_sim.methods.*` (this package) |
| Hornresp text interchange | `hornlab_sim.hornresp` |
| Aperture radiation impedance matrices | `hornlab_sim.methods.radiation_impedance` |
| Reusable tetrahedral chamber FEM and reduced FEM-BEM condensation | `hornlab_sim.methods.acoustic_fem` |
| Project geometry, params, driver presets, adapters, and project-flavored CLIs | consuming project |
| BEM mesher | `hornlab-waveguide-mesher` |
| BEM solver | `hornlab-metal-bem` |

If a calculation is project-specific, it belongs in the consuming project. If
it is general lumped or coupling physics, it belongs here.
