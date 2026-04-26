# E08 — Red-black GS-SOR: omega U-curve

Machine: SCC scc-j06.  Single-thread CPU run (the omega sweep is about
the algorithm, not the hardware).
Config: N=128, TOL=1e-5, MAX_ITERS=100000.  Binary: `sor2d_rb --sweep`.
Methodology: 150 omega values in [0.50, 1.99] in 0.01 steps (denser than
the doc's 0.02 default); each row reports
`omega, iters, seconds, cycles` for one solve.

## Headline finding

| metric                   | value         |
|:-------------------------|:--------------|
| theory ω_opt = 2/(1+sin(π/(N-1))) | **1.9517** |
| empirical ω_opt          | **1.95**      |
| iters at empirical ω_opt | **254**       |
| iters at ω = 1.0 (Jacobi)| 3,434         |
| iters at ω = 0.5         | 7,225         |
| iters at ω = 1.99        | 1,230 (rising again) |
| **SOR speedup vs Jacobi**| **13.5× fewer iterations** |

Theory and experiment agree to within **0.1%**.

## The U-curve story

For the symmetric positive-definite Laplacian, classical SOR has an
optimal over-relaxation factor

  ω_opt = 2 / (1 + sin(π / (N-1)))

derived from the spectral radius of the iteration matrix.  Plotting
iterations-to-converge vs. ω gives the canonical U-curve:

- **ω < 1**: under-relaxation.  Each cell update only partially commits
  to the new value; convergence is slower than plain Gauss-Seidel.
- **ω = 1**: pure Gauss-Seidel (no over-relaxation).  Converges in
  ~3,434 iters at N=128.
- **ω in (1, ω_opt)**: over-relaxation.  Each update overshoots toward
  the equilibrium, accelerating convergence.  Iters drops monotonically.
- **ω = ω_opt = 1.95**: the spectral-radius minimum.  **254 iters** —
  13.5× fewer than Jacobi.
- **ω > ω_opt**: still convergent, but the iteration becomes
  oscillatory; iters climb sharply.  At ω = 1.99 already up to 1,230.
- **ω ≥ 2**: the iteration diverges (not plotted; binary refuses /
  caps at MAX_ITERS).

## Why this matters for the talk (the numerical-algorithm slide)

The omega U-curve is **the** classical SOR result and the cleanest way
to motivate why this algorithm exists.  In one figure you can show:

1. Jacobi/Gauss-Seidel (ω=1) is **not** the right answer — there's an
   over-relaxation regime that's an order of magnitude faster.
2. The optimal ω depends on grid size, but a closed-form formula gets
   you to within 0.1% of the empirical minimum.
3. The U-curve is sharply asymmetric — under-relaxation (ω < ω_opt)
   degrades convergence linearly, but over-relaxation (ω > ω_opt) is
   a cliff: at ω=1.99 you've already given back most of the win.
4. ω is the **only knob you adjust** to get this 13.5× speedup;
   everything else stays the same.  This is "algorithm tuning" — the
   single most cost-effective optimization in the whole project.

## Cross-reference: omega in our temporal-blocking kernels

The other tiers (sor2d_cpu, sor2d_omp, sor2d_pth, sor2d_gpu) use a
**Jacobi-like ping-pong** stencil at omega=0.9 (damped Jacobi: stable
only for ω in (0, 1] on large grids).  This means those tiers can't
exploit the U-curve win — they live somewhere on the under-relaxation
side of the curve and pay 11.4× more iterations than this red-black
GS-SOR variant would for the same convergence.

The temporal-blocking optimizations win on **per-iter throughput** (E01
through E06).  The omega-tuning here wins on **iters-to-convergence**.
The two are complementary; combining them — temporal blocking with
red-black GS-SOR — would multiply the wins.  We didn't implement that
because the shrinking-trapezoid scheme has to be re-derived for paired
red/black half-sweeps; report.md lists it as future work.

## Surprise

The U-curve minimum is *sharper than I expected*.  ω = 1.95 → 254
iters; ω = 1.99 → 1,230 iters.  That's a 4.8× penalty for being 0.04
off the optimum.  This says: **omega-tuning is high-stakes**.  In
production code with a fixed N you compute ω_opt analytically; if N
changes (multigrid, AMR), ω needs to be retuned each level.

Plot: `omega_ucurve.png` — log-y plot of iters vs ω (left axis) with
total cycles (right axis), theory and empirical optima marked.

## STATUS update

E08: done — empirical ω_opt = **1.95** (254 iters) matches theory
**1.9517** to 0.1%; **13.5× fewer iters** than Jacobi (ω=1).
