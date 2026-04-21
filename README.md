# Multi-dimension temporally blocked SOR

Extends the Lab 5–7 SOR work along two axes:

- **Parallelism tier** — single-thread CPU, pthreads, OpenMP, and CUDA GPU
  variants of the same stencil, so per-tier speedups can be attributed
  cleanly.
- **Algorithmic variant** — baseline Jacobi-like ping-pong sweep vs.
  temporally blocked (time-skewed) sweep that runs `T` sub-steps on a tile
  before writing it back, cutting DRAM traffic by ≈`T` at the cost of
  redundant halo work.

A separate red-black Gauss-Seidel variant (`sor2d_rb.c`) recovers real
over-relaxation (`omega in (1, 2)`) and exposes the classic Lab-5 U-curve
of iterations-to-converge vs. omega.

## Layout

```
project/
├── Makefile
├── README.md
├── report.md              # longer write-up
├── instructions.md        # full project goal / scope
└── src/
    ├── common.h           # timers, init, validation, PPM writer
    ├── sor2d_cpu.c        # 2D single-thread   baseline + temporal
    ├── sor3d_cpu.c        # 3D single-thread   baseline + temporal
    ├── sor2d_omp.c        # 2D OpenMP          baseline + temporal
    ├── sor3d_omp.c        # 3D OpenMP          baseline + temporal
    ├── sor2d_pth.c        # 2D pthreads        baseline (strip + barrier)
    ├── sor2d_rb.c         # 2D red-black GS SOR, omega-sweep mode
    └── sor2d_gpu.cu       # 2D CUDA            baseline + shared-memory temporal
```

## Comparison matrix

| | 2D baseline | 2D temporal | 3D baseline | 3D temporal | omega > 1 |
|---|:-:|:-:|:-:|:-:|:-:|
| single-thread   | `sor2d_cpu` | `sor2d_cpu` | `sor3d_cpu` | `sor3d_cpu` | `sor2d_rb` |
| pthreads        | `sor2d_pth` |     —       |     —       |     —       |     —      |
| OpenMP          | `sor2d_omp` | `sor2d_omp` | `sor3d_omp` | `sor3d_omp` |     —      |
| CUDA GPU        | `sor2d_gpu` | `sor2d_gpu` |     —       |     —       |     —      |

The temporal-blocking kernels all use the same Jacobi-like ping-pong
stencil (stable for `omega in (0, 1]`, so they run at `omega = 0.9`); the
red-black variant is the one place where true over-relaxation lives.

## Build & run

```
make                                       # CPU + OMP + pthread + red-black
make gpu                                   # CUDA binary (needs nvcc)

build/sor2d_cpu <N> <iters> [B=64] [T=4]   [--ppm path]
build/sor3d_cpu <N> <iters> [B=32] [T=2]
build/sor2d_omp <N> <iters> [B=64] [T=4]   [--ppm path]
build/sor3d_omp <N> <iters> [B=32] [T=2]
build/sor2d_pth <N> <iters> [nthreads=4]   [--ppm path]
build/sor2d_rb  <N> [--sweep | --omega w]  [--ppm path]
build/sor2d_gpu <N> <iters>                [--ppm path]

OMP_NUM_THREADS=16 build/sor2d_omp 4098 48 256 8
```

`iters` must be a multiple of `T` for the temporal kernels (or `HALO_T=4`
for the GPU binary). Each run validates its output against an in-binary
reference and prints `max|.. − ..|`.

### Omega sweep

```
make omega                                  # writes build/omega_sweep.csv
```

Produces a `(omega, iterations, seconds)` table for `omega in [0.50,
1.99]`. The minimum typically lands near the theoretical
`omega_opt = 2 / (1 + sin(pi / (N-1)))` — around 1.95 for `N=128`. Any
plotting tool (`gnuplot`, `matplotlib`) can read the CSV directly.

### Visualization

Every 2D driver accepts `--ppm <path>` and dumps the final field as a
binary grayscale PPM (P5), normalized to its own min/max. Open in any
image viewer, or convert: `convert out.ppm out.png`.

```
build/sor2d_cpu 512 64 --ppm build/cpu.ppm
build/sor2d_rb  512 --omega 1.95 --ppm build/rb.ppm
```

## A note on omega (ping-pong form)

The temporal-blocking kernels use the Lab-7-style ping-pong Jacobi update
`dst = s - omega*(s - avg(nbrs))`. That is equivalent to the damped Jacobi
operator `M = (1 - omega)*I + omega*J` with `rho(J) ≈ 1` on a large Poisson
grid, so it is stable only for `omega in (0, 1]`. We use `omega = 0.9`.
Lab 7's `omega = 1.85` in this form is unconditionally unstable; it only
"passes" the Lab-7 correctness check because the CPU and GPU diverge in
lockstep and `max|a − b|` with matching `NaN` inputs returns 0.

True SOR with `omega in (1, 2)` requires Gauss-Seidel ordering — hence
`sor2d_rb.c`. Extending the temporal kernels to red-black Gauss-Seidel
(shrinking trapezoid in pairs of half-sweeps) is the natural next step,
and would unify the two stories.

## Temporal-blocking scheme

For each super-step of `T` sweeps, the domain is tiled into `BxB` (`BxBxB`)
output tiles. Per tile:

1. Load a `(B+2T)^d` scratch from `src` with clamp-to-edge.
2. Run `T` in-scratch sweeps; the valid update region shrinks by 1 per
   side each sub-step, so after `T` sub-steps the central `B^d` is the
   correct `T`-step-advanced state. A `memcpy(Bp, A, ...)` before each
   sub-step carries through the shrinking frame (and the
   global-boundary-excluded strips) so the inner loop is branch-free.
3. Copy the central `B^d` block back to `dst`.

Super-steps ping-pong at the caller level between the two top-level
buffers, so there is no `O(N^d)` copy between super-steps.

## Status

`report.md` has the full write-up: single-thread CPU, OpenMP thread-scaling
to 16 threads (1.82× temporal speedup at 16 threads, N=4098), and V100 GPU
numbers (~130 Gup/s temporal, ~2–3× over the coalesced Lab-7-style
baseline at realistic `N`). This README is the quick-start; `report.md` is
the detail.
