# Multi-dimension temporally blocked SOR

Extends the Lab 5–7 SOR work with **time skewing** — running multiple sweeps
on a tile before writing back — in both 2D (5-point stencil) and 3D (7-point
stencil). The iteration form is the Jacobi-like ping-pong used in Lab 7
(`dst[i,j] = src[i,j] - omega*(src[i,j] - avg(neighbors))`), so each super-step
of `T` sweeps reads only old values and tiles are independent within a
super-step.

## Layout

```
project/
├── Makefile
├── README.md
├── src/
│   ├── common.h          # timers, init, validation
│   ├── sor2d_cpu.c       # 2D single-thread baseline + temporally blocked
│   ├── sor3d_cpu.c       # 3D single-thread baseline + temporally blocked
│   ├── sor2d_omp.c       # 2D OpenMP baseline + temporally blocked
│   ├── sor3d_omp.c       # 3D OpenMP baseline + temporally blocked
│   └── sor2d_gpu.cu      # 2D CUDA baseline + shared-memory temporal
└── build/                # generated
```

## Build & run

```
make                                   # build all CPU + OMP binaries
make gpu                               # build the CUDA binary (needs nvcc)
build/sor2d_cpu <N> <iters> [B] [T]    # N x N grid,  default B=64  T=4
build/sor3d_cpu <N> <iters> [B] [T]    #       cubic, default B=32  T=2
OMP_NUM_THREADS=16 build/sor2d_omp <N> <iters> [B] [T]
OMP_NUM_THREADS=16 build/sor3d_omp <N> <iters> [B] [T]
build/sor2d_gpu <N> <iters>            # TILE/HALO_T are compile-time constants
```

`iters` must be a multiple of `T` (or `HALO_T` for the GPU binary). Each run
validates against an in-binary reference (CPU baseline for CPU/OMP, CPU
reference for GPU) and prints `max|.. − ..|` with a warning on non-finite
values.

## A note on omega

The iteration form used throughout is the Lab-7-style ping-pong Jacobi
update, `dst = s - omega*(s - avg(nbrs))`, equivalent to the damped Jacobi
operator `M = (1 − omega)I + omega*J`. For a Poisson Laplacian on a large
grid `rho(J) ≈ 1`, so this form is stable only for `omega ∈ (0, 1]`. We use
`omega = 0.9`. Lab 7's `omega = 1.85` diverges in this form — it only
"passes" its own correctness check because the CPU and GPU diverge in
lockstep and `max|a − b|` with `NaN` inputs returns 0 (NaN comparisons are
false). `common.h::max_abs_diff` here explicitly flags non-finite values.
True SOR with `omega ∈ (1, 2)` requires Gauss-Seidel ordering (e.g. red-black
coloring), which is a straightforward but more involved change — see Next
steps.

## Temporal-blocking scheme

For each super-step of `T` sweeps, the domain is tiled into `B×B` (`B×B×B`)
output tiles. For each tile:

1. A `(B+2T)^d` scratch is loaded from `src` with clamp-to-edge.
2. `T` in-scratch sweeps run. At sub-step `t`, the valid update region
   shrinks by 1 on each side, so after `T` sub-steps the central `B^d` is
   the correct `T`-step-advanced state. A pre-`memcpy(Bp, A, ...)` before
   each sub-step carries through the shrinking frame and any cells excluded
   because they would map to a global-boundary coordinate — this keeps the
   inner update loop branch-free so the compiler can vectorize it.
3. The central `B^d` block is copied back to `dst`.

Super-steps ping-pong at the caller level between the two top-level
buffers, so there is no `O(N^d)` copy between super-steps.

## Current status

### Single-threaded CPU (Xeon Gold 6242, Skylake, 22 MB L3)

| Test | N | iters | B | T | baseline | temporal | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D | 2050 | 96 | 128 | 8 | 1.81 Gup/s | 1.19 Gup/s | 0.66× |
| 2D | 4098 | 48 | 256 | 8 | 1.44 Gup/s | 1.28 Gup/s | 0.89× |
| 3D |  258 | 20 |  32 | 4 | 0.97 Gup/s | 0.40 Gup/s | 0.41× |

Single-core does not saturate DRAM, so temporal blocking's reuse win is
masked by halo overhead. Bit-identical to baseline in every configuration.

### OpenMP, thread-scaling sweep (2D, N=4098, B=256, T=8)

| threads | baseline Gup/s | temporal Gup/s | speedup |
|--:|--:|--:|--:|
| 1 | 1.41 | 1.47 | 1.04× |
| 2 | 2.68 | 2.88 | 1.07× |
| 4 | 4.78 | 5.32 | 1.11× |
| 8 | 7.20 | 9.60 | 1.33× |
| 16 | 9.57 | **17.41** | **1.82×** |

Temporal's advantage grows monotonically with thread count: more cores =
more DRAM pressure = bigger win from cutting traffic by ~T. Bit-identical
to the baseline at every thread count.

### GPU (NVIDIA V100-SXM2-16GB)

`sor2d_gpu.cu`, TILE=32, HALO_T=4, INTER=24, iters=96.

| N | GPU base (ms) | GPU temporal (ms) | speedup | GPU base Gup/s | GPU temporal Gup/s |
|--:|--:|--:|--:|--:|--:|
|  258 | 3.30 | 0.12 | 26.9× | 1.9 | 51.3 |
| 1026 | 2.37 | 0.83 |  2.8× | 42.5 | 120.7 |
| 2050 | 6.23 | 3.08 |  2.0× | 64.6 | 130.7 |

Max difference vs. CPU reference is ~3×10⁻⁶ absolute (~3×10⁻⁷ relative) at
all sizes — standard single-precision round-off. The 26.9× at N=258 is
launch-overhead dominated: the baseline issues 96 kernel launches, the
temporal version only 24. At realistic sizes the steady-state GPU speedup
is ~2–3×, and temporal temporal blocking reaches ~130 GUpdates/s on the
V100 — about 2× what the coalesced sweep-per-launch baseline can do.

### Why isn't the CPU version beating the baseline yet?

Temporal blocking only wins when DRAM bandwidth is the bottleneck. Three
forces fight it here:

1. **Single-threaded baseline isn't DRAM-bound.** At N=2050 the two ping-pong
   buffers total 32 MB, so hardware prefetchers still land many accesses in
   L3 even for the plain loop. One core cannot saturate the socket's ~90 GB/s
   DRAM pipe; the baseline runs at ~35 GB/s effective, which looks like
   streaming from the caches, not the DRAM wall that temporal blocking is
   designed to hide.
2. **Halo overhead.** In 2D, `(B+2T)^2 / B^2` at `B=128, T=8` is ~1.27×
   extra loads per tile. In 3D with `B=32, T=4`, it is `(40/32)^3 ≈ 1.95×`
   extra work — so 3D bleeds about half its potential speedup to the halo
   before any reuse benefit shows up.
3. **`memcpy(Bp, A, …)` pre-copy.** It is correct (avoids branching in the
   inner update) but reads and writes the full scratch per sub-step, adding
   ~1× of extra traffic per super-step. At small `T` this erases the
   reuse win outright.

### Where temporal blocking *will* win on this scaffold

The measurements above are a floor, not a ceiling. Expected wins:

- **Multi-thread with OpenMP.** The payoff scales with DRAM pressure. With
  16 cores streaming, a plain SOR saturates the memory subsystem long before
  it saturates FPUs — this is the regime where time skewing famously gives
  2–4× wins. Tile loops are already independent, so an OMP `parallel for
  collapse(2)` around the tile loop is a ~10-line change.
- **GPU shared memory.** Each thread block loads a `(B+2T)^2` tile into
  shared memory, runs `T` in-shared-memory sweeps (using `__syncthreads()`
  between sub-steps, with the same shrinking trapezoid), then writes the
  central `B^2` back. This is the natural counterpart of the Lab 7/8
  shared-memory MMM kernel and should give much larger wins than the
  baseline Lab 7 `sor_sweep_3a` kernel, because DRAM↔L2↔SM traffic per
  sweep drops by ~T.
- **Bigger `T` and smarter carry-through.** Replacing the full scratch
  `memcpy(Bp, A, …)` with an update-region-sized one (only frame + boundary
  exclusion strip), and hoisting the global-boundary check out of the loop
  so bound-adjacent tiles have a fast path, should close the remaining gap
  at `T ≥ 8`.

## Next steps

1. ~~OpenMP pragmas around the tile loop.~~ *Done. 1.82× at 16 threads, N=4098.*
2. ~~GPU shared-memory temporal kernel (2D).~~ *Done. 2–3× over coalesced
   Lab-7-style baseline at realistic N on V100.*
3. **3D GPU kernel (`sor3d_gpu.cu`).** Natural extension of `sor2d_gpu.cu`.
   3D shared memory is tighter — at TILE=16, HALO_T=2 the scratch is
   `16³ × 2 × 4 = 32 KB`, which just fits in a V100 block's 96 KB SM. Needs
   a careful B/T choice.
4. **Red-black SOR variant** to recover `omega ∈ (1, 2)` and meaningful
   over-relaxation. Each temporal sub-step becomes a pair of half-sweeps
   (red-then-black), the trapezoid shrinks by 1 per half-sweep, and the
   tile-level parallelism stays the same.
5. **(B, T) sweep heatmap** at the winning configuration (16-thread CPU,
   V100 GPU) to identify the sweet spot instead of guessing.
6. **Convergence-to-residual study.** Number of super-steps to reach a
   fixed `||r||_2`; the ping-pong form should give bit-identical residual
   curves to the baseline, which is the check that separates "correct fast
   algorithm" from "correct algorithm, different fixed point."
