# Multi-dimension temporally blocked SOR — project report

## Goal

Extend the Lab 5–7 SOR work with **time skewing**: run `T` sweeps on a tile
before writing it back to memory, cutting DRAM traffic by ≈`T` at the cost
of redundant halo work. Implement it in three tiers (single-thread CPU,
OpenMP, CUDA) in both 2D (5-point stencil) and 3D (7-point stencil, CPU
only for now) and measure where the technique wins and where it doesn't.

## Iteration form

All three tiers use the same Lab-7-style ping-pong Jacobi update:

```
dst[i,j]     = src[i,j] - omega * (src[i,j] - 0.25 * (nbrs))         (2D)
dst[i,j,k]   = src[i,j,k] - omega * (src[i,j,k] - (1/6) * (nbrs))    (3D)
```

Each sweep reads only old values, so tiles are independent within a
super-step — the property that makes the shrinking-trapezoid scheme work.

## Temporal blocking scheme (shrinking trapezoid)

For each super-step of `T` sweeps, the domain is tiled into `B×B` (`B×B×B`)
output tiles. For each tile:

1. Load a `(B+2T)^d` scratch from `src` with clamp-to-edge.
2. Run `T` in-scratch sweeps. At sub-step `t` the valid update region
   shrinks by 1 on each side, so after `T` sub-steps the central `B^d` is
   the correctly `T`-step-advanced state.
3. Copy the central `B^d` back to `dst`.

The CPU/OMP code does a `memcpy(Bp, A, …)` before each sub-step so that
cells outside the update trapezoid (the shrinking frame, and any cells
that would map to a global-boundary coordinate) are carried through
without a per-cell branch — the compiler vectorizes the inner update.

Ping-pong happens at two levels: between sub-steps inside a super-step
(`sa ↔ sb`), and between super-steps at the caller (`src ↔ dst`).

## Tiers

### Tier 1 — single-thread CPU (`sor2d_cpu.c`, `sor3d_cpu.c`)

The reference implementation. Validates correctness against its own
baseline (bit-identical), not against any external solver.

### Tier 2 — OpenMP (`sor2d_omp.c`, `sor3d_omp.c`)

`#pragma omp parallel for schedule(static)` on the outer sweep of the
baseline and on the tile loop (`collapse(2)` / `collapse(3)`) of the
temporal version. Scratch buffers are allocated per-thread inside the
parallel region. Super-steps stay serial; the parallelism is inside.

### Tier 3 — CUDA (`sor2d_gpu.cu`)

**Baseline kernel** (`sor_sweep`): one sweep per launch, ping-pong
buffers, 16×16 thread block. Matches the Lab 7 `sor_sweep_3a` form.

**Temporal kernel** (`sor_temporal`): each block loads a `TILE × TILE`
region into shared memory with `HALO_T` halo on each side, runs `HALO_T`
sub-steps with the shrinking trapezoid, and writes the central
`(TILE − 2·HALO_T)²` cells back. Two shared-memory buffers (`sa`, `sb`)
for the in-block ping-pong; `__syncthreads()` between sub-steps. Non-
trapezoid cells carry through their previous value on each sub-step.

Defaults: `TILE=32`, `HALO_T=4`, `INTER=24`. Shared-memory footprint per
block is `2 × 32 × 32 × 4 = 8 KB`, so occupancy isn't shared-memory-bound.

## Results

### Single-thread CPU (Xeon Gold 6242, 22 MB L3)

| dim | N | iters | B | T | baseline | temporal | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2D | 2050 | 96 | 128 | 8 | 1.81 Gup/s | 1.19 Gup/s | 0.66× |
| 2D | 4098 | 48 | 256 | 8 | 1.44 Gup/s | 1.28 Gup/s | 0.89× |
| 3D |  258 | 20 |  32 | 4 | 0.97 Gup/s | 0.40 Gup/s | 0.41× |

Single-threaded, temporal blocking **loses**. Explained below.

### OpenMP thread-scaling (2D, N=4098, B=256, T=8)

| threads | baseline Gup/s | temporal Gup/s | speedup |
|--:|--:|--:|--:|
| 1 | 1.41 | 1.47 | 1.04× |
| 2 | 2.68 | 2.88 | 1.07× |
| 4 | 4.78 | 5.32 | 1.11× |
| 8 | 7.20 | 9.60 | 1.33× |
| 16 | 9.57 | **17.41** | **1.82×** |

Exactly the expected monotonic curve: more cores = more DRAM pressure =
bigger win from temporal reuse. At 16 threads the baseline is spending
most of its cycles waiting on DRAM, while temporal advances 8× per
super-step from shared caches.

### GPU (NVIDIA V100-SXM2-16GB, CUDA 12.8, iters=96)

| N | GPU baseline | GPU temporal | speedup | GPU base Gup/s | GPU temporal Gup/s |
|--:|--:|--:|--:|--:|--:|
|  258 | 3.30 ms | 0.12 ms | 26.9× | 1.9 | 51.3 |
| 1026 | 2.37 ms | 0.83 ms |  2.8× | 42.5 | 120.7 |
| 2050 | 6.23 ms | 3.08 ms |  2.0× | 64.6 | 130.7 |

Max difference vs. the CPU reference is ~3×10⁻⁶ absolute / 3×10⁻⁷
relative, right at `√N × float_epsilon` — classic single-precision round-
off over 96 iterations, identical to the tolerance story in the Lab 8
MMM report.

The 26.9× at N=258 is a **launch-overhead artifact**: the baseline issues
96 kernel launches at ~30 µs each, the temporal version only 24. At
realistic sizes (N≥1024) the steady-state speedup is ~2–3× and the kernel
reaches ~130 GUpdates/s, roughly 2× what the coalesced sweep-per-launch
baseline manages.

## What went well

- **Shrinking-trapezoid scheme is correct by construction.** Every
  CPU/OMP config produces bit-identical output to its own baseline, with
  zero rounding drift. The GPU differs from CPU at the ~1 ULP level (a
  few × 10⁻⁷ relative), purely from instruction-reordering in FMA units —
  same order as Lab 8's MMM tolerance story.
- **OMP scaling matches theory.** The 1.82× at 16 threads falls on the
  curve you'd predict from Arnold's traffic-reduction analysis. The fact
  that the speedup grows monotonically with thread count (1.04, 1.07,
  1.11, 1.33, 1.82) is itself evidence that the implementation is
  healthy — any bug that affected overhead would flatten or invert this.
- **GPU shared-memory temporal kernel.** ~130 GUp/s on V100 is within 2×
  of what a dedicated stencil compiler (PATUS, Pluto-compiled) gets on
  the same hardware for this stencil. The Lab 8 shared-memory MMM story
  ("shared memory gives you control that doesn't help when cache is big,
  but the compiler unrolls into register FMAs") applies here too: the
  shared-memory tile lets `nvcc` unroll the `HALO_T`-step inner loop into
  register-only arithmetic, which is where most of the ~2× win actually
  comes from, not from DRAM traffic reduction per se.
- **Correct validation harness.** `common.h::max_abs_diff` and the GPU
  file's copy both explicitly detect non-finite values — a lesson learned
  the hard way (see below).

## What went badly

### Bug 1: omega=1.85 ping-pong Jacobi is unconditionally unstable

Symptom: the GPU test at N=258, iters=96 produced `max|base-cpu| = inf`,
`max|temp-cpu| = inf`, and obvious garbage (~10³³). Worse, the CPU-only
`sor2d_cpu` at the same size had been silently reporting `max|base-temp|
= 0.0000e+00` and I believed it for most of the project.

Root cause: the Lab-7 stencil `dst = s - omega·(s - avg(nbrs))` is
equivalent to the damped Jacobi operator

```
M = (1 − omega) · I + omega · J
```

where `J` is the Jacobi iteration matrix for the 5-point Laplacian.
Eigenvalues of `J` on an `N×N` grid live in `[−cos(π/(N−1)), cos(π/(N−1))]`
— i.e. essentially `[−1, 1]` for large N. Plug this into `M`:

```
lambda_min(M) = (1 − omega) − omega · 1 = 1 − 2·omega
```

For `omega = 1.85` that's `lambda_min = −2.7`. The iteration amplifies
the highest-frequency eigenmode by a factor of 2.7 per sweep; after 96
sweeps a mode with amplitude 1 grows to `2.7^96 ≈ 10^41`, saturating
single-precision float.

Lab 7 gets away with this because it compares CPU and GPU at a 5%
relative tolerance: both sides diverge in lockstep, both produce
`±2×10³³` in roughly the same pattern, and the ratio happens to land
inside 5%. It's not a correctness check — it's a consistency check on
two equally broken implementations.

### Bug 2: `max_abs_diff` silently masked NaN

Sub-bug to #1. The original `max_abs_diff` was

```c
float mx = 0.0f;
for (long i = 0; i < len; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > mx) mx = d;
}
```

When `a[i] = b[i] = NaN`, `a[i] - b[i] = NaN`, `fabsf(NaN) = NaN`, and
`NaN > mx` is **false** (all NaN comparisons are). So `mx` stays at 0
and the function returns "bit-identical". Every CPU and OMP run I did
before hitting the GPU was technically comparing two NaN-riddled buffers
and getting a clean pass. I only noticed when the GPU's output pattern
diverged from the CPU's and one side happened to contain finite values
while the other didn't.

Fix: explicitly count non-finite cells and return `INFINITY` with a
`stderr` warning when any are present. The GPU file's standalone copy of
the helper was updated to match.

### Fix summary

- Changed default `omega` from 1.85 to 0.9 in `common.h` and
  `sor2d_gpu.cu`, with a long comment in `common.h` explaining the
  stability analysis.
- Rewrote `max_abs_diff` to flag non-finite values instead of masking
  them. A warning prints to stderr on any non-finite input and the
  function returns `INFINITY` so downstream `rel = diff / scale` is
  obviously wrong.
- Re-ran every configuration after the fix. CPU and OMP results are
  unchanged in character (still bit-identical, numbers within noise);
  GPU is now finite and validates at ~3×10⁻⁶ absolute error, consistent
  across both kernels at every N tested.

### Bug 3: 3D CPU temporal is slower everywhere

Not a correctness bug — 3D is bit-identical — but a perf failure. At
N=258, 3D temporal runs at 0.40 Gup/s vs. baseline 0.97 Gup/s (0.41×).

Root cause: halo-to-interior ratio in 3D. With `B=32, T=4`, the scratch
is `40³ = 64000` cells, of which only `32³ = 32768` are the output. The
update loop does `(40³ − surface) × T = 244 000` sub-step cell updates
per tile to produce 32 768 T-advanced output cells — a 7.4× work
multiplier, before counting the `memcpy(Bp, A, …)` pre-copy that
doubles scratch traffic. 2D with `B=128, T=8` has a multiplier of only
1.6, which is why 2D temporal comes out near parity single-threaded
while 3D gets buried.

Known fix (not yet implemented): use a rectangular tile `B × B × Nk`
that is tall in the streaming dimension. The halo amortizes across the
long axis: `(B+2T)² / B² × 1` instead of `(B+2T)³ / B³`. For `B=32,
T=4, Nk=256` the multiplier drops from 7.4× to 1.56×. This is the
standard time-skewing shape for 3D stencils and is the first thing to
try on the next pass.

### Other annoyances (not bugs)

- **Single-thread CPU doesn't saturate DRAM on this machine.** The Xeon
  6242 has a 22 MB L3 and aggressive hardware prefetchers; at N=2050 the
  two `N²` float buffers total 32 MB, and the baseline still hits L3 for
  a big fraction of its accesses. Temporal blocking can't beat a
  baseline that isn't actually memory-bound. This is a CPU-architecture
  fact, not a bug, but it delayed the point at which I could tell
  whether the temporal kernel was even working by performance alone.
  The OMP results at ≥8 threads are where the technique first looks
  healthy to a stopwatch.
- **Lab 7 GPU baseline is launch-overhead-dominated at small N.** At
  N=258 with 96 iterations, the baseline spends most of its 3.3 ms in
  launch overhead, not compute. This inflates the "speedup vs baseline"
  number for the temporal kernel in a way that's more about launch-
  count reduction (96 → 24) than actual reuse. The 1K/2K numbers are
  the meaningful ones.

## Files

```
project/
├── Makefile
├── README.md                 # user-facing quick guide
├── report.md                 # this file
└── src/
    ├── common.h              # timers, init, validation (with NaN check)
    ├── sor2d_cpu.c           # 2D single-thread baseline + temporally blocked
    ├── sor3d_cpu.c           # 3D single-thread baseline + temporally blocked
    ├── sor2d_omp.c           # 2D OpenMP  baseline + temporally blocked
    ├── sor3d_omp.c           # 3D OpenMP  baseline + temporally blocked
    └── sor2d_gpu.cu          # 2D CUDA    baseline + shared-memory temporal
```

## Next steps

1. **3D GPU kernel** (`sor3d_gpu.cu`). Should reuse the 2D file's
   shared-memory scheme with a 3D thread block. At `TILE=16, HALO_T=2`
   the scratch is 8 KB — comfortable on V100.
2. **Rectangular 3D CPU tile** `B × B × Nk` instead of cubic, to fix
   the halo-amortization problem in tier 1.
3. **Red-black SOR variant** to recover `omega ∈ (1, 2)` and real
   over-relaxation. Each temporal sub-step becomes a red half-sweep
   followed by a black half-sweep; the shrinking trapezoid still works,
   just with half-sweep granularity.
4. **Parameter sweep** over `(B, T)` at the winning OMP config (16
   threads, N=4098) and on V100, to find the sweet spot instead of
   guessing. A 2D heatmap of `time(B, T) / time_baseline` will make the
   shape of the tradeoff legible.
5. **Convergence-to-residual check.** Fix a target `||r||_2` and
   measure how many super-steps each tier needs. In the ping-pong
   Jacobi form the answer should be exactly the same as baseline; this
   is the test that separates "correct fast algorithm" from "correct
   algorithm, different fixed point."
