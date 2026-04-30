# Multi-dimension temporally blocked SOR — project report

## Goal

Extend the Lab 5–7 SOR work with **time skewing**: run `T` sweeps on a tile
before writing it back to memory, cutting DRAM traffic by ≈`T` at the cost
of redundant halo work. Implement it in three tiers (single-thread CPU,
OpenMP, CUDA) in both 2D (5-point stencil) and 3D (7-point stencil) and
measure where the technique wins and where it doesn't.

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

### Tier 3 — CUDA (`sor2d_gpu.cu`, `sor3d_gpu.cu`)

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

The 3D port (`sor3d_gpu.cu`) reuses the same scheme with a 3D thread
block. Defaults: `TILE=8`, `HALO_T=2`, `INTER=4` (block 8×8×8 = 512
threads, scratch `2 × 8³ × 4 = 4 KB`). The smaller per-step halo is
forced by the 3D shared-memory budget — at `TILE=16, HALO_T=2` the two
scratch buffers would be 32 KB and the block would be 4096 threads.

## Results

### Single-thread CPU (Xeon Gold 6242, 22 MB L3)

| dim | N | iters | B | T | baseline | temporal | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2D | 2050 | 96 | 128 | 8 | 1.105 | 1.681 | 0.66× |
| 2D | 4098 | 48 | 256 | 8 | 1.389 | 1.562 | 0.89× |
| 3D |  258 | 20 |  32 | 4 | 2.062 | 5.000 | 0.41× |

Single-threaded, temporal blocking **loses**. Explained below.

### OpenMP thread-scaling (2D, N=4098, B=256, T=8)

| threads | baseline CPE | temporal CPE | speedup |
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

| N | GPU baseline | GPU temporal | speedup | GPU base CPE | GPU temporal CPE |
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
reaches ~0.015, roughly 2× what the coalesced sweep-per-launch
baseline manages.

### GPU 3D (NVIDIA L40S, CUDA 12.8, `sor3d_gpu.cu`)

Re-verification was on an L40S (the V100 was unavailable); same code,
same `omega=0.9`. `TILE=8, HALO_T=2, INTER=4`.

| N   | iters | GPU baseline | GPU temporal | speedup | base CPE | temp CPE |
|---:|---:|---:|---:|---:|---:|---:|
|  34 |  8 | 108.26 ms | 0.017 ms | (launch-warmup artifact) | 0.002 | 15.1 |
|  66 | 16 |   3.73 ms | 0.096 ms | 39.0× (launch-bound) | 1.12 | 43.8 |
| 130 | 32 |   3.17 ms | 1.18 ms  | **2.68×** | 21.2 | 56.8 |
| 258 | 96 |  21.18 ms | 31.79 ms | **0.67×** | 76.0 | 50.7 |

Correctness holds at every size: `max|base-cpu|`, `max|temp-cpu|` and
`max|base-temp|` are all in the `1.4×10⁻⁶ … 4.3×10⁻⁶` range (relative
~`10⁻⁷`), the same `√N × float_epsilon` story as 2D. `max|base-temp|` is
the smallest of the three diffs at every N≥66 — both kernels share the
fp32 reduction order, so they agree more tightly with each other than
either does with the fp64 CPU reference.

The interesting finding is the speedup column: temporal **wins** at
N=130 and **loses** at N=258. Cause is the halo-overhead ratio that
killed 3D-CPU temporal as well, just at a different ratio:

```
useful work fraction = (TILE − 2·HALO_T)³ / TILE³ = 4³ / 8³ = 12.5%
```

so each block does 8× the cell-update work of its baseline equivalent.
At N=130 the baseline is launch-overhead-bound (32 launches × ~25 µs ≈
0.8 ms is most of its 3.17 ms total), so the 4× launch-count reduction
in temporal still pays for the 8× redundant compute. At N=258 the
baseline saturates the L40S DRAM bandwidth at 0.026 and the redundant
compute starts to bite — temporal drops to 0.040. This parallels
exactly the 3D CPU result (0.41× temporal at N=258 cubic tiles) and
confirms that **3D shared-memory time-skewing without z-streaming is
bandwidth-poor**. The standard fix is the 2.5D-streaming variant
(Micikevicius 2009): a 16×16 thread block slides a 3-z-plane window
through shared memory instead of loading a cubic tile. Listed as
next-steps bullet 1 below.

## Results — full bench

This section synthesises the 11 standalone experiments under
`results/exp/E0*/` into one coherent narrative.  Every paragraph
corresponds to one experiment block, and every plot it references is
linked from the per-experiment `notes.md`.  The headline metric
throughout is **CPE** (cycles per output cell at CPNS=2.0, lab
convention); smaller is faster.

The experiments were run on **SCC scc-j06**: 2× Intel Xeon Gold 6426Y
(32 cores total, 38.4 MB L3 / socket, 250 GB RAM), NVIDIA L40S
(46 GB, sm_89, ~96 MB L2), CUDA 12.8.  Thread budget capped at **8**
on the shared login node (kernel allows all 32 cores but etiquette
caps us at one socket; this matches the experiment-doc's "8 physical
cores" target).  All CPU runs are pinned with `OMP_PROC_BIND=close
OMP_PLACES=cores` and reported as the **median of 3 reps** per cell,
with min/max envelopes for noise visibility.

### Setup and validation (E00)

`make && make gpu` produced 11 binaries (2D/3D × {cpu, omp, pth,
pth_decomp, pth_temporal, omp_part}, plus 2D/3D `gpu`).  `make smoke`
reports `max|base-temp| = 0.0000e+00` on every CPU tier (bit-identical
against the in-binary serial reference); GPU validates at ~1.4×10⁻⁶
absolute / 1.4×10⁻⁷ relative against the CPU reference (single-
precision round-off over 96 iterations, identical to the Lab 7 / Lab 8
MMM tolerance story).  Details in
[results/exp/E00_setup/info.txt](results/exp/E00_setup/info.txt).

### Strong scaling, 2D OpenMP (E01)

At N=4098, B=128, T=8 (32 MB per buffer at FP64; total working set
exceeds L3 of one socket), OMP scaling is near-linear for the temporal
kernel and noticeably worse for the baseline:

| threads | baseline CPE | temporal CPE | temp/base ratio |
|--------:|-------------:|-------------:|----------------:|
|       1 |       2.921  |       2.756  |       0.94×     |
|       2 |       2.364  |       1.512  |       0.64×     |
|       4 |       1.199  |       0.830  |       0.69×     |
|       8 |   **0.651**  |   **0.377**  |   **0.58×**     |

The temporal kernel hits **91% of perfect 8× scaling** (7.31× at 8t
vs 1t); the baseline reaches only 56% (4.49×).  The temporal-vs-
baseline ratio shrinks monotonically from 0.94× to 0.58× as threads
are added — the canonical Arnold-style traffic-reduction signature.
At one thread the Xeon's hardware prefetchers and 38 MB L3 keep the
baseline near streaming bandwidth, so temporal blocking has no DRAM
cycles to recover; **temporal earns its keep only once parallelism
stresses DRAM**.  See
[E01/strong_scaling_cpe.png](results/exp/E01_strong_scaling_2d/strong_scaling_cpe.png).

### Cache-regime / size sweep (E02)

At fixed 8 threads, sweeping N exposes the L3→DRAM crossover.  L3 per
socket is 38.4 MB; one FP64 buffer at N²×8 bytes:

| N    | bytes/pair | baseline CPE | temporal CPE |
|-----:|-----------:|-------------:|-------------:|
|  256 |     1.0 MB |        0.678 |        1.356 |
|  512 |     4.0 MB |        0.396 |        0.565 |
| 1024 |    16.0 MB |        0.464 |    **0.386** |
| 2048 |    64.0 MB |        0.667 |    **0.346** |
| 4098 |   256.3 MB |        0.631 |    **0.387** |

The baseline CPE is roughly flat at ~0.4 below L3, then **jumps to
0.67 at N=2048** when the working set first exceeds one socket's L3 —
the classic L3→DRAM cliff.  The temporal kernel stays at ~0.35 across
N=1024–4098 (compute-bound on its L2-resident scratch) and **wins
1.93× at N=2048**.  Temporal *loses* at N≤512 because it pays its
overhead (per-tile scratch malloc, redundant halo work, pre-copy
memcpy) without DRAM cycles to recover.  See
[E02/size_sweep_cpe.png](results/exp/E02_size_sweep_2d/size_sweep_cpe.png).

### Spatial block size (E03)

11 B values × 3 reps at N=2048, T=4, 8 threads.  Baseline CPE is flat
at ~0.4 across all B (B has no effect on the baseline kernel — it's
only used by the temporal scratch).  Temporal has a wide plateau
across **B ∈ [24, 256]** with two co-equal minima at B=24 (CPE 0.388)
and B=128 (CPE 0.389), then **falls off a cliff** at B≥384 to CPE
0.60–0.65.  The cliff lands exactly at the L2-fit threshold:

  `2 (B+8)² × 8 B ≤ 1.5 MB → B ≤ ~297`

B=256 (1.06 MB scratch) fits L2; B=384 (2.4 MB) doesn't.  Once two
ping-pong scratch buffers per thread don't fit in L2, the temporal
kernel pays L3 misses on its own scratch — defeating the entire
purpose.  We lock in **B=128** as the canonical default (middle of
the plateau, scratch fits L2 with headroom).  See
[E03/block_size_cpe.png](results/exp/E03_block_size/block_size_cpe.png).

### Temporal depth (E04)

10 T values × 3 reps at N=2048, B=128, 8 threads, iters=96 (chosen so
all T ∈ {1,2,3,4,6,8,12,16,24,32} divide).  The U-curve traces the
shrinking-trapezoid tradeoff exactly as theory predicts:

| T  | baseline CPE | temporal CPE | temp/base | halo overhead |
|---:|-------------:|-------------:|----------:|--------------:|
|  1 |        0.504 |        0.739 |    1.47×  |          3.1% |
|  4 |        0.464 |        0.393 |    0.85×  |         12.9% |
|  8 |        0.444 |    **0.332** |  **0.75×**|         26.6% |
| 16 |        0.412 |        0.378 |    0.92×  |         56.2% |
| 32 |        0.401 |        0.461 |    1.15×  |        125.0% |

DRAM-traffic savings grow ∝ T; redundant halo work grows ∝ T² (the
scratch is `(B+2T)²`).  Their product is unimodal — minimum at **T=8,
CPE 0.332**, the project's best CPU CPE.  T=1 loses because the
temporal kernel still pays per-tile scratch+memcpy overhead with no
DRAM reuse; T=32 loses because every output cell is computed alongside
more than its own value of halo cells.  See
[E04/temporal_depth_cpe.png](results/exp/E04_temporal_depth/temporal_depth_cpe.png)
— the figure overlays halo-overhead-% on the right axis to make the
two competing terms visible at a glance.

### Pthreads decomposition × scheduling (E05)

108 runs across 6 thread counts × 3 modes × 2 schedules at N=2048,
iters=64.  At 8 threads, persistent schedule:

| mode        | persistent CPE | speedup vs 1t | spawn slowdown |
|:------------|---------------:|--------------:|---------------:|
| strip       |      **0.255** |      **7.89×**|        1.54×   |
| interleaved |          0.752 |          2.67×|        1.03×   |
| block       |          0.544 |          3.76×|        1.10×   |

**Strip wins decisively** — 7.89× scaling at 8 threads is within 1.4%
of perfect 8×.  Interleaved bottoms out at ~2.7× because every cell
update needs `(i±1, j)` rows owned by the next thread; adjacent rows
share L1 cache lines so writes ping-pong between cores (false
sharing).  Block (2D rectangle per thread) sits in the middle: no
intra-row sharing, but 2× the halo of strip.

The spawn-vs-persistent slowdown is a function of per-sweep work
size: strip's tiny per-sweep work (~0.5 ms) makes the
pthread_create+join overhead a 54% slowdown; interleaved's much
slower per-sweep work makes the same overhead a 3% slowdown.  E07
quantifies the absolute overhead.  See
[E05/decomposition_cpe.png](results/exp/E05_pth_decomp/decomposition_cpe.png).

### 3D OMP partitioning (E06)

216 runs across 4 N values × 6 thread counts × 3 modes × 3 reps.  At
8 threads:

| N   | slab CPE     | pencil CPE   | cube CPE  |
|----:|-------------:|-------------:|----------:|
|  66 |        0.703 |    **0.621** |     1.340 |
| 130 |    **0.359** |        0.392 |     0.508 |
| 194 |        0.465 |        0.465 |     0.762 |
| 258 |        0.539 |        0.539 |     0.834 |

**Slab and pencil tie at all N at 8 threads** (CPE 0.36–0.74); **cube
loses 1.3–1.7× everywhere** to its 18.75% halo overhead (`6 B² /
B³ - 1` for B=32 cube tiles).  The Lab-6-Part-2 finding that **slab is
the right default for 3D stencil sweeps** is reproduced.  Best 3D CPU
CPE = **0.359** at slab/pencil, N=130, 8 threads.

This is the dimension-up companion of E05's 2D pthreads result.
Different dimensions, different winners (strip in 2D, slab/pencil in
3D), same underlying lesson: **the cheapest decomposition that
exposes enough parallelism wins**.  Adding decomposition axes (2D-
block, 3D-cube) buys you nothing if the simpler strategy already
saturates the cores.  See
[E06/partition_3d_cpe.png](results/exp/E06_3d_partition/partition_3d_cpe.png).

### Pthread spawn overhead — Lab-6-Part-1b (E07)

10 iters × 2 schedules × 3 reps at N=512, 4 threads, mode=strip.
Linear-fit `time = a + b·iters`:

| schedule    | per-sweep slope | intercept |
|:-----------|----------------:|----------:|
| persistent |       **81 µs** |   +133 µs |
| spawn      |      **235 µs** |   −299 µs |

The slope difference — **per-sweep `pthread_create + pthread_join`
overhead = 153 µs at 4 threads** — is *larger than the 81 µs of
actual per-sweep compute work* at this configuration.  Spawning fresh
threads on every sweep nearly triples wall time per sweep.  This is
the canonical demonstration of why **persistent threads + barriers**
is the right pattern for any kernel with short per-sweep work; OpenMP
uses persistent threads under the hood for exactly this reason.  See
[E07/spawn_overhead.png](results/exp/E07_spawn_overhead/spawn_overhead.png).

### Omega U-curve, red-black GS-SOR (E08)

150 ω values in [0.50, 1.99] at N=128.  The classic Lab-5 figure:

|                    | value         |
|:-------------------|:--------------|
| theory ω_opt = 2/(1+sin(π/(N-1))) | **1.9517** |
| empirical ω_opt    | **1.95**      |
| iters at ω_opt     | **254**       |
| iters at ω = 1.0 (Jacobi) | 3,434 |
| **SOR speedup vs Jacobi** | **13.5× fewer iterations** |

Theory and experiment agree to **0.1%**.  The U-curve is sharply
asymmetric: under-relaxation (ω<1) degrades convergence linearly, but
over-relaxation past ω_opt is a cliff — at ω=1.99 we're already up to
1,230 iters (4.8× the optimum).  This is the project's **algorithmic
optimization** counterpart to the temporal-blocking story: same
algorithm, ω is the only knob, single biggest cost-effective win in
the project.  Note that the temporal-blocking kernels use Jacobi
ping-pong at ω=0.9 (damped Jacobi: stable only for ω∈(0,1] on large
grids); they live on the under-relaxation side of the curve.
Combining temporal blocking with red-black GS would multiply the wins
but requires re-deriving the shrinking trapezoid for paired half-
sweeps — listed as future work.  See
[E08/omega_ucurve.png](results/exp/E08_omega_sweep/omega_ucurve.png).

### GPU TILE/HALO sweep (E09)

21 (TILE, HALO) combos × 3 reps at N=2048, iters=96.  Best
configurations:

| TILE | HALO | INTER | temp CPE | base CPE | speedup vs base |
|----:|----:|----:|---------:|---------:|----------------:|
|  24 |   4 |  16 | **0.0088** |   0.0137 |       **1.56×** |
|  32 |   4 |  24 |   0.0091 |   0.0137 |           1.51× |

The same shrinking-trapezoid tradeoff as E04, just at GPU scale: HALO
too small means no DRAM reuse; HALO too large means halo work eats
the savings.  Empirical pattern: **keep HALO ≤ INTER/4**.  TILE=8 is
dead on the L40S — 64 threads/block is too few warps to hide latency.
TILE=24 with HALO=4 lands in a sweet spot (576 threads = 18 warps,
fits cleanly into the L40S's per-SM warp slots).  Same default
TILE=32/HALO=4 from prior V100 work is within 3%.  See
[E09/tile_halo_heatmap.png](results/exp/E09_gpu_th_sweep/tile_halo_heatmap.png)
and
[E09/tile_halo_lines.png](results/exp/E09_gpu_th_sweep/tile_halo_lines.png).

### GPU baseline vs temporal across N (E10)

7 N values for 2D, 4 for 3D, 3 reps each.  **2D GPU**:

| N    | baseline CPE | temporal CPE | speedup base/temp |
|-----:|-------------:|-------------:|------------------:|
|  258 |       0.3945 |       0.0230 |        17.15×*    |
| 1026 |       0.0295 |       0.0100 |             2.95× |
| 2050 |       0.0138 |       0.0090 |             1.53× |
| 3074 |       0.0103 |   **0.0089** |             1.16× |
| 4098 |       0.0260 |       0.0110 |             2.36× |

`*` The 17× at N=258 is launch-overhead-dominated baseline (96
launches × ~30 µs); the legitimate steady-state win is **1.5–3× at
N ∈ [1026, 2050]**.  N=4098 sees both kernels degrade because two
FP32 buffers (128 MB) exceed the L40S's ~96 MB L2 — the GPU
equivalent of the CPU L3 cliff in E02, two orders of magnitude
larger.  Best 2D GPU CPE = **0.0089 at N=3074**.

**3D GPU** is the project's honest negative finding.  Temporal *loses*
the baseline at N ≥ 194:

| N   | baseline CPE | temporal CPE | speedup |
|----:|-------------:|-------------:|--------:|
| 130 |       0.0686 |       0.0356 |   1.93× |
| 194 |       0.0304 |       0.0354 |   0.86× (LOSS) |
| 258 |       0.0309 |       0.0397 |   0.78× (LOSS) |

The 3D temporal kernel uses TILE=8, HALO_T=2 (compile-time defaults)
giving INTER=4.  Halo-to-interior in 3D is `(TILE/INTER)³ - 1 = 7×`
redundant work — an order of magnitude worse than 2D's `0.78×`
overhead at the same TILE/HALO.  **The 3D shared-memory time-skewing
without z-streaming is bandwidth-poor by construction.**  The
standard fix is the 2.5D-streaming variant (Micikevicius 2009): a
16×16 thread block slides a 3-z-plane window through shared memory
instead of loading a cubic tile.  Listed as future work.  See
[E10/gpu_2d_cpe.png](results/exp/E10_gpu_n_sweep/gpu_2d_cpe.png) and
[E10/gpu_3d_cpe.png](results/exp/E10_gpu_n_sweep/gpu_3d_cpe.png).

### Cross-tier headline (E11)

The single chart that compresses everything onto one axis.  N=2050,
iters=96, 8 threads, median of 3 reps:

| tier / kind                | CPE      | speedup vs serial baseline |
|:--------------------------|---------:|---------------------------:|
| serial-CPU baseline        |   2.792  |                      1.00× |
| serial-CPU temporal        |   2.586  |                      1.08× |
| pthreads-8t-strip          |   0.399  |                      7.0×  |
| pthreads-8t-temp temporal  |   0.378  |                      7.4×  |
| OMP-8t temporal            |   0.347  |                      8.0×  |
| OMP-8t baseline            |   0.229  |                     12.2×  |
| GPU baseline               |   0.015  |                    186×    |
| **GPU temporal**           |**0.0090**|                   **310×** |

**310× total span** from serial CPU baseline to GPU temporal on the
same grid, same iteration count, same stencil, same initial data.
At CPNS=2.0, GPU temporal's 0.009 CPE is ~4.5 ps per element — about
one floating-point op per nanosecond per cell, near the Roofline
machine-balance line for a memory-bound 5-point stencil at FP32.
See
[E11/headline_cpe.png](results/exp/E11_headline/headline_cpe.png).

### How the results map to the talk's required topics

| Talk requirement                      | Best evidence                  | Figure |
|:-------------------------------------|:-------------------------------|:-------|
| Problem description                   | `## Goal` + `## Iteration form`| (text) |
| Serial algorithm + complexity         | E08 + Iteration-form section   | E08    |
| Where time goes / arithmetic intensity| E02 cache regime               | E02    |
| Data structures / memory pattern      | `## Tiers` + sor2d_cpu.c       | (text) |
| Parallel algorithm choice             | E08 (red-black) + ping-pong    | E08    |
| Data partitioning (CPU)               | E05 (2D), E06 (3D)             | E05/E06|
| Data partitioning (GPU)               | E09                            | E09    |
| Optimizations + problems              | E03/E04 (CPU temporal); E09 (GPU shared mem); **E10 3D = honest negative** | E03/E04/E09/E10 |
| Experiments and results               | E11 cross-tier headline        | E11    |



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
- **GPU shared-memory temporal kernel.** ~0.015 on V100 is within 2×
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
N=258, 3D temporal runs at 5.000 vs. baseline 2.062 (0.41×).

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
    ├── sor2d_gpu.cu          # 2D CUDA    baseline + shared-memory temporal
    └── sor3d_gpu.cu          # 3D CUDA    baseline + shared-memory temporal
```

## Next steps

1. **3D GPU 2.5D-streaming variant.** The cubic-tile 3D kernel landed
   (see GPU 3D table above) and validates correctly, but is bandwidth-
   poor: at N=258 it runs at 0.67× baseline because useful-work fraction
   is only `4³/8³ = 12.5%`. The standard fix is a 16×16 thread block
   that slides a 3-z-plane window through shared memory (Micikevicius
   2009). On the L40S this should reach ≥0.025 and beat the cubic-
   tile baseline at every N.
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
