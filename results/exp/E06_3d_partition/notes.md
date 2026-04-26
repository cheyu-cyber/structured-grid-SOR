# E06 — 3D OMP partitioning study (slab / pencil / cube)

Machine: SCC scc-j06, 2× Xeon Gold 6426Y (32 cores, 38.4 MB L3 / socket).
Thread budget 8.
Config: iters=20 (kept short — 3D is heavier than 2D), B=32 (only used
by `cube`).
Methodology: 4 N values × 6 thread counts × 3 modes × 3 repeats = 216
runs; OMP pinned with `OMP_PROC_BIND=close OMP_PLACES=cores`; table
reports the median CPE.

CPE = `time_s × CPNS × 1e9 / ((N-2)³ × iters)` with CPNS=2.0 (lab
convention).  The binary now prints CPE directly in its CSV row, so
data.csv comes straight from binary output (no derived metric).

## Throughput at 8 threads (median of 3)

| N   | slab CPE     | pencil CPE   | cube CPE  | best winner |
|----:|-------------:|-------------:|----------:|:-----------:|
|  66 |        0.703 |    **0.621** |     1.340 | pencil      |
| 130 |    **0.359** |        0.392 |     0.508 | slab        |
| 194 |        0.465 |        0.465 |     0.762 | tie         |
| 258 |        0.539 |        0.539 |     0.834 | tie         |

Smaller CPE = faster.

**Best 3D CPE in project: 0.359 (slab, N=130, 8 threads).**

## Strong scaling (slab) — CPE_1t / CPE_nt

| threads | N=66  | N=130     | N=194 | N=258 |
|--------:|------:|----------:|------:|------:|
|       1 | 1.00× |     1.00× | 1.00× | 1.00× |
|       2 | 2.23× |     1.94× | 1.69× | 1.67× |
|       4 | 2.45× |     3.63× | 3.25× | 3.12× |
|       8 | 3.69× | **6.37×** | 5.61× | 5.08× |

Slab scales best at **N=130** (6.37× at 8 threads).  At N=258 the
DRAM-bound regime caps scaling at ~5×.

## Observation (the 3D partitioning story)

Three decomposition strategies, three results:

- **Slab** (`#pragma omp parallel for` over outer i): every thread owns
  a contiguous `Ni/nt × N × N` slab.  Largest cache footprint per
  thread (`Ni/nt × N²` cells), simplest dispatch, halo per slab is
  `2 × N²` cells.  Wins or ties everywhere at 8 threads.
- **Pencil** (`collapse(2)` over (i, j)): each thread owns slices of
  jk-pencils.  Halo grows along k only (~`4 N h` cells per pencil
  group).  **Statistically tied with slab** for N ≥ 130; slightly wins
  at small N=66 because pencil's smaller per-thread footprint helps
  load-balance.
- **Cube** (`collapse(3)` over `B³` tiles, B=32): smallest tile, fits
  L1/L2 per thread, but **halo-per-tile is `6 B²` cells** (6 sides) =
  18.75% redundant work.  Plus, with `collapse(3)` the work-distribution
  loop has 64 tiles at N=130 vs only 8 tiles at N=66, so at small N
  cube has very poor load balance (CPE 1.34 at N=66).
  **Cube loses everywhere by 1.3–1.7×.**

The Lab-6-Part-2 finding that **slab is the right default for 3D
stencil sweeps** is reproduced.  Pencil is a fine alternative at small
N; cube only makes sense when the working set is so large that even
slab tiles don't fit any cache — not the case for any N we tested
(N=258 still fits L3 of one socket).

## Surprise

Cube's 1-thread CPE at N=130 (3.343) is *worse* than slab's at 1
thread (2.286) — even though cube has the same 1-thread arithmetic
work.  The difference is the `collapse(3)` over 64 tiles even at 1
thread imposes per-tile boundary copies and pre-copy overhead, while
slab is one big sweep.  Cube isn't just bad for parallelism — it's a
worse single-threaded algorithm too.

## Cross-reference with E05 (2D pthreads)

In 2D (E05) **strip wins**, **interleaved loses to false sharing**,
**block is in the middle**.  In 3D (E06) **slab=pencil tie**, **cube
loses to halo overhead**.  Different dimensions, different winners,
but the underlying lesson is the same: **the cheapest decomposition
that exposes enough parallelism wins**.  Adding decomposition axes
(2D-block, 3D-cube) buys you nothing if the simpler strategy already
saturates the cores.

Plots: `partition_3d_cpe.png` — 4-panel CPE vs threads (one panel per
N), 3 lines per panel (slab / pencil / cube), shaded min/max envelope,
ideal-scaling reference.

## STATUS update

E06: done — **slab and pencil tie at 8 threads** for all N (CPE
~0.36–0.74); **cube loses 1.3–1.7× everywhere** to halo overhead.
Best 3D CPE = **0.359** at slab, N=130, 8 threads.
