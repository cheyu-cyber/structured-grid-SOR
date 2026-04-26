# E09 — GPU TILE / HALO sweep

Machine: SCC scc-j06, NVIDIA L40S (46 GB, sm_89), CUDA 12.8.
Config: N=2048, iters=96 (chosen so HALO ∈ {1,2,3,4,6,8,12} all divide).
Constraint: `2*HALO < TILE`.  Binary: `sor2d_gpu --tile <T> --halo <H>`.
Methodology: 21 (TILE, HALO) combos × 3 reps = 63 runs.  Headline metric
is **CPE** (cycles/element at CPNS=2.0, lab convention).  Per-kernel CPE
captures both DRAM efficiency and per-element compute cost on the GPU.

## Best configurations

| TILE | HALO | INTER | temp CPE | base CPE | speedup vs base |
|----:|----:|----:|---------:|---------:|----------------:|
|  24 |   4 |  16 | **0.0088** |   0.0137 |       **1.56×** |
|  32 |   4 |  24 |   0.0091 |   0.0137 |           1.51× |
|  16 |   2 |  12 |   0.0094 |   0.0136 |           1.45× |
|  24 |   2 |  20 |   0.0099 |   0.0142 |           1.43× |

Smaller CPE = faster.  **Best GPU CPE in the project: 0.0088
(TILE=24, HALO=4, INTER=16).**

## The (TILE, HALO) story

Two competing forces, exactly mirroring E04's CPU story but at a
different scale:

1. **DRAM-traffic savings ∝ HALO**.  Each super-step does HALO sweeps
   on shared-memory data; one DRAM pass replaces HALO passes.
2. **Halo overhead grows like (HALO/INTER)²**.  The shared-memory tile
   computes (TILE)² cells per block but only (INTER)² of them are
   committed; the rest is redundant work.

Plus two GPU-specific terms:

3. **Thread-block utilization**.  TILE² threads/block.  TILE=8 → 64
   threads/block (under-utilized warps); TILE=16 → 256; TILE=24 → 576;
   TILE=32 → 1024 (max).  Bigger blocks hide latency better but pay
   higher launch overhead.
4. **Shared-memory footprint** = `2 × TILE² × 4 B` (two FP32 ping-pong
   buffers).  At TILE=32 this is 8 KB / block — well under L40S's
   100 KB / SM, so occupancy isn't shared-mem-bound at any combo we
   tested.

## Per-TILE behavior

- **TILE=8** is *bad* everywhere (CPE 0.030–0.089).  Only 64 threads
  per block — 2 warps — too few to hide DRAM latency on the L40S.
  HALO=3 (INTER=2) is especially terrible (0.089) because the halo
  swallows almost the entire tile.
- **TILE=16** has its minimum at HALO=2 (CPE 0.0094, INTER=12).  HALO=4
  is statistically tied (0.0113); HALO=6 (INTER=4) collapses to 0.0412
  for the same reason.
- **TILE=24** has its minimum at **HALO=4** (CPE 0.0088, INTER=16) —
  the project's overall GPU best.  HALO=2 and HALO=3 are both ~0.010;
  HALO=8 (INTER=8) jumps to 0.025.
- **TILE=32** has its minimum at HALO=4 (CPE 0.0091, INTER=24).  All
  HALO ∈ {2,3,4,6} are within 25% of each other; HALO=12 (INTER=8)
  collapses to 0.0497.

## Pattern: the optimal INTER:HALO ratio is around 4-6

| best per-TILE config | INTER : HALO ratio | CPE     |
|:--------------------|------------------:|--------:|
| TILE=16, HALO=2     | 6.0 : 1           | 0.0094  |
| TILE=24, HALO=4     | 4.0 : 1           | 0.0088  |
| TILE=32, HALO=4     | 6.0 : 1           | 0.0091  |

Empirical lesson: **keep HALO ≤ INTER/4**.  Above that, redundant work
catches up to DRAM savings.  This is the same `(B+2T)²/B²` bound that
governed CPU temporal blocking in E03/E04 — translated into TILE/HALO
language for the GPU.

## Surprise

**TILE=24/HALO=4 wins at L40S** (ours), but Lab-7-era literature and
report.md's V100 results recommend TILE=32/HALO=4.  The two are within
3% on this L40S; given thread-block grain quantization (1024-thread
limit, warp count rounding), TILE=24 lands in a sweet spot where 576
threads = 18 warps fits cleanly into the L40S's 64 warp-slots-per-SM.
At TILE=32 we use 1024 threads = 32 warps and back off occupancy a
bit.  Either way the practical default is **TILE ∈ {24, 32}, HALO=4**.

## Cross-reference with E04 (CPU temporal depth)

CPU side (E04): 8 threads, B=128, best **T=8** (= HALO).
GPU side (E09): one device, TILE=24, best **HALO=4**.

Why the difference?  On the CPU the per-thread scratch is 144 KB
(fits L2 = 1.5 MB) and the loop is unrolled into vector FMAs; halo
overhead ((128+16)²/128² = 27%) is small relative to L1-L3 reuse, so
T=8 is fine.  On the GPU the per-block shared memory (2 × 24² × 4 =
4.5 KB) is tiny relative to compute, so even small extra HALO is
expensive — best HALO is half of best T.

Plots:
- `tile_halo_heatmap.png` — TILE × HALO heatmap, viridis_r colormap
  (dark = best CPE), with numerical CPE overlaid in each cell.
- `tile_halo_lines.png` — CPE vs HALO, one line per TILE, log-y, with
  baseline-kernel CPE as a horizontal reference and the global minimum
  starred.

## STATUS update

E09: done — best GPU CPE **0.0088** at **(TILE=24, HALO=4)**, **1.56×
faster than the per-launch baseline kernel** (CPE 0.0137).  TILE=32/4
ties statistically at 0.0091.  TILE=8 underutilizes warps everywhere;
HALO too large (INTER ≤ 8) collapses on every TILE.
