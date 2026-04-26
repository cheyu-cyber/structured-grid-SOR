# E02 — size sweep, 2D OMP at 8 threads

Machine: SCC scc-j06, 2× Xeon Gold 6426Y (32 cores total, 38.4 MB L3
per socket, 250 GB RAM).  Thread budget 8.
Config: iters=64, B=128, T=8, threads=8.  Pinned with
`OMP_PROC_BIND=close OMP_PLACES=cores`.
Methodology: 3 repeats per N, table reports the median CPE.
`data_t = double` (8 B), so each buffer is `N² × 8`; "bytes_pair_mb" is
both buffers.

## Throughput (median of 3, CPE = cycles/element at CPNS=2.0)

| N    | bytes/pair | baseline CPE | temporal CPE | temp/base CPE |
|-----:|-----------:|-------------:|-------------:|--------------:|
|  256 |     1.0 MB |        0.678 |        1.356 |        2.00× |
|  512 |     4.0 MB |        0.396 |        0.565 |        1.43× |
| 1024 |    16.0 MB |        0.464 |    **0.386** |        0.83× |
| 2048 |    64.0 MB |        0.667 |    **0.346** |    **0.52×** |
| 4098 |   256.3 MB |        0.631 |    **0.387** |        0.61× |

Smaller CPE = faster.  Below 1 means temporal wins.

## Crossover analysis

L3 per socket = 38.4 MB.  Working-set thresholds:
- Fits in **L1/L2 per thread**: N ≤ ~256.
- Fits in **L3 of one socket** (≤ 38.4 MB): N ≤ ~2200.
- **Exceeds L3** of one socket: N ≥ ~2200.

Observed:
- **N=256, N=512**: temporal *loses* (CPE 1.36, 0.57 vs baseline 0.68,
  0.40).  Working set fits L2/L3, baseline runs at cache bandwidth, and
  the temporal kernel pays its overhead (per-tile scratch malloc,
  redundant halo work, pre-copy memcpy) without DRAM cycles to recover.
- **N=1024 (16 MB)**: temporal pulls ahead (CPE 0.386 vs 0.464, 17%
  better).  Working set fits L3 of one socket but not L2; baseline
  starts paying L3 bandwidth.
- **N=2048 (64 MB)**: baseline CPE jumps from 0.464 → 0.667 (43%
  worse), the classic L3→DRAM crossover.  Temporal stays at CPE 0.346
  → **temporal beats baseline by 1.93×** (1/0.52).  This is THE
  headline data point for temporal blocking on this machine.
- **N=4098 (256 MB)**: temporal still wins 1.63× (CPE 0.387 vs 0.631).
  The ratio is slightly smaller than at N=2048 because at very large N
  the temporal kernel also starts paying L3 misses on its scratch
  reuse across super-steps.

The temporal CPE is roughly flat at ~0.35–0.39 across N=1024–4098 —
that's the "compute-bound on L2-resident scratch" regime.  Baseline CPE
rises as N grows past L3.

## Observation

Temporal blocking only earns its keep once the working set exceeds L3.
Below L3 the baseline runs at L3 bandwidth and the temporal kernel
loses to its own overhead.  At N=2048–4098 the baseline transitions to
DRAM-bound and temporal — which advances 8 sweeps per super-step on
cached scratch — wins ~1.6–1.9×.

## Surprise

Baseline CPE is *minimum* at N=512 (0.40), not at the smallest N=256
(0.68).  Likely the small-N runs spend a meaningful fraction of their
time on the boundary-pass-through copy and on the 64-iter loop
overhead; at N=512 the inner loop is long enough to amortise that and
short enough to fit in L2.  Once N ≥ 1024 the L3 hit-rate dominates.

Plots: `size_sweep_cpe.png` — CPE vs N (log x), shaded L3-crossover band.

## STATUS update

E02: done — temporal beats baseline by **1.93× CPE** at N=2048 (L3→DRAM
crossover); 1.63× at N=4098.
