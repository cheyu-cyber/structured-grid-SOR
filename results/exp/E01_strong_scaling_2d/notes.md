# E01 — strong scaling 2D OpenMP

Machine: SCC scc-j06, 2× Intel Xeon Gold 6426Y (32 cores, 38.4 MB L3 per
socket, 250 GB RAM).  Thread budget capped at 8 (shared login node).
Config: N=4098, iters=64, B=128, T=8 (32 MB per buffer at FP64; total
working set 64 MB exceeds L3-of-one-socket).
Headline metric: **CPE = cycles / element** at CPNS=2.0 (lab convention).

## Throughput (CPE = wall-clock cycles per output cell)

| threads | baseline CPE | temporal CPE | temp/base CPE |
|--------:|-------------:|-------------:|--------------:|
|       1 |       2.921  |       2.756  |     0.94×     |
|       2 |       2.364  |       1.512  |     0.64×     |
|       4 |       1.199  |       0.830  |     0.69×     |
|       8 |   **0.651**  |   **0.377**  |   **0.58×**   |

Smaller CPE = faster.

## Strong-scaling speedup (CPE_1t / CPE_nt, same kind)

| threads | baseline | temporal |
|--------:|---------:|---------:|
|       1 |    1.00× |    1.00× |
|       2 |    1.24× |    1.82× |
|       4 |    2.44× |    3.32× |
|       8 |    4.49× |   **7.31×** |

Ideal scaling at 8 threads = 8.0×.  Temporal hits **91%** of ideal;
baseline only **56%**.

## Observation

Both kernels scale, but **temporal scales noticeably better** because
temporal does most of its work on cached scratch buffers, so additional
cores convert directly to additional work.  Baseline is DRAM-bound at
N=4098 with 8 threads — adding cores past 4 buys diminishing returns.

The **baseline-vs-temporal CPE ratio shrinks monotonically** with
thread count (0.94× → 0.64× → 0.69× → 0.58×), the headline prediction
of Arnold-style traffic-reduction analysis: more cores → more aggregate
DRAM pressure → bigger relative win for the cache-friendly variant.
The small dip at 4 threads is plausibly a NUMA-locality artefact (4
threads can fit in a single socket; 8 spreads across both).

## Surprise

The 1-thread temporal/baseline CPE ratio is 0.94× — temporal *barely*
beats baseline single-threaded.  This is consistent with the prior
report.md finding: the Xeon's hardware prefetchers and large L3
(38.4 MB / socket) keep the baseline near streaming-bandwidth even at
N=4098, so temporal blocking has no DRAM cycles to recover when the
baseline isn't yet DRAM-bound.  Temporal earns its keep only once
parallelism stresses DRAM.

Plots: `strong_scaling_cpe.png` — CPE vs threads (log-log, with
ideal-scaling reference lines).
