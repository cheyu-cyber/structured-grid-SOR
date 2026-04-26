# E05 — pthreads decomposition × scheduling (Lab-5-Part-4 redo)

Machine: SCC scc-j06, 2× Xeon Gold 6426Y (32 cores, 38.4 MB L3 / socket).
Thread budget 8.
Config: N=2048, iters=64.  Binary: `sor2d_pth_decomp`.
Methodology: 6 thread counts × 3 modes × 2 scheds × 3 reps = 108 runs.
Headline metric: **CPE = cycles/element** at CPNS=2.0.

## Throughput at 8 threads (median of 3)

| mode        | persistent CPE | spawn CPE | spawn / persistent |
|:------------|---------------:|----------:|-------------------:|
| strip       |      **0.255** |     0.393 |          **1.54×** |
| interleaved |          0.752 |     0.776 |              1.03× |
| block       |          0.544 |     0.596 |              1.10× |

Smaller CPE = faster.  `spawn / persistent` is the per-sweep
`pthread_create + pthread_join` overhead expressed as a slowdown factor.

## Strong scaling (persistent), CPE_1t / CPE_nt

| threads | strip   | interleaved | block   |
|--------:|--------:|------------:|--------:|
|       1 |   1.00× |       1.00× |   1.00× |
|       2 |   1.96× |       1.18× |   1.48× |
|       3 |   2.91× |       1.51× |   2.11× |
|       4 |   4.09× |       1.85× |   2.76× |
|       6 |   5.80× |       2.44× |   3.69× |
|       8 |  **7.89×** |    2.67× |   3.76× |

**Strip-persistent at 8 threads scales 7.89× — within 1.4% of perfect 8×
scaling.**  Interleaved bottoms out at ~2.7×; block at ~3.8×.

## Observation (the partitioning story)

Three decomposition strategies, three different scaling stories:

- **Strip** (each thread owns a contiguous block of `(N-2)/nt` rows):
  best spatial locality.  Each thread reads at most one shared boundary
  row from each neighbour, so cache-line contention is minimal.  Scales
  near-linearly to 8 threads.
- **Interleaved** (thread t handles rows t, t+nt, t+2nt, ...): every
  cell update needs `(i±1, j)` rows owned by the next/previous thread.
  Adjacent rows share L1 cache lines (64 B / 8 doubles = 8 cells of one
  row); when thread t writes row n and thread t+1 reads row n+1, those
  two rows share half their cache lines and ping-pong between cores.
  Saturates at ~2.7× speedup.
- **Block** (2D rectangle per thread, `pi × pj` ≈ √nt × √nt): each
  thread has 2× the halo of strip (4 sides instead of 2) but no
  intra-row sharing.  Scales 3.76× — middle of the road.

The Lab-5-Part-4 finding that **strip wins for 2D SOR** is reproduced.

## Spawn-overhead story

`spawn` schedule does `pthread_create` + `pthread_join` per sweep
(64 sweeps).  `persistent` does it once total.  The overhead matters
more when the per-sweep work is smaller:

- **strip** at 8 threads: persistent 0.255 / spawn 0.393 → **54%
  slowdown** from spawn.  Spawn cost is meaningful relative to a 0.5 ms
  sweep.
- **interleaved** at 8 threads: persistent 0.752 / spawn 0.776 → **3%
  slowdown**.  The per-sweep work is so much larger that spawn cost
  vanishes by comparison.
- **block** at 8 threads: 10% slowdown.  Intermediate.

This is exactly the Lab-6-Part-1b pattern: when the kernel is fast,
thread-spawn cost is a meaningful fraction of total time.  The cure is
**persistent threads + barriers**, the pattern OpenMP uses internally
and the one our `sor2d_omp.c` and `sor2d_pth.c` use.

## Surprise

Strip's near-linear scaling (7.89× at 8 threads) is *better* than what
we observed for OMP at the same N=4098 in E01 (4.49× / 7.31× for
baseline / temporal).  Two factors:
- E01 used N=4098 (larger DRAM footprint), so the per-thread work was
  more memory-bound than this E05 N=2048 config.
- E05's strip pthread workers are deliberately barrier-pure (no scratch
  alloc, no copy_boundary kernel), so they have less overhead than the
  OMP baseline.

Useful headline: **the pthreads-strip scaffold matches OMP at scaling,
and at small N actually beats it slightly.**

Plots: `decomposition_cpe.png` — CPE vs threads (log-log), 6 lines (3
modes × 2 scheds), with shaded min/max envelope and ideal-scaling
reference (black dotted) for strip-persistent.

## STATUS update

E05: done — **strip-persistent wins** at 8 threads (CPE 0.255, 7.89×
linear scaling).  Interleaved scales worst (cache-line ping-pong);
block in the middle.  Spawn-vs-persistent slowdown = 1.54× for strip,
1.03× for interleaved, 1.10× for block — Lab-6-Part-1b pthread_create
overhead clearly visible.
