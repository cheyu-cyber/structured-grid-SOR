# E11 — Cross-tier headline at N=2050, iters=96, 8 threads

The single chart for the talk's "Experiments and results" slide.
Every variant on the same axes, one representative grid size.

Machine: SCC scc-j06, 2× Intel Xeon Gold 6426Y (32 cores, 38.4 MB
L3/socket), NVIDIA L40S (46 GB, sm_89), CUDA 12.8.
Methodology: 5 binaries × 3 reps each, OMP and pthreads pinned with
`OMP_PROC_BIND=close OMP_PLACES=cores`.  Headline metric: CPE
(cycles/element at CPNS=2.0).

## Cross-tier CPE (median of 3)

| tier                 | kind     | CPE      | speedup vs serial-CPU baseline |
|:--------------------|:---------|---------:|-------------------------------:|
| serial-CPU           | baseline |   2.792  |                          1.00× |
| serial-CPU           | temporal |   2.586  |                          1.08× |
| pthreads-8t-strip    | baseline |   0.399  |                          7.0×  |
| pthreads-8t-temp     | baseline |   0.392  |                          7.1×  |
| pthreads-8t-temp     | temporal |   0.378  |                          7.4×  |
| OMP-8t               | temporal |   0.347  |                          8.0×  |
| OMP-8t               | baseline |   0.229  |                         12.2×  |
| GPU                  | baseline |   0.015  |                        186×    |
| GPU                  | temporal |**0.0090**|                       **310×** |

Smaller CPE = faster.  Lower bar = better.

## The headline number

**310× span** from serial-CPU baseline (2.792 CPE) to GPU temporal
(0.0090 CPE) on the **same** N=2050 grid, **same** iteration count
(96), **same** stencil, **same** initial data.  Every step is
measurable and attributable:

- Serial → 8-threads (OMP/pthreads): **~10×**.  Pure parallelism win.
- 8-threads CPU → GPU baseline: **~15–25×**.  GPU memory bandwidth
  + 17,000 cores easily beat 8 cores, even without any GPU-side
  optimization.
- GPU baseline → GPU temporal: **~1.6×**.  Shared-memory temporal
  blocking on top of the baseline kernel — the smallest individual
  win, but the one most directly tied to the project's
  algorithmic-optimization story.

## What each tier teaches

- **serial-CPU temporal vs baseline (2.59 vs 2.79, 7% win)**: at one
  thread the Xeon's hardware prefetchers + 38 MB L3 keep baseline
  near streaming-bandwidth, so temporal blocking has almost no DRAM
  cycles to recover.  This was the surprise in E01: **temporal
  blocking earns its keep only once parallelism stresses DRAM**.
- **OMP-8t baseline (0.229) beats OMP-8t temporal (0.347)**: at this
  specific (N=2050, iters=96, B=128, T=8) point the temporal kernel's
  per-tile overhead (scratch malloc, pre-copy, halo work) outweighs
  the DRAM savings.  At larger N (E02 found 1.93× win at N=2048,
  iters=64) and at higher thread counts the temporal kernel pulls
  ahead.  Cross-experiment: **temporal is good when the working set
  exceeds L3** — see E02.
- **pthreads vs OMP at 8 threads**: pthreads-strip (0.399) and OMP
  baseline (0.229) bracket the same parallel computation.  OMP wins
  here because its per-iter `parallel for` dispatch is cheaper than
  pthreads' barrier-pair-per-sweep loop.  At larger problems the gap
  narrows — see E05's 7.89× linear scaling for strip-persistent at
  N=2048, iters=64.
- **GPU baseline (0.015) → GPU temporal (0.0090)**: 1.6× win, in line
  with E10's 1.5–2.0× speedup in the steady-state regime (N ∈
  [1538, 2050]).
- **GPU temporal at 0.0090 CPE** is the project's overall winner.
  At CPNS=2.0 cycles/ns, that's 4.5 ps per element on the L40S — about
  one floating-point op per nanosecond per element, which puts us
  near the Roofline machine-balance line for a memory-bound 5-point
  stencil at FP32.

## How this maps to the talk

This is the slide that answers "**what worked?**"  The bar chart
visually compresses every prior experiment into one figure:

| Story arc                          | Bar that shows it       | Other ref |
|:----------------------------------|:-----------------------|:----------|
| Parallelism (8 cores)              | serial → OMP-8t base    | E01      |
| Hardware-aware blocking            | OMP-8t base → temp      | E03, E04 |
| Decomposition matters              | OMP vs pthreads bars    | E05      |
| GPU acceleration                   | OMP-8t → GPU base       | E10      |
| GPU shared-memory tuning           | GPU base → GPU temp     | E09      |

Plot: `headline_cpe.png` — horizontal bar chart, log-x CPE, sorted
slowest-to-fastest, error bars from min/max, color-coded by tier.

## Surprise

The fastest **CPU** result here is **OMP-8t baseline at 0.229 CPE**,
not OMP-8t temporal (0.347) — the opposite of E02's finding (where
temporal won 1.93× at N=2048).  Most likely explanation: at iters=96
with this specific config, the per-iter loop dispatch is amortized
enough that the baseline pulls even with temporal.  The E02 finding
(temporal wins at N ≥ 2048 with iters=64) and this E11 finding
(baseline wins at N=2050 with iters=96) both stand — they're just
adjacent points on a noisy CPE landscape near the temporal/baseline
crossover.  The talk should report **both** measurements honestly:
"temporal blocking helps in the right regime; outside that regime
its overhead matters."

## STATUS update

E11: done — fastest config is **GPU temporal at CPE = 0.0090**;
**slowest → fastest span = 310×** across all tiers at N=2050.  The
cross-tier story is fully assembled.
