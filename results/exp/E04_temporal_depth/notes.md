# E04 — temporal depth T (halo-overhead crossover)

Machine: SCC scc-j06, 2× Xeon Gold 6426Y (32 cores, 38.4 MB L3 / socket).
Thread budget 8.
Config: N=2048, B=128 (locked from E03), iters=96 (all T values of
interest divide evenly), threads=8 with `OMP_PROC_BIND=close`.
Methodology: 10 T values (denser than the 5-point doc default), 3
repeats each; table reports the median CPE (cycles/element, CPNS=2.0).

## Throughput vs T (median of 3)

| T  | baseline CPE | temporal CPE | temp/base CPE | halo overhead |
|---:|-------------:|-------------:|--------------:|--------------:|
|  1 |        0.504 |        0.739 |        1.47×  |          3.1% |
|  2 |        0.469 |        0.487 |        1.04×  |          6.3% |
|  3 |        0.494 |        0.424 |        0.86×  |          9.6% |
|  4 |        0.464 |        0.393 |        0.85×  |         12.9% |
|  6 |        0.448 |        0.360 |        0.80×  |         19.6% |
|  8 |        0.444 |    **0.332** |    **0.75×**  |         26.6% |
| 12 |        0.469 |        0.348 |        0.74×  |         41.0% |
| 16 |        0.412 |        0.378 |        0.92×  |         56.2% |
| 24 |        0.418 |        0.393 |        0.94×  |         89.1% |
| 32 |        0.401 |        0.461 |        1.15×  |        125.0% |

Smaller CPE = faster.  Below 1 means temporal wins.

`halo overhead = ((B + 2T)² / B²) − 1`, the fractional redundant work
done in the halo region.

## The shrinking-trapezoid tradeoff (the talk story)

Two competing trends as T grows:

1. **DRAM-traffic savings ∝ T**.  One super-step of T sweeps reads each
   tile from DRAM once instead of T times.
2. **Redundant work ∝ T²** (in 2D).  The scratch is `(B + 2T)²`; the
   ratio of scratch volume to output volume is `(1 + 2T/B)²`,
   super-linear in T.

The product is unimodal in CPE.  Minimum at **T=8** with CPE = **0.332
cycles / element**, **1.34× faster** than baseline at the same T (CPE
0.444).  CPE rises both sides:
- **T=1**: CPE 0.74 — the degenerate case, no DRAM reuse, but temporal
  still pays per-tile scratch alloc + memcpy + boundary pre-copy.
  Temporal *loses* here (1.47× slower than baseline).
- **T=32**: CPE 0.46 — **125% redundant work** (every output cell is
  computed alongside more than its own value of halo cells).  Temporal
  again *loses* (1.15× slower than baseline).

T=12 is statistically tied with T=8 (CPE 0.348 vs 0.332), but T=8 is
the safer pick: smaller halo overhead so it's more robust to N changes.

## Observation

The classic shrinking-trapezoid story comes through cleanly: temporal
blocking trades quadratic-in-T halo work for linear-in-T DRAM savings.
The crossovers are exactly where a back-of-envelope model predicts —
T=1 loses to overhead, T=32 loses to halo, minimum CPE in the middle.

## Surprise

The minimum is sharper than expected: T=8 wins decisively at CPE
0.332, T=16 already drops to CPE 0.378.  The temporal sweet spot is
narrow on this machine; when extending to 3D or larger N, T tuning
matters and a quick T-sweep should be re-done before locking a value.

Best CPE seen anywhere in the project: **0.332 cycles/element**.

Plots: `temporal_depth_cpe.png` — CPE vs T with halo-overhead %
overlaid (right axis, red dashed).

## STATUS update

E04: done — min CPE **0.332** at T=8 (B=128), **1.34× faster** than
baseline at same T.  T=1 loses to overhead, T=32 loses to halo;
unimodal curve confirms the trapezoid analysis.
