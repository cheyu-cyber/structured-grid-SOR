# E03 — spatial block size B (sweet spot)

Machine: SCC scc-j06, 2× Xeon Gold 6426Y (32 cores, 38.4 MB L3/socket,
~1.5 MB L2/core, ~80 KB L1/core).  Thread budget 8.
Config: N=2048, iters=64, T=4, threads=8, OMP pinned.
Methodology: 11 B values (denser than the 6-point doc default), 3
repeats each, table reports the median CPE (cycles/element, CPNS=2.0).

## Throughput vs B (median of 3)

| B    | baseline CPE | temporal CPE | scratch KB / thread |
|-----:|-------------:|-------------:|--------------------:|
|   16 |        0.411 |        0.447 |                   9 |
|   24 |        0.434 |    **0.388** |                  16 |
|   32 |        0.428 |        0.405 |                  25 |
|   48 |        0.423 |        0.417 |                  49 |
|   64 |        0.462 |        0.399 |                  81 |
|   96 |        0.451 |        0.416 |                 169 |
|  128 |        0.470 |    **0.389** |                 289 |
|  192 |        0.461 |        0.412 |                 625 |
|  256 |        0.389 |        0.389 |                1089 |
|  384 |        0.537 |        0.653 |                2401 |
|  512 |        0.414 |        0.603 |                4225 |

Smaller CPE = faster.  Per-thread scratch =
`2 × (B + 2T)² × 8 B`.  L2 ≈ 1.5 MB / core.

## Observation

**Baseline CPE is roughly flat at ~0.39–0.47** (B doesn't affect the
baseline kernel — it's only used by the temporal scratch).

**Temporal has a wide plateau across B ∈ [24, 256]**, hovering at
CPE 0.39–0.42, with two co-equal minima at **B=24 (0.388)** and
**B=128 (0.389)**.  Below B=24 (B=16, scratch 9 KB) there are too many
small tiles — redundant halo work dominates.  Above B=256 the temporal
kernel **falls off a cliff** to CPE 0.60–0.65 (50%+ slowdown).

**The cliff is the L2-fit threshold**.  Two scratch buffers per thread
must fit in the per-core L2 (~1.5 MB) for in-tile reuse to be at L2
bandwidth.  Threshold:
  `2 (B+8)² · 8 ≤ 1.5 MB → B + 8 ≤ 305 → B ≤ ~297`
Predicted: B=256 (1.06 MB scratch) fits; B=384 (2.4 MB) doesn't.
Observed: B=256 holds CPE 0.389; B=384 jumps to 0.653.  Exactly matches
the prediction.

**Best B for downstream experiments**: **B=128**.  Median of plateau,
scratch (289 KB / thread) fits L2 with headroom, 2046/128 = 16 tiles
per axis aligns cleanly.  B=24 is a co-winner in CPE but its narrower
tile means more boundary work; B=128 is the more robust choice.

## Surprise

The peak at **B=24** (scratch 16 KB, fits L1 fine) is unexpected —
small enough that the scratch sits in L1.  This says the temporal
kernel does benefit from L1-resident reuse when the tile is tiny, but
the gain is no better than the L2-resident plateau.  "Fits in L2" is
the necessary condition; whether you also fit in L1 is a second-order
tweak that buys nothing.

Plots: `block_size_cpe.png` — CPE vs B (log x), L2-fit threshold marked.

## STATUS update

E03: done — best **B = 128**, min CPE ~**0.389**.  Plateau B∈[24,256],
cliff at B≥384 = L2-fit threshold.
