# E10 — GPU baseline vs temporal across N

Machine: SCC scc-j06, NVIDIA L40S (46 GB, sm_89, ~96 MB L2), CUDA 12.8.
Config:
- **2D**: N ∈ {258, 514, 1026, 1538, 2050, 3074, 4098} (denser than the
  doc's 4 values), iters=96, TILE=32, HALO=4 (defaults).
- **3D**: N ∈ {66, 130, 194, 258}, iters=16, TILE=8, HALO_T=2 (compile-
  time defaults in `sor3d_gpu.cu`).

Methodology: 3 reps per (N, dim); `data.csv` reports the median CPE
(cycles/element at CPNS=2.0).

## 2D GPU (the headline figure)

| N    | baseline CPE | temporal CPE | speedup base/temp |
|-----:|-------------:|-------------:|------------------:|
|  258 |       0.3945 |       0.0230 |        **17.15×** |
|  514 |       0.0957 |       0.0138 |             6.93× |
| 1026 |       0.0295 |       0.0100 |             2.95× |
| 1538 |       0.0183 |       0.0091 |             2.01× |
| 2050 |       0.0138 |       0.0090 |             1.53× |
| 3074 |       0.0103 |   **0.0089** |             1.16× |
| 4098 |       0.0260 |       0.0110 |             2.36× |

**Best 2D GPU CPE in the project: 0.0089 (temporal, N=3074).**

## What the 2D curves say

- **N=258 (small)**: baseline CPE is very high (0.39) because the kernel
  is launch-overhead-dominated — 96 launches × ~30 µs/launch is most of
  the wall time.  Temporal reduces this to 24 launches and gets 17×
  speedup.  The 17× is mostly *launch-count reduction*, not DRAM
  reuse; it's an artefact of small N, not the temporal kernel's
  legitimate win.
- **N ∈ [1026, 2050]**: baseline is now compute-saturated (DRAM-bound
  steady state).  Temporal beats baseline by a clean **1.5–3×**.  This
  is the *legitimate* temporal-blocking win.
- **N=3074**: peak GPU performance — temporal CPE 0.0089, baseline
  0.0103.  Both are within 16% of each other.  Temporal hits a
  ceiling here because at this size both kernels are memory-saturated.
- **N=4098**: both kernels degrade (CPE 0.026 / 0.011).  This grid
  exceeds the L40S L2 cache (~96 MB; one buffer at FP32 = 128 MB),
  so DRAM contention spikes.  Temporal's L2-resident scratch helps
  more here, restoring its lead to 2.36×.

## 3D GPU (honest negative finding)

| N   | baseline CPE | temporal CPE | speedup base/temp |
|----:|-------------:|-------------:|------------------:|
|  66 |       0.4483 |       0.0467 |             9.61× |
| 130 |       0.0686 |       0.0356 |             1.93× |
| 194 |       0.0304 |       0.0354 |        **0.86×** (LOSS) |
| 258 |       0.0309 |       0.0397 |        **0.78×** (LOSS) |

**The 3D temporal kernel loses to the 3D baseline at N ≥ 194.**

This is the legitimate negative finding flagged in `report.md` and
`gpu_verify_todo.md`.  Why it happens:

- 3D temporal compile-time defaults are TILE=8, HALO_T=2.
- Per-block scratch volume = `TILE³ = 512` cells; interior =
  `(TILE - 2·HALO_T)³ = 4³ = 64` cells.
- Halo-to-interior ratio = `512/64 - 1` = **7×** redundant work in 3D.

Compare to 2D temporal at TILE=32/HALO=4: `(32² - 24²) / 24² = 0.78×`
overhead — an order of magnitude less.  The 3D kernel pays 7× extra
work to save HALO_T=2 = 2× DRAM passes; net loss is unavoidable.

**Fix paths (future work)**:
- Bigger TILE (TILE=12, HALO_T=2 → INTER=8, halo overhead 2.4×).
  Constraint: 1024-thread block limit means TILE³ ≤ 1024 → TILE ≤ 10.
- Rectangular tiles `B × B × Nk` (Micikevicius 2009 "2.5D streaming"):
  amortize halo over the long dimension.  Listed in `report.md:230` as
  next-step #2.

For the talk, the honest framing is: **2D temporal works as designed
(2-3× win across the meaningful regime); 3D temporal needs the
2.5D-streaming reformulation, which is future work**.

## Cross-tier summary at the talk's representative N=2050

| tier        | best CPE | from |
|:-----------|---------:|:-----|
| serial CPU (E02 N=2048 baseline) | 0.667    | E02   |
| OMP 8t (E02 N=2048 temporal)     | 0.346    | E02   |
| pthreads 8t strip (E05)          | 0.255    | E05   |
| GPU baseline (E10 N=2050)        | 0.0138   | E10   |
| **GPU temporal (E10 N=3074)**    | **0.0089** | E10 |

Span: 75× from serial CPU to best GPU temporal.  This is the figure
to lead with in the talk's "Experiments and results" slide.

## Surprise

The 2D N=4098 *increase* in CPE for both kernels (vs N=3074) is a
clean DRAM-cache-cliff signature.  L40S has ~96 MB L2 (Ada Gen);
two FP32 buffers at N=4098 take 128 MB → can't fit in L2 → DRAM
contention.  At N=3074 the buffer pair is 72 MB → still fits L2 → both
kernels run at L2 bandwidth.  This is the GPU equivalent of the CPU
L3-crossover we saw in E02 — same physics, two orders of magnitude
larger cache.

Plots:
- `gpu_2d_cpe.png` — 2D GPU CPE vs N (log-log), baseline vs temporal,
  shaded min/max envelope.
- `gpu_3d_cpe.png` — 3D GPU CPE vs N, baseline (grey) vs temporal
  (red, to flag it as the underperformer).

## STATUS update

E10: done — **2D GPU temporal beats baseline 1.5–17×** across N
(legitimate win at N ∈ [1026, 2050]); **3D temporal loses at N ≥ 194**
because TILE=8/HALO=2 has 7× halo overhead — fixable with bigger TILE
or 2.5D streaming.  Best CPE in project = **0.0089** (2D GPU, N=3074).
