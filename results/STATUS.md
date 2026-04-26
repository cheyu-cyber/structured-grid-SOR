# Experiment status

Machine: SCC scc-j06 — 2× Intel Xeon Gold 6426Y (32 physical cores total),
250 GB RAM, 2× NVIDIA L40S (46 GB each).  CUDA 12.8.

Thread budget: capped at **8 threads** for experiments.  Kernel allows all
32 cores (`Cpus_allowed_list: 0-31`, cgroup quota unlimited), but this is
the shared SCC login node — 8 keeps us under one socket and well-behaved
relative to other users.  Matches the experiment doc, which was written
for an "8 physical cores" target.

Cap E01's `threads` sweep at {1, 2, 4, 8} (drop the 16 point — its
original purpose was to show oversubscription on an 8-core box, which
doesn't apply here anyway).  All other experiments already cap at 8.

| ID | Name                                | Status   | Finding |
|----|-------------------------------------|----------|---------|
| E00| Setup                               | done     | 32-core/2-L40S box; smoke bit-identical, GPU base/temp finite |
| E01| Strong scaling 2D OMP               | done     | CPE 2.92→0.38 (1t→8t temp), 7.31× scaling vs 1t; temp/base CPE = 0.58× at 8t |
| E02| Size sweep 2D                       | done     | min temporal CPE 0.346 at N=2048 (L3→DRAM crossover); base/temp = 1.93× |
| E03| Spatial block size B                | done     | best B=128, min CPE 0.389; plateau B∈[24,256], cliff at B≥384 (L2-fit) |
| E04| Temporal depth T                    | done     | min CPE **0.332** at T=8, B=128 (1.34× faster than baseline); T=1 / T=32 lose |
| E05| Pthreads decomposition              | done     | strip-persistent CPE=0.255 at 8t (7.89× linear); spawn overhead 1.54× on strip |
| E06| 3D OMP partitioning                 | done     | slab=pencil tie at 8t (CPE 0.359 best at N=130); cube loses 1.3–1.7× to halo |
| E07| Pthread spawn overhead              | done     | pthread_create+join = **153 µs/sweep** at 4t (vs 81 µs compute work; 1.89× tax) |
| E08| Omega U-curve                       | done     | empirical ω_opt=1.95 (254 iters) matches theory 1.9517 to 0.1%; 13.5× fewer iters than Jacobi |
| E09| GPU TILE/HALO sweep                 | done     | best (TILE=24, HALO=4), CPE 0.0088, 1.56× over baseline kernel; HALO=4 wins for TILE≥16 |
| E10| GPU baseline vs temporal vs N       | done     | 2D temp 1.5–17× over base (best CPE 0.0089 at N=3074); 3D temp loses at N≥194 (7× halo overhead, fixable) |
| E11| Cross-tier headline                 | done     | fastest = GPU temporal CPE 0.0090; slowest→fastest span = **310×** at N=2050 |
| E12| Convergence (skipped)               | skipped  | no residual binary |

## Final summary

11 of 12 experiments done (E12 skipped — no residual binary).  The
**talk-headline** is E11's **310× span from serial-CPU baseline (CPE
2.792) to GPU temporal (CPE 0.0090)** at the same N=2050.

Best CPE per tier:

| tier                          | best CPE   | source |
|:------------------------------|-----------:|:-------|
| serial CPU (1 thread)         |     2.586  | E11    |
| OMP, 8 threads                |     0.229  | E11    |
| pthreads strip-persistent, 8t |     0.255  | E05    |
| GPU baseline (one-sweep)      |     0.0103 | E10    |
| **GPU temporal (shared mem)** | **0.0089** | E10    |

The synthesised write-up lives in `report.md` under the new
**`## Results — full bench`** section.  Each experiment's per-block
notes + plot is reachable via the figure-link table at the end of
that section.
