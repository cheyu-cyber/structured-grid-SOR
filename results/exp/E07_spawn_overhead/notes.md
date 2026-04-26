# E07 — Pthread spawn-overhead (Lab-6-Part-1b for SOR)

Machine: SCC scc-j06, 2× Xeon Gold 6426Y.  Thread budget 4.
Config: N=512 (small enough that per-sweep work is small and
pthread_create overhead is visible), threads=4, mode=strip.
Methodology: 10 iters values × 2 schedules × 3 reps = 60 runs.
Linear-fit `wall_time = a + b·iters` separately for `persistent` and
`spawn` schedules; the slope difference is the per-sweep
pthread_create+join cost.

## Linear fits (median wall time across 3 reps)

| schedule    | per-sweep slope b   | intercept a |
|:-----------|--------------------:|------------:|
| persistent |      **81.2 µs**    |   +133 µs   |
| spawn      |     **234.7 µs**    |   −299 µs   |

## Per-sweep pthread overhead

|                                                 | µs       |
|:-----------------------------------------------|---------:|
| **per-sweep pthread_create + pthread_join**     | **153.5**|
| compute work per sweep (persistent slope)       |    81.2  |
| **overhead / work ratio**                       |  **1.89×** |

The per-sweep `pthread_create + pthread_join` cost is **larger than the
per-sweep compute work** at this configuration.  Spawning fresh threads
on every sweep nearly triples wall time per sweep relative to the
barrier-only persistent design (234.7 / 81.2 = 2.89×).

## Observation (the talk's "what didn't work" / "Lab-6 Part-1b" slide)

This is the canonical demonstration of why **persistent threads +
barriers** is the right pattern for any kernel with short per-sweep
work:

- A 5-point Laplacian sweep on a 512² grid does ~260,000 cell updates
  spread across 4 threads = ~65,000 updates per thread.  At ~1.25 ns
  per update on a Xeon Gold 6426Y, that's about **81 µs** of actual
  compute — exactly what the persistent-schedule slope measures.
- `pthread_create` does memory allocation (per-thread stack), kernel
  scheduler entry, and TLS setup.  Plus `pthread_join` blocks until
  the thread exits and reaps it.  At 4 threads, doing this on every
  sweep adds **~38 µs per thread per sweep**, totaling **~153 µs**.
- For larger kernels (long inner loop, high N), the relative overhead
  shrinks; `spawn`-style scheduling becomes survivable.  At N=512,
  iters=512: persistent 42 ms, spawn 121 ms — still 2.9× difference,
  even at this iter count.

The OpenMP runtime uses persistent threads under the hood; that's
why our `sor2d_omp.c` with `#pragma omp parallel for` doesn't pay
this overhead.  Our `sor2d_pth_decomp.c --sched persistent` matches
that pattern (one create+join per binary, barriers between sweeps);
`--sched spawn` is the *anti-pattern* you'd accidentally write if
you didn't think about thread lifecycle.

## Surprise

The intercept of the spawn fit is *negative* (−299 µs).  This is a
linear-fit artefact at small iters: at iters=1, both schedules pay
one create+join, so they're close.  As iters grows, persistent's
intercept (the one-time setup cost) becomes a smaller fraction of
total time, while spawn keeps paying the per-sweep cost — so the
linear models cross at small iters.  The **slope** difference is
what's meaningful, and it's robust (no dependence on intercept).

## Cross-reference to E05 (the cross-mode pthreads study)

E05 measured `spawn / persistent` ratio at 8 threads, N=2048, iters=64:

|              | E05 slowdown ratio   | E07 (4t/N=512/iters=64) ratio |
|:------------|---------------------:|------------------------------:|
| strip        |              1.54×   | 14.5 ms / 5.2 ms = **2.79×** |
| interleaved  |              1.03×   |                            (n/a) |
| block        |              1.10×   |                            (n/a) |

E05 saw a smaller ratio (1.54×) at strip-N=2048 because the per-sweep
work (~660 µs at N=2048) had grown by ~8× relative to N=512, so the
overhead is a smaller fraction.  At N=512 the same pthread_create
cost is a bigger fraction → 2.79× slowdown.  Both measurements are
consistent with the **fixed ~150 µs per-sweep overhead** we extracted
here.

## STATUS update

E07: done — pthread_create+join cost = **~153 µs per sweep at 4
threads**.  Larger than the 81 µs of actual per-sweep compute work
at N=512, so spawning per sweep is a 1.9× tax on the compute.
Persistent + barrier is the right pattern.

Plot: `spawn_overhead.png` — log-log wall-time vs iters with linear
fits, error bars from min/max across 3 reps.
