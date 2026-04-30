# Previous session — summary of changes

## Context

Starting state: the project had 2D/3D temporally blocked SOR with
single-thread CPU, OpenMP, and CUDA GPU tiers, all using a Jacobi-like
ping-pong stencil at `omega = 0.9`. The `instructions.md` goal asks for a
GPU Poisson solver for fluid/image-editing, with a comparison story across
the parallelism tiers. Missing pieces for a minimum starter: pthreads
variant, a real-SOR (`omega > 1`) variant with omega-tuning experiment,
and any form of visualization.

## What I changed

### New files

- **`src/sor2d_pth.c`** — pthreads baseline (strip decomposition over
  interior rows, `pthread_barrier_t` between sweeps, thread 0 does the
  buffer swap). Same Jacobi-ping-pong stencil as everything else so
  cross-tier comparisons are apples-to-apples. Verified at N=1024,
  iters=64, 4 threads: 3.78× over serial, bit-identical output.
- **`src/sor2d_rb.c`** — classical red-black Gauss-Seidel SOR, in-place,
  supports `omega > 1`. Iterates until mean `|change|` per interior cell
  drops below `TOL = 1e-5` (or a `MAX_ITERS = 100000` cap). Three modes:
  - default: run at theoretical `omega_opt = 2 / (1 + sin(pi/(N-1)))`
  - `--omega <w>`: run at a specified omega
  - `--sweep`: run 75 values from `omega = 0.50` to `1.99` in `0.02`
    steps, print `(omega, iters, seconds)` CSV
  Verified at N=64: minimum at `omega=1.90` (139 iters) matches theory
  `omega_opt=1.9050`; diverges at `omega >= 2.0` as expected.

### Modified files

- **`src/common.h`**
  - Bumped `_POSIX_C_SOURCE` from 199309L to 200112L so
    `pthread_barrier_t` is visible (199309L was only enough for
    `clock_gettime`).
  - Added `write_ppm_gray(path, a, N)` — writes a binary grayscale PPM
    (P5), normalizing to `[0, 255]` using `a[]`'s own min/max; skips
    non-finite values gracefully.
- **`src/sor2d_cpu.c`, `src/sor2d_omp.c`, `src/sor2d_gpu.cu`** — added
  optional `--ppm <path>` flag that dumps the final baseline field. The
  GPU file inlines its own copy of `write_ppm_gray` since it doesn't
  include `common.h`.
- **`Makefile`** — added `sor2d_pth` and `sor2d_rb` targets with `-pthread`
  where needed; extended `smoke` to exercise them; added a new `omega`
  target that writes `build/omega_sweep.csv` for plotting the classic
  Lab-5 U-curve.
- **`README.md`** — rewrote to document the full comparison matrix
  (single-thread / pthreads / OpenMP / CUDA × baseline / temporal × 2D /
  3D, plus the red-black variant), the `--ppm` visualization flow, and
  the omega-sweep experiment. Kept the ping-pong-omega stability note.

### Git hygiene

- `.gitignore` expanded earlier in the session to also cover `build/`,
  common C/CUDA artifacts, editor/OS files, and logs. (Now updated: no
  longer ignores `instructions.md`.)

## Why these choices, briefly

The existing temporal-blocking code is solid but uses damped Jacobi
(`omega in (0, 1]`), which gives up the main benefit of SOR. Rather than
rewriting the temporal kernels to red-black (a large, error-prone change
— the shrinking trapezoid has to operate in pairs of half-sweeps), I
added a **separate** red-black file that owns the "real SOR" story. The
temporal files keep their clean ping-pong form and bit-identical
validation; `sor2d_rb.c` is the place to discuss over-relaxation and
omega tuning. This matches the EC527 lab split (Lab 5 test_SOR_OMEGA.c
for tuning; Lab 7 red-black kernel for the parallel story) and keeps
each file doing one thing clearly.

## Verification

Built with `-Wall -Wextra`, no warnings. Smoke tests pass bit-identically
for all four CPU-tier targets. GPU file unchanged except for the
`--ppm` hook (not built — no nvcc hardware here). Omega sweep produces
the expected U-curve bottoming at `omega_opt`.

## Suggested next steps

1. **3D GPU kernel** (`sor3d_gpu.cu`) — natural extension of the 2D
   shared-memory temporal kernel; at `TILE=16, HALO_T=2` the scratch is
   8 KB, comfortable on V100.
2. **Red-black temporal kernel** — pair the shrinking-trapezoid scheme
   with half-sweep granularity so temporal blocking can run at
   `omega > 1`. This is the single change that unifies the two stories.
3. **Rectangular 3D CPU tile** `B × B × Nk` — amortize halo over the
   long axis, fix the 3D temporal `0.41×` regression at cubic `B`.
4. **Application layer** — Poisson image editing is the simplest target
   that actually uses the solver to produce a visible result; the
   `write_ppm_gray` helper is already in place.
