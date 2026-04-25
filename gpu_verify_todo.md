# GPU verification TODO — to run on the V100 (or any nvcc machine)

This local dev machine has no `nvcc` and no CUDA driver, so neither
`sor2d_gpu.cu` nor the new `sor3d_gpu.cu` have been built or executed
in this session.  The 2D file was last verified on the V100 in the
prior session (~130 GUp/s temporal, ~3×10⁻⁶ rel error vs. CPU); the 3D
file has *never* been run.

## Build

```
make gpu                                  # builds both 2D and 3D GPU
build/sor3d_gpu 130 16
```

## Smoke / correctness

| N   | iters | What to check |
|----:|------:|---------------|
|  34 |     8 | `max\|base-cpu\|` and `max\|temp-cpu\|` should be ~10⁻⁶ relative.  Any `WARNING: non-finite` line means the temporal kernel diverged — first thing to inspect: clamp-to-edge in the corner cells. |
|  66 |    16 | Same.  Should also confirm `max\|base-temp\|` is the smaller of the three diffs, since both GPU paths use the same fp32 reduction order. |
| 130 |    32 | First "real" size.  CPU reference here takes ~5 s — be patient. |
| 258 |    96 | Where the temporal speedup story should start to show.  Expected: GPU baseline 5–15 ms, GPU temporal 2–5 ms.  CPU reference ~60 s+. |

`iters` must be a multiple of `HALO_T` (=2 in the file as written).

## Likely 3D-specific gotchas

1. **Block size 8×8×8 = 512 threads.**  Hard-coded.  If the V100 is at
   default register settings this is comfortable; if some kernel is
   register-bound and won't launch you'll see `cudaErrorLaunchOutOfResources`
   in the `CUDA_SAFE_CALL` from a kernel.  Drop to TILE=6, INTER=4
   (block 6×6×6=216) as a fallback.

2. **Halo-overhead is brutal in 3D.**  TILE=8, HALO_T=2 means useful
   work fraction = 64/512 = 12.5%.  If `temporal` is *slower* than
   `baseline` at every N, that's the expected story — it would parallel
   the 3D-CPU report.md result (0.41× temporal at N=258 cubic tiles)
   and validate that 3D needs 2.5D streaming for a real win.  Don't
   "fix" a slowdown by tweaking TILE without first reading
   Micikevicius 2009.

3. **`cudaMalloc(&d, 4)` warmup pattern.**  Same as `sor2d_gpu.cu`.
   If nvcc complains about `void**` from `float*`, fall back to the
   explicit cast: `cudaMalloc((void**)&d, 4)`.  (The 2D file would have
   the same complaint if it complained at all.)

4. **`__syncthreads()` between the load and the first sub-step.**
   Already in the kernel.  If the temporal output looks "smeared" along
   the boundary of every block, the issue is almost always a missing
   sync between sub-steps — re-read lines around `cur = 1 - cur`.

5. **`iters % HALO_T != 0`** is rejected.  Run the sweep harness with
   `--iters` divisible by 2.

## After it works

- Replace the placeholder in `report.md` "**3D GPU kernel**" next-steps
  bullet with measured numbers.
- Add the 3D row to the variant heatmap by including `sor3d_gpu` runs
  in the `make bench` output.

## If 3D temporal is consistently slower than baseline

That is a legitimate finding (3D shared-memory tiling without z-streaming
is bandwidth-poor) and should be reported as such, not hidden.  The next
step then becomes Tier B sketch:

  > 3D GPU 2.5D-streaming variant.  16×16 thread block, slide a window
  > of 3 z-planes through shared memory.  This is the standard 3D GPU
  > stencil shape and should reach ≥50 GUp/s on V100.

## Tier B2 verification: sor2d_gpu TILE/HALO runtime args

`src/sor2d_gpu.cu` was refactored from `#define TILE/HALO_T` to
`--tile T --halo H` runtime flags.  Defaults preserve the original
behaviour (`tile=32`, `halo=4`).  Things to verify on the V100:

```
build/sor2d_gpu 2050 96                       # default; should match prior numbers
build/sor2d_gpu 2050 96 --tile 32 --halo 4    # same as default, sanity
build/sor2d_gpu 2050 96 --tile 32 --halo 2    # lower halo, INTER=28
build/sor2d_gpu 2050 96 --tile 16 --halo 2    # smaller tile
```

- **Defaults**: numbers should be within noise of the prior session
  (~64 GUp/s baseline at N=2050, ~130 GUp/s temporal — see report.md).
  If the new code is *slower* at default args, something regressed in
  the dynamic-shared-memory rewrite.
- **`iters % halo == 0`** is enforced.  At halo=2 use iters=96 (✓);
  at halo=6 use iters=96 (✓); at halo=4 use iters=96 (✓).
- **Block dim limit**: TILE=32 → 1024 threads (max).  TILE > 32 will
  error in main; safe.

The `bench` target in `scripts/sweep.sh` includes a `gpu_th_sweep`
section that drives the (TILE, HALO) ∈ {16,32} × {2,4,6} grid.  After
running, plot the (TILE, HALO) vs gup_s heatmap by extending plot.py
(currently unimplemented — chart slot reserved).

## Pre-existing bug in sor3d_omp.c (caught while writing sor3d_pth_temporal)

`src/sor3d_omp.c::sor3d_temporal_superstep_omp` uses inconsistent
strides in the temporal kernel: the load loop indexes `sa` with stride
`S` (the allocated buffer is `S^3`), but the compute loop indexes with
strides `Sj` and `Sk`.  When `Si == Sj == Sk == S` (every tile when
`(N-2)` is a multiple of `B`, i.e. all tested smoke configurations)
this is a no-op — the addresses match.  At edge tiles where
`Si < S` or `Sj < S` or `Sk < S`, the compute reads/writes wrong
cells.

**Repro**: pick `N` such that `(N-2) % B != 0`.  Currently every
tested config is a multiple, so this has been silent.  E.g.
`build/sor3d_omp 100 4 32 2` would have an edge tile.

**Fix**: change all `Sj`/`Sk` strides in `sor3d_omp.c` to `S`, matching
the buffer allocation.  `sor3d_pth_temporal.c` already does this.
