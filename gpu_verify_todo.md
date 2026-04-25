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
