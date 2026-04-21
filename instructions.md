# EC527 Project — Advanced Structured Grid SOR

## Project Goals

Build a **real-time GPU fluid solver** whose pressure-projection step is driven by an advanced, heavily optimized SOR/multigrid solver. The project has three layered goals:

1. **Application goal** — produce a working, visually compelling incompressible Navier–Stokes simulation (2D, with optional 3D extension). Use Chorin projection so the pressure Poisson solve becomes the computational core and the natural place to apply all SOR optimizations.

2. **Numerical goal** — go beyond vanilla SOR by implementing and comparing multiple solvers for the pressure Poisson equation: GPU red-black SOR (baseline), GPU multigrid V-cycle with SOR smoother, and at least one SOTA acceleration (asynchronous relaxation or mixed-precision iterative refinement).

3. **Performance-engineering goal** — apply GPU-specific optimizations to the main solver: shared-memory temporal blocking, warp shuffles, persistent kernels or CUDA Graphs, and (stretch) Tensor Core stencil formulation. Measure and attribute the speedup from each layer.

The final story arc is: **algorithm improvement (multigrid) + hardware-aware optimizations (temporal blocking, sync elimination) + a real application that motivates the whole stack.**

---

## What We Already Have From Labs

These are the baseline tools and prior code to build on, not the project itself:

- **Lab 1–2**: Loop interchange, blocking, cache behavior measurement
- **Lab 3**: SIMD/AVX intrinsics (`__m256d`, alignment handling, reduction patterns)
- **Lab 4**: CPU pipeline, multiple accumulators, unrolling
- **Lab 5**: 2D SOR with optimal omega tuning, red-black ordering, spatial cache blocking, pthreads with strip/interleaved decomposition and barriers
- **Lab 7**: CUDA GPU SOR (one-thread-per-cell, red-black, kernel relaunch per iteration)

Every project piece must go clearly beyond these. "3D instead of 2D" alone is not enough — dimensionality is a vehicle for new techniques, not a technique itself.

---

## Core Technical Concepts

### 1. 2D vs 3D SOR

**2D SOR** uses the 5-point Laplacian stencil:

```
u_new[i][j] = (1/4) * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
```

Over-relax: `u[i][j] += omega * (u_new - u[i][j])`. Optimal omega ≈ `2 / (1 + sin(π/N))`.

**3D SOR** uses the 7-point stencil:

```
u_new[i][j][k] = (1/6) * (6 face neighbors)
```

Critical differences:

- **Memory stride**: in `data[i*NY*NZ + j*NZ + k]`, the k-neighbor is at stride 1 (great), j-neighbor at stride NZ (okay), i-neighbor at stride NY*NZ (can be hundreds of KB, destroys cache locality).
- **Cache blocking matters more** — block in three dimensions (a small cube) rather than one strip. Tile working set is B³ instead of B².
- **Parallelism decomposition has more options** — slabs (split i), pencils (split i and j), or 3D sub-blocks. Surface-to-volume ratio determines communication cost.

### 2. Vectorization: the dependency problem

Standard Gauss-Seidel SOR has a sequential dependency along the scan order. `u[i][j+1]` needs the just-updated `u[i][j]`, so packing 4 consecutive cells into an AVX register and updating them in parallel is wrong.

**Three solutions:**

- **Jacobi variant**: read from old array, write to new array. Fully vectorizable but needs ~2× more iterations to converge than Gauss-Seidel.
- **Red-black SOR (preferred)**: checkerboard coloring makes all cells of one color independent. Each color sweep vectorizes cleanly. Already implemented serially in Lab 5; extending to SIMD is natural.
- **Implementation options for red-black SIMD**:
  - A) Gather/scatter (slow, avoid)
  - B) Separate red and black storage (fast, complex indexing)
  - C) Masked writes (practical, moderate performance) — load full rows, compute for all cells, blend/mask to write back only the current color

### 3. Temporal blocking (time-tiling)

**The idea**: instead of sweeping the whole grid once per iteration (which blows out cache every iteration), run K time steps on a small tile while it's still hot in cache.

**Dependency constraint**: a cell at time t=K on the tile boundary would need data from cells outside the tile at times t<K. Solution: **trapezoidal tiles** that narrow as time advances. Two adjacent tiles form a diamond shape; the inverted trapezoid fills the gap.

**2D implementation**:
- 3D spacetime (i, j, t)
- Trapezoid has a single narrowing slope in each spatial direction
- Three nested loops: spatial tile, time within tile, cells within tile
- Tile width W, time depth K → about W·K cells updated with only W+2K cache lines loaded → K× amortized bandwidth reduction

**3D implementation**:
- 4D spacetime (i, j, k, t) — tile is a truncated 4D hyper-pyramid
- Halo grows on 6 faces instead of 4 edges — halo-to-interior ratio is much worse for small W
- Asymmetric tile shapes usually win: wide in unit-stride direction (k), narrow in the high-stride direction (i)
- Many research implementations (Pluto, YASK, Girih) use auto-generated index math because writing it by hand is error-prone
- **Recommendation**: stick to 2D temporal blocking unless you have strong reason otherwise; it already demonstrates the principle cleanly

### 4. Parallelization methods comparison

| Method | Level | Strengths | Use for this project |
|---|---|---|---|
| **Pthreads** | Inter-core (CPU) | Full control, persistent threads possible | Only if OpenMP can't express what you need |
| **OpenMP** | Inter-core (CPU) | Pragma-based, compact, fast to iterate | **Default choice** for CPU parallelism |
| **AVX intrinsics** | Intra-core (SIMD) | 4× doubles or 8× floats per instruction | Inner-loop vectorization within each thread |
| **CUDA** | GPU | Thousands of threads, massive throughput | **Primary target** for the main solver |

**Key relationship**: OpenMP is implemented on top of pthreads. The `#pragma omp parallel for reduction(+:sum) schedule(static)` does what Lab 5 Part 4 did with 200 lines of barrier/strip code.

**Composition**: OpenMP for outer parallelism + AVX intrinsics for inner vectorization + CUDA for the GPU solver. Each layer is independently measurable.

---

## SOTA GPU Techniques to Apply

### Shared-memory temporal blocking
Load a tile + halo into shared memory (96–228 KB per SM on modern GPUs, ~10× faster than L2). Run K time steps inside the kernel using `__syncthreads()` between steps. Eliminates ~10 µs kernel launch overhead per iteration. Typical K = 2–8 before halo overhead eats the benefit.

### 2.5D streaming (for 3D stencils)
Micikevicius 2009. Each thread block owns a pencil in the z-direction. Keep only 3 planes (z-1, z, z+1) in shared memory, shift as you advance through z. Turns 3D stencil memory traffic into a streaming 2D pattern.

### Warp shuffles
`__shfl_up_sync` / `__shfl_down_sync` exchange register values directly between the 32 threads of a warp in 1 cycle — no shared memory, no bank conflicts. Use for innermost-axis neighbors; use shared memory for cross-warp neighbors.

### Persistent kernels + grid-wide sync
Launch the kernel once, spin in a loop inside, synchronize across the whole grid via `grid.sync()` from cooperative groups. Eliminates launch overhead entirely. Constraint: all blocks must fit on the GPU simultaneously or you deadlock. Works well for multigrid where launch overhead dominates at coarse levels.

### CUDA Graphs
Capture a multi-kernel iteration once, replay with ~1 µs overhead. Great fit for multigrid V-cycle (different kernel configs per level).

### Asynchronous (chaotic) relaxation on GPU
Remove grid-wide sync entirely. Each block iterates with whatever's in global memory. Convergence still guaranteed for diagonally dominant systems (our Laplacian qualifies). GPU hardware is excellent at hiding latency through TLP, so dropping sync is often a net win despite more iterations.

### Mixed-precision / Tensor Core stencils (stretch)
Tensor Cores do 16×16×16 matmul in one instruction but only in FP16/BF16/TF32. Recent research (ConvStencil PPoPP 2024, Liu 2022) reformulates stencils as small matmuls via im2col-style reorganization → 2–5× over hand-tuned CUDA. Pairs naturally with mixed-precision iterative refinement (FP16 inner iterations + FP64 residual correction).

### Multi-GPU (stretch)
Split grid across GPUs, halo exchange via NCCL or `cudaMemcpyPeerAsync`, overlap halo exchange with interior compute using streams. Only matters for grids > single-GPU memory.

---

## Application Layer: Incompressible Fluid Simulation

Use **Chorin projection** for 2D or 3D incompressible Navier–Stokes:

1. **Advect** velocity: `u* = advect(u, dt)` (semi-Lagrangian, cheap)
2. **Apply external forces**: `u* += dt * f` (cheap)
3. **Pressure Poisson solve**: solve `∇²p = (ρ/dt) * ∇·u*` — **this is where SOR lives, ~80–95% of total time**
4. **Project**: `u = u* − (dt/ρ) * ∇p` (cheap)

Boundary conditions: no-slip on walls (velocity = 0), Neumann for pressure (∂p/∂n = 0 on walls, p = 0 at outflow).

**Why this is the right application**:
- Makes the Poisson solve the bottleneck in an organic way
- Every optimization directly accelerates a visible simulation
- Well-understood reference (Stam's "Stable Fluids" 1999) with lots of prior art to validate against
- Visual output makes the project memorable

**Demo candidates**: smoke plume rising, Kármán vortex street behind a cylinder, lid-driven cavity, flow around an airfoil cross-section.

### Backup / simpler application
If fluid simulation scope feels too large, **Poisson image editing** (Pérez 2003) gives a cleaner, more scoped alternative: seamless image cloning via gradient-domain solving. Same SOR solver, smaller project surface, very visual.

---

## Recommended Project Structure

### Phase 1 — Baseline (Week 1)
- Port Lab 7 CUDA SOR to the project structure with clean benchmarking harness
- Extend to correct boundary conditions for Poisson (Dirichlet or Neumann as needed)
- CPU reference solver (OpenMP red-black SOR) for correctness validation

### Phase 2 — Fluid solver skeleton (Week 1–2)
- Implement 2D incompressible Navier–Stokes with Chorin projection
- Velocity advection (semi-Lagrangian)
- Call the Phase-1 Poisson solver for the pressure step
- Visualization pipeline (output PPM frames or OpenGL interop)

### Phase 3 — Algorithmic improvement: Multigrid V-cycle (Week 2)
- Implement restriction (full-weighting or injection)
- Implement prolongation (bilinear interpolation)
- 2-level V-cycle with red-black SOR as smoother at each level
- Extend to 3-level if time permits
- **Expected speedup**: 5–20× fewer iterations vs plain SOR for same tolerance

### Phase 4 — GPU optimizations (Week 3)
Pick at least two of:
- Shared-memory temporal blocking for the finest-level smoother
- Warp shuffles for innermost-axis neighbors
- CUDA Graphs for the V-cycle
- Persistent kernels with cooperative groups
- Asynchronous relaxation variant (measure convergence vs. wall time tradeoff)

### Phase 5 — Measurement and stretch goals (Week 4)
- Full benchmark matrix: grid sizes, solvers, optimizations
- `nvprof` / `ncu` profiling to confirm bandwidth/compute utilization
- Convergence quality measurements (residual vs iteration)
- Stretch: mixed-precision iterative refinement, Tensor Core stencil, 3D extension

---

## Deliverables

### Code
- CPU reference implementation (OpenMP red-black SOR) for correctness checking
- GPU baseline (Lab 7–style red-black SOR extended for Poisson BCs)
- GPU multigrid V-cycle with SOR smoother
- GPU-optimized variant with temporal blocking + sync elimination
- 2D fluid solver linking it all together

### Report sections
1. **Background**: SOR, red-black, multigrid intuition, GPU memory hierarchy
2. **Related work**: brief literature survey (leverage the 2020–2026 literature survey already drafted; Micikevicius 2009 for 2.5D streaming; ConvStencil 2024 for Tensor Cores; Anzt/Chow async work)
3. **Methods**: clearly describe each solver variant and optimization layer
4. **Results**:
   - Speedup vs baseline at multiple grid sizes (fit-in-L2, fit-in-L3, exceed-L3-but-fit-single-GPU)
   - Attribution: how much of the speedup comes from algorithm (multigrid) vs hardware (temporal blocking) vs sync-elimination
   - Convergence behavior of each variant
   - Roofline plot or at least bandwidth utilization numbers
5. **Application demo**: screenshots / frame sequences from the fluid simulation at different Reynolds numbers or grid resolutions
6. **Discussion**: what surprised you, which optimizations gave the biggest real-world win, what didn't work

### Visuals
- Fluid simulation frames or short animation (smoke plume, vortex street)
- Speedup bar charts
- Residual-vs-iteration convergence plots
- Roofline or bandwidth-utilization plot

---

## Scope-Control Checkpoints

If behind schedule, cut in this order (keep upstream, drop downstream):

1. Drop Tensor Core experiments (stretch only)
2. Drop 3D extension, stay 2D throughout
3. Drop multi-optimization comparison — pick one SOTA technique and go deep
4. Fall back to Poisson image editing application instead of fluid simulation
5. Keep: baseline GPU SOR + multigrid V-cycle + one GPU-specific optimization + some application

The minimum viable project is **"GPU multigrid SOR solver with shared-memory temporal blocking, applied to Poisson image editing."** Everything else is scope on top of that.

---

## Key References to Read

- **Stam 1999**, "Stable Fluids" — canonical real-time fluid simulation paper
- **Micikevicius 2009**, "3D Finite Difference Computation on GPUs using CUDA" — 2.5D streaming
- **Trottenberg, Oosterlee, Schüller 2000**, *Multigrid* — the standard reference
- **Briggs, Henson, McCormick 2000**, *A Multigrid Tutorial* — more approachable intro
- **Anzt et al. 2013+** — asynchronous iterative methods on GPUs
- **ConvStencil, PPoPP 2024** — Tensor Core stencil formulation
- **Pérez et al. 2003**, "Poisson Image Editing" — if the image-editing application is chosen

---

## Open Questions To Resolve Early

- CPU-only baseline or GPU-only baseline for speedup comparisons? (GPU baseline is more honest)
- 2D throughout or commit to 3D? (2D recommended for scope)
- Which SOTA technique(s) are the "main" ones? (Multigrid + temporal blocking recommended)
- Fluid simulation boundary conditions: closed box, periodic, inflow/outflow?
- Precision: FP32, FP64, or mixed? (FP32 is fine for the fluid demo; FP64 needed for tight convergence benchmarks)
