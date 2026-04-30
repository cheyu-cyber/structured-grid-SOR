You are helping me write the final report for an undergraduate-level
high-performance computing course (BU EC527). The course covers serial
CPU optimization, multicore parallelism, and GPU computing for
numerically intensive kernels.
================================================================
PROJECT TOPIC
My final project is on Structured-Grid SOR / 5-point Laplacian
stencil with temporal blocking, taken from a serial CPU baseline all
the way to a GPU-accelerated temporally-blocked implementation. The
kernel solves the 2D Laplace equation ∇²u = 0 on an N×N grid via
iterative relaxation; the optimization narrative is about turning a
memory-bandwidth-bound stencil sweep into something that re-uses
on-chip data across multiple time-steps via temporal (a.k.a. time-
skewed) blocking.
================================================================
COURSE OBJECTIVES & GRADING CRITERIA (what the report has to do)
The report is graded by the EC527 instructor against the criteria
below. Treat this as the rubric and make sure each item is
covered somewhere in the draft.
The course's stated purpose for the project: "give in-depth
experience with high performance programming, especially by using
the methods you're learning this semester." So the report should
visibly draw on course material — Bryant & O'Hallaron-style CPE
analysis, cache hierarchy reasoning, the partitioning steps from
the parallel-programming lectures (Decomposition / Assignment /
Orchestration / Mapping), pthreads & OpenMP, and CUDA.
What the write-up must contain (verbatim from the project
handout, slightly reordered to match our section structure):
(a) Description of the problem.
(b) What the serial code/algorithm looks like — what is the
algorithm, and what is its complexity?
(c) Where does the time go? What is the arithmetic intensity?
(d) What are the primary data structures? What is the memory
reference pattern?
(e) Did we modify the algorithm to run in parallel (and which
parallel algorithm did we select if there is a choice)?
→ For us this is non-trivial: sequential SOR has a
loop-carried dependency, so we moved to red-black SOR
and/or Jacobi-form for the parallel and temporal-blocked
variants. Make this transition explicit.
(f) For the parallel and GPU parts, how were the data and
computations partitioned?
(g) Overview of the optimized codes — what are the
optimizations and what problems came up?
(h) Experiments and results — what worked, what didn't, why,
what limit did we hit?
(i) Reference codes and validation — ideally we compare against
something external (MATLAB, MKL, a textbook reference
solution to ∇²u = 0 with our boundary conditions). At
minimum: the analytical solution to a simple Dirichlet
problem, or convergence of the residual norm to zero, used
as a correctness check across all variants. Call out the
validation strategy explicitly in §2 and §3.
What the grader is looking for (also from the handout):

Clarity. Easy to follow, rational progression, reasonable
assumptions stated. The Weinberg-style walk-through-reasoning
voice serves this directly.
Difficulty of application. SOR is on the "straightforward,
covered-in-class" end of the spectrum, which the handout
explicitly says means "the optimizations need to be
substantial and thorough." We compensate with breadth
(serial → pthreads → OpenMP → GPU, plus temporal blocking
on each tier) and depth (sweeps over B, T, tile, threads).
The report should make this trade-off visible: acknowledge
SOR is well-trodden, then justify the project by the
thoroughness of the optimization sweeps and the
temporal-blocking implementation across architectures.
Number of architectures tried. Handout says "one per group
member is OK, more is better." We have four execution
tiers (serial, pthreads, OMP, CUDA) — make the breadth
visible in the headline results table.
Optimizations attempted. The handout lists the relevant
axes: algorithm (sometimes), memory mapping, accounting for
the pipeline, vectorization, partitioning, synchronization,
parallelization method. The report should explicitly touch:

algorithm: SOR vs. red-black SOR vs. Jacobi
memory mapping: spatial blocking, temporal blocking,
GPU shared-memory tiling
pipeline: inner-loop scheduling (mention even if not
deeply optimized)
vectorization: whether -O3 -march=native auto-vectorizes
the stencil; check the assembly for ymm/zmm
partitioning: row-block partitioning across CPU threads,
(TILE, T) partitioning across CUDA blocks
synchronization: red-black phase barrier; CUDA
__syncthreads() between sub-steps
parallelization method: pthreads vs. OpenMP head-to-
head, then CUDA. The OMP-worse-than-pthreads result is
a textbook talking point — own it.


Reference / validation. Build a correctness check into the
report (residual converging to zero, or comparison against a
known analytical solution like u(x,y) = sinh(πy)sin(πx) on
the unit square with appropriate boundaries). Show that all
four variants agree to within FP32 tolerance.
Quality of work. "What worked? What didn't? Why? What limit
did we hit? How many ideas from the semester did we try —
how well did we understand them?" The Weinberg tone is
designed for exactly this: walk-through reasoning, honest
negatives, hardware-justified optimization choices. The
graded version is the version that diagnoses why something
didn't work, not the version that hides it.

Apply-the-course-methods checklist (the handout mentions these
explicitly under "Overall"). The report should visibly use:

Decomposition: how we broke the stencil sweep into
independent units of work.
Assignment: which thread/block gets which units.
Orchestration: synchronization, halo exchange, ping-pong
buffers.
Mapping: how the assignment lands on the actual cores /
SMs / cache hierarchy.

Length / format expectations from the handout: presentation was
6–14 minutes (already delivered, slides exist); the write-up is
"basically the same as the presentation guidelines, except your
audience is the instructor and you have reached definite
conclusions." So the report is the presentation expanded to
~22 LaTeX pages with the depth and definite conclusions the talk
couldn't fit.
================================================================
STRUCTURE (sections in this exact order)

Description of the Algorithm

1.1 Mathematical foundation in plain language. Derive SOR from
first principles: start from the continuous Laplace equation
∇²u = ∂²u/∂x² + ∂²u/∂y² = 0, do the second-order centered
finite-difference discretization on an N×N grid (giving the
classic 5-point stencil u_{i,j} = ¼(u_{i-1,j} + u_{i+1,j} +
u_{i,j-1} + u_{i,j+1}) for the Jacobi form, or the SOR update
u^{k+1}{i,j} = (1-ω)u^k{i,j} + (ω/4)(neighbors) for
Gauss-Seidel-with-relaxation). Show why SOR converges faster
than Jacobi for the right ω (spectral-radius argument; for an
N×N Laplacian the optimal ω → 2 - O(π/N)). Show the math.
1.2 The specific algorithmic form used. Write out pseudocode
for (a) the baseline sweep, (b) red-black SOR (which is what
makes parallelism legal — flag this!), and (c) the temporally-
blocked variant. Define every variable: N (grid side), B
(spatial tile size), T (temporal halo / number of sub-steps
fused), ω (relaxation parameter), TOL (convergence tolerance),
iters (sweep count).
1.3 An "Implementation Reality Check" subsection that honestly
names where theory meets finite-precision arithmetic. Worked
example: a 3×3 interior on a 5×5 grid with fixed Dirichlet
boundaries (e.g. boundary = 1, interior init = 0). Walk through
two SOR sweeps explicitly with ω = 1.6, showing the actual
numbers — be willing to write fractions like "26/131" to
contrast exact arithmetic with what FP32 actually stores. Then
comment on the SOR-vs-temporal-blocking tension: classical
sequential SOR has a loop-carried dependency along the sweep
order, so the "temporal blocking" we actually implement is on
either red-black SOR or the Jacobi form — the report should
own this honestly rather than paper over it.


Scalar CPU Version

2.1 Identify the hot inner loop: the 5-point stencil update. It's
the optimization priority because it's executed N²·iters times
and it has arithmetic intensity AI ≈ 0.75 FLOP/byte (5–7 FLOPs
per cell, 8 bytes of DRAM traffic per cell per sweep =
1 read + 1 write × float32). The roofline puts us deep in the
bandwidth-bound regime, so every optimization is really about
reducing DRAM bytes per useful FLOP. Include 2–3 micro-
benchmarks of variants of the inner loop (row-major vs. ji-order,
scalar vs. vectorizable form, with/without restrict). Present a
small CPE comparison table.
2.2 The next-level operation: a full sweep over the grid. Try at
least two algorithmic approaches:
(a) plain double-loop sweep (the baseline), and
(b) spatially-blocked sweep with cache-tile size B,
(c) temporally-blocked sweep that loads a (B+2T)×(B+2T) tile,
advances T sub-steps in place, and writes the central B×B
back to dst.
The work multiplier of (c) is ((B+2T)/B)^d (d = 2 here), which
bounds the redundant-compute overhead. Include CPE tables for
each variant at a couple of N's.
2.3 "Loose Ends" — the O(N²) boundary copies, the per-tile
scratch malloc inside the temporal kernel, the convergence-norm
reduction (||u^{k+1} - u^k||_∞), and the final result memcpy.
Briefly explain why these aren't the optimization priority but
also why the per-tile malloc actually does show up as
overhead at small N (this is one of our measured surprises —
temporal loses to baseline at N ≤ 512 because the overhead
beats the bandwidth savings).
2.4 Stitching it together. Show real source code from
test_SOR.c / test_SOR_part4.c (15-40 lines) with comments
mapping each line to the math from §1: the ω·(neighbor sum)/4
update, the (1-ω) damping term, the boundary-skip indexing,
the tile-scratch allocation, the T sub-step loop.
2.5 Code and Results

2.5.1 Discussion of CPE — present the size-sweep plot
(CPE vs N at N ∈ {256, 1024, 4096}, B=128, T=8, threads=8,
iters=64). Comment on the regime change: baseline is fine
up to N=1024 where the entire grid still fits in L2
(2MB/thread × 8 threads = 16MB; 1024² × 4B = 4MB fits with
room to spare per-thread when partitioned). At N=4096 the
working set is 64MB and we're DRAM-bound, so temporal
blocking starts to pay. Also present the block-size
sweep (CPE vs B at N=2048, T=4, iters=64, threads=8) and
the halo/T sweep (CPE vs T at N=2048, B=128, iters=96,
threads=8). The T-sweep has a clean optimum around T=8–12
(CPE 0.332) before the work multiplier ((B+2T)/B)² overtakes
the bandwidth savings; quote the actual table from §INPUTS.
2.5.2 Discussion of Iteration Number and Error — present a
plot of iters-to-tolerance vs N for the SOR sweep with
ω = 1.60 and ω chosen near the asymptotic optimum
(2 - 2π/N). Try to fit it to the theoretical O(N)
convergence rate of optimally-tuned SOR (vs. O(N²) for
Jacobi/Gauss-Seidel). If the fit fails — for example, if
red-black temporal blocking changes the effective ω or the
error norm doesn't drop monotonically due to FP32 rounding
in long sweeps — say so and investigate.


2.6 Overall Comments — bullet list of 4-6 key findings,
followed by 2-3 candidate next steps.


To the GPU!

3.1 Memory allocation and transfer — list every buffer:
d_src and d_dst (each N²·sizeof(float) = 16 MB at N=2048;
much larger at N=4096), the convergence norm reduction
buffer, and the constant-memory copies of {ω, N, T}. The
dominant transfer cost at N=2048 is the initial H→D copy of
d_src (16 MB over PCIe Gen4 ~25 GB/s ≈ 0.6 ms ≈ ~2M cycles
at 3.5 GHz). After that the iteration loop is on-device.
3.2 The parallelized hot loop. Describe how work partitions
across CUDA blocks and threads. Note the vocabulary mismatch
(CPU "thread" → CUDA "block"; CPU "block of work" → CUDA
"tile") explicitly — this caught me out and is worth telling
the reader about. Include 4 tuning tables (one per matrix
size N ∈ {512, 1024, 2048, 4096}) sweeping (TILE_WIDTH,
T) parameters. The N=2048 sweep is the one already measured:
tile ∈ {8, 16, 32}, T ∈ {1, 2, 3, 4, 6, 8, 12}; optimum is
near tile=16, T=4 at CPE ≈ 0.009. Pick out the optimum from
each.
3.3 Partitioning the rest of the algorithm — explain the
decision between (a) one kernel per stencil sub-step (clean,
easy convergence-norm reductions, but N²·iters DRAM round-
trips) vs. (b) one big temporal-blocked kernel that holds a
(TILE+2T)² shared-memory tile and runs T sub-steps inside one
launch (fewer DRAM trips, but harder convergence checking and
bound by 48 KB shared memory per SM on sm_89). Justify the
temporal-blocking choice and explain the shared-memory
budgeting: TILE=16, T=4 → halo=20² = 400 floats × 2 buffers
(ping-pong) = 3200 bytes per block, well under 48 KB.
3.4 Running it end-to-end — present a results table showing
compute cycles, wall-to-wall cycles (including memcpy),
average iteration count, effective CPE both with and without
memory transfer time, for N=2048. The headline numbers from
my measurements (cite §INPUTS): GPU baseline 0.015 CPE (186×
vs. serial), GPU temporal 0.009 CPE (310×).
3.5 Discussion of Results — be honest if results are
surprising. Things to dig into: (a) the GPU temporal/baseline
ratio is only ~1.7×, not the order-of-magnitude we got on
CPU. Why? The L40S has roughly 864 GB/s HBM-class bandwidth
and very large L2 (96 MB on Ada AD102, slightly less on
L40S), so the GPU baseline is already mostly hitting cache,
leaving less for temporal blocking to recover. (b) FP32 vs
FP64 — the roofline shifts; quote what we used. (c) The
temporal kernel needs __syncthreads() after every sub-step,
which serializes warps within a block; quantify roughly. The
Weinberg report opens this section with "Oi! Well those
results aren't good at all" — match that candor where the
numbers actually disappoint, and resist the urge to spin
impressive speedups (310× sounds great but the apples-to-
apples comparison is GPU-temporal vs. OMP-temporal, which is
~37×, not 310×).


Closing Thoughts — 4-6 paragraphs reflecting on:

what worked (temporal blocking on the bandwidth-bound CPU
regime, GPU shared-memory tiling)
what didn't (serial-CPU temporal only got 1.08× — overhead
ate most of the savings; OMP baseline at 0.667 CPE was
actually worse than pthreads baseline at 0.399 CPE,
suggesting the OMP scheduler / false-sharing story isn't
free)
precision/numerical issues (FP32 accumulation in long sweeps,
ω drift, red-black ordering changing the effective spectral
radius)
3-4 concrete future directions (diamond/hexagonal time
skewing, multi-GPU with halo-exchange, mixed FP16/FP32 in
the inner sweep with FP32 accumulator, comparison to a
proper multigrid solver as a sanity check on "iterations to
tolerance").


Compilation instructions — exact gcc and nvcc commands with all
flags (-O3 -march=native -ffast-math -fopenmp for CPU;
-O3 -arch=sm_89 --use_fast_math for GPU on the L40S), plus a
list of preprocessor knobs (#defines) the reader can flip:
N, B (block size), T (temporal halo), OMEGA, TOL, ITERS,
USE_REDBLACK, USE_TEMPORAL, USE_FP64.
List of Files — bullet list of every source file with a one-line
description (test_SOR.c, test_SOR_OMEGA.c, test_SOR_part4.c,
the GPU .cu files, etc.).

================================================================
TONE AND VOICE
This is the most important part. The Weinberg report is technical
but personal — match these specific qualities:

First person plural ("we") for technical work, first person
singular ("I") for design decisions. "We computed" but "I had
a CPE of 1.17 for my matrix transpose lab."
Casual interjections that don't undercut rigor. Examples from
Weinberg: "wow is it close!", "Oi! Well those results aren't
good at all", "(Hopefully!)", "though it's not the most
significant saving", "I was bored one day, but besides getting
it to work I didn't want to pursue it further".
Honest negative framing. When something doesn't work, say so
directly and investigate. Phrases like "It's unfortunate that
X isn't better than Y, but wow is it close!" or "this isn't
too surprising for larger matrices because..."
Walk-through reasoning. Don't just state conclusions. Show
the chain of thought: "The reason for this is likely directly
tied with X. When considering Y, the entire Z can be loaded
into L1 cache, while there's no hope for W to fit even in L3."
Hedge appropriately. "If I could beg a guess as to why this
made a difference, it could be either A, but more likely it
could be B." This is how working engineers actually talk
about uncertain results.
Explicit forward references. "We'll discuss this soon!"
"More on this when we get to results." Treat the reader as a
fellow engineer following along.
Easter eggs and asides are encouraged. The Weinberg report
mentions commented-out code "as a fun easter egg" — this
humanizes the report.

================================================================
CONTENT REQUIREMENTS

Every quantitative claim gets a number. CPE values, FLOP counts,
cycle counts, iteration counts, problem sizes. Tables wherever
more than 2 numbers exist.
At least 3 plots/figures. For each: describe what's on the axes,
what the data shows, and what the curve fit (if any) suggests.
When fits fail, say so. Required figures at minimum:
(1) roofline diagram with the 0.75 FLOP/byte arithmetic
intensity marked,
(2) CPE vs N (size sweep, baseline vs temporal),
(3) CPE vs B (block size sweep at N=2048),
(4) CPE vs T (halo sweep at N=2048, B=128),
(5) GPU CPE vs (tile, T) heatmap or family of curves.
Every optimization technique gets justified by hardware:
"blocking didn't help because the working set already fits in
L1", "prefetching helped at large N because of bandwidth
limits", "temporal blocking helped at N≥2048 because the
N²·float32 working set spilled out of L2", etc. Connect
software choices to memory hierarchy explicitly. Specifically
reason about:

Xeon Gold 6426Y: L1 = 80 KB / core, L2 = 2 MB / core,
L3 = 37.5 MB shared, 8 threads used.
Per-thread L2 budget: 2 MB. A (B+2T)² float32 tile at
B=128, T=8 is 144² × 4B ≈ 83 KB — fits L1 comfortably.
At N=2048, the full grid is 2048² × 4B = 16 MB — fits L3
but blows L2; this is exactly the regime where temporal
blocking should and does win.
At N=4096, full grid is 64 MB — spills L3, becomes purely
DRAM-bandwidth-bound.


Show real code in the body of the report. Not pseudocode.
Pull from test_SOR.c, test_SOR_part4.c, and the .cu file.
Include comments that map code lines to algorithmic steps.
A worked numerical example for SOR — pick a 5×5 grid with
fixed boundary = 1, interior init = 0, ω = 1.6 — and walk
through 2-3 sweeps explicitly, showing the actual numbers.
Identify and explain at least one negative or surprising
result. Several candidates from the measurements:
(a) serial-CPU temporal only beats serial-CPU baseline by
1.08× — basically no win. Why? Per-tile scratch malloc
+ redundant halo work cancel the bandwidth savings on
a single core where DRAM is already lightly loaded.
(b) OMP-8t baseline (0.667 CPE) is worse than pthreads-8t
baseline (0.399 CPE). Diagnose: scheduler overhead /
false sharing on the dst-row write?
(c) Temporal blocking loses at N ≤ 512 — the small-N
regime fits L2 even without blocking, so we pay
overhead for nothing.
Don't hide them. Investigate.
Connect to numerical precision when relevant. We use FP32. With
~10⁻⁷ unit roundoff, after ~10⁴ stencil ops per cell over many
sweeps the error magnitude is bounded by something like
10⁻⁷ × √(iters · 5) — for iters=10⁴ that's ~10⁻⁵, comfortably
below TOL=10⁻⁵. But tighter tolerances or larger N would push
us to FP64. Note this.

================================================================
FORMATTING

LaTeX (article class), single column, 22 pages target length.
Section numbering as shown in the structure above.
Tables formatted with a small caption beneath each ("Table 1.
This table shows the CPEs for...")
Figures formatted with caption beneath ("Figure 1. A plot of...")
Code blocks use \begin{verbatim} or \texttt{} for inline.
Math in proper display equations where appropriate.

================================================================
WHAT TO AVOID

Don't write a marketing-style abstract. Weinberg has none —
the report opens directly with "Description of the Algorithm".
Don't sanitize the negatives. If something didn't work,
diagnose why; don't paper over it.
Don't over-format. Subsection headings, tables, and figures
carry the structure. Avoid bulleted lists for everything.
Don't write hype phrases like "we present a novel approach"
or "state-of-the-art". This is a course project; frame it
like one.
Don't claim more than you measured. If you didn't run it at
N=10000, don't claim it scales to N=10000.
Don't quote 310× without immediately contextualizing it as
GPU-temporal-FP32 vs. serial-CPU-baseline-FP32; the
apples-to-apples GPU-vs-OMP-temporal comparison is the
honest one.

================================================================
PROJECT-SPECIFIC INPUTS (already collected — use these directly,
don't re-ask)

Algorithm and big-O

Kernel: 5-point Laplacian stencil, applied as SOR
(Gauss-Seidel + relaxation parameter ω) or red-black SOR
for parallel/temporal-blocked variants.
Per-sweep work: O(N²) cells × 5–7 FLOPs/cell.
Iterations to tolerance for optimally-tuned SOR on the 2D
Laplacian: O(N) (vs. O(N²) for Jacobi/Gauss-Seidel).
Total work: O(N³) for tuned SOR, O(N⁴) for naive Jacobi.
Arithmetic intensity: ≈ 0.75 FLOP/byte (memory-bandwidth
bound).


Hardware

CPU: Intel Xeon Gold 6426Y, 8 threads used.

L1 = 80 KB per core
L2 = 2 MB per core
L3 = 37.5 MB shared
CPNS = 2.0 (cycles per nanosecond) — used in CPE
calculation in the timing harness.


GPU: NVIDIA L40S, 46 GB, sm_89 (Ada Lovelace).

CUDA 12.8.
48 KB shared memory per block (default), 96 MB L2 on the
part.


Precision: FP32 throughout (single-precision float).
FP64 considered as future direction.


Serial-CPU CPE measurements (representative, N=2048 unless
noted)

serial-CPU baseline (single thread, no blocking): 2.792 CPE
serial-CPU temporal: 2.586 CPE  (1.08× speedup — barely)
pthreads-8t baseline: 0.399 CPE  (7.0× vs serial)
OMP-8t baseline: 0.667 CPE  (4.2× vs serial — note this is
worse than pthreads)
OMP-8t temporal: 0.332 CPE  (8.7× vs serial)
GPU baseline (kernel-per-sweep): 0.015 CPE  (186× vs serial)
GPU temporal: 0.009 CPE  (310× vs serial)

Size sweep (config: iters=64, B=128, T=8, threads=8):
N=256:  baseline ≈ ?, temporal ≈ ?  (temporal loses here)
N=1024: baseline ≈ 0.45, temporal ≈ 0.50 (still loses)
N=4096: baseline ≈ 1.4, temporal ≈ 0.55 (clear win)
(exact bar-chart values from the size-sweep figure;
temporal-loses-at-small-N is the key observation.)
Block-size sweep (N=2048, T=4, iters=64, threads=8):
B ∈ {16, 32, 64, 128, 256, 512}; baseline is essentially
flat at ~0.5 CPE (spatial blocking alone doesn't help);
temporal has a shallow optimum near B=128 dipping toward
~0.4 CPE.
Halo / T sweep (N=2048, B=128, iters=96, threads=8):
T   baseline_CPE   temporal_CPE   ratio (temp/base)
1   0.504          0.739          1.47× (temporal worse)
2   0.469          0.487          1.04×
3   0.494          0.424          0.86×
4   0.464          0.393          0.85×
6   0.448          0.360          0.80×
8   0.444          0.332          0.75× (best ratio)
12  0.469          0.348          0.74×
16  0.412          0.378          0.92×
24  0.418          0.393          0.94×
32  0.401          0.461          1.15× (work multiplier
eats the win)
GPU tuning sweep (N=2048, iters=96, on L40S)

Vocabulary note: CPU "thread" ↔ CUDA "block";
"block of work" ↔ CUDA "tile". Tell the reader.
Sweep grid: tile ∈ {8, 16, 32}, T ∈ {1, 2, 3, 4, 6, 8, 12}.
Selected results (CPE):
tile=8,  T=1:  ≈0.030
tile=8,  T=2:  ≈0.034
tile=8,  T=3:  ≈0.089 (sharp regression — likely
shared-memory bank conflicts or
occupancy cliff; investigate)
tile=16, T=1:  ≈0.030
tile=16, T=4:  ≈0.010 (near-optimum)
tile=16, T=6:  ≈0.014
tile=32, T=1–4: ≈0.010 across the board, then rises
slightly at T=8, T=12
overall best: tile=16, T=4 at CPE ≈ 0.009 (this is the
headline 310× number).


Failure modes and surprises (call these out explicitly)

Serial-CPU temporal barely wins (1.08×). Per-tile scratch
malloc inside the temporal kernel is a real overhead.
Temporal blocking is worse than baseline for N ≤ 512.
OMP baseline is worse than pthreads baseline. Plausibly
scheduler overhead, false sharing on dst rows, or first-
touch policy issues.
GPU temporal-vs-baseline ratio is only ~1.7× — much smaller
than CPU's, because the L40S already has so much L2/HBM
bandwidth that the baseline isn't truly DRAM-starved.
SOR temporal blocking strictly only works on the red-black
coloring or the Jacobi form; sequential row-major SOR has
loop-carried dependencies that forbid it. Acknowledge this
explicitly in §1.3.


Real code snippets to include

From test_SOR.c: the inner SOR loop with the ω·(neighbor
sum)/4 update.
From test_SOR_part4.c: the blocked / temporal-blocked
variant with the (B+2T)² scratch buffer.
From the GPU .cu source: the shared-memory load + halo
exchange + T-step inner loop.



================================================================
START
You have everything you need above. Begin by drafting Section 1
(Description of the Algorithm), and we'll iterate from there. Do
NOT ask follow-up input questions — every answer is in the
PROJECT-SPECIFIC INPUTS section. If you find a number missing,
flag it inline as "[TODO: measure]" rather than fabricating one.