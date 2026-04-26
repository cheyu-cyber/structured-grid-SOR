# Experiment execution guide — full-bench / single-GPU machine

**Target machine**: 8 physical cores, 1 GPU (CUDA), Linux.
**Workflow**: small, independent experiments.  One experiment per Claude
session.  After each experiment, raw data + a short notes file + (optional)
plot are saved under `results/exp/E0N_<name>/`, and `results/STATUS.md`
gets one line updated.

The session-local rule: **don't try to run all experiments in one session.**
Reading and following ONE experiment block end-to-end is the unit of work.

---

## How Claude should run this

When the user says **"run E04"** (or similar):

1. Open this file, scroll to that experiment block.  Don't reread other
   blocks.
2. Run the bash block exactly as written.  If a command fails, fix the
   smallest possible thing and continue — don't redesign the experiment.
3. After the run finishes, fill in `results/exp/E0N_<name>/notes.md`
   from the template at the bottom of the experiment block.
4. Update the line for E0N in `results/STATUS.md`.
5. Tell the user one sentence: "E0N done — <findings>" and stop.  Do
   NOT proceed to the next experiment.

If the user says **"run E04 then E05 then E06"** (multi-batch), still
do them one at a time, updating status between each.  Do not run the
whole queue silently.

---

## E00 — Setup (run once on the new machine)

**Goal**: build everything, install python deps, verify smoke.
**Time**: ~3 min.

```bash
cd ~/src/structured-grid-SOR

# 1. Build CPU/OMP/pthreads.
make clean
make
make gpu                                    # builds sor2d_gpu, sor3d_gpu

# 2. Python deps (only if not already there).
if [ ! -d ~/.venv ]; then python3 -m venv ~/.venv; fi
~/.venv/bin/pip install --quiet pandas matplotlib numpy

# 3. Smoke.
make smoke

# 4. Save system info.
mkdir -p results/exp/E00_setup
{
    echo "=== uname"; uname -a
    echo "=== nproc"; nproc
    echo "=== /proc/cpuinfo (model + cache)"
    grep -E 'model name|cache size' /proc/cpuinfo | sort -u
    echo "=== free -h"; free -h
    echo "=== nvidia-smi"
    nvidia-smi || echo "no nvidia-smi"
    echo "=== nvcc --version"
    nvcc --version || echo "no nvcc"
} > results/exp/E00_setup/info.txt 2>&1

cp results/exp/E00_setup/info.txt results/exp/E00_setup/notes.md
```

**Quick checks**:
- `make` produces 9 CPU/OMP/pthread binaries in `build/`.
- `make gpu` produces `build/sor2d_gpu` and `build/sor3d_gpu`.
- `make smoke` reports `max|...| = 0.0000e+00` on every line — bit-
  identical against serial reference.
- `~/.venv/bin/python -c "import pandas, matplotlib, numpy; print('ok')"`
  prints `ok`.

**STATUS update**: `E00: done`

---

## E01 — Strong scaling, 2D OpenMP

**Goal**: speedup vs thread count at fixed N, baseline + temporal.
Plots the canonical "is parallelism working" curve.
**Variable**: threads ∈ {1, 2, 4, 8, 16}.  16 is oversubscribed
(2× cores) — included to show the regime where scaling breaks.
**Fixed**: N=4098, iters=64, B=128, T=8.
**Time**: ~3 min.

```bash
DIR=results/exp/E01_strong_scaling_2d
mkdir -p $DIR
> $DIR/data.csv

for nt in 1 2 4 8 16; do
    OMP_NUM_THREADS=$nt build/sor2d_omp 4098 64 128 8 \
        | tee $DIR/raw_nt${nt}.txt
done

# Pull out (baseline, temporal) time + compute CPE for each thread count.
{
    echo "threads,kind,time_s,cpe"
    for nt in 1 2 4 8 16; do
        awk -v nt=$nt -v N=4098 -v iters=64 '
            /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
                cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
                print nt",baseline,"t","cpe }
            /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
                cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
                print nt",temporal,"t","cpe }
        ' $DIR/raw_nt${nt}.txt
    done
} > $DIR/data.csv
```

**Quick checks**:
- Speedup at 8 threads should be 4-7× for both kinds (DRAM-bound 2D
  doesn't reach 8×).
- Temporal speedup-over-baseline should grow with thread count.
- 16 threads should be no faster than 8 (oversubscription).

**Notes template** for `$DIR/notes.md`:
```
# E01 — strong scaling 2D OpenMP

Machine: <from E00>
Config: N=4098, iters=64, B=128, T=8

Speedup table (vs 1 thread):
| threads | baseline | temporal |
|---------|----------|----------|
| 1       | 1.00x    | <fill>   |
| 2       | <fill>   | <fill>   |
| 4       | <fill>   | <fill>   |
| 8       | <fill>   | <fill>   |
| 16      | <fill>   | <fill>   |

Observation: <one paragraph>
Surprise: <if any>
```

**STATUS update**: `E01: done — <best speedup>x at <threads> threads`

---

## E02 — Size sweep at fixed threads (2D, cache-regime story)

**Goal**: how throughput changes from L2-fit to DRAM-bound.
**Variable**: N ∈ {256, 512, 1024, 2048, 4098}.
**Fixed**: threads = max useful (8 cores; the cap from E01).
**Time**: ~5 min.

```bash
DIR=results/exp/E02_size_sweep_2d
mkdir -p $DIR
> $DIR/data.csv
echo "N,kind,time_s,cpe,bytes_pair_mb" > $DIR/data.csv

for N in 256 512 1024 2048 4098; do
    OMP_NUM_THREADS=8 build/sor2d_omp $N 64 128 8 \
        | tee $DIR/raw_N${N}.txt

    bytes=$(awk -v N=$N 'BEGIN{printf "%.1f", 2 * N * N * 4 / (1024*1024)}')
    awk -v N=$N -v b=$bytes -v iters=64 '
        /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print N",baseline,"t","cpe","b }
        /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print N",temporal,"t","cpe","b }
    ' $DIR/raw_N${N}.txt >> $DIR/data.csv
done
```

**Quick checks**:
- Look up L3 size in `results/exp/E00_setup/info.txt`.  Bytes_pair_mb
  for N=2048 = 32 MB; mark which N first exceeds L3.
- Temporal/baseline ratio should grow as N exceeds L3 (the headline
  story for temporal blocking).

**Notes template** for `$DIR/notes.md`:
```
# E02 — size sweep, 2D OMP at 8 threads

L3 size: <from cpuinfo>
Crossover (baseline goes off-cache) at: N=<>
Crossover (temporal beats baseline by 2x): N=<>

Observation: <paragraph>
```

**STATUS update**: `E02: done — temporal beats baseline by Nx at N=<>`

---

## E03 — Spatial block size B (sweet spot)

**Goal**: where blocking helps and where it stops helping.
**Variable**: B ∈ {16, 32, 64, 128, 256, 512}.  16 is sub-L1, 512 fills
or exceeds L2 per thread.
**Fixed**: N=2048, iters=64, T=4, threads=8.
**Time**: ~5 min.

```bash
DIR=results/exp/E03_block_size
mkdir -p $DIR
echo "B,kind,time_s,cpe" > $DIR/data.csv

for B in 16 32 64 128 256 512; do
    # iters must be multiple of T=4 (yes for 64).
    OMP_NUM_THREADS=8 build/sor2d_omp 2048 64 $B 4 \
        | tee $DIR/raw_B${B}.txt
    awk -v B=$B -v N=2048 -v iters=64 '
        /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print B",baseline,"t","cpe }
        /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print B",temporal,"t","cpe }
    ' $DIR/raw_B${B}.txt >> $DIR/data.csv
done
```

**Quick checks**:
- Baseline rows should be ~constant (B has no effect on baseline; it's
  only used by the temporal kernel).  If they vary much, system noise.
- Temporal should peak somewhere in {64, 128, 256} and degrade outside.

**Notes template**: same shape as above; record best B and its CPE.

**STATUS update**: `E03: done — best B=<>, min CPE=<>`

---

## E04 — Temporal depth T (halo-overhead crossover)

**Goal**: at what T does halo redundant work cancel the DRAM savings?
**Variable**: T ∈ {1, 2, 4, 8, 16}.  iters must be a multiple of T —
use iters = 64.
**Fixed**: N=2048, B = best from E03 (default 128 if E03 not run yet),
threads=8.
**Time**: ~3 min.

```bash
DIR=results/exp/E04_temporal_depth
mkdir -p $DIR
B=128                                       # update from E03 if known
echo "T,kind,time_s,cpe" > $DIR/data.csv

for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=8 build/sor2d_omp 2048 64 $B $T \
        | tee $DIR/raw_T${T}.txt
    awk -v T=$T -v N=2048 -v iters=64 '
        /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print T",baseline,"t","cpe }
        /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
            cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
            print T",temporal,"t","cpe }
    ' $DIR/raw_T${T}.txt >> $DIR/data.csv
done
```

**Quick checks**:
- T=1 temporal should match baseline (one sweep per super-step → no
  reuse).
- T should peak at 4 or 8 and degrade at 16 (halo dominates).

**STATUS update**: `E04: done — best T=<> at B=<>`

---

## E05 — Pthreads decomposition study (Lab-5-Part-4 redo)

**Goal**: strip vs interleaved vs 2D-block; persistent vs spawn-per-sweep.
**Variable**: mode × sched × threads.
**Fixed**: N=2048, iters=64.
**Time**: ~10 min (24 runs × ~25s).

```bash
DIR=results/exp/E05_pth_decomp
mkdir -p $DIR
echo "N,threads,mode,sched,time_s,cpe" > $DIR/data.csv

for nt in 1 2 4 8; do
    for mode in strip interleaved block; do
        for sched in persistent spawn; do
            tag="N2048_nt${nt}_${mode}_${sched}"
            build/sor2d_pth_decomp 2048 64 \
                --threads $nt --mode $mode --sched $sched \
                | tee $DIR/raw_${tag}.txt
            grep '^CSV,' $DIR/raw_${tag}.txt | \
                awk -F, -v nt=$nt -v m=$mode -v s=$sched \
                  '{print $4","nt","m","s","$9","$10}' \
                >> $DIR/data.csv
        done
    done
done
```

**Quick checks**:
- At 8 threads, all 3 decomp modes should be within 20% of each other
  for `persistent` (Lab-5-Part-4 finding: hstrip slightly wins at large
  N).
- `spawn` should be 1.5–3× *slower* than `persistent` at every config —
  that gap is the per-sweep `pthread_create` cost.

**STATUS update**: `E05: done — best decomp=<> at 8t; spawn overhead=<>x`

---

## E06 — 3D OMP partitioning (slab / pencil / cube)

**Goal**: which 3D partitioning wins as N grows.
**Variable**: mode × N × threads.
**Fixed**: iters=20.  B=32 only matters for cube.
**Time**: ~10 min.

```bash
DIR=results/exp/E06_3d_partition
mkdir -p $DIR
echo "N,threads,mode,time_s,cpe" > $DIR/data.csv

for N in 66 130 258; do
    for nt in 1 2 4 8; do
        for mode in slab pencil cube; do
            tag="N${N}_nt${nt}_${mode}"
            OMP_NUM_THREADS=$nt build/sor3d_omp_part $N 20 \
                --mode $mode --block 32 \
                | tee $DIR/raw_${tag}.txt
            grep '^CSV,' $DIR/raw_${tag}.txt | \
                awk -F, -v nt=$nt -v m=$mode \
                  '{print $4","nt","m","$9","$10}' \
                >> $DIR/data.csv
        done
    done
done
```

**Quick checks**:
- At small N (66) cube should lose (halo overhead dominates).
- At large N (258) and high threads, slab and pencil should be comparable.
- Cube should *never* beat slab in baseline.  If it does, you're
  cache-thrashing in slab — note it.

**STATUS update**: `E06: done — winner at N=258, 8t = <mode>`

---

## E07 — Pthread spawn overhead (Lab-6-Part-1b for SOR)

**Goal**: extract the per-sweep `pthread_create` cost as a number.
**Variable**: iters ∈ {1, 4, 16, 64, 256}.  Fit `time = a + b·iters`,
where `a` is the per-launch overhead, `b` the per-sweep work.
**Fixed**: N=512 (small, so per-sweep work is small and overhead is
visible), threads=4, mode=strip.
**Time**: ~3 min.

```bash
DIR=results/exp/E07_spawn_overhead
mkdir -p $DIR
echo "iters,sched,time_s" > $DIR/data.csv

for iters in 1 4 16 64 256; do
    for sched in persistent spawn; do
        tag="iters${iters}_${sched}"
        # iters is a multiple of 1 trivially; spawn handles any iters.
        build/sor2d_pth_decomp 512 $iters \
            --threads 4 --mode strip --sched $sched \
            | tee $DIR/raw_${tag}.txt
        t=$(grep '^CSV,' $DIR/raw_${tag}.txt | awk -F, '{print $9}')
        echo "${iters},${sched},${t}" >> $DIR/data.csv
    done
done
```

**Quick checks**:
- `persistent` line: y = a_p + b·iters; `a_p` should be ~50-100 µs
  (one pthread_create + one pthread_join).
- `spawn` line: y = a_s + b'·iters; `b'` is the slope, `a_s` near 0.
- The slope difference (`b' - b`) is the per-sweep pthread_create cost.
  Expected: ~30 µs per sweep at 4 threads.

**STATUS update**: `E07: done — pthread_create cost ~<>µs/sweep`

---

## E08 — Omega U-curve (red-black GS)

**Goal**: replicate Lab 5's classic iterations-vs-omega U-curve.
**Variable**: omega ∈ [0.50, 1.99] in steps of 0.02 (75 points;
`sor2d_rb --sweep` does it).
**Fixed**: N=128 (theoretical `omega_opt ≈ 1.95`).
**Time**: ~1 min.

```bash
DIR=results/exp/E08_omega_sweep
mkdir -p $DIR
build/sor2d_rb 128 --sweep > $DIR/data.csv
```

**Quick checks**:
- File has 75 rows: `omega, iters_to_converge, seconds`.
- `iters_to_converge` should be a U-curve, minimum near
  `omega = 2 / (1 + sin(π/(N-1)))`.  For N=128: `omega_opt ≈ 1.951`.
- Above `omega = 2`, divergence (binary should refuse / cap at MAX_ITERS).

**Plot** (optional):
```bash
~/.venv/bin/python -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('$DIR/data.csv', header=None,
                 names=['omega','iters','sec'])
plt.plot(df['omega'], df['iters'])
plt.xlabel('omega'); plt.ylabel('iterations to converge')
plt.title('Omega U-curve, N=128')
plt.grid(alpha=0.3)
plt.savefig('$DIR/omega_ucurve.png', dpi=140)
"
```

**STATUS update**: `E08: done — omega_opt ~<> with <> iters`

---

## E09 — GPU TILE / HALO sweep

**Goal**: (TILE, HALO) heatmap for the 2D GPU temporal kernel.
**Variable**: TILE ∈ {16, 32}, HALO ∈ {2, 4, 6}.  iters=96 (multiple
of all three).
**Fixed**: N=2048.  Constraint: 2*HALO < TILE.
**Time**: ~2 min.

```bash
DIR=results/exp/E09_gpu_th_sweep
mkdir -p $DIR
echo "tile,halo,kind,time_s,cpe" > $DIR/data.csv

for tile in 16 32; do
    for halo in 2 4 6; do
        # Skip degenerate combos.
        inter=$((tile - 2 * halo))
        if [ $inter -le 0 ]; then continue; fi
        if [ $((96 % halo)) -ne 0 ]; then continue; fi
        tag="T${tile}_H${halo}"
        build/sor2d_gpu 2048 96 --tile $tile --halo $halo \
            | tee $DIR/raw_${tag}.txt
        grep '^CSV,' $DIR/raw_${tag}.txt | \
            awk -F, -v T=$tile -v H=$halo \
              '{print T","H","$7","$9","$10}' \
            >> $DIR/data.csv
    done
done
```

**Quick checks**:
- Best CPE is typically at TILE=32, HALO=4 (matches prior session
  defaults).
- Higher HALO at fixed TILE helps until the halo:interior ratio
  dominates.

**STATUS update**: `E09: done — best (TILE,HALO)=<>, min CPE=<>`

---

## E10 — GPU baseline vs temporal at multiple N

**Goal**: when does the temporal kernel beat the baseline.
**Variable**: N ∈ {258, 1026, 2050, 4098} (note odd-by-1 to avoid
power-of-2 cache aliasing).
**Fixed**: iters=96, TILE=32, HALO=4 (defaults).
**Time**: ~2 min.

```bash
DIR=results/exp/E10_gpu_n_sweep
mkdir -p $DIR
echo "N,kind,time_s,cpe" > $DIR/data.csv

for N in 258 1026 2050 4098; do
    tag="N${N}"
    build/sor2d_gpu $N 96 | tee $DIR/raw_${tag}.txt
    grep '^CSV,' $DIR/raw_${tag}.txt | \
        awk -F, -v N=$N '{print N","$7","$9","$10}' \
        >> $DIR/data.csv
done

# Same for 3D GPU.
echo "N,kind,time_s,cpe" > $DIR/data_3d.csv
for N in 66 130 258; do
    tag="3d_N${N}"
    build/sor3d_gpu $N 16 | tee $DIR/raw_${tag}.txt   # iters=16 (mult of HALO=2)
    grep '^CSV,' $DIR/raw_${tag}.txt | \
        awk -F, -v N=$N '{print N","$7","$9","$10}' \
        >> $DIR/data_3d.csv
done
```

**Quick checks**:
- 2D temporal speedup vs baseline grows with N (from launch-overhead-
  dominated at N=258 to reuse-dominated at N=4098).
- 3D temporal may be *slower* than 3D baseline at every N — that's the
  TILE=8, HALO_T=2 halo overhead and is the legitimate finding (see
  `gpu_verify_todo.md`).

**STATUS update**: `E10: done — 2D temp/base = <>x at N=4098; 3D temp/base = <>x at N=258`

---

## E11 — Cross-tier headline comparison

**Goal**: one chart that puts every variant on the same axes at one
representative N.  This is the figure for the report's "summary"
section.
**Variable**: tier (single-thread / pthreads / OMP / GPU) × kind
(baseline / temporal).
**Fixed**: N=2050, iters=96 (matches the GPU side; CPU runs accept
this).  Threads = 8.
**Time**: ~5 min.

```bash
DIR=results/exp/E11_headline
mkdir -p $DIR

# Order: serial 2D, OMP 2D base+temp, pthreads-temporal 2D base+temp, GPU 2D base+temp.
echo "tier,kind,time_s,cpe" > $DIR/data.csv

# 1. Serial CPU.
build/sor2d_cpu 2050 96 128 8 | tee $DIR/raw_cpu.txt
awk -v N=2050 -v iters=96 '
    /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
        cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
        print "serial,baseline,"t","cpe }
    /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
        cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
        print "serial,temporal,"t","cpe }
' $DIR/raw_cpu.txt >> $DIR/data.csv

# 2. OMP at 8 threads.
OMP_NUM_THREADS=8 build/sor2d_omp 2050 96 128 8 | tee $DIR/raw_omp.txt
awk -v N=2050 -v iters=96 '
    /baseline[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
        cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
        print "omp,baseline,"t","cpe }
    /temporal[[:space:]]*:/ { for(i=1;i<=NF;i++) if($i==":") t=$(i+1);
        cpe = t * 2.0 * 1e9 / ((N-2)*(N-2)*iters);
        print "omp,temporal,"t","cpe }
' $DIR/raw_omp.txt >> $DIR/data.csv

# 3. Pthreads temporal at 8 threads.
build/sor2d_pth_temporal 2050 96 128 8 --threads 8 | tee $DIR/raw_pth.txt
grep '^CSV,' $DIR/raw_pth.txt | awk -F, '{print "pthreads,"$7","$9","$10}' >> $DIR/data.csv

# 4. GPU.
build/sor2d_gpu 2050 96 | tee $DIR/raw_gpu.txt
grep '^CSV,' $DIR/raw_gpu.txt | awk -F, '{print "gpu,"$7","$9","$10}' >> $DIR/data.csv

cat $DIR/data.csv
```

**Quick checks**:
- Order of magnitude expected (CPE, lower=faster, depending on machine):
  serial CPE ~1–2 / OMP-8 ~0.1–0.2 / pthreads-8 ~0.1–0.2 / GPU baseline ~0.02–0.05
  / GPU temporal ~0.01–0.02 (these are rough — depends on machine and CPNS).
- Temporal on every tier should beat the baseline of that tier (or
  match, single-threaded — and that's the reportable "single-thread
  CPU isn't DRAM-bound enough for temporal to help" finding).

**STATUS update**: `E11: done — fastest = <tier> <kind> at min CPE=<>`

---

## E12 — Convergence vs iteration (residual ‖r‖₂)

**Goal**: prove the temporal kernels reach the same fixed point as the
baseline at the same iteration count.  This is the test that
distinguishes "correct fast algorithm" from "correct algorithm,
different fixed point."
**Status**: **NOT READY** — no binary currently emits residual.  Skip
unless someone adds a residual-tracking binary.  Suggested impl:
modify `sor2d_cpu.c` to compute `‖dst − src‖∞` after each sweep and
print it.  Then run baseline and temporal at the same (N, iters) and
compare the residual trajectories.

If implementing, the experiment becomes:
```bash
DIR=results/exp/E12_convergence
mkdir -p $DIR
build/sor2d_cpu_residual 512 256 64 8 > $DIR/data_baseline.csv
build/sor2d_omp_residual 512 256 64 8 > $DIR/data_temporal.csv
# data_*.csv is iter,residual; plot both.
```

**STATUS update**: `E12: skipped (binary does not track residual)`

---

## After all experiments — synthesis (one final session)

**Goal**: combine per-experiment notes into a "Results" section in
`report.md`, with the right figure references.
**Time**: ~10 min.

Steps:

1. Copy per-experiment data CSVs into one canonical location:
   ```bash
   cat results/exp/E*/data*.csv > results/all_experiments.csv
   ```
2. Re-run `make plots` so figures reflect all data.
3. Read each `results/exp/E*/notes.md` and write a paragraph in
   `report.md` under a new `## Results — full bench` section.  Use
   one figure per experiment, captioned with the finding.
4. Update `STATUS.md` with the final summary line.

---

## STATUS.md template (run E00 first to seed it)

```bash
cat > results/STATUS.md <<'EOF'
# Experiment status

| ID | Name                                | Status   | Finding |
|----|-------------------------------------|----------|---------|
| E00| Setup                               | not run  |         |
| E01| Strong scaling 2D OMP               | not run  |         |
| E02| Size sweep 2D                       | not run  |         |
| E03| Spatial block size B                | not run  |         |
| E04| Temporal depth T                    | not run  |         |
| E05| Pthreads decomposition              | not run  |         |
| E06| 3D OMP partitioning                 | not run  |         |
| E07| Pthread spawn overhead              | not run  |         |
| E08| Omega U-curve                       | not run  |         |
| E09| GPU TILE/HALO sweep                 | not run  |         |
| E10| GPU baseline vs temporal vs N       | not run  |         |
| E11| Cross-tier headline                 | not run  |         |
| E12| Convergence (skipped)               | skipped  | no residual binary |
EOF
```

After each experiment, replace the row with:
```
| E0N| <name> | done | <one-line finding> |
```

---

## Iteration policy

- One experiment per session.  After Claude finishes one experiment,
  it should report results in 1-2 sentences and stop.
- If an experiment fails (e.g. a binary crashes), Claude should fix the
  smallest possible thing (a typo, a missing arg) and retry once.  If
  it still fails, mark `STATUS` as `failed: <reason>` and stop.
- Don't change experiment definitions in this file mid-run.  If an
  experiment's design needs to change, write down the change and run
  the new design as `E0N+12` rather than rewriting `E0N`.
- Don't add experiments without the user's go-ahead.  If a finding
  suggests a follow-up experiment, propose it (one sentence) and stop.
