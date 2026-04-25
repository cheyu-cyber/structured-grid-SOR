#!/usr/bin/env bash
#
# Big-grid parameter sweep harness.
#
# Drives every binary in build/ across (N, threads, mode, B, T) and
# accumulates a single results/results.csv with one row per run.
#
# Usage:
#   scripts/sweep.sh                  # full sweep
#   scripts/sweep.sh --quick          # smaller grid set, faster
#   scripts/sweep.sh --only=PATTERN   # only run binaries matching PATTERN
#                                     # (e.g. --only=pth_decomp)
#   scripts/sweep.sh --threads="1 4"  # override thread set
#   scripts/sweep.sh --sizes="512 2048" # override 2D size set
#
# Each binary's stdout is also tee'd to results/raw/<binary>.<run>.txt for
# debugging.  CSV rows come from two sources:
#   - the *_part / *_decomp binaries print a `CSV,...` line directly;
#     sweep.sh just greps and appends.
#   - existing binaries (sor2d_cpu, sor2d_omp, sor3d_cpu, sor3d_omp,
#     sor2d_pth, sor2d_gpu) print human-readable output; sweep.sh parses
#     it with awk and emits a synthesised CSV row.
#
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build"
RESULTS="$ROOT/results"
RAW="$RESULTS/raw"
CSV="$RESULTS/results.csv"

mkdir -p "$RAW"

# ----- defaults --------------------------------------------------------------
QUICK=0
ONLY=""
SIZES_2D="512 1024 2048 4098"
SIZES_3D="66 130 258"
THREADS="1 2 4 8 16"
ITERS_2D=64
ITERS_3D=20
B_2D=128
T_2D=8
B_3D=32

for arg in "$@"; do
    case $arg in
        --quick)            QUICK=1 ;;
        --only=*)           ONLY="${arg#--only=}" ;;
        --sizes=*)          SIZES_2D="${arg#--sizes=}" ;;
        --sizes-3d=*)       SIZES_3D="${arg#--sizes-3d=}" ;;
        --threads=*)        THREADS="${arg#--threads=}" ;;
        --iters=*)          ITERS_2D="${arg#--iters=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [ $QUICK -eq 1 ]; then
    SIZES_2D="512 1024"
    SIZES_3D="66 130"
    THREADS="1 4"
    ITERS_2D=32
fi

want() {
    [ -z "$ONLY" ] && return 0
    [[ "$1" == *"$ONLY"* ]]
}

# ----- helpers ---------------------------------------------------------------
# CSV columns:
#   binary, dim, N, iters, threads, mode, extra, time_s, gup_s, max_diff
echo "binary,dim,N,iters,threads,mode,extra,time_s,gup_s,max_diff" > "$CSV"

emit_csv_from_log() {
    # Forward any `CSV,...` lines the binary itself printed.
    grep '^CSV,' "$1" | sed 's/^CSV,//' >> "$CSV" || true
}

parse_existing() {
    # Args: log_path binary dim N iters threads kind extra
    # Existing binaries print:
    #   baseline : 0.6024 s  (5557.86 Mupdates/s, ...
    #   max|base-temp| = 1.2e-06   ...
    local log="$1" bin="$2" dim="$3" N="$4" iters="$5" thr="$6"
    local kind="$7" extra="$8"
    local label
    case "$kind" in
        baseline) label="baseline" ;;
        temporal) label="temporal" ;;
        pthread)  label="pthread"  ;;
        *)        label="$kind"    ;;
    esac
    # Extract `<label> : <time> s  (<gups> Mupdates/s,...`
    awk -v lbl="$label" -v bin="$bin" -v dim="$dim" -v N="$N" \
        -v iters="$iters" -v thr="$thr" -v kind="$kind" -v extra="$extra" '
        $0 ~ "^[ \t]+" lbl "[[:space:]]*:" {
            for (i=1; i<=NF; i++) {
                if ($i == ":") { t=$(i+1); }
                if ($i == "Mupdates/s,") { mu=$(i-1); }
            }
            gup = mu / 1000.0
            time_s = t
        }
        /max\|/ { for (i=1; i<=NF; i++) if ($i=="=") { d=$(i+1); break } }
        END {
            if (time_s != "" && gup != "") {
                printf "%s,%d,%s,%s,%s,%s,%s,%.6e,%.6e,%s\n",
                    bin, dim, N, iters, thr, kind, extra, time_s, gup, d
            }
        }' "$log" >> "$CSV"
}

run_log() {
    # Args: tag command...
    local tag="$1"; shift
    local log="$RAW/${tag}.txt"
    echo "  -> $tag"
    if "$@" > "$log" 2>&1; then
        emit_csv_from_log "$log"
    else
        echo "    !! failed; see $log" >&2
    fi
    echo "$log"
}

# ----- 1) pthreads decomposition study (new binary, prints CSV) --------------
if want sor2d_pth_decomp && [ -x "$BUILD/sor2d_pth_decomp" ]; then
    echo "[1/4] pthreads decomposition study"
    for N in $SIZES_2D; do
        for nt in $THREADS; do
            for mode in strip interleaved block; do
                for sched in persistent spawn; do
                    tag="pth_decomp_N${N}_nt${nt}_${mode}_${sched}"
                    run_log "$tag" \
                        "$BUILD/sor2d_pth_decomp" $N $ITERS_2D \
                        --threads $nt --mode $mode --sched $sched \
                        > /dev/null
                done
            done
        done
    done
fi

# ----- 2) 3D OMP partitioning study (new binary, prints CSV) -----------------
if want sor3d_omp_part && [ -x "$BUILD/sor3d_omp_part" ]; then
    echo "[2/4] 3D OpenMP partitioning study"
    for N in $SIZES_3D; do
        for nt in $THREADS; do
            for mode in slab pencil cube; do
                tag="3d_part_N${N}_nt${nt}_${mode}"
                OMP_NUM_THREADS=$nt run_log "$tag" \
                    "$BUILD/sor3d_omp_part" $N $ITERS_3D \
                    --mode $mode --block $B_3D \
                    > /dev/null
            done
        done
    done
fi

# ----- 3) Existing 2D / 3D OMP at varying thread counts ----------------------
if want sor2d_omp && [ -x "$BUILD/sor2d_omp" ]; then
    echo "[3/4] 2D OpenMP (baseline + temporal) thread-scaling"
    for N in $SIZES_2D; do
        for nt in $THREADS; do
            tag="2d_omp_N${N}_nt${nt}"
            log="$RAW/${tag}.txt"
            echo "  -> $tag"
            OMP_NUM_THREADS=$nt "$BUILD/sor2d_omp" $N $ITERS_2D $B_2D $T_2D \
                > "$log" 2>&1 || { echo "    !! failed" >&2; continue; }
            parse_existing "$log" sor2d_omp 2 $N $ITERS_2D $nt baseline ""
            parse_existing "$log" sor2d_omp 2 $N $ITERS_2D $nt temporal "B=$B_2D;T=$T_2D"
        done
    done
fi

if want sor3d_omp && [ -x "$BUILD/sor3d_omp" ]; then
    echo "[3/4] 3D OpenMP (baseline + temporal) thread-scaling"
    for N in $SIZES_3D; do
        for nt in $THREADS; do
            tag="3d_omp_N${N}_nt${nt}"
            log="$RAW/${tag}.txt"
            echo "  -> $tag"
            OMP_NUM_THREADS=$nt "$BUILD/sor3d_omp" $N $ITERS_3D $B_3D 2 \
                > "$log" 2>&1 || { echo "    !! failed" >&2; continue; }
            parse_existing "$log" sor3d_omp 3 $N $ITERS_3D $nt baseline ""
            parse_existing "$log" sor3d_omp 3 $N $ITERS_3D $nt temporal "B=$B_3D;T=2"
        done
    done
fi

# ----- 4) Existing single-thread CPU / pthread (strip-only) / GPU ------------
if want sor2d_cpu && [ -x "$BUILD/sor2d_cpu" ]; then
    echo "[4/4] 2D single-thread CPU"
    for N in $SIZES_2D; do
        tag="2d_cpu_N${N}"
        log="$RAW/${tag}.txt"
        echo "  -> $tag"
        "$BUILD/sor2d_cpu" $N $ITERS_2D $B_2D $T_2D \
            > "$log" 2>&1 || { echo "    !! failed" >&2; continue; }
        parse_existing "$log" sor2d_cpu 2 $N $ITERS_2D 1 baseline ""
        parse_existing "$log" sor2d_cpu 2 $N $ITERS_2D 1 temporal "B=$B_2D;T=$T_2D"
    done
fi

if want sor3d_cpu && [ -x "$BUILD/sor3d_cpu" ]; then
    echo "[4/4] 3D single-thread CPU"
    for N in $SIZES_3D; do
        tag="3d_cpu_N${N}"
        log="$RAW/${tag}.txt"
        echo "  -> $tag"
        "$BUILD/sor3d_cpu" $N $ITERS_3D $B_3D 2 \
            > "$log" 2>&1 || { echo "    !! failed" >&2; continue; }
        parse_existing "$log" sor3d_cpu 3 $N $ITERS_3D 1 baseline ""
        parse_existing "$log" sor3d_cpu 3 $N $ITERS_3D 1 temporal "B=$B_3D;T=2"
    done
fi

if want sor2d_pth && [ -x "$BUILD/sor2d_pth" ]; then
    echo "[4/4] 2D pthreads (strip-only, original)"
    for N in $SIZES_2D; do
        for nt in $THREADS; do
            tag="2d_pth_N${N}_nt${nt}"
            log="$RAW/${tag}.txt"
            echo "  -> $tag"
            "$BUILD/sor2d_pth" $N $ITERS_2D $nt \
                > "$log" 2>&1 || { echo "    !! failed" >&2; continue; }
            parse_existing "$log" sor2d_pth 2 $N $ITERS_2D $nt pthread "strip"
        done
    done
fi

if want sor2d_gpu && [ -x "$BUILD/sor2d_gpu" ]; then
    echo "[4/4] 2D CUDA GPU"
    for N in $SIZES_2D; do
        tag="2d_gpu_N${N}"
        log="$RAW/${tag}.txt"
        echo "  -> $tag"
        "$BUILD/sor2d_gpu" $N $ITERS_2D \
            > "$log" 2>&1 || { echo "    !! failed (no GPU?)" >&2; continue; }
        parse_existing "$log" sor2d_gpu 2 $N $ITERS_2D 0 baseline ""
        parse_existing "$log" sor2d_gpu 2 $N $ITERS_2D 0 temporal ""
    done
fi

# sor3d_gpu prints CSV lines directly; sweep.sh only needs to forward them.
if want sor3d_gpu && [ -x "$BUILD/sor3d_gpu" ]; then
    echo "[4/4] 3D CUDA GPU"
    for N in $SIZES_3D; do
        tag="3d_gpu_N${N}"
        run_log "$tag" "$BUILD/sor3d_gpu" $N $ITERS_3D > /dev/null
    done
fi

echo
echo "Wrote $CSV"
nrows=$(($(wc -l < "$CSV") - 1))
echo "  $nrows runs recorded"
