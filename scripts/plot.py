#!/usr/bin/env python3
"""
Render the four canonical charts from results/results.csv.

Headline metric: CPE (cycles per element, CPNS=2.0).  Smaller = faster.

Charts (each saved to results/figs/*.png):
  1. strong_scaling.png   Speedup vs. thread count, one line per (binary, N).
                          Reference y=x line included.
  2. variant_heatmap.png  Heatmap of min CPE by (variant, N).  Variants stack
                          serial / pthread / OpenMP / GPU; columns are N
                          regimes (fits-L2 / fits-L3 / exceeds-L3).
  3. decomp_compare.png   Pthreads decomposition study: bar chart of
                          CPE by (mode, schedule) at each N.  Direct
                          Lab-5-Part-4 redo.
  4. partition_3d.png     3D OMP partitioning study: CPE by (mode, N)
                          at the highest thread count.  Slab vs pencil
                          vs cube at large grids.

Optional fifth chart if data is present:
  5. spawn_overhead.png   pthread spawn-per-sweep vs. persistent — the
                          gap is the per-sweep pthread_create cost.

Run after sweep.sh:
    scripts/plot.py
    scripts/plot.py --csv path/to/results.csv --out path/to/figs/
"""
import argparse
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.stderr.write(f"missing dep: {e}\n  pip install pandas matplotlib numpy\n")
    sys.exit(1)


def load(csv_path):
    df = pd.read_csv(csv_path)
    # Numeric coercion: silently drop rows where conversion fails.
    for c in ("dim", "N", "iters", "threads", "time_s", "cpe", "max_diff"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s", "cpe"])
    return df


# ---------------------------------------------------------------- Chart 1 ----
def chart_strong_scaling(df, out):
    """Speedup vs thread count, one line per (binary, N).

    Reference: y=x diagonal.  Anything below the diagonal is sub-linear;
    anything at or above is the goal.
    """
    sub = df[df["threads"] >= 1].copy()

    # Build a (binary, mode, N, threads) -> time_s table to compute
    # speedup against the threads=1 row of the same (binary, mode, N).
    keys = ["binary", "mode", "N"]
    base = (sub[sub["threads"] == 1]
            .groupby(keys, as_index=False)["time_s"].min()
            .rename(columns={"time_s": "t1"}))
    if base.empty:
        print("  (no threads=1 baseline rows; skipping strong_scaling)")
        return
    sub = sub.merge(base, on=keys, how="inner")
    sub["speedup"] = sub["t1"] / sub["time_s"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = 0
    for (binary, mode, N), g in sub.groupby(["binary", "mode", "N"]):
        g = g.sort_values("threads")
        if len(g) < 2:
            continue
        label = f"{binary}/{mode} N={int(N)}"
        ax.plot(g["threads"], g["speedup"], marker="o", label=label)
        plotted += 1

    if plotted == 0:
        print("  (no multi-thread series; skipping strong_scaling)")
        plt.close(fig)
        return

    tmax = sub["threads"].max()
    ax.plot([1, tmax], [1, tmax], color="gray", lw=0.7,
            ls="--", label="ideal (y=x)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup over 1 thread")
    ax.set_title("Strong scaling")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "strong_scaling.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'strong_scaling.png'} ({plotted} series)")


# ---------------------------------------------------------------- Chart 2 ----
def chart_variant_heatmap(df, out):
    """CPE heatmap: rows = variant, columns = N.

    `Variant` collapses (binary, mode) to one label.  We pick, per
    (variant, N), the *min* CPE seen at any thread count — so the cell
    answers "what's the best this variant can do at this N" (lower CPE
    = faster).  Colormap is reversed (viridis_r) so dark = fastest.
    """
    # Filter to 2D for now; 3D has different N regime, would distort scale.
    sub = df[df["dim"] == 2].copy()
    if sub.empty:
        print("  (no 2D rows; skipping variant_heatmap)")
        return

    sub["variant"] = sub.apply(
        lambda r: f"{r['binary']}:{r['mode']}", axis=1)

    grid = (sub.groupby(["variant", "N"], as_index=False)["cpe"].min()
                .pivot(index="variant", columns="N", values="cpe"))
    grid = grid.reindex(sorted(grid.index, key=str))
    grid = grid.reindex(columns=sorted(grid.columns))

    fig, ax = plt.subplots(figsize=(1.0 + 0.7*len(grid.columns),
                                    0.5 + 0.35*len(grid.index)))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"N={int(c)}" for c in grid.columns],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", color="white", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("min CPE across threads (cycles/element, lower=faster)")
    ax.set_title("2D variants × grid size — min CPE")
    fig.tight_layout()
    fig.savefig(out / "variant_heatmap.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'variant_heatmap.png'} "
          f"({grid.shape[0]} variants × {grid.shape[1]} sizes)")


# ---------------------------------------------------------------- Chart 3 ----
def chart_decomp_compare(df, out):
    """Pthread decomposition study: 3 modes × 2 schedules per N.

    Bars grouped by N; one bar group per (mode, sched) combination at the
    representative thread count (the largest in the data).
    """
    sub = df[df["binary"] == "sor2d_pth_decomp"].copy()
    if sub.empty:
        print("  (no sor2d_pth_decomp rows; skipping decomp_compare)")
        return

    nt = int(sub["threads"].max())
    sub = sub[sub["threads"] == nt]
    if sub.empty:
        print("  (no rows at max threads; skipping decomp_compare)")
        return

    # mode column for new binary encodes "<mode>" and extra column has
    # schedule.  Re-derive a (mode, sched) pair from the columns.
    # Note: sor2d_pth_decomp.c writes: "mode=...,sched=..." into mode+extra.
    # In CSV row positions: mode, extra → here mode=mode, extra=sched.
    sub = sub.rename(columns={"mode": "decomp", "extra": "sched"})

    pivot = (sub.groupby(["decomp", "sched", "N"], as_index=False)["cpe"]
                .min())

    Ns = sorted(pivot["N"].unique())
    combos = [("strip", "persistent"), ("strip", "spawn"),
              ("interleaved", "persistent"), ("interleaved", "spawn"),
              ("block", "persistent"), ("block", "spawn")]
    width = 0.13
    x = np.arange(len(Ns))

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(Ns), 5))
    for k, (m, s) in enumerate(combos):
        ys = []
        for N in Ns:
            row = pivot[(pivot["decomp"] == m)
                        & (pivot["sched"] == s)
                        & (pivot["N"] == N)]
            ys.append(row["cpe"].iloc[0] if len(row) else 0.0)
        ax.bar(x + (k - 2.5) * width, ys, width, label=f"{m} / {s}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={int(N)}" for N in Ns])
    ax.set_ylabel("CPE (cycles / element, lower=faster)")
    ax.set_title(f"Pthreads decomposition × scheduling at {nt} threads")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "decomp_compare.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'decomp_compare.png'} "
          f"({len(Ns)} sizes × {len(combos)} variants)")


# ---------------------------------------------------------------- Chart 4 ----
def chart_partition_3d(df, out):
    """3D OMP partitioning: CPE by (mode, N) at the largest thread count."""
    sub = df[df["binary"] == "sor3d_omp_part"].copy()
    if sub.empty:
        print("  (no sor3d_omp_part rows; skipping partition_3d)")
        return

    nt = int(sub["threads"].max())
    sub = sub[sub["threads"] == nt]
    if sub.empty:
        print("  (no rows at max threads; skipping partition_3d)")
        return

    pivot = (sub.groupby(["mode", "N"], as_index=False)["cpe"]
                .min()
                .pivot(index="mode", columns="N", values="cpe"))
    Ns = sorted(pivot.columns)
    modes = ["slab", "pencil", "cube"]

    width = 0.25
    x = np.arange(len(Ns))
    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(Ns), 5))
    for k, m in enumerate(modes):
        ys = [pivot.loc[m, N] if (m in pivot.index and N in pivot.columns)
              else 0.0 for N in Ns]
        ax.bar(x + (k - 1) * width, ys, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={int(N)}" for N in Ns])
    ax.set_ylabel("CPE (cycles / element, lower=faster)")
    ax.set_title(f"3D OMP partitioning at {nt} threads")
    ax.legend()
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "partition_3d.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'partition_3d.png'}")


# ---------------------------------------------------------------- Chart 5 ----
def chart_spawn_overhead(df, out):
    """spawn-per-sweep vs. persistent: the gap = per-sweep pthread_create cost.

    For each (N, threads, mode), plot persistent_time and spawn_time
    side-by-side.  The interesting story is small N where the spawn cost
    is comparable to or exceeds the compute cost.
    """
    sub = df[df["binary"] == "sor2d_pth_decomp"].copy()
    if sub.empty:
        return
    sub = sub.rename(columns={"mode": "decomp", "extra": "sched"})

    # Pick one decomp (strip) for clarity; the overhead question is
    # decomp-independent.
    sub = sub[sub["decomp"] == "strip"]
    if sub.empty:
        return

    pivot = (sub.groupby(["sched", "N", "threads"], as_index=False)["time_s"]
                .min())
    nt_set = sorted(pivot["threads"].unique())
    Ns = sorted(pivot["N"].unique())
    if not nt_set or not Ns:
        return

    fig, axes = plt.subplots(1, len(nt_set),
                             figsize=(2.5 * len(nt_set) + 1, 4.5),
                             sharey=False, squeeze=False)
    for ax, nt in zip(axes[0], nt_set):
        for s, ls, mk in (("persistent", "-", "o"), ("spawn", "--", "s")):
            ys = []
            for N in Ns:
                r = pivot[(pivot["sched"] == s)
                          & (pivot["N"] == N)
                          & (pivot["threads"] == nt)]
                ys.append(r["time_s"].iloc[0] if len(r) else np.nan)
            ax.plot(Ns, ys, ls=ls, marker=mk, label=s)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{int(nt)} threads")
        ax.set_xlabel("N (grid)")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("time (s) — strip mode, lower is better")
    fig.suptitle("Persistent vs. spawn-per-sweep — gap = pthread_create overhead")
    fig.tight_layout()
    fig.savefig(out / "spawn_overhead.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out/'spawn_overhead.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.stderr.write(f"missing {csv_path}; run scripts/sweep.sh first\n")
        sys.exit(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")

    chart_strong_scaling(df, out)
    chart_variant_heatmap(df, out)
    chart_decomp_compare(df, out)
    chart_partition_3d(df, out)
    chart_spawn_overhead(df, out)

    print(f"figs in {out}/")


if __name__ == "__main__":
    main()
