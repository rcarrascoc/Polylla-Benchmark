#!/usr/bin/env python3
"""
PPolylla Benchmark Plotting Tool
Generates standalone benchmark visualizations:
  1. speedup_log_<type>.png: Speedup vs Sequential Baseline (Logarithmic Y-scale, with final speedup badges).
  2. speedup_linear_<type>.png: Speedup vs Sequential Baseline (Linear Y-scale, with final speedup badges).
  3. time_linear_<type>.png: Execution time in milliseconds (Linear Scale).
  4. time_log_<type>.png: Execution time in milliseconds (Logarithmic Scale).

Usage:
  python3 scripts/plot.py                               # Automatically plots the latest sweep_*.csv
  python3 scripts/plot.py --all                         # Plots all sweep_*.csv found in results/
  python3 scripts/plot.py results/sweep_uniform_seed42.csv
  python3 scripts/plot.py --csv results/sweep_grid_seed42.csv
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

def format_vertices_axis(ax):
    """Formats the X-axis for human-readable vertex counts (10K, 100K, 1M, 10M, etc.)."""
    ax.ticklabel_format(style="plain", axis="x")
    def format_k_m(val, tick_number):
        if val >= 1_000_000:
            return f"{val/1_000_000:g}M"
        elif val >= 1_000:
            return f"{val/1_000:g}K"
        else:
            return f"{val:g}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k_m))

def annotate_endpoints(ax, items, x_last, fig, min_gap_pt=18.0, baseline_val=1.0):
    """
    Annotates the final endpoints of curves with callout badges that do not overlap.
    items: list of dicts with {'val': float, 'color': str, 'text': str}
    """
    if not items:
        return

    fig.canvas.draw()
    dpi = fig.dpi

    for item in items:
        px, py = ax.transData.transform((x_last, item["val"]))
        item["py_pt"] = py * 72.0 / dpi
        item["target_py_pt"] = item["py_pt"]

    items.sort(key=lambda it: it["py_pt"])

    # If baseline is defined, keep lowest badge cleanly above baseline
    if baseline_val is not None:
        base_py_pt = ax.transData.transform((x_last, baseline_val))[1] * 72.0 / dpi
        if items[0]["target_py_pt"] < base_py_pt + 8.0:
            items[0]["target_py_pt"] = base_py_pt + 8.0

    # Iterative relaxation to guarantee physical vertical separation in points
    for _ in range(60):
        for i in range(len(items) - 1):
            if items[i + 1]["target_py_pt"] - items[i]["target_py_pt"] < min_gap_pt:
                items[i + 1]["target_py_pt"] = items[i]["target_py_pt"] + min_gap_pt

    bbox = ax.get_window_extent()
    bottom_pt = bbox.ymin * 72.0 / dpi
    top_pt = bbox.ymax * 72.0 / dpi
    for item in items:
        item["target_py_pt"] = np.clip(item["target_py_pt"], bottom_pt + 8, top_pt - 8)

    for item in items:
        dy = item["target_py_pt"] - item["py_pt"]
        shifted = abs(dy) > 2.0
        ax.annotate(
            item["text"],
            xy=(x_last, item["val"]),
            xycoords="data",
            xytext=(10, dy),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=item["color"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=item["color"], lw=1.3, alpha=0.95),
            arrowprops=dict(
                arrowstyle="->",
                color=item["color"],
                lw=1.0,
                linestyle="--",
                shrinkA=3,
                shrinkB=3,
            ) if shifted else None,
        )

def plot_benchmark_data(df: pd.DataFrame, mesh_type: str = "uniform", seed: int = 42, output_dir: Path = PLOTS_DIR):
    """Generates speedup and time (linear and log) plots from a pandas DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort DataFrame by vertex count
    df = df.sort_values(by="n_vertices").copy()

    # Identify column names for Data (handling legacy 'omp_loops_time_ms' if needed)
    data_time_col = None
    for col in ["omp_data_time_ms", "omp_loops_time_ms", "cpu_time_ms"]:
        if col in df.columns and df[col].notnull().any():
            data_time_col = col
            break

    task_time_col = "omp_tasks_time_ms" if "omp_tasks_time_ms" in df.columns and df["omp_tasks_time_ms"].notnull().any() else None
    seq_time_col = "seq_time_ms" if "seq_time_ms" in df.columns and df["seq_time_ms"].notnull().any() else None

    # Compute Speedups strictly over baseline (T_baseline / T_variant)
    has_seq = seq_time_col is not None and (df[seq_time_col] > 0).any()
    if has_seq:
        seq_time = df[seq_time_col]
        df["calc_speedup_gpu_alg"] = seq_time / df["gpu_time_ms"]
        if data_time_col:
            df["calc_speedup_data"] = seq_time / df[data_time_col]
        if task_time_col:
            df["calc_speedup_task"] = seq_time / df[task_time_col]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------------------
    # 1. SPEEDUP PLOTS (Strictly every variable over Sequential Baseline)
    # -------------------------------------------------------------------------
    if has_seq:
        # Helper to plot speedup curves onto a given axes
        def draw_speedup_curves(ax, log_scale=False):
            ax.plot(df["n_vertices"], df["calc_speedup_gpu_alg"],
                    marker="o", markersize=8, color="#2ca02c", linewidth=2.5,
                    label="GPU (GPolylla)")
            if task_time_col:
                ax.plot(df["n_vertices"], df["calc_speedup_task"],
                        marker="v", markersize=7, color="#9467bd", linewidth=2.2,
                        label="OpenMP Task (MPolylla)")
            if data_time_col:
                ax.plot(df["n_vertices"], df["calc_speedup_data"],
                        marker="^", markersize=7, color="#1f77b4", linewidth=2.2,
                        label="OpenMP Data (MPolylla)")
            ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Sequential Baseline (1.0x)")

            x_min, x_max = df["n_vertices"].min(), df["n_vertices"].max()
            x_span = x_max - x_min if x_max > x_min else x_max
            ax.set_xlim(x_min - 0.03 * x_span, x_max + 0.16 * x_span)

            y_max = df["calc_speedup_gpu_alg"].max()
            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.7, top=y_max * 2.5)
            else:
                ax.set_ylim(bottom=-0.02 * y_max, top=y_max * 1.08)

            ax.set_xlabel("Number of Vertices", fontsize=11, fontweight="bold")
            format_vertices_axis(ax)

        endpoints_to_annotate = [
            {"val": df["calc_speedup_gpu_alg"].iloc[-1], "color": "#2ca02c", "text": f"{df['calc_speedup_gpu_alg'].iloc[-1]:.1f}x"}
        ]
        if task_time_col:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_task"].iloc[-1], "color": "#9467bd", "text": f"{df['calc_speedup_task'].iloc[-1]:.1f}x"}
            )
        if data_time_col:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_data"].iloc[-1], "color": "#1f77b4", "text": f"{df['calc_speedup_data'].iloc[-1]:.1f}x"}
            )
        x_last = df["n_vertices"].iloc[-1]

        # 1a. Speedup Logarithmic
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        draw_speedup_curves(ax, log_scale=True)
        ax.set_ylabel("Speedup Factor (Log Scale over Baseline)", fontsize=12, fontweight="bold")
        ax.set_title(f"Polylla Speedup vs Baseline (Log Scale)\nMesh: {mesh_type.capitalize()} (Seed: {seed})", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, [dict(e) for e in endpoints_to_annotate], x_last, fig)
        speedup_log_path = output_dir / f"speedup_log_{mesh_type}.png"
        fig.savefig(speedup_log_path, bbox_inches="tight")
        fig.savefig(output_dir / f"speedup_{mesh_type}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Saved Speedup Log Plot:        {speedup_log_path}")

        # 1b. Speedup Linear
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        draw_speedup_curves(ax, log_scale=False)
        ax.set_ylabel("Speedup Factor (Linear over Baseline)", fontsize=12, fontweight="bold")
        ax.set_title(f"Polylla Speedup vs Baseline (Linear Scale)\nMesh: {mesh_type.capitalize()} (Seed: {seed})", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, [dict(e) for e in endpoints_to_annotate], x_last, fig)
        speedup_lin_path = output_dir / f"speedup_linear_{mesh_type}.png"
        fig.savefig(speedup_lin_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Saved Speedup Linear Plot:     {speedup_lin_path}")
    else:
        print("  [-] Skipping Speedup plot (Sequential baseline data not available)")

    # -------------------------------------------------------------------------
    # 2. TIME PLOTS (Linear and Logarithmic, separate)
    # -------------------------------------------------------------------------
    def draw_time_curves(ax, log_scale=False):
        if has_seq:
            ax.plot(df["n_vertices"], df[seq_time_col],
                    marker="x", markersize=8, color="#7f7f7f", linewidth=2.2,
                    label="Sequential Baseline (Polylla-Mesh-DCEL)")
        if data_time_col:
            ax.plot(df["n_vertices"], df[data_time_col],
                    marker="^", markersize=7, color="#1f77b4", linewidth=2.2,
                    label="OpenMP Data (MPolylla)")
        if task_time_col:
            ax.plot(df["n_vertices"], df[task_time_col],
                    marker="v", markersize=7, color="#9467bd", linewidth=2.2,
                    label="OpenMP Task (MPolylla)")
        ax.plot(df["n_vertices"], df["gpu_time_ms"],
                marker="o", markersize=8, color="#2ca02c", linewidth=2.5,
                label="GPU (GPolylla)")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Number of Vertices", fontsize=11, fontweight="bold")
        format_vertices_axis(ax)

    # 2a. Time Linear Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    draw_time_curves(ax, log_scale=False)
    ax.set_ylabel("Execution Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(f"Polylla Execution Time (Linear Scale)\nMesh: {mesh_type.capitalize()} (Seed: {seed})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    fig.tight_layout()
    time_linear_path = output_dir / f"time_linear_{mesh_type}.png"
    fig.savefig(time_linear_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved Time Linear Plot:        {time_linear_path}")

    # 2b. Time Logarithmic Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    draw_time_curves(ax, log_scale=True)
    ax.set_ylabel("Execution Time in ms (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title(f"Polylla Execution Time (Logarithmic Scale)\nMesh: {mesh_type.capitalize()} (Seed: {seed})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    fig.tight_layout()
    time_log_path = output_dir / f"time_log_{mesh_type}.png"
    fig.savefig(time_log_path, bbox_inches="tight")
    fig.savefig(output_dir / f"time_{mesh_type}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved Time Logarithmic Plot:   {time_log_path}")

def plot_file(csv_path: Path, output_dir: Path = PLOTS_DIR):
    """Parses a CSV file and triggers plot generation."""
    if not csv_path.exists():
        print(f"[Error] File not found: {csv_path}")
        return False

    print(f"\n[Plotter] Processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Could not read CSV: {e}")
        return False

    filename = csv_path.stem
    mesh_type = "uniform"
    seed = 42

    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part in ["uniform", "grid"]:
            mesh_type = part
        if part.startswith("seed"):
            try:
                seed = int(part[4:])
            except ValueError:
                pass

    plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=output_dir)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Plotting Tool for PPolylla Benchmark Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 scripts/plot.py                               # Automatically plots the latest sweep_*.csv
  python3 scripts/plot.py --all                         # Plots all sweep_*.csv found in results/
  python3 scripts/plot.py results/sweep_uniform_seed42.csv
  python3 scripts/plot.py --csv results/sweep_uniform_seed42.csv --outdir custom_plots/
"""
    )
    parser.add_argument("csv_positional", nargs="?", default=None, help="Path to CSV file (optional positional argument)")
    parser.add_argument("--csv", "-c", type=str, default=None, help="Path to CSV file (flag argument)")
    parser.add_argument("--outdir", "-o", type=str, default=str(PLOTS_DIR), help="Output directory for plots (default: plots/)")
    parser.add_argument("--all", "-a", action="store_true", help="Plot all sweep CSV files in results/")
    parser.add_argument("--plot-only", action="store_true", help="Compatibility flag for runner script")

    args, unknown = parser.parse_known_args()
    target_csv = args.csv or args.csv_positional
    output_dir = Path(args.outdir)

    if target_csv:
        csv_path = Path(target_csv)
        plot_file(csv_path, output_dir=output_dir)
    else:
        # Scan results directory for sweep CSVs
        csv_files = sorted(RESULTS_DIR.glob("sweep_*.csv"), key=os.path.getmtime, reverse=True)
        if not csv_files:
            csv_files = list(RESULTS_DIR.glob("*.csv"))

        if not csv_files:
            print(f"[Plotter] No CSV benchmark results found in {RESULTS_DIR}")
            print("Run a sweep benchmark first, e.g.:")
            print("  ./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000")
            sys.exit(1)

        if args.all or len(csv_files) == 1:
            for f in csv_files:
                plot_file(f, output_dir=output_dir)
        else:
            print(f"[Plotter] Multiple CSVs found. Plotting latest: {csv_files[0].name}")
            print(f"         (Tip: use --all to plot all {len(csv_files)} files)")
            plot_file(csv_files[0], output_dir=output_dir)

    print("\n[Plotter] Done! All plots updated in directory: " + str(output_dir) + "\n")

if __name__ == "__main__":
    main()
