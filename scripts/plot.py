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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# Ensure TrueType (Type 42) font embedding for PDF output (required by IEEE/ACM)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_PDF_DIR = PROJECT_ROOT / "plots_pdf"

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

def annotate_endpoints(ax, items, x_last, fig, min_gap_pt=14.0, baseline_val=1.0):
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
        if items[0]["target_py_pt"] < base_py_pt + 6.0:
            items[0]["target_py_pt"] = base_py_pt + 6.0

    # Iterative relaxation to guarantee physical vertical separation in points
    for _ in range(60):
        for i in range(len(items) - 1):
            if items[i + 1]["target_py_pt"] - items[i]["target_py_pt"] < min_gap_pt:
                items[i + 1]["target_py_pt"] = items[i]["target_py_pt"] + min_gap_pt

    bbox = ax.get_window_extent()
    bottom_pt = bbox.ymin * 72.0 / dpi
    top_pt = bbox.ymax * 72.0 / dpi
    for item in items:
        item["target_py_pt"] = np.clip(item["target_py_pt"], bottom_pt + 6, top_pt - 6)

    for item in items:
        dy = item["target_py_pt"] - item["py_pt"]
        shifted = abs(dy) > 2.0
        ax.annotate(
            item["text"],
            xy=(x_last, item["val"]),
            xycoords="data",
            xytext=(8, dy),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=item["color"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=item["color"], lw=1.1, alpha=0.95),
            arrowprops=dict(
                arrowstyle="->",
                color=item["color"],
                lw=0.9,
                linestyle="--",
                shrinkA=2,
                shrinkB=2,
            ) if shifted else None,
        )

def plot_benchmark_data(df: pd.DataFrame, mesh_type: str = "uniform", seed: int = 42,
                        output_dir: Path = PLOTS_DIR, seeds_info: str = "", runs: int = 1,
                        formats: list = None, show_title: bool = True):
    """Generates speedup and time (linear and log) plots from a pandas DataFrame."""
    if formats is None:
        formats = ["png"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = (5.5, 4.6) if not show_title else (5.5, 5.0)

    # Sort DataFrame by vertex count
    df = df.sort_values(by="n_vertices").copy()

    # Detect runs or seeds from DataFrame columns if available
    if "runs" in df.columns and pd.notnull(df["runs"].iloc[0]):
        try:
            runs = int(float(df["runs"].iloc[0]))
        except Exception:
            pass
    elif "runs_per_seed" in df.columns and pd.notnull(df["runs_per_seed"].iloc[0]):
        try:
            runs = int(float(df["runs_per_seed"].iloc[0]))
        except Exception:
            pass

    if not seeds_info and "seeds_list" in df.columns and pd.notnull(df["seeds_list"].iloc[0]):
        seeds_info = str(df["seeds_list"].iloc[0])

    # Construct title subtitle
    if seeds_info:
        title_sub = f"Mesh: {mesh_type.capitalize()} (Seeds: {seeds_info}" + (f", {runs} runs/seed)" if runs > 1 else ")")
    elif runs > 1:
        title_sub = f"Mesh: {mesh_type.capitalize()} (Seed: {seed}, {runs} runs averaged)"
    else:
        title_sub = f"Mesh: {mesh_type.capitalize()} (Seed: {seed})"

    # Identify column names for Data (handling legacy 'omp_loops_time_ms' if needed)
    data_time_col = None
    for col in ["omp_data_time_ms", "omp_loops_time_ms", "cpu_time_ms"]:
        if col in df.columns and df[col].notnull().any():
            data_time_col = col
            break

    dcel_time_col = "omp_dcel_time_ms" if "omp_dcel_time_ms" in df.columns and df["omp_dcel_time_ms"].notnull().any() else None
    task_time_col = "omp_tasks_time_ms" if "omp_tasks_time_ms" in df.columns and df["omp_tasks_time_ms"].notnull().any() else None
    seq_time_col = "seq_time_ms" if "seq_time_ms" in df.columns and df["seq_time_ms"].notnull().any() else None

    # Standard deviation columns
    seq_std_col = "seq_time_std_ms" if "seq_time_std_ms" in df.columns and df["seq_time_std_ms"].notnull().any() else None
    dcel_std_col = "omp_dcel_time_std_ms" if "omp_dcel_time_std_ms" in df.columns and df["omp_dcel_time_std_ms"].notnull().any() else None
    data_std_col = "omp_data_time_std_ms" if "omp_data_time_std_ms" in df.columns and df["omp_data_time_std_ms"].notnull().any() else None
    task_std_col = "omp_tasks_time_std_ms" if "omp_tasks_time_std_ms" in df.columns and df["omp_tasks_time_std_ms"].notnull().any() else None
    gpu_std_col = "gpu_time_std_ms" if "gpu_time_std_ms" in df.columns and df["gpu_time_std_ms"].notnull().any() else None

    # Compute Speedups strictly over baseline (T_baseline / T_variant)
    has_seq = seq_time_col is not None and (df[seq_time_col] > 0).any()
    if has_seq:
        seq_time = df[seq_time_col]
        df["calc_speedup_gpu_alg"] = seq_time / df["gpu_time_ms"]
        if dcel_time_col:
            df["calc_speedup_dcel"] = seq_time / df[dcel_time_col]
        if data_time_col:
            df["calc_speedup_data"] = seq_time / df[data_time_col]
        if task_time_col:
            df["calc_speedup_task"] = seq_time / df[task_time_col]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Determine number of cores used for OpenMP
    omp_cores = None
    if "cpu_threads" in df.columns and df["cpu_threads"].notnull().any():
        try:
            val = df["cpu_threads"].dropna().iloc[0]
            if pd.notnull(val) and int(float(val)) > 0:
                omp_cores = int(float(val))
        except Exception:
            pass

    if omp_cores is None:
        omp_cores = os.cpu_count() or 1

    core_unit = "core" if omp_cores == 1 else "cores"
    omp_suffix = f" ({omp_cores} {core_unit})"

    # -------------------------------------------------------------------------
    # 1. SPEEDUP PLOTS (Strictly every variable over Sequential Baseline)
    # -------------------------------------------------------------------------
    if has_seq:
        def draw_speedup_curves(ax, log_scale=False):
            ax.plot(df["n_vertices"], df["calc_speedup_gpu_alg"],
                    marker="o", markersize=6.5, color="#2ca02c", linewidth=2.2,
                    label="GPolylla")
            if dcel_time_col:
                ax.plot(df["n_vertices"], df["calc_speedup_dcel"],
                        marker="s", markersize=5.5, color="#ff7f0e", linewidth=1.8,
                        label=f"MPolylla DCEL{omp_suffix}")
            if data_time_col:
                ax.plot(df["n_vertices"], df["calc_speedup_data"],
                        marker="^", markersize=5.5, color="#1f77b4", linewidth=1.8,
                        label=f"MPolylla Data{omp_suffix}")
            if task_time_col:
                ax.plot(df["n_vertices"], df["calc_speedup_task"],
                        marker="v", markersize=5.5, color="#9467bd", linewidth=1.8,
                        label=f"MPolylla Task{omp_suffix}")
            ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.3, label="Polylla-Mesh-DCEL")

            x_min, x_max = df["n_vertices"].min(), df["n_vertices"].max()
            x_span = x_max - x_min if x_max > x_min else x_max
            ax.set_xlim(x_min - 0.03 * x_span, x_max + 0.18 * x_span)

            y_max = df["calc_speedup_gpu_alg"].max()
            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.7, top=y_max * 2.8)
            else:
                ax.set_ylim(bottom=-0.02 * y_max, top=y_max * 1.08)

            ax.set_xlabel("Number of Vertices", fontsize=10.5, fontweight="bold")
            format_vertices_axis(ax)
            ax.tick_params(labelsize=9)

        endpoints_to_annotate = [
            {"val": df["calc_speedup_gpu_alg"].iloc[-1], "color": "#2ca02c", "text": f"{df['calc_speedup_gpu_alg'].iloc[-1]:.1f}x"}
        ]
        if dcel_time_col:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_dcel"].iloc[-1], "color": "#ff7f0e", "text": f"{df['calc_speedup_dcel'].iloc[-1]:.1f}x"}
            )
        if data_time_col:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_data"].iloc[-1], "color": "#1f77b4", "text": f"{df['calc_speedup_data'].iloc[-1]:.1f}x"}
            )
        if task_time_col:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_task"].iloc[-1], "color": "#9467bd", "text": f"{df['calc_speedup_task'].iloc[-1]:.1f}x"}
            )
        x_last = df["n_vertices"].iloc[-1]

        # 1a. Speedup Logarithmic
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        draw_speedup_curves(ax, log_scale=True)
        ax.set_ylabel("Speedup Factor (Log Scale over Baseline)", fontsize=10.5, fontweight="bold")
        if show_title:
            ax.set_title(f"Polylla Speedup vs Baseline (Log Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, [dict(e) for e in endpoints_to_annotate], x_last, fig)
        for fmt in formats:
            speedup_log_path = output_dir / f"speedup_log_{mesh_type}.{fmt}"
            fig.savefig(speedup_log_path, bbox_inches="tight", dpi=300)
            if fmt == "png":
                fig.savefig(output_dir / f"speedup_{mesh_type}.png", bbox_inches="tight")
            print(f"  [+] Saved Speedup Log Plot:        {speedup_log_path}")
        plt.close(fig)

        # 1b. Speedup Linear
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        draw_speedup_curves(ax, log_scale=False)
        ax.set_ylabel("Speedup Factor (Linear over Baseline)", fontsize=10.5, fontweight="bold")
        if show_title:
            ax.set_title(f"Polylla Speedup vs Baseline (Linear Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, [dict(e) for e in endpoints_to_annotate], x_last, fig)
        for fmt in formats:
            speedup_lin_path = output_dir / f"speedup_linear_{mesh_type}.{fmt}"
            fig.savefig(speedup_lin_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Speedup Linear Plot:     {speedup_lin_path}")
        plt.close(fig)
    else:
        print("  [-] Skipping Speedup plot (Sequential baseline data not available)")

    # -------------------------------------------------------------------------
    # 2. TIME PLOTS (Linear and Logarithmic, separate)
    # -------------------------------------------------------------------------
    def draw_time_curves(ax, log_scale=False):
        if has_seq:
            ax.plot(df["n_vertices"], df[seq_time_col],
                    marker="x", markersize=6.5, color="#7f7f7f", linewidth=1.8,
                    label="Polylla-Mesh-DCEL")
            if seq_std_col and (df[seq_std_col] > 0).any():
                ax.fill_between(df["n_vertices"],
                                np.maximum(0.001, df[seq_time_col] - df[seq_std_col]),
                                df[seq_time_col] + df[seq_std_col],
                                color="#7f7f7f", alpha=0.18)

        if dcel_time_col:
            ax.plot(df["n_vertices"], df[dcel_time_col],
                    marker="s", markersize=5.5, color="#ff7f0e", linewidth=1.8,
                    label=f"MPolylla DCEL{omp_suffix}")
            if dcel_std_col and (df[dcel_std_col] > 0).any():
                ax.fill_between(df["n_vertices"],
                                np.maximum(0.001, df[dcel_time_col] - df[dcel_std_col]),
                                df[dcel_time_col] + df[dcel_std_col],
                                color="#ff7f0e", alpha=0.18)

        if data_time_col:
            ax.plot(df["n_vertices"], df[data_time_col],
                    marker="^", markersize=5.5, color="#1f77b4", linewidth=1.8,
                    label=f"MPolylla Data{omp_suffix}")
            if data_std_col and (df[data_std_col] > 0).any():
                ax.fill_between(df["n_vertices"],
                                np.maximum(0.001, df[data_time_col] - df[data_std_col]),
                                df[data_time_col] + df[data_std_col],
                                color="#1f77b4", alpha=0.18)

        if task_time_col:
            ax.plot(df["n_vertices"], df[task_time_col],
                    marker="v", markersize=5.5, color="#9467bd", linewidth=1.8,
                    label=f"MPolylla Task{omp_suffix}")
            if task_std_col and (df[task_std_col] > 0).any():
                ax.fill_between(df["n_vertices"],
                                np.maximum(0.001, df[task_time_col] - df[task_std_col]),
                                df[task_time_col] + df[task_std_col],
                                color="#9467bd", alpha=0.18)

        ax.plot(df["n_vertices"], df["gpu_time_ms"],
                marker="o", markersize=6.5, color="#2ca02c", linewidth=2.2,
                label="GPolylla")
        if gpu_std_col and (df[gpu_std_col] > 0).any():
            ax.fill_between(df["n_vertices"],
                            np.maximum(0.001, df["gpu_time_ms"] - df[gpu_std_col]),
                            df["gpu_time_ms"] + df[gpu_std_col],
                            color="#2ca02c", alpha=0.18)

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Number of Vertices", fontsize=10.5, fontweight="bold")
        format_vertices_axis(ax)
        ax.tick_params(labelsize=9)

    # 2a. Time Linear Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_curves(ax, log_scale=False)
    ax.set_ylabel("Execution Time (ms)", fontsize=10.5, fontweight="bold")
    if show_title:
        ax.set_title(f"Polylla Execution Time (Linear Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        time_linear_path = output_dir / f"time_linear_{mesh_type}.{fmt}"
        fig.savefig(time_linear_path, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Time Linear Plot:        {time_linear_path}")
    plt.close(fig)

    # 2b. Time Logarithmic Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_curves(ax, log_scale=True)
    ax.set_ylabel("Execution Time in ms (Log Scale)", fontsize=10.5, fontweight="bold")
    if show_title:
        ax.set_title(f"Polylla Execution Time (Logarithmic Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        time_log_path = output_dir / f"time_log_{mesh_type}.{fmt}"
        fig.savefig(time_log_path, bbox_inches="tight", dpi=300)
        if fmt == "png":
            fig.savefig(output_dir / f"time_{mesh_type}.png", bbox_inches="tight")
        print(f"  [+] Saved Time Logarithmic Plot:   {time_log_path}")
    plt.close(fig)

def plot_file(csv_path: Path, output_dir: Path = PLOTS_DIR, formats: list = None, show_title: bool = True):
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
    seeds_info = ""
    runs = 1

    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part in ["uniform", "grid"]:
            mesh_type = part
        if part.startswith("seed") and not part.startswith("seeds"):
            try:
                seed = int(part[4:])
            except ValueError:
                pass

    if "seeds_" in filename:
        idx = filename.find("seeds_")
        if idx != -1:
            seeds_info = filename[idx + 6:].replace("_", ", ")

    if "runs" in df.columns and pd.notnull(df["runs"].iloc[0]):
        try:
            runs = int(float(df["runs"].iloc[0]))
        except Exception:
            pass
    elif "runs_per_seed" in df.columns and pd.notnull(df["runs_per_seed"].iloc[0]):
        try:
            runs = int(float(df["runs_per_seed"].iloc[0]))
        except Exception:
            pass

    plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=output_dir,
                        seeds_info=seeds_info, runs=runs, formats=formats, show_title=show_title)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Plotting Tool for PPolylla Benchmark Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 scripts/plot.py                               # Automatically plots the latest sweep_*.csv (PNG with title in plots/)
  python3 scripts/plot.py --all                         # Plots all sweep_*.csv found in results/
  python3 scripts/plot.py --pdf --all                   # Plots all sweep_*.csv as vector PDFs without title in plots_pdf/
  python3 scripts/plot.py results/sweep_uniform_seed42.csv
  python3 scripts/plot.py --csv results/sweep_uniform_seed42.csv --outdir custom_plots/
"""
    )
    parser.add_argument("csv_positional", nargs="?", default=None, help="Path to CSV file (optional positional argument)")
    parser.add_argument("--csv", "-c", type=str, default=None, help="Path to CSV file (flag argument)")
    parser.add_argument("--outdir", "-o", type=str, default=None, help="Output directory for plots (default: plots/ or plots_pdf/)")
    parser.add_argument("--all", "-a", action="store_true", help="Plot all sweep CSV files in results/")
    parser.add_argument("--pdf", action="store_true", help="Generate publication-ready PDF plots without titles in plots_pdf/")
    parser.add_argument("--no-title", action="store_true", help="Omit figure title (ideal for LaTeX paper captions)")
    parser.add_argument("--format", "-f", type=str, default=None, help="Output format: png, pdf, or png,pdf (comma-separated)")
    parser.add_argument("--plot-only", action="store_true", help="Compatibility flag for runner script")

    args, unknown = parser.parse_known_args()
    target_csv = args.csv or args.csv_positional

    if args.pdf:
        default_outdir = PLOTS_PDF_DIR
        formats = ["pdf"]
        show_title = False
    else:
        default_outdir = PLOTS_DIR
        formats = ["png"]
        show_title = True

    if args.format:
        formats = [fmt.strip().lower() for fmt in args.format.split(",") if fmt.strip()]

    if args.no_title:
        show_title = False

    output_dir = Path(args.outdir) if args.outdir else default_outdir

    if target_csv:
        csv_path = Path(target_csv)
        plot_file(csv_path, output_dir=output_dir, formats=formats, show_title=show_title)
    else:
        # Scan results directory for sweep CSVs (excluding raw unaggregated dumps)
        candidate_files = [f for f in RESULTS_DIR.glob("sweep_*.csv") if not f.name.endswith("_raw.csv")]
        if not candidate_files:
            candidate_files = [f for f in RESULTS_DIR.glob("*.csv") if not f.name.endswith("_raw.csv")]

        if not candidate_files:
            print(f"[Plotter] No CSV benchmark results found in {RESULTS_DIR}")
            print("Run a sweep benchmark first, e.g.:")
            print("  ./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000")
            sys.exit(1)

        if args.all:
            # Group candidate files by mesh type to select the best canonical dataset per type
            mesh_types = set()
            for f in candidate_files:
                for part in f.stem.split("_"):
                    if part in ["uniform", "grid"]:
                        mesh_types.add(part)

            selected_files = []
            for m in sorted(mesh_types):
                canonical = RESULTS_DIR / f"sweep_{m}.csv"
                if canonical.exists():
                    selected_files.append(canonical)
                else:
                    matching = [f for f in candidate_files if f"_{m}" in f.stem or f.stem.startswith(f"sweep_{m}")]
                    if matching:
                        matching.sort(key=os.path.getmtime, reverse=True)
                        selected_files.append(matching[0])

            if not selected_files:
                selected_files = sorted(candidate_files, key=os.path.getmtime, reverse=True)

            for f in selected_files:
                plot_file(f, output_dir=output_dir, formats=formats, show_title=show_title)
        else:
            sorted_files = sorted(candidate_files, key=os.path.getmtime, reverse=True)
            print(f"[Plotter] Multiple CSVs found. Plotting latest: {sorted_files[0].name}")
            print(f"         (Tip: use --all to plot all mesh types)")
            plot_file(sorted_files[0], output_dir=output_dir, formats=formats, show_title=show_title)

    print("\n[Plotter] Done! All plots updated in directory: " + str(output_dir) + "\n")

if __name__ == "__main__":
    main()
