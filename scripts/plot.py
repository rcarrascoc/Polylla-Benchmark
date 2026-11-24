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
    has_gpu = "gpu_time_ms" in df.columns and (df["gpu_time_ms"].fillna(0) > 0).any()

    # Compute Speedups strictly over baseline (T_baseline / T_variant)
    has_seq = seq_time_col is not None and (df[seq_time_col] > 0).any()
    if has_seq:
        seq_time = df[seq_time_col]
        if has_gpu:
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
            if has_gpu and "calc_speedup_gpu_alg" in df.columns:
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

            active_cols = [c for c in ["calc_speedup_gpu_alg", "calc_speedup_dcel", "calc_speedup_data", "calc_speedup_task"] if c in df.columns]
            y_max = max([df[c].max() for c in active_cols] + [1.0])
            if log_scale:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.7, top=y_max * 2.8)
            else:
                ax.set_ylim(bottom=-0.02 * y_max, top=y_max * 1.08)

            ax.set_xlabel("Number of Vertices", fontsize=10.5, fontweight="bold")
            format_vertices_axis(ax)
            ax.tick_params(labelsize=9)

        endpoints_to_annotate = []
        if has_gpu and "calc_speedup_gpu_alg" in df.columns:
            endpoints_to_annotate.append(
                {"val": df["calc_speedup_gpu_alg"].iloc[-1], "color": "#2ca02c", "text": f"{df['calc_speedup_gpu_alg'].iloc[-1]:.1f}x"}
            )
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

        if has_gpu and "gpu_time_ms" in df.columns:
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

def plot_file(csv_path: Path, output_dir: Path = None, formats: list = None,
              show_title: bool = None, both: bool = False):
    """Parses a CSV file and triggers plot generation (dual PNG + PDF by default)."""
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

    is_dual = both or (output_dir is None and formats is None and show_title is None)
    if is_dual:
        print("[Plotter] Generating standard PNG visualizations in plots/...")
        plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=PLOTS_DIR,
                            seeds_info=seeds_info, runs=runs, formats=["png"], show_title=True)
        print("[Plotter] Generating publication-ready vector PDFs (no title) in plots_pdf/...")
        plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=PLOTS_PDF_DIR,
                            seeds_info=seeds_info, runs=runs, formats=["pdf"], show_title=False)
    else:
        target_dir = output_dir if output_dir is not None else PLOTS_DIR
        target_formats = formats if formats is not None else ["png"]
        target_title = show_title if show_title is not None else True
        plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=target_dir,
                            seeds_info=seeds_info, runs=runs, formats=target_formats, show_title=target_title)
    return True

def plot_combined_files(csv_paths: list, output_dir: Path = PLOTS_DIR, formats: list = None,
                        show_title: bool = True, custom_name: str = None):
    """Combines multiple CSV files into a unified dataset and plots it."""
    valid_paths = [p for p in csv_paths if p.exists()]
    if not valid_paths:
        print("[Error] No valid CSV files found to combine.")
        return False

    print(f"\n[Plotter] Combining {len(valid_paths)} CSV files:")
    dfs = []
    mesh_types = set()
    runs = 1
    seeds = set()
    for p in valid_paths:
        print(f"  -> Reading: {p}")
        try:
            df = pd.read_csv(p)
            dfs.append(df)
            for part in p.stem.split("_"):
                if part in ["uniform", "grid"]:
                    mesh_types.add(part)
                if part.startswith("seed") and not part.startswith("seeds"):
                    seeds.add(part[4:])
            if "runs" in df.columns and pd.notnull(df["runs"].iloc[0]):
                runs = max(runs, int(float(df["runs"].iloc[0])))
        except Exception as e:
            print(f"  [Warning] Could not read {p}: {e}")

    if not dfs:
        return False

    combined = pd.concat(dfs, ignore_index=True)
    numeric_cols = [c for c in combined.select_dtypes(include=[np.number]).columns if c != "n_vertices"]
    if "n_vertices" in combined.columns:
        combined = combined.groupby("n_vertices", as_index=False)[numeric_cols].mean()
        combined = combined.sort_values(by="n_vertices").reset_index(drop=True)

    tag = custom_name or ("_".join(sorted(mesh_types)) + "_combined" if mesh_types else "combined")
    seeds_info = ", ".join(sorted(seeds)) if seeds else ""
    plot_benchmark_data(combined, mesh_type=tag, output_dir=output_dir, formats=formats,
                        show_title=show_title, runs=runs, seeds_info=seeds_info)
    return True

def plot_compare_datasets(csv_paths: list, output_dir: Path = PLOTS_DIR, formats: list = None,
                          show_title: bool = True, custom_name: str = None):
    """Generates comparison plots contrasting multiple CSV datasets on the same figures."""
    if formats is None:
        formats = ["png"]
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_paths = [p for p in csv_paths if p.exists()]
    if len(valid_paths) < 2:
        print("[Warning] Need at least 2 CSV files to compare.")
        if len(valid_paths) == 1:
            return plot_file(valid_paths[0], output_dir=output_dir, formats=formats, show_title=show_title)
        return False

    datasets = []
    for p in valid_paths:
        try:
            df = pd.read_csv(p).sort_values(by="n_vertices").copy()
            label = p.stem.replace("sweep_", "")
            for m in ["uniform", "grid"]:
                if m in p.stem:
                    label = m.capitalize()
                    break
            datasets.append((label, p.stem, df))
        except Exception as e:
            print(f"[Warning] Could not load {p}: {e}")

    if len(datasets) < 2:
        return False

    print(f"\n[Plotter] Generating multi-dataset comparison for: {[d[0] for d in datasets]}")

    variants = [
        ("GPolylla", "gpu_time_ms", "#2ca02c", "o"),
        ("MPolylla DCEL", "omp_dcel_time_ms", "#ff7f0e", "s"),
        ("MPolylla Data", "omp_data_time_ms", "#1f77b4", "^"),
        ("MPolylla Task", "omp_tasks_time_ms", "#9467bd", "v"),
        ("Polylla-Mesh-DCEL", "seq_time_ms", "#7f7f7f", "x"),
    ]
    linestyles = ["-", "--", "-.", ":"]

    tag = custom_name or "_vs_".join([d[0].lower() for d in datasets])
    fig_w, fig_h = (6.2, 5.2) if not show_title else (6.2, 5.6)

    # 1. Comparison Time Logarithmic Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    for d_idx, (d_label, _, df) in enumerate(datasets):
        ls = linestyles[d_idx % len(linestyles)]
        for v_name, col, color, marker in variants:
            actual_col = col
            if col == "omp_data_time_ms" and col not in df.columns:
                for alt in ["omp_loops_time_ms", "cpu_time_ms"]:
                    if alt in df.columns: actual_col = alt; break
            if actual_col in df.columns and (df[actual_col].fillna(0) > 0).any():
                ax.plot(df["n_vertices"], df[actual_col],
                        marker=marker, markersize=5.5, color=color, linestyle=ls, linewidth=1.8,
                        label=f"{v_name} ({d_label})")

    ax.set_yscale("log")
    ax.set_xlabel("Number of Vertices", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Execution Time in ms (Log Scale)", fontsize=10.5, fontweight="bold")
    format_vertices_axis(ax)
    ax.tick_params(labelsize=9)
    if show_title:
        title_str = " vs ".join([d[0] for d in datasets])
        ax.set_title(f"Polylla Execution Time Comparison\n({title_str})", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        out_p = output_dir / f"comparison_time_log_{tag}.{fmt}"
        fig.savefig(out_p, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Comparison Time Plot:     {out_p}")
    plt.close(fig)

    # 2. Comparison Speedup Logarithmic Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    for d_idx, (d_label, _, df) in enumerate(datasets):
        ls = linestyles[d_idx % len(linestyles)]
        seq_col = "seq_time_ms" if "seq_time_ms" in df.columns and (df["seq_time_ms"] > 0).any() else None
        if not seq_col:
            continue
        for v_name, col, color, marker in variants:
            if v_name == "Polylla-Mesh-DCEL":
                continue
            actual_col = col
            if col == "omp_data_time_ms" and col not in df.columns:
                for alt in ["omp_loops_time_ms", "cpu_time_ms"]:
                    if alt in df.columns: actual_col = alt; break
            if actual_col in df.columns and (df[actual_col].fillna(0) > 0).any():
                sp = df[seq_col] / df[actual_col]
                ax.plot(df["n_vertices"], sp,
                        marker=marker, markersize=5.5, color=color, linestyle=ls, linewidth=1.8,
                        label=f"{v_name} ({d_label})")

    ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.3, label="Baseline (1.0x)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Vertices", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Speedup Factor over Baseline (Log Scale)", fontsize=10.5, fontweight="bold")
    format_vertices_axis(ax)
    ax.tick_params(labelsize=9)
    if show_title:
        title_str = " vs ".join([d[0] for d in datasets])
        ax.set_title(f"Polylla Speedup Comparison vs Baseline\n({title_str})", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        out_p = output_dir / f"comparison_speedup_log_{tag}.{fmt}"
        fig.savefig(out_p, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Comparison Speedup Plot:  {out_p}")
    plt.close(fig)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Plotting Tool for PPolylla Benchmark Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 scripts/plot.py                               # Automatically plots the latest sweep_*.csv (PNG in plots/)
  python3 scripts/plot.py results/sweep_uniform.csv     # Plots a single CSV directly
  python3 scripts/plot.py file1.csv file2.csv           # Plots multiple CSVs in batch (separate plots for each)
  python3 scripts/plot.py file1.csv file2.csv --combine # Combines/merges multiple CSVs into a single continuous plot
  python3 scripts/plot.py file1.csv file2.csv --compare # Compares multiple CSVs on the same figure (e.g. Uniform vs Grid)
  python3 scripts/plot.py --all                         # Plots all sweep_*.csv found in results/
  python3 scripts/plot.py --pdf --all                   # Plots all sweep_*.csv as vector PDFs without title in plots_pdf/
"""
    )
    parser.add_argument("csv_files", nargs="*", default=[], help="Path to one or more CSV files")
    parser.add_argument("--csv", "-c", action="append", default=[], help="Path to CSV file (can be repeated or comma-separated)")
    parser.add_argument("--outdir", "-o", type=str, default=None, help="Output directory for plots (default: plots/ or plots_pdf/)")
    parser.add_argument("--all", "-a", action="store_true", help="Plot all sweep CSV files in results/")
    parser.add_argument("--combine", "--merge", action="store_true", help="Merge multiple CSV files into a single unified dataset and plot it")
    parser.add_argument("--compare", action="store_true", help="Compare multiple CSV files on the same figure")
    parser.add_argument("--name", type=str, default=None, help="Custom tag/name for combined or comparison output files")
    parser.add_argument("--pdf", action="store_true", help="Generate publication-ready PDF plots without titles in plots_pdf/")
    parser.add_argument("--png", action="store_true", help="Generate PNG plots with titles in plots/")
    parser.add_argument("--both", action="store_true", help="Explicitly generate both PNG in plots/ and PDF in plots_pdf/")
    parser.add_argument("--no-title", action="store_true", help="Omit figure title (ideal for LaTeX paper captions)")
    parser.add_argument("--format", "-f", type=str, default=None, help="Output format: png, pdf, or png,pdf (comma-separated)")
    parser.add_argument("--plot-only", action="store_true", help="Compatibility flag for runner script")

    args, unknown = parser.parse_known_args()

    # Collect all targeted CSV files
    target_csvs = list(args.csv_files)
    for c_item in args.csv:
        for p in c_item.split(","):
            p_clean = p.strip()
            if p_clean and p_clean not in target_csvs:
                target_csvs.append(p_clean)

    output_dir = None
    formats = None
    show_title = None
    both = args.both

    if args.pdf:
        output_dir = Path(args.outdir) if args.outdir else PLOTS_PDF_DIR
        formats = ["pdf"]
        show_title = False
    elif args.png:
        output_dir = Path(args.outdir) if args.outdir else PLOTS_DIR
        formats = ["png"]
        show_title = True
    elif args.format:
        formats = [fmt.strip().lower() for fmt in args.format.split(",") if fmt.strip()]
        if args.outdir:
            output_dir = Path(args.outdir)
        else:
            output_dir = PLOTS_PDF_DIR if formats == ["pdf"] else PLOTS_DIR
        show_title = False if (formats == ["pdf"] or args.no_title) else True
    elif args.outdir:
        output_dir = Path(args.outdir)
        formats = ["png"]
        show_title = True
    else:
        # Default behavior: generate BOTH PNG in plots/ and PDF in plots_pdf/
        both = True

    if args.no_title:
        show_title = False

    if target_csvs:
        csv_paths = [Path(p) for p in target_csvs]
        if args.combine:
            plot_combined_files(csv_paths, output_dir=output_dir, formats=formats,
                                show_title=show_title, custom_name=args.name, both=both)
        elif args.compare:
            plot_compare_datasets(csv_paths, output_dir=output_dir, formats=formats,
                                  show_title=show_title, custom_name=args.name, both=both)
        else:
            for p in csv_paths:
                plot_file(p, output_dir=output_dir, formats=formats, show_title=show_title, both=both)
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

        if args.combine:
            plot_combined_files(candidate_files, output_dir=output_dir, formats=formats,
                                show_title=show_title, custom_name=args.name, both=both)
        elif args.compare:
            plot_compare_datasets(candidate_files, output_dir=output_dir, formats=formats,
                                  show_title=show_title, custom_name=args.name, both=both)
        elif args.all:
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
                plot_file(f, output_dir=output_dir, formats=formats, show_title=show_title, both=both)
        else:
            sorted_files = sorted(candidate_files, key=os.path.getmtime, reverse=True)
            print(f"[Plotter] Multiple CSVs found. Plotting latest: {sorted_files[0].name}")
            print(f"         (Tip: use --all to plot all, or --compare to contrast them)")
            plot_file(sorted_files[0], output_dir=output_dir, formats=formats, show_title=show_title, both=both)

    out_summary = str(output_dir) if output_dir else f"{PLOTS_DIR} (PNG) & {PLOTS_PDF_DIR} (PDF)"
    print(f"\n[Plotter] Done! All plots updated in: {out_summary}\n")

if __name__ == "__main__":
    main()
