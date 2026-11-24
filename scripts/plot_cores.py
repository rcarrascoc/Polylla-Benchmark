#!/usr/bin/env python3
"""
PPolylla OpenMP Cores & Thread Scalability Plotting Tool
Generates standalone strong-scaling visualizations for OpenMP variants:
  1. cores_<type>_<size>_time.{png,pdf} / _time_linear.{png,pdf}: Execution Time (ms) vs Number of Threads (Line chart).
  2. cores_<type>_<size>_time_log.{png,pdf}: Execution Time (ms) vs Number of Threads (Logarithmic Line chart).
  3. cores_<type>_<size>_time_bar.{png,pdf}: Execution Time (ms) vs Number of Threads (Grouped Bar chart).
  4. cores_<type>_<size>_time_bar_log.{png,pdf}: Execution Time (ms) vs Number of Threads (Logarithmic Bar chart).
  5. cores_<type>_<size>_speedup.{png,pdf} / _speedup_linear.{png,pdf}: Speedup vs Sequential Baseline vs Threads.
  6. cores_<type>_<size>_speedup_log.{png,pdf}: Speedup vs Sequential Baseline (Logarithmic Scale).
  7. cores_<type>_<size>_self_speedup.{png,pdf}: Self-Speedup (T(1) / T(p)) vs Number of Threads.
  8. cores_<type>_<size>_self_speedup_log.{png,pdf}: Self-Speedup (Logarithmic Scale).
  9. cores_<type>_<size>_efficiency.{png,pdf}: Parallel Efficiency (%) vs Number of Threads.

Complies with publication standards: TrueType (Type 42) font embedding for IEEE/ACM,
standardized publication aspect ratio, 2-column legend below plot, and automated dual-output
(standard PNG with title in plots/ and publication vector PDF without title in plots_pdf/).
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

def annotate_endpoints(ax, items, x_last, fig, min_gap_pt=14.0, baseline_val=None):
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

def plot_cores_data(df: pd.DataFrame, mesh_type: str = "uniform", size: int = 1000000,
                    seed: int = 42, output_dir: Path = PLOTS_DIR, seeds_info: str = "",
                    runs: int = 1, affinity_info: str = "", formats: list = None,
                    show_title: bool = True):
    """Generates strong-scaling execution time, speedup, and efficiency plots for OpenMP cores."""
    if formats is None:
        formats = ["png"]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = (5.5, 4.6) if not show_title else (5.5, 5.0)

    # Sort DataFrame by thread count
    thread_col = "threads" if "threads" in df.columns else "cpu_threads"
    df = df.sort_values(by=thread_col).copy()
    threads_series = df[thread_col]

    # Subtitle info
    size_str = f"{size/1_000_000:g}M" if size >= 1_000_000 else (f"{size/1_000:g}K" if size >= 1_000 else str(size))
    if seeds_info:
        title_sub = f"Mesh: {mesh_type.capitalize()} (N={size_str}, Seeds: {seeds_info}" + (f", {runs} runs/seed)" if runs > 1 else ")")
    elif runs > 1:
        title_sub = f"Mesh: {mesh_type.capitalize()} (N={size_str}, Seed: {seed}, {runs} runs avg)"
    else:
        title_sub = f"Mesh: {mesh_type.capitalize()} (N={size_str}, Seed: {seed})"

    if affinity_info:
        title_sub += f"\nAffinity: {affinity_info}"

    # Column mappings
    dcel_time_col = "omp_dcel_time_ms" if "omp_dcel_time_ms" in df.columns and df["omp_dcel_time_ms"].notnull().any() else None
    data_time_col = "omp_data_time_ms" if "omp_data_time_ms" in df.columns and df["omp_data_time_ms"].notnull().any() else ("omp_loops_time_ms" if "omp_loops_time_ms" in df.columns else None)
    task_time_col = "omp_tasks_time_ms" if "omp_tasks_time_ms" in df.columns and df["omp_tasks_time_ms"].notnull().any() else None
    seq_time_col = "seq_time_ms" if "seq_time_ms" in df.columns and df["seq_time_ms"].notnull().any() else None
    gpu_time_col = "gpu_time_ms" if "gpu_time_ms" in df.columns and df["gpu_time_ms"].notnull().any() else None

    # Standard deviation columns
    dcel_std_col = "omp_dcel_time_std_ms" if "omp_dcel_time_std_ms" in df.columns and df["omp_dcel_time_std_ms"].notnull().any() else None
    data_std_col = "omp_data_time_std_ms" if "omp_data_time_std_ms" in df.columns and df["omp_data_time_std_ms"].notnull().any() else None
    task_std_col = "omp_tasks_time_std_ms" if "omp_tasks_time_std_ms" in df.columns and df["omp_tasks_time_std_ms"].notnull().any() else None

    has_seq = seq_time_col is not None and (df[seq_time_col] > 0).any()
    seq_time_val = df[seq_time_col].dropna().iloc[0] if has_seq else None

    has_gpu = gpu_time_col is not None and (df[gpu_time_col] > 0).any()
    gpu_time_val = df[gpu_time_col].dropna().iloc[0] if has_gpu else None

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    x_min, x_max = threads_series.min(), threads_series.max()
    x_ticks = sorted(list(threads_series.unique()))

    # Compute Speedups vs Sequential
    if has_seq:
        if dcel_time_col:
            df["speedup_dcel_seq"] = seq_time_val / df[dcel_time_col]
        if data_time_col:
            df["speedup_data_seq"] = seq_time_val / df[data_time_col]
        if task_time_col:
            df["speedup_task_seq"] = seq_time_val / df[task_time_col]

    # Compute Self-Speedup vs 1 Thread and Efficiency
    row_1 = df[threads_series == 1]
    has_1_thread = not row_1.empty

    if has_1_thread:
        if dcel_time_col and row_1[dcel_time_col].notnull().any():
            t1_dcel = row_1[dcel_time_col].iloc[0]
            df["self_sp_dcel"] = t1_dcel / df[dcel_time_col]
            df["eff_dcel"] = (df["self_sp_dcel"] / threads_series) * 100.0
        if data_time_col and row_1[data_time_col].notnull().any():
            t1_data = row_1[data_time_col].iloc[0]
            df["self_sp_data"] = t1_data / df[data_time_col]
            df["eff_data"] = (df["self_sp_data"] / threads_series) * 100.0
        if task_time_col and row_1[task_time_col].notnull().any():
            t1_task = row_1[task_time_col].iloc[0]
            df["self_sp_task"] = t1_task / df[task_time_col]
            df["eff_task"] = (df["self_sp_task"] / threads_series) * 100.0

    prefix = f"cores_{mesh_type}_{size}"

    # -------------------------------------------------------------------------
    # 1. TIME LINE PLOTS (Execution Time vs Threads: Linear & Logarithmic)
    # -------------------------------------------------------------------------
    def draw_time_lines(ax, log_scale=False):
        if dcel_time_col:
            ax.plot(threads_series, df[dcel_time_col], marker="s", markersize=5.5,
                    color="#ff7f0e", linewidth=1.8, label="MPolylla DCEL")
            if dcel_std_col and (df[dcel_std_col] > 0).any():
                ax.fill_between(threads_series,
                                np.maximum(0.001, df[dcel_time_col] - df[dcel_std_col]),
                                df[dcel_time_col] + df[dcel_std_col],
                                color="#ff7f0e", alpha=0.18)

        if data_time_col:
            ax.plot(threads_series, df[data_time_col], marker="^", markersize=5.5,
                    color="#1f77b4", linewidth=1.8, label="MPolylla Data")
            if data_std_col and (df[data_std_col] > 0).any():
                ax.fill_between(threads_series,
                                np.maximum(0.001, df[data_time_col] - df[data_std_col]),
                                df[data_time_col] + df[data_std_col],
                                color="#1f77b4", alpha=0.18)

        if task_time_col:
            ax.plot(threads_series, df[task_time_col], marker="v", markersize=5.5,
                    color="#9467bd", linewidth=1.8, label="MPolylla Task")
            if task_std_col and (df[task_std_col] > 0).any():
                ax.fill_between(threads_series,
                                np.maximum(0.001, df[task_time_col] - df[task_std_col]),
                                df[task_time_col] + df[task_std_col],
                                color="#9467bd", alpha=0.18)

        if has_seq:
            ax.axhline(seq_time_val, color="#7f7f7f", linestyle="--", linewidth=1.5,
                       label=f"Polylla-Mesh-DCEL ({seq_time_val:.1f} ms)")

        if has_gpu:
            ax.axhline(gpu_time_val, color="#2ca02c", linestyle=":", linewidth=1.8,
                       label=f"GPolylla ({gpu_time_val:.1f} ms)")

        ax.set_xlabel("Number of OpenMP Threads / Cores", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.tick_params(labelsize=9)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Execution Time in ms (Log Scale)", fontsize=10.5, fontweight="bold")
            ax.grid(True, which="both", linestyle="--", alpha=0.6)
        else:
            ax.set_ylabel("Execution Time (ms)", fontsize=10.5, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)

    # 1a. Time Linear Line Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_lines(ax, log_scale=False)
    if show_title:
        ax.set_title(f"OpenMP Polylla Execution Time vs Cores\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()

    for fmt in formats:
        time_path = output_dir / f"{prefix}_time.{fmt}"
        time_lin_path = output_dir / f"{prefix}_time_linear.{fmt}"
        fig.savefig(time_path, bbox_inches="tight", dpi=300)
        fig.savefig(time_lin_path, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Cores Time Plot:         {time_path}")
    plt.close(fig)

    # 1b. Time Logarithmic Line Plot
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_lines(ax, log_scale=True)
    if show_title:
        ax.set_title(f"OpenMP Polylla Execution Time vs Cores (Log Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()

    for fmt in formats:
        time_log_path = output_dir / f"{prefix}_time_log.{fmt}"
        fig.savefig(time_log_path, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Cores Time Log Plot:     {time_log_path}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 1c. TIME BAR PLOTS (Grouped Bar Chart: Linear & Logarithmic)
    # -------------------------------------------------------------------------
    x_indices = np.arange(len(threads_series))
    bar_variants = []
    if dcel_time_col:
        bar_variants.append({
            "col": dcel_time_col,
            "std": dcel_std_col,
            "label": "MPolylla DCEL",
            "color": "#ff7f0e",
        })
    if data_time_col:
        bar_variants.append({
            "col": data_time_col,
            "std": data_std_col,
            "label": "MPolylla Data",
            "color": "#1f77b4",
        })
    if task_time_col:
        bar_variants.append({
            "col": task_time_col,
            "std": task_std_col,
            "label": "MPolylla Task",
            "color": "#9467bd",
        })

    n_bars = len(bar_variants)
    total_group_width = 0.74
    bar_w = total_group_width / max(1, n_bars)

    def draw_time_bars(ax, log_scale=False):
        for idx, v in enumerate(bar_variants):
            offset = (idx - (n_bars - 1) / 2.0) * bar_w
            times = df[v["col"]].values
            stds = df[v["std"]].values if (v["std"] and v["std"] in df.columns) else None
            yerr = stds if (stds is not None and (stds > 0).any()) else None

            ax.bar(
                x_indices + offset, times, width=bar_w * 0.92,
                yerr=yerr, capsize=2.5, error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
                label=v["label"], color=v["color"], edgecolor="white", linewidth=0.6, zorder=3
            )

        if has_seq:
            ax.axhline(seq_time_val, color="#7f7f7f", linestyle="--", linewidth=1.5,
                       label=f"Polylla-Mesh-DCEL ({seq_time_val:.1f} ms)", zorder=4)

        if has_gpu:
            ax.axhline(gpu_time_val, color="#2ca02c", linestyle=":", linewidth=1.8,
                       label=f"GPolylla ({gpu_time_val:.1f} ms)", zorder=4)

        ax.set_xlabel("Number of OpenMP Threads / Cores", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(t) for t in threads_series])
        ax.tick_params(labelsize=9)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Execution Time in ms (Log Scale)", fontsize=10.5, fontweight="bold")
            ax.grid(True, which="both", linestyle="--", alpha=0.6, axis="y", zorder=0)
        else:
            ax.set_ylabel("Execution Time (ms)", fontsize=10.5, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6, axis="y", zorder=0)

    # 1c.1 Linear Bar Chart
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_bars(ax, log_scale=False)
    if show_title:
        ax.set_title(f"OpenMP Polylla Execution Time vs Cores\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        bar_path = output_dir / f"{prefix}_time_bar.{fmt}"
        fig.savefig(bar_path, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Cores Time Bar Plot:     {bar_path}")
    plt.close(fig)

    # 1c.2 Logarithmic Bar Chart
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    draw_time_bars(ax, log_scale=True)
    if show_title:
        ax.set_title(f"OpenMP Polylla Execution Time vs Cores (Log Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    for fmt in formats:
        bar_log_path = output_dir / f"{prefix}_time_bar_log.{fmt}"
        fig.savefig(bar_log_path, bbox_inches="tight", dpi=300)
        print(f"  [+] Saved Cores Time Bar Log Plot: {bar_log_path}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. SPEEDUP PLOTS VS SEQUENTIAL BASELINE (Linear & Logarithmic)
    # -------------------------------------------------------------------------
    if has_seq:
        def draw_speedup_curves(ax, log_scale=False):
            endpoints = []
            if dcel_time_col and "speedup_dcel_seq" in df.columns:
                ax.plot(threads_series, df["speedup_dcel_seq"], marker="s", markersize=5.5,
                        color="#ff7f0e", linewidth=1.8, label="MPolylla DCEL")
                endpoints.append({"val": df["speedup_dcel_seq"].iloc[-1], "color": "#ff7f0e", "text": f"{df['speedup_dcel_seq'].iloc[-1]:.1f}x"})

            if data_time_col and "speedup_data_seq" in df.columns:
                ax.plot(threads_series, df["speedup_data_seq"], marker="^", markersize=5.5,
                        color="#1f77b4", linewidth=1.8, label="MPolylla Data")
                endpoints.append({"val": df["speedup_data_seq"].iloc[-1], "color": "#1f77b4", "text": f"{df['speedup_data_seq'].iloc[-1]:.1f}x"})

            if task_time_col and "speedup_task_seq" in df.columns:
                ax.plot(threads_series, df["speedup_task_seq"], marker="v", markersize=5.5,
                        color="#9467bd", linewidth=1.8, label="MPolylla Task")
                endpoints.append({"val": df["speedup_task_seq"].iloc[-1], "color": "#9467bd", "text": f"{df['speedup_task_seq'].iloc[-1]:.1f}x"})

            # Ideal linear speedup reference line
            ax.plot([x_min, x_max], [x_min, x_max], color="#bbbbbb", linestyle="-.", linewidth=1.2, label="Ideal Linear (y = x)")
            ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Baseline (1.0x)")

            ax.set_xlabel("Number of OpenMP Threads / Cores", fontsize=10.5, fontweight="bold")
            ax.set_xticks(x_ticks)
            ax.tick_params(labelsize=9)

            x_span = x_max - x_min if x_max > x_min else x_max
            ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.18 * x_span)

            if log_scale:
                ax.set_yscale("log")
                ax.set_ylabel("Speedup vs Baseline (Log Scale)", fontsize=10.5, fontweight="bold")
                ax.grid(True, which="both", linestyle="--", alpha=0.6)
            else:
                ax.set_ylabel("Speedup Factor vs Sequential Baseline", fontsize=10.5, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.6)
            return endpoints

        # 2a. Speedup Linear Plot
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        endpoints = draw_speedup_curves(ax, log_scale=False)
        if show_title:
            ax.set_title(f"OpenMP Polylla Speedup vs Baseline\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, endpoints, threads_series.iloc[-1], fig, baseline_val=1.0)

        for fmt in formats:
            sp_path = output_dir / f"{prefix}_speedup.{fmt}"
            sp_lin_path = output_dir / f"{prefix}_speedup_linear.{fmt}"
            fig.savefig(sp_path, bbox_inches="tight", dpi=300)
            fig.savefig(sp_lin_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Cores Speedup Plot:      {sp_path}")
        plt.close(fig)

        # 2b. Speedup Logarithmic Plot
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        endpoints = draw_speedup_curves(ax, log_scale=True)
        if show_title:
            ax.set_title(f"OpenMP Polylla Speedup vs Baseline (Log Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, endpoints, threads_series.iloc[-1], fig, baseline_val=1.0)

        for fmt in formats:
            sp_log_path = output_dir / f"{prefix}_speedup_log.{fmt}"
            fig.savefig(sp_log_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Cores Speedup Log Plot:  {sp_log_path}")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. SELF-SPEEDUP PLOTS (T(1) / T(p): Linear & Logarithmic)
    # -------------------------------------------------------------------------
    if has_1_thread:
        def draw_self_speedup_curves(ax, log_scale=False):
            endpoints = []
            if "self_sp_dcel" in df.columns:
                ax.plot(threads_series, df["self_sp_dcel"], marker="s", markersize=5.5,
                        color="#ff7f0e", linewidth=1.8, label="MPolylla DCEL")
                endpoints.append({"val": df["self_sp_dcel"].iloc[-1], "color": "#ff7f0e", "text": f"{df['self_sp_dcel'].iloc[-1]:.1f}x"})

            if "self_sp_data" in df.columns:
                ax.plot(threads_series, df["self_sp_data"], marker="^", markersize=5.5,
                        color="#1f77b4", linewidth=1.8, label="MPolylla Data")
                endpoints.append({"val": df["self_sp_data"].iloc[-1], "color": "#1f77b4", "text": f"{df['self_sp_data'].iloc[-1]:.1f}x"})

            if "self_sp_task" in df.columns:
                ax.plot(threads_series, df["self_sp_task"], marker="v", markersize=5.5,
                        color="#9467bd", linewidth=1.8, label="MPolylla Task")
                endpoints.append({"val": df["self_sp_task"].iloc[-1], "color": "#9467bd", "text": f"{df['self_sp_task'].iloc[-1]:.1f}x"})

            ax.plot([x_min, x_max], [x_min, x_max], color="#bbbbbb", linestyle="-.", linewidth=1.2, label="Ideal Linear (y = x)")
            ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Single-Thread (1.0x)")

            ax.set_xlabel("Number of OpenMP Threads / Cores", fontsize=10.5, fontweight="bold")
            ax.set_xticks(x_ticks)
            ax.tick_params(labelsize=9)

            x_span = x_max - x_min if x_max > x_min else x_max
            ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.18 * x_span)

            if log_scale:
                ax.set_yscale("log")
                ax.set_ylabel("Self-Speedup (T(1) / T(p)) (Log Scale)", fontsize=10.5, fontweight="bold")
                ax.grid(True, which="both", linestyle="--", alpha=0.6)
            else:
                ax.set_ylabel("Self-Speedup (T(1) / T(p))", fontsize=10.5, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.6)
            return endpoints

        # 3a. Self-Speedup Linear Plot
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        endpoints = draw_self_speedup_curves(ax, log_scale=False)
        if show_title:
            ax.set_title(f"OpenMP Polylla Self-Speedup (Strong Scaling)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, endpoints, threads_series.iloc[-1], fig, baseline_val=1.0)

        for fmt in formats:
            self_sp_path = output_dir / f"{prefix}_self_speedup.{fmt}"
            fig.savefig(self_sp_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Cores Self-Speedup Plot: {self_sp_path}")
        plt.close(fig)

        # 3b. Self-Speedup Logarithmic Plot
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        endpoints = draw_self_speedup_curves(ax, log_scale=True)
        if show_title:
            ax.set_title(f"OpenMP Polylla Self-Speedup (Log Scale)\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()
        annotate_endpoints(ax, endpoints, threads_series.iloc[-1], fig, baseline_val=1.0)

        for fmt in formats:
            self_sp_log_path = output_dir / f"{prefix}_self_speedup_log.{fmt}"
            fig.savefig(self_sp_log_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Cores Self-Speedup Log:  {self_sp_log_path}")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. PARALLEL EFFICIENCY PLOT (E = S(p)/p * 100%)
    # -------------------------------------------------------------------------
    if has_1_thread:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        if "eff_dcel" in df.columns:
            ax.plot(threads_series, df["eff_dcel"], marker="s", markersize=5.5,
                    color="#ff7f0e", linewidth=1.8, label="MPolylla DCEL")

        if "eff_data" in df.columns:
            ax.plot(threads_series, df["eff_data"], marker="^", markersize=5.5,
                    color="#1f77b4", linewidth=1.8, label="MPolylla Data")

        if "eff_task" in df.columns:
            ax.plot(threads_series, df["eff_task"], marker="v", markersize=5.5,
                    color="#9467bd", linewidth=1.8, label="MPolylla Task")

        ax.axhline(100.0, color="#7f7f7f", linestyle="--", linewidth=1.3, label="Ideal Efficiency (100%)")

        ax.set_xlabel("Number of OpenMP Threads / Cores", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Parallel Efficiency (%)", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.tick_params(labelsize=9)
        ax.set_ylim(bottom=0.0, top=110.0)

        if show_title:
            ax.set_title(f"OpenMP Polylla Parallel Efficiency\n{title_sub}", fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5, frameon=True, framealpha=0.95)
        fig.tight_layout()

        for fmt in formats:
            eff_path = output_dir / f"{prefix}_efficiency.{fmt}"
            fig.savefig(eff_path, bbox_inches="tight", dpi=300)
            print(f"  [+] Saved Cores Efficiency Plot:   {eff_path}")
        plt.close(fig)

def plot_cores_file(csv_path: Path, output_dir: Path = None, formats: list = None,
                    show_title: bool = None, both: bool = False):
    """
    Parses a cores CSV file and triggers plot generation.
    If output_dir and formats are not explicitly provided (default run),
    it generates BOTH:
      1. PNG plots with titles in plots/
      2. Publication-ready vector PDFs without titles in plots_pdf/
    """
    if not csv_path.exists():
        print(f"[Error] File not found: {csv_path}")
        return False

    print(f"\n[Cores Plotter] Processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Could not read CSV: {e}")
        return False

    filename = csv_path.stem
    mesh_type = "uniform"
    size = 1000000
    seed = 42
    seeds_info = ""
    runs = 1
    affinity_info = ""

    parts = filename.split("_")
    for part in parts:
        if part in ["uniform", "grid"]:
            mesh_type = part
        elif part.isdigit() and int(part) >= 1000:
            size = int(part)

    if "requested_size" in df.columns and pd.notnull(df["requested_size"].iloc[0]):
        try:
            size = int(float(df["requested_size"].iloc[0]))
        except Exception:
            pass

    if "seeds_list" in df.columns and pd.notnull(df["seeds_list"].iloc[0]):
        seeds_info = str(df["seeds_list"].iloc[0])

    if "runs" in df.columns and pd.notnull(df["runs"].iloc[0]):
        try:
            runs = int(float(df["runs"].iloc[0]))
        except Exception:
            pass

    if "cpu_cores_affinity" in df.columns and pd.notnull(df["cpu_cores_affinity"].iloc[0]):
        affinity_info = str(df["cpu_cores_affinity"].iloc[0])

    # Determine whether dual generation (PNG + PDF) should run
    is_dual = both or (output_dir is None and formats is None and show_title is None)

    if is_dual:
        print("[Cores Plotter] Generating standard PNG visualizations in plots/...")
        plot_cores_data(df, mesh_type=mesh_type, size=size, seed=seed, output_dir=PLOTS_DIR,
                        seeds_info=seeds_info, runs=runs, affinity_info=affinity_info,
                        formats=["png"], show_title=True)
        print("[Cores Plotter] Generating publication-ready vector PDFs (no title) in plots_pdf/...")
        plot_cores_data(df, mesh_type=mesh_type, size=size, seed=seed, output_dir=PLOTS_PDF_DIR,
                        seeds_info=seeds_info, runs=runs, affinity_info=affinity_info,
                        formats=["pdf"], show_title=False)
    else:
        target_dir = output_dir if output_dir is not None else PLOTS_DIR
        target_formats = formats if formats is not None else ["png"]
        target_title = show_title if show_title is not None else True

        plot_cores_data(df, mesh_type=mesh_type, size=size, seed=seed, output_dir=target_dir,
                        seeds_info=seeds_info, runs=runs, affinity_info=affinity_info,
                        formats=target_formats, show_title=target_title)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Plotting Tool for PPolylla OpenMP Cores Scalability Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Automatically plots latest cores_*.csv (generating BOTH PNG in plots/ and PDF in plots_pdf/):
  python3 scripts/plot_cores.py

  # Plot a specific cores scaling CSV:
  python3 scripts/plot_cores.py results/cores_uniform_1000000.csv

  # Plot multiple cores scaling CSVs in batch:
  python3 scripts/plot_cores.py results/cores_uniform_10000.csv results/cores_grid_10000.csv

  # Only generate publication-ready vector PDFs in plots_pdf/ (no top title):
  python3 scripts/plot_cores.py --pdf

  # Plot all cores_*.csv found in results/:
  python3 scripts/plot_cores.py --all
"""
    )
    parser.add_argument("csv_files", nargs="*", default=[], help="Path to one or more cores CSV files (positional)")
    parser.add_argument("--csv", "-c", action="append", default=[], help="Path to cores CSV file (can be repeated or comma-separated)")
    parser.add_argument("--outdir", "-o", type=str, default=None, help="Output directory for plots (default: plots/ and plots_pdf/)")
    parser.add_argument("--all", "-a", action="store_true", help="Plot all cores CSV files in results/")
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

    # Determine execution mode
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
        for target in target_csvs:
            csv_path = Path(target)
            plot_cores_file(csv_path, output_dir=output_dir, formats=formats,
                            show_title=show_title, both=both)
    else:
        csv_files = [f for f in sorted(RESULTS_DIR.glob("cores_*.csv"), key=os.path.getmtime, reverse=True) if not f.name.endswith("_raw.csv")]
        if not csv_files:
            print(f"[Cores Plotter] No cores scaling CSV files found in {RESULTS_DIR}")
            print("Run a cores benchmark first, e.g.:")
            print("  python3 scripts/benchmark_cores.py --size 1000000 --threads 1,2,4,8,12,16,20")
            sys.exit(1)

        if args.all:
            for f in csv_files:
                plot_cores_file(f, output_dir=output_dir, formats=formats,
                                show_title=show_title, both=both)
        else:
            print(f"[Cores Plotter] Multiple files found. Plotting latest: {csv_files[0].name}")
            print(f"               (Tip: use --all to plot all cores CSVs)")
            plot_cores_file(csv_files[0], output_dir=output_dir, formats=formats,
                            show_title=show_title, both=both)

    out_summary = str(output_dir) if output_dir else f"{PLOTS_DIR} (PNG) & {PLOTS_PDF_DIR} (PDF)"
    print(f"\n[Cores Plotter] Done! All plots updated in: {out_summary}\n")

if __name__ == "__main__":
    main()
