#!/usr/bin/env python3
"""
PPolylla Benchmark Suite: Automated runner, sweep evaluator, and plotter
Comparing:
  1. Sequential CPU (Polylla-Mesh-DCEL) [Baseline]
  2. OpenMP Multi-Core CPU (MPolylla)
  3. CUDA GPU (GPolylla)
"""

import os
import sys
import argparse
import subprocess
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
BINARY = BUILD_DIR / "ppolylla_benchmark"

def ensure_built():
    """Compiles the benchmark executable if it does not exist."""
    BUILD_DIR.mkdir(exist_ok=True)
    if not BINARY.exists():
        print(f"[Build] {BINARY} not found. Running CMake and Make...")
        cmd_cmake = ["cmake", "-DCMAKE_CUDA_ARCHITECTURES=89", ".."]
        subprocess.check_call(cmd_cmake, cwd=BUILD_DIR)
        cmd_make = ["make", f"-j{os.cpu_count()}"]
        subprocess.check_call(cmd_make, cwd=BUILD_DIR)
        print("[Build] Build completed successfully.\n")

def run_single(mesh_type: str, size: int, seed: int, threads: int = 0, no_seq: bool = False, output_json: str = None):
    """Executes a single benchmark run."""
    ensure_built()
    RESULTS_DIR.mkdir(exist_ok=True)

    if not output_json:
        output_json = RESULTS_DIR / f"benchmark_{mesh_type}_{size}_seed{seed}.json"
    else:
        output_json = Path(output_json)

    cmd = [
        str(BINARY),
        "--type", mesh_type,
        "--size", str(size),
        "--seed", str(seed),
        "--output", str(output_json)
    ]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    if no_seq:
        cmd.append("--no-seq")

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[Error] Benchmark run failed with return code {result.returncode}")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None

def run_sweep(mesh_type: str, sizes: list, seed: int, threads: int = 0, no_seq: bool = False, plot: bool = True):
    """Executes a sweep over multiple mesh sizes and optionally plots the results."""
    ensure_built()
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    records = []
    for sz in sizes:
        print(f"\n>>> Running sweep point: type={mesh_type}, size={sz}, seed={seed} <<<")
        data = run_single(mesh_type, sz, seed, threads, no_seq=no_seq)
        if data:
            has_seq = data.get("has_sequential_baseline", False)
            has_task = data.get("has_omp_task", False)
            seq_time = data["sequential"]["time_total_algorithm"] if has_seq else None
            loops_data = data.get("cpu_loops", data.get("cpu", {}))
            tasks_data = data.get("cpu_tasks", {})

            rec = {
                "requested_size": data["requested_size"],
                "n_vertices": data["gpu"]["n_vertices"],
                "n_faces": data["gpu"]["n_faces"],
                "n_polygons": data["gpu"]["n_polygons"],
                "seq_time_ms": seq_time,
                "omp_loops_time_ms": loops_data.get("time_total_algorithm", 0.0),
                "omp_tasks_time_ms": tasks_data.get("time_total_algorithm", 0.0) if has_task else None,
                "gpu_time_ms": data["gpu"]["time_total_algorithm"],
                "gpu_total_with_copies_ms": data["gpu"]["time_total_with_copies"],
                "gpu_copy_in_ms": data["gpu"]["time_copy_to_device"],
                "gpu_copy_out_ms": data["gpu"]["time_back_to_host"],
                "speedup_gpu_vs_seq": data.get("speedup_gpu_vs_sequential", 0.0),
                "speedup_gpu_with_copies_vs_seq": data.get("speedup_gpu_with_copies_vs_sequential", 0.0),
                "speedup_omp_vs_seq": data.get("speedup_omp_vs_sequential", 0.0),
                "speedup_task_vs_seq": data.get("speedup_omp_task_vs_sequential", 0.0),
                "speedup_gpu_vs_omp": data.get("speedup_gpu_vs_omp", 0.0),
                "speedup_gpu_vs_task": data.get("speedup_gpu_vs_omp_task", 0.0),
                "speedup_task_vs_omp": data.get("speedup_omp_task_vs_omp", 0.0),
            }
            records.append(rec)

    if not records:
        print("[Sweep] No benchmark data collected.")
        return

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / f"sweep_{mesh_type}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Sweep] Summary table saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 135)
    print(f"                                      SWEEP SUMMARY ({mesh_type.upper()}, SEED={seed})")
    print("=" * 135)
    header = f"{'Size (N)':>10} | {'Seq (ms)':>11} | {'OMP Loop':>11} | {'OMP Task':>11} | {'GPU (ms)':>11} | {'GPU+Trans':>11} | {'GPU/Seq':>9} | {'Task/Seq':>9} | {'Task/Loop':>9}"
    print(header)
    print("-" * 135)
    for _, row in df.iterrows():
        seq_str = f"{row['seq_time_ms']:>11.3f}" if row['seq_time_ms'] is not None else f"{'-':>11}"
        task_str = f"{row['omp_tasks_time_ms']:>11.3f}" if row['omp_tasks_time_ms'] is not None else f"{'-':>11}"
        gpu_seq_str = f"{row['speedup_gpu_vs_seq']:>8.2f}x" if row['speedup_gpu_vs_seq'] > 0 else f"{'-':>9}"
        task_seq_str = f"{row['speedup_task_vs_seq']:>8.2f}x" if row['speedup_task_vs_seq'] > 0 else f"{'-':>9}"
        task_omp_str = f"{row['speedup_task_vs_omp']:>8.2f}x" if row['speedup_task_vs_omp'] > 0 else f"{'-':>9}"
        print(f"{int(row['requested_size']):>10} | {seq_str} | {row['omp_loops_time_ms']:>11.3f} | {task_str} | {row['gpu_time_ms']:>11.3f} | {row['gpu_total_with_copies_ms']:>11.3f} | {gpu_seq_str} | {task_seq_str} | {task_omp_str}")
    print("=" * 135 + "\n")

    if plot:
        generate_plots(df, mesh_type, seed)

def generate_plots(df: pd.DataFrame, mesh_type: str, seed: int):
    """Generates comparison charts."""
    PLOTS_DIR.mkdir(exist_ok=True)

    has_seq = df["seq_time_ms"].notnull().all() and (df["speedup_gpu_vs_seq"] > 0).any()
    has_task = "omp_tasks_time_ms" in df.columns and df["omp_tasks_time_ms"].notnull().any()

    # 1. Speedup Plot
    plt.figure(figsize=(10, 6))
    if has_seq:
        plt.plot(df["n_vertices"], df["speedup_gpu_vs_seq"], marker="o", color="#2ca02c", linewidth=2.5, label="GPU (GPolylla) vs Sequential Baseline")
        plt.plot(df["n_vertices"], df["speedup_gpu_with_copies_vs_seq"], marker="s", color="#8c564b", linestyle="--", linewidth=2, label="GPU (With Transfers) vs Sequential")
        plt.plot(df["n_vertices"], df["speedup_omp_vs_seq"], marker="^", color="#1f77b4", linewidth=2, label="CPU Loops (MPolylla main) vs Sequential")
        if has_task:
            plt.plot(df["n_vertices"], df["speedup_task_vs_seq"], marker="v", color="#9467bd", linewidth=2, label="CPU Tasks (MPolylla Task) vs Sequential")
    plt.plot(df["n_vertices"], df["speedup_gpu_vs_omp"], marker="d", color="#d62728", linestyle="-.", linewidth=2, label="GPU vs OpenMP Loops")
    if has_task:
        plt.plot(df["n_vertices"], df["speedup_gpu_vs_task"], marker="p", color="#e377c2", linestyle=":", linewidth=2, label="GPU vs OpenMP Tasks")
    plt.axhline(1.0, color="gray", linestyle=":", label="1.0x Parity")
    plt.xlabel("Number of Vertices", fontsize=12)
    plt.ylabel("Speedup Factor (X)", fontsize=12)
    plt.title(f"Polylla Parallel Speedup\nMesh: {mesh_type.capitalize()}, Seed: {seed}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    speedup_path = PLOTS_DIR / f"speedup_{mesh_type}.png"
    plt.savefig(speedup_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved speedup plot to: {speedup_path}")

    # 2. Time Comparison Plot (Log Scale)
    plt.figure(figsize=(10, 6))
    if has_seq:
        plt.plot(df["n_vertices"], df["seq_time_ms"], marker="x", color="#7f7f7f", linewidth=2, label="Sequential Baseline (Polylla-Mesh-DCEL)")
    plt.plot(df["n_vertices"], df["omp_loops_time_ms"], marker="^", color="#1f77b4", linewidth=2, label="OpenMP Loops (MPolylla main)")
    if has_task:
        plt.plot(df["n_vertices"], df["omp_tasks_time_ms"], marker="v", color="#9467bd", linewidth=2, label="OpenMP Tasks (MPolylla Task)")
    plt.plot(df["n_vertices"], df["gpu_total_with_copies_ms"], marker="s", color="#ff7f0e", linestyle="--", linewidth=2, label="GPU Total (GPolylla + Transfers)")
    plt.plot(df["n_vertices"], df["gpu_time_ms"], marker="o", color="#2ca02c", linewidth=2.5, label="GPU Alg Only (GPolylla)")
    plt.xlabel("Number of Vertices", fontsize=12)
    plt.ylabel("Execution Time in ms (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.title(f"Polylla Execution Time: Sequential vs OpenMP Loops vs Tasks vs GPU\nMesh: {mesh_type.capitalize()}, Seed: {seed}", fontsize=13)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    time_path = PLOTS_DIR / f"time_{mesh_type}.png"
    plt.savefig(time_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved time comparison plot to: {time_path}")

def main():
    parser = argparse.ArgumentParser(description="PPolylla Parallel Benchmark Suite (Sequential vs OpenMP vs CUDA)")
    parser.add_argument("-t", "--type", choices=["grid", "uniform"], default="uniform", help="Mesh type (grid or uniform)")
    parser.add_argument("-s", "--size", type=int, default=100000, help="Number of vertices for single run (default: 100000)")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-j", "--threads", type=int, default=0, help="OpenMP threads (default: 0 = all cores)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no-seq", action="store_true", help="Skip sequential CPU baseline")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over multiple sizes")
    parser.add_argument("--sizes", type=str, default="10000,50000,100000,500000,1000000", help="Comma-separated sizes for sweep")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    if args.sweep:
        sizes = [int(x.strip()) for x in args.sizes.split(",")]
        run_sweep(args.type, sizes, args.seed, args.threads, no_seq=args.no_seq, plot=not args.no_plot)
    else:
        run_single(args.type, args.size, args.seed, args.threads, no_seq=args.no_seq, output_json=args.output)

if __name__ == "__main__":
    main()
