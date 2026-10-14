#!/usr/bin/env python3
"""
PPolylla Benchmark Suite: Automated runner, sweep evaluator, and plotter
Comparing GPolylla (CUDA GPU) vs MPolylla (OpenMP CPU).
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

def run_single(mesh_type: str, size: int, seed: int, threads: int = 0, output_json: str = None):
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

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[Error] Benchmark run failed with return code {result.returncode}")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None

def run_sweep(mesh_type: str, sizes: list, seed: int, threads: int = 0, plot: bool = True):
    """Executes a sweep over multiple mesh sizes and optionally plots the results."""
    ensure_built()
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    records = []
    for sz in sizes:
        print(f"\n>>> Running sweep point: type={mesh_type}, size={sz}, seed={seed} <<<")
        data = run_single(mesh_type, sz, seed, threads)
        if data:
            records.append({
                "requested_size": data["requested_size"],
                "n_vertices": data["gpu"]["n_vertices"],
                "n_faces": data["gpu"]["n_faces"],
                "n_polygons": data["gpu"]["n_polygons"],
                "gpu_time_ms": data["gpu"]["time_total_algorithm"],
                "gpu_total_with_copies_ms": data["gpu"]["time_total_with_copies"],
                "gpu_copy_in_ms": data["gpu"]["time_copy_to_device"],
                "gpu_copy_out_ms": data["gpu"]["time_back_to_host"],
                "cpu_time_ms": data["cpu"]["time_total_algorithm"],
                "speedup_alg": data["speedup_algorithm"],
                "speedup_with_copies": data["speedup_with_copies"],
                # Phase Breakdown GPU
                "gpu_label_max": data["gpu"]["time_label_max_edges"],
                "gpu_label_frontier": data["gpu"]["time_label_frontier_edges"],
                "gpu_repair_frontier": data["gpu"]["time_label_extra_frontier_edge"],
                "gpu_travel_1": data["gpu"]["time_traversal_1"],
                "gpu_travel_2": data["gpu"]["time_traversal_2"],
                "gpu_overwrite_seed": data["gpu"]["time_overwrite_seed"],
                # Phase Breakdown CPU
                "cpu_label_max": data["cpu"]["time_label_max_edges"],
                "cpu_label_frontier": data["cpu"]["time_label_frontier_edges"],
                "cpu_repair_frontier": data["cpu"]["time_label_extra_frontier_edge"],
                "cpu_travel_1": data["cpu"]["time_traversal_1"],
                "cpu_travel_2": data["cpu"]["time_traversal_2"],
                "cpu_overwrite_seed": data["cpu"]["time_overwrite_seed"]
            })

    if not records:
        print("[Sweep] No benchmark data collected.")
        return

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / f"sweep_{mesh_type}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Sweep] Summary table saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"                     SWEEP SUMMARY ({mesh_type.upper()}, SEED={seed})")
    print("=" * 95)
    print(f"{'Size (N)':>12} | {'GPU (ms)':>12} | {'GPU+Trans (ms)':>14} | {'CPU (ms)':>12} | {'Speedup Alg':>12} | {'Speedup Total':>14}")
    print("-" * 95)
    for _, row in df.iterrows():
        print(f"{int(row['requested_size']):>12} | {row['gpu_time_ms']:>12.3f} | {row['gpu_total_with_copies_ms']:>14.3f} | {row['cpu_time_ms']:>12.3f} | {row['speedup_alg']:>11.2f}x | {row['speedup_with_copies']:>13.2f}x")
    print("=" * 95 + "\n")

    if plot:
        generate_plots(df, mesh_type, seed)

def generate_plots(df: pd.DataFrame, mesh_type: str, seed: int):
    """Generates comparison charts."""
    PLOTS_DIR.mkdir(exist_ok=True)

    # 1. Speedup Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["n_vertices"], df["speedup_alg"], marker="o", color="#2ca02c", linewidth=2, label="Speedup (Algorithm only)")
    plt.plot(df["n_vertices"], df["speedup_with_copies"], marker="s", color="#d62728", linestyle="--", linewidth=2, label="Speedup (With Transfers)")
    plt.axhline(1.0, color="gray", linestyle=":", label="1.0x Parity")
    plt.xlabel("Number of Vertices", fontsize=12)
    plt.ylabel("Speedup (CPU Time / GPU Time)", fontsize=12)
    plt.title(f"Polylla Speedup: GPolylla (GPU) vs MPolylla (CPU)\nMesh: {mesh_type.capitalize()}, Seed: {seed}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    speedup_path = PLOTS_DIR / f"speedup_{mesh_type}.png"
    plt.savefig(speedup_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved speedup plot to: {speedup_path}")

    # 2. Time Comparison Plot (Log Scale)
    plt.figure(figsize=(8, 5))
    plt.plot(df["n_vertices"], df["cpu_time_ms"], marker="o", color="#1f77b4", linewidth=2, label="CPU (MPolylla)")
    plt.plot(df["n_vertices"], df["gpu_total_with_copies_ms"], marker="^", color="#ff7f0e", linestyle="--", linewidth=2, label="GPU (GPolylla + Transfers)")
    plt.plot(df["n_vertices"], df["gpu_time_ms"], marker="s", color="#2ca02c", linewidth=2, label="GPU (GPolylla Alg Only)")
    plt.xlabel("Number of Vertices", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.yscale("log")
    plt.title(f"Polylla Execution Time: GPU vs CPU (Log Scale)\nMesh: {mesh_type.capitalize()}, Seed: {seed}", fontsize=13)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    time_path = PLOTS_DIR / f"time_{mesh_type}.png"
    plt.savefig(time_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved time comparison plot to: {time_path}")

def main():
    parser = argparse.ArgumentParser(description="PPolylla Parallel Benchmark Suite (GPolylla vs MPolylla)")
    parser.add_argument("-t", "--type", choices=["grid", "uniform"], default="uniform", help="Mesh type (grid or uniform)")
    parser.add_argument("-s", "--size", type=int, default=100000, help="Number of vertices for single run (default: 100000)")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-j", "--threads", type=int, default=0, help="OpenMP threads (default: 0 = all cores)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over multiple sizes")
    parser.add_argument("--sizes", type=str, default="10000,50000,100000,500000,1000000", help="Comma-separated sizes for sweep")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    if args.sweep:
        sizes = [int(x.strip()) for x in args.sizes.split(",")]
        run_sweep(args.type, sizes, args.seed, args.threads, plot=not args.no_plot)
    else:
        run_single(args.type, args.size, args.seed, args.threads, args.output)

if __name__ == "__main__":
    main()
