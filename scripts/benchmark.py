#!/usr/bin/env python3
"""
PPolylla Benchmark Suite: Automated runner, sweep evaluator, and plotter
Comparing:
  1. Sequential CPU (Polylla-Mesh-DCEL) [Baseline]
  2. OpenMP Multi-Core CPU (MPolylla DCEL, Data, Tasks)
  3. CUDA GPU (GPolylla)
Supports multi-run repetitions (average time and std dev) and multi-seed evaluations.
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
BINARY = BUILD_DIR / "ppolylla_benchmark"

def ensure_built():
    """Compiles the benchmark executable if it does not exist."""
    BUILD_DIR.mkdir(exist_ok=True)
    if not BINARY.exists():
        required_submodules = [
            PROJECT_ROOT / "submodules/GPolylla/CMakeLists.txt",
            PROJECT_ROOT / "submodules/MPolylla/CMakeLists.txt",
            PROJECT_ROOT / "submodules/Polylla-Mesh-DCEL/CMakeLists.txt",
            PROJECT_ROOT / "submodules/MPolylla-Task/CMakeLists.txt",
            PROJECT_ROOT / "submodules/MPolylla-Mesh-DCEL/CMakeLists.txt",
        ]
        if any(not p.exists() for p in required_submodules):
            print("[Submodules] Initializing and updating git submodules...")
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=PROJECT_ROOT)

        cuda_arch = os.environ.get("CUDA_ARCH", "native")
        print(f"[Build] {BINARY} not found. Running CMake (CUDA_ARCH={cuda_arch}) and Make...")
        cmd_cmake = ["cmake", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}", ".."]
        if os.environ.get("NO_CUDA") == "1" or "--no-gpu" in sys.argv:
            cmd_cmake.append("-DENABLE_CUDA=OFF")
        subprocess.check_call(cmd_cmake, cwd=BUILD_DIR)
        cmd_make = ["make", f"-j{os.cpu_count() or 1}"]
        subprocess.check_call(cmd_make, cwd=BUILD_DIR)
        print("[Build] Build completed successfully.\n")

def run_single(mesh_type: str, size: int, seed: int, runs: int = 1, threads: int = 0, cores: str = "", no_seq: bool = False, no_gpu: bool = False, gpu_only: bool = False, output_json: str = None):
    """Executes a single benchmark run with optional repetitions."""
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
        "--runs", str(runs),
        "--output", str(output_json)
    ]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    if cores:
        cmd.extend(["--cores", str(cores)])
    if gpu_only:
        cmd.append("--gpu-only")
    else:
        if no_seq:
            cmd.append("--no-seq")
        if no_gpu:
            cmd.append("--no-gpu")

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        if result.returncode == -9:
            print(f"[Error] Benchmark run failed with return code -9 (SIGKILL / Out-Of-Memory Killer).")
            print(f"        The system ran out of RAM! The kernel killed the process. Try reducing --size.")
        else:
            print(f"[Error] Benchmark run failed with return code {result.returncode}")
        return None

    if output_json.exists():
        with open(output_json) as f:
            return json.load(f)
    return None

def run_sweep(mesh_type: str, sizes: list, seeds: list, runs: int = 1, threads: int = 0, cores: str = "", no_seq: bool = False, no_gpu: bool = False, gpu_only: bool = False, plot: bool = True):
    """Executes a sweep over multiple mesh sizes and seeds, and optionally plots results."""
    ensure_built()
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    records = []
    for sz in sizes:
        for sd in seeds:
            print(f"\n>>> Running sweep point: type={mesh_type}, size={sz}, seed={sd}, runs={runs} <<<")
            data = run_single(mesh_type, sz, sd, runs=runs, threads=threads, cores=cores, no_seq=no_seq, no_gpu=no_gpu, gpu_only=gpu_only)
            if data:
                has_seq = data.get("has_sequential_baseline", False)
                has_dcel = data.get("has_omp_dcel", False)
                has_omp = data.get("has_omp_data", False)
                has_task = data.get("has_omp_task", False)
                has_gpu = data.get("has_gpu", False)
                seq_dict = data.get("sequential", {})
                dcel_dict = data.get("cpu_dcel", {})
                data_dict = data.get("cpu_data", data.get("cpu_loops", data.get("cpu", {})))
                tasks_dict = data.get("cpu_tasks", {})
                gpu_dict = data.get("gpu", {})

                seq_time = seq_dict.get("time_total_algorithm") if has_seq else None
                seq_std = seq_dict.get("time_total_algorithm_std", 0.0) if has_seq else 0.0

                omp_dcel_time = dcel_dict.get("time_total_algorithm", 0.0) if has_dcel else None
                omp_dcel_std = dcel_dict.get("time_total_algorithm_std", 0.0) if has_dcel else 0.0

                omp_data_time = data_dict.get("time_total_algorithm", 0.0) if has_omp else None
                omp_data_std = data_dict.get("time_total_algorithm_std", 0.0) if has_omp else 0.0

                omp_task_time = tasks_dict.get("time_total_algorithm", 0.0) if has_task else None
                omp_task_std = tasks_dict.get("time_total_algorithm_std", 0.0) if has_task else 0.0

                gpu_time = gpu_dict.get("time_total_algorithm", 0.0) if has_gpu else None
                gpu_std = gpu_dict.get("time_total_algorithm_std", 0.0) if has_gpu else 0.0

                gpu_total_with_copies = gpu_dict.get("time_total_with_copies", 0.0) if has_gpu else None
                gpu_copies_std = gpu_dict.get("time_total_with_copies_std", 0.0) if has_gpu else 0.0

                ref_dict = gpu_dict if has_gpu and gpu_dict.get("n_vertices", 0) > 0 else (data_dict if data_dict.get("n_vertices", 0) > 0 else seq_dict)

                rec = {
                    "requested_size": data["requested_size"],
                    "seed": sd,
                    "runs": runs,
                    "n_vertices": ref_dict.get("n_vertices", sz),
                    "n_faces": ref_dict.get("n_faces", 0),
                    "n_polygons": ref_dict.get("n_polygons", 0),
                    "cpu_threads": data.get("cpu_threads", threads),
                    "cpu_cores_affinity": data.get("cpu_cores_affinity", "all"),
                    "seq_time_ms": seq_time,
                    "seq_time_std_ms": seq_std,
                    "omp_dcel_time_ms": omp_dcel_time,
                    "omp_dcel_time_std_ms": omp_dcel_std,
                    "omp_data_time_ms": omp_data_time,
                    "omp_data_time_std_ms": omp_data_std,
                    "omp_loops_time_ms": omp_data_time,  # backwards compatibility
                    "omp_tasks_time_ms": omp_task_time,
                    "omp_tasks_time_std_ms": omp_task_std,
                    "gpu_time_ms": gpu_time,
                    "gpu_time_std_ms": gpu_std,
                    "gpu_total_with_copies_ms": gpu_total_with_copies,
                    "gpu_total_with_copies_std_ms": gpu_copies_std,
                    "gpu_copy_in_ms": gpu_dict.get("time_copy_to_device", 0.0),
                    "gpu_copy_out_ms": gpu_dict.get("time_back_to_host", 0.0),
                    "speedup_gpu_vs_seq": data.get("speedup_gpu_vs_sequential", 0.0),
                    "speedup_gpu_with_copies_vs_seq": data.get("speedup_gpu_with_copies_vs_sequential", 0.0),
                    "speedup_dcel_vs_seq": data.get("speedup_omp_dcel_vs_sequential", 0.0),
                    "speedup_data_vs_seq": data.get("speedup_omp_data_vs_sequential", data.get("speedup_omp_vs_sequential", 0.0)),
                    "speedup_omp_vs_seq": data.get("speedup_omp_vs_sequential", 0.0),
                    "speedup_task_vs_seq": data.get("speedup_omp_task_vs_sequential", 0.0),
                    "speedup_gpu_vs_dcel": data.get("speedup_gpu_vs_omp_dcel", 0.0),
                    "speedup_gpu_vs_omp": data.get("speedup_gpu_vs_omp", 0.0),
                    "speedup_gpu_vs_task": data.get("speedup_gpu_vs_omp_task", 0.0),
                    "speedup_task_vs_omp": data.get("speedup_omp_task_vs_omp", 0.0),
                }
                records.append(rec)

    if not records:
        print("[Sweep] No benchmark data collected.")
        return

    raw_df = pd.DataFrame(records)

    is_multi_seed = len(seeds) > 1
    seeds_slug = "_".join(str(s) for s in seeds)
    seeds_label = ", ".join(str(s) for s in seeds)

    if is_multi_seed:
        # Save raw per-seed data
        raw_csv_path = RESULTS_DIR / f"sweep_{mesh_type}_seeds_{seeds_slug}_raw.csv"
        raw_df.to_csv(raw_csv_path, index=False)

        # Aggregate across seeds for each requested_size
        summary_records = []
        for sz, group in raw_df.groupby("requested_size", sort=False):
            n_v = int(round(group["n_vertices"].mean()))
            n_f = int(round(group["n_faces"].mean()))
            n_p = int(round(group["n_polygons"].mean()))

            has_seq = group["seq_time_ms"].notnull().all()
            has_dcel = group["omp_dcel_time_ms"].notnull().all()
            has_task = group["omp_tasks_time_ms"].notnull().all()

            mean_seq = group["seq_time_ms"].mean() if has_seq else None
            std_seq = group["seq_time_ms"].std() if (has_seq and len(group) > 1) else (group["seq_time_std_ms"].mean() if has_seq else 0.0)

            mean_dcel = group["omp_dcel_time_ms"].mean() if has_dcel else None
            std_dcel = group["omp_dcel_time_ms"].std() if (has_dcel and len(group) > 1) else (group["omp_dcel_time_std_ms"].mean() if has_dcel else 0.0)

            mean_data = group["omp_data_time_ms"].mean()
            std_data = group["omp_data_time_ms"].std() if len(group) > 1 else group["omp_data_time_std_ms"].mean()

            mean_task = group["omp_tasks_time_ms"].mean() if has_task else None
            std_task = group["omp_tasks_time_ms"].std() if (has_task and len(group) > 1) else (group["omp_tasks_time_std_ms"].mean() if has_task else 0.0)

            mean_gpu = group["gpu_time_ms"].mean()
            std_gpu = group["gpu_time_ms"].std() if len(group) > 1 else group["gpu_time_std_ms"].mean()

            mean_gpu_copies = group["gpu_total_with_copies_ms"].mean()
            std_gpu_copies = group["gpu_total_with_copies_ms"].std() if len(group) > 1 else group["gpu_total_with_copies_std_ms"].mean()

            # Speedups based on mean times
            sp_gpu_seq = (mean_seq / mean_gpu) if (has_seq and mean_gpu > 0) else 0.0
            sp_gpu_copies_seq = (mean_seq / mean_gpu_copies) if (has_seq and mean_gpu_copies > 0) else 0.0
            sp_dcel_seq = (mean_seq / mean_dcel) if (has_seq and has_dcel and mean_dcel > 0) else 0.0
            sp_data_seq = (mean_seq / mean_data) if (has_seq and mean_data > 0) else 0.0
            sp_task_seq = (mean_seq / mean_task) if (has_seq and has_task and mean_task > 0) else 0.0

            s_rec = {
                "requested_size": sz,
                "n_vertices": n_v,
                "n_faces": n_f,
                "n_polygons": n_p,
                "seeds_count": len(seeds),
                "seeds_list": seeds_label,
                "runs_per_seed": runs,
                "runs": runs,
                "cpu_threads": group["cpu_threads"].iloc[0],
                "cpu_cores_affinity": group["cpu_cores_affinity"].iloc[0],
                "seq_time_ms": mean_seq,
                "seq_time_std_ms": std_seq if pd.notnull(std_seq) else 0.0,
                "omp_dcel_time_ms": mean_dcel,
                "omp_dcel_time_std_ms": std_dcel if pd.notnull(std_dcel) else 0.0,
                "omp_data_time_ms": mean_data,
                "omp_data_time_std_ms": std_data if pd.notnull(std_data) else 0.0,
                "omp_loops_time_ms": mean_data,
                "omp_tasks_time_ms": mean_task,
                "omp_tasks_time_std_ms": std_task if pd.notnull(std_task) else 0.0,
                "gpu_time_ms": mean_gpu,
                "gpu_time_std_ms": std_gpu if pd.notnull(std_gpu) else 0.0,
                "gpu_total_with_copies_ms": mean_gpu_copies,
                "gpu_total_with_copies_std_ms": std_gpu_copies if pd.notnull(std_gpu_copies) else 0.0,
                "speedup_gpu_vs_seq": sp_gpu_seq,
                "speedup_gpu_with_copies_vs_seq": sp_gpu_copies_seq,
                "speedup_dcel_vs_seq": sp_dcel_seq,
                "speedup_data_vs_seq": sp_data_seq,
                "speedup_omp_vs_seq": sp_data_seq,
                "speedup_task_vs_seq": sp_task_seq,
            }
            summary_records.append(s_rec)

        df = pd.DataFrame(summary_records)
        csv_path = RESULTS_DIR / f"sweep_{mesh_type}_seeds_{seeds_slug}.csv"
        df.to_csv(csv_path, index=False)
        # Also update default sweep file
        df.to_csv(RESULTS_DIR / f"sweep_{mesh_type}.csv", index=False)
        print(f"\n[Sweep] Multi-seed summary saved to: {csv_path}")
    else:
        df = raw_df
        csv_path = RESULTS_DIR / f"sweep_{mesh_type}_seed{seeds[0]}.csv"
        df.to_csv(csv_path, index=False)
        df.to_csv(RESULTS_DIR / f"sweep_{mesh_type}.csv", index=False)
        print(f"\n[Sweep] Summary table saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 160)
    title_run_info = f"{runs} RUNS/SAMPLE" if runs > 1 else "1 RUN"
    if is_multi_seed:
        title_hdr = f"SWEEP SUMMARY ({mesh_type.upper()}, SEEDS=[{seeds_label}], {title_run_info})"
    else:
        title_hdr = f"SWEEP SUMMARY ({mesh_type.upper()}, SEED={seeds[0]}, {title_run_info})"
    print(f"{title_hdr:^160}")
    print("=" * 160)

    show_std = (runs > 1) or is_multi_seed
    if show_std:
        header = f"{'Size (N)':>10} | {'Seq (ms)':>16} | {'OMP DCEL':>16} | {'OMP Data':>16} | {'OMP Task':>16} | {'GPU (ms)':>16} | {'GPU+Trans':>16} | {'GPU/Seq':>8} | {'Task/Seq':>8} | {'Data/Seq':>8} | {'DCEL/Seq':>8}"
    else:
        header = f"{'Size (N)':>10} | {'Seq (ms)':>11} | {'OMP DCEL':>11} | {'OMP Data':>11} | {'OMP Task':>11} | {'GPU (ms)':>11} | {'GPU+Trans':>11} | {'GPU/Seq':>8} | {'Task/Seq':>8} | {'Data/Seq':>8} | {'DCEL/Seq':>8}"
    print(header)
    print("-" * 160)

    for _, row in df.iterrows():
        if show_std:
            def fmt_val_std(val, std):
                if val is None or pd.isna(val):
                    return f"{'-':>16}"
                return f"{val:7.2f} ± {std:5.2f}"

            seq_str = fmt_val_std(row.get("seq_time_ms"), row.get("seq_time_std_ms", 0.0))
            dcel_str = fmt_val_std(row.get("omp_dcel_time_ms"), row.get("omp_dcel_time_std_ms", 0.0))
            data_str = fmt_val_std(row.get("omp_data_time_ms"), row.get("omp_data_time_std_ms", 0.0))
            task_str = fmt_val_std(row.get("omp_tasks_time_ms"), row.get("omp_tasks_time_std_ms", 0.0))
            gpu_str = fmt_val_std(row.get("gpu_time_ms"), row.get("gpu_time_std_ms", 0.0)) if (row.get("gpu_time_ms") is not None and not pd.isna(row.get("gpu_time_ms")) and row.get("gpu_time_ms", 0.0) > 0) else f"{'-':>16}"
            gpu_cp_str = fmt_val_std(row.get("gpu_total_with_copies_ms"), row.get("gpu_total_with_copies_std_ms", 0.0)) if (row.get("gpu_total_with_copies_ms") is not None and not pd.isna(row.get("gpu_total_with_copies_ms")) and row.get("gpu_total_with_copies_ms", 0.0) > 0) else f"{'-':>16}"
        else:
            seq_str = f"{row['seq_time_ms']:>11.3f}" if (row.get('seq_time_ms') is not None and not pd.isna(row['seq_time_ms'])) else f"{'-':>11}"
            dcel_str = f"{row['omp_dcel_time_ms']:>11.3f}" if (row.get('omp_dcel_time_ms') is not None and not pd.isna(row['omp_dcel_time_ms'])) else f"{'-':>11}"
            data_str = f"{row['omp_data_time_ms']:>11.3f}" if (row.get('omp_data_time_ms') is not None and not pd.isna(row['omp_data_time_ms'])) else f"{'-':>11}"
            task_str = f"{row['omp_tasks_time_ms']:>11.3f}" if (row.get('omp_tasks_time_ms') is not None and not pd.isna(row['omp_tasks_time_ms'])) else f"{'-':>11}"
            gpu_str = f"{row['gpu_time_ms']:>11.3f}" if (row.get('gpu_time_ms') is not None and not pd.isna(row['gpu_time_ms']) and row['gpu_time_ms'] > 0) else f"{'-':>11}"
            gpu_cp_str = f"{row['gpu_total_with_copies_ms']:>11.3f}" if (row.get('gpu_total_with_copies_ms') is not None and not pd.isna(row['gpu_total_with_copies_ms']) and row['gpu_total_with_copies_ms'] > 0) else f"{'-':>11}"

        gpu_seq_str = f"{row['speedup_gpu_vs_seq']:>7.2f}x" if row['speedup_gpu_vs_seq'] > 0 else f"{'-':>8}"
        task_seq_str = f"{row['speedup_task_vs_seq']:>7.2f}x" if row['speedup_task_vs_seq'] > 0 else f"{'-':>8}"
        data_seq_str = f"{row['speedup_data_vs_seq']:>7.2f}x" if row['speedup_data_vs_seq'] > 0 else f"{'-':>8}"
        dcel_seq_str = f"{row.get('speedup_dcel_vs_seq', 0.0):>7.2f}x" if row.get('speedup_dcel_vs_seq', 0.0) > 0 else f"{'-':>8}"

        print(f"{int(row['requested_size']):>10} | {seq_str} | {dcel_str} | {data_str} | {task_str} | {gpu_str} | {gpu_cp_str} | {gpu_seq_str} | {task_seq_str} | {data_seq_str} | {dcel_seq_str}")
    print("=" * 160 + "\n")

    if plot:
        generate_plots(df, mesh_type, seed=seeds[0], seeds_info=(seeds_label if is_multi_seed else ""), runs=runs)

def generate_plots(df: pd.DataFrame, mesh_type: str, seed: int = 42, seeds_info: str = "", runs: int = 1):
    """Delegates to standalone plot module for unified plotting (generating both PNG and PDF)."""
    try:
        from plot import plot_benchmark_data, PLOTS_DIR, PLOTS_PDF_DIR
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot import plot_benchmark_data, PLOTS_DIR, PLOTS_PDF_DIR

    print("[Plots] Generating standard PNG visualizations in plots/...")
    plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=PLOTS_DIR, seeds_info=seeds_info, runs=runs, formats=["png"], show_title=True)
    print("[Plots] Generating publication-ready vector PDFs (no title) in plots_pdf/...")
    plot_benchmark_data(df, mesh_type=mesh_type, seed=seed, output_dir=PLOTS_PDF_DIR, seeds_info=seeds_info, runs=runs, formats=["pdf"], show_title=False)

def main():
    parser = argparse.ArgumentParser(description="PPolylla Parallel Benchmark Suite (Sequential vs OpenMP vs CUDA)")
    parser.add_argument("-t", "--type", choices=["grid", "uniform"], default="uniform", help="Mesh type (grid or uniform)")
    parser.add_argument("-s", "--size", type=int, default=100000, help="Number of vertices for single run (default: 100000)")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed for single run (default: 42)")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated list of random seeds (e.g. '42,43,44'). Overrides --seed.")
    parser.add_argument("--runs", "--repeat", type=int, default=1, dest="runs", help="Number of repetitions per experiment to calculate average time (default: 1)")
    parser.add_argument("-j", "--threads", type=int, default=0, help="OpenMP threads (default: 0 = all cores or count of --cores)")
    parser.add_argument("-c", "--cores", type=str, default="", help="CPU core affinity for OpenMP (e.g. '0-7', '8-19', 'p-cores', 'e-cores')")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no-seq", action="store_true", help="Skip sequential CPU baseline")
    parser.add_argument("--no-gpu", action="store_true", help="Skip CUDA GPU benchmark")
    parser.add_argument("--gpu-only", "--only-gpu", action="store_true", dest="gpu_only", help="Run only CUDA GPU benchmark (skip all CPU implementations)")
    parser.add_argument("--sweep", action="store_true", help="Run sweep over multiple sizes")
    parser.add_argument("--sizes", type=str, default="10000,50000,100000,500000,1000000", help="Comma-separated sizes for sweep")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing CSV results")
    parser.add_argument("--csv", type=str, default=None, help="Specific CSV file to plot (when using --plot-only)")

    args = parser.parse_args()

    if args.plot_only or args.csv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import plot as plotter
        if args.csv:
            plotter.plot_file(Path(args.csv))
        else:
            plotter.main()
        return

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [args.seed]

    if args.sweep:
        sizes = [int(x.strip()) for x in args.sizes.split(",")]
        run_sweep(args.type, sizes, seeds, runs=args.runs, threads=args.threads, cores=args.cores, no_seq=args.no_seq, no_gpu=args.no_gpu, gpu_only=args.gpu_only, plot=not args.no_plot)
    else:
        if len(seeds) == 1:
            run_single(args.type, args.size, seeds[0], runs=args.runs, threads=args.threads, cores=args.cores, no_seq=args.no_seq, no_gpu=args.no_gpu, gpu_only=args.gpu_only, output_json=args.output)
        else:
            cmd = [
                str(BINARY),
                "--type", args.type,
                "--size", str(args.size),
                "--seeds", ",".join(str(s) for s in seeds),
                "--runs", str(args.runs),
            ]
            if args.threads > 0:
                cmd.extend(["--threads", str(args.threads)])
            if args.cores:
                cmd.extend(["--cores", str(cores)])
            if args.gpu_only:
                cmd.append("--gpu-only")
            else:
                if args.no_seq:
                    cmd.append("--no-seq")
                if args.no_gpu:
                    cmd.append("--no-gpu")
            if args.output:
                cmd.extend(["--output", str(args.output)])
            print(f"\n[Run Multi-Seed] Executing: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
