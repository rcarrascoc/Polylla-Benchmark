#!/usr/bin/env python3
"""
PPolylla OpenMP Cores & Thread Scalability Benchmark Suite
Evaluates strong scaling and parallel efficiency of OpenMP variants:
  1. MPolylla DCEL (MPolylla-Mesh-DCEL)
  2. MPolylla Data (MPolylla branch main)
  3. MPolylla Task (MPolylla branch Task)
Across different thread counts, with optional CPU core affinity (P-cores, E-cores, all),
multi-run averaging (mean and std dev), and multi-seed evaluations.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_PDF_DIR = PROJECT_ROOT / "plots_pdf"
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

def get_default_threads(cores_spec: str = "") -> list:
    """Returns smart default thread steps based on core affinity and CPU topology."""
    total_cpus = os.cpu_count() or 1
    spec_lower = cores_spec.lower().strip()

    if "p-core" in spec_lower:
        # Standard Intel hybrid: 8 P-cores
        return [1, 2, 4, 6, 8]
    elif "e-core" in spec_lower:
        # Standard Intel hybrid: 12 or 16 E-cores
        return [1, 2, 4, 6, 8, 12]
    elif "-" in cores_spec:
        try:
            parts = cores_spec.split("-")
            count = abs(int(parts[1]) - int(parts[0])) + 1
            if count <= 4:
                return list(range(1, count + 1))
            elif count <= 8:
                return [1, 2, 4, 6, 8][:count]
            else:
                steps = [1, 2, 4, 8, 12, 16, 20, 24, 32]
                res = [s for s in steps if s < count]
                res.append(count)
                return sorted(list(set(res)))
        except Exception:
            pass

    if total_cpus <= 4:
        return list(range(1, total_cpus + 1))
    elif total_cpus <= 8:
        return [1, 2, 4, 6, 8][:total_cpus]
    elif total_cpus <= 16:
        return [1, 2, 4, 8, 12, 16]
    elif total_cpus <= 20:
        return [1, 2, 4, 8, 12, 16, 20]
    else:
        steps = [1, 2, 4, 8, 16, 24, 32, 64]
        res = [s for s in steps if s < total_cpus]
        res.append(total_cpus)
        return sorted(list(set(res)))

def check_ram_capacity(size: int) -> bool:
    """Checks if requested mesh size fits safely within available physical RAM."""
    try:
        total_ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        total_ram_gb = total_ram_bytes / (1024**3)
    except Exception:
        total_ram_gb = 32.0

    # DCEL and MeshGen peak memory is approx 460-480 bytes per vertex
    needed_gb = (size * 460) / (1024**3)
    if needed_gb > total_ram_gb * 0.80:
        max_safe = max(1000000, int((total_ram_gb * 0.65 * (1024**3)) / 460) // 5000000 * 5000000)
        print("\n" + "!" * 110)
        print(f"  [RAM WARNING] Requested size N={size:,} requires approx {needed_gb:.1f} GB of RAM.")
        print(f"                This system has only {total_ram_gb:.1f} GB physical RAM.")
        print(f"                Algorithms (especially DCEL) will exceed RAM and be terminated by Linux OOM Killer (code -9).")
        print(f"  --> Recommended maximum size on this system: ~{max_safe:,} vertices (e.g. 10M, 20M, or 40M).")
        print("!" * 110 + "\n")
        return False
    return True

def run_cmd(cmd: list) -> dict:
    """Executes benchmark binary and returns parsed JSON data."""
    print(f"\n[Run] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        if result.returncode == -9:
            print(f"[Error] Benchmark run failed with return code -9 (SIGKILL / Out-Of-Memory Killer).")
            print(f"        The system ran out of RAM! The kernel killed the process. Try reducing --size.")
        else:
            print(f"[Error] Benchmark run failed with return code {result.returncode}")
        return None

    # Find the output json file
    output_idx = -1
    for i, arg in enumerate(cmd):
        if arg in ["--output", "-o"]:
            output_idx = i + 1
            break

    if output_idx != -1 and output_idx < len(cmd):
        out_path = Path(cmd[output_idx])
        if out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    return None

def run_cores_benchmark(mesh_type: str = "uniform", size: int = 1000000,
                        seeds: list = None, runs: int = 1, threads_list: list = None,
                        cores_spec: str = "", no_seq: bool = False, no_gpu: bool = False,
                        plot: bool = True, generate_pdf: bool = True):
    """Orchestrates OpenMP scalability benchmark across thread counts."""
    ensure_built()
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    if generate_pdf:
        PLOTS_PDF_DIR.mkdir(exist_ok=True)

    if seeds is None or len(seeds) == 0:
        seeds = [42]

    if threads_list is None or len(threads_list) == 0:
        threads_list = get_default_threads(cores_spec)

    check_ram_capacity(size)

    print("\n" + "=" * 110)
    print("                  PPolylla OpenMP Cores & Thread Scalability Benchmark Suite                  ")
    print("=" * 110)
    print(f"  Mesh Distribution:    {mesh_type}")
    print(f"  Fixed Mesh Size (N):  {size:,} vertices")
    print(f"  Random Seeds:         {seeds}")
    print(f"  Repetitions per test: {runs} run(s)")
    print(f"  Thread Progression:   {threads_list}")
    print(f"  CPU Core Affinity:    {cores_spec if cores_spec else 'All available cores'}")
    print(f"  Sequential Baseline:  {'Disabled' if no_seq else 'Enabled (run once per seed)'}")
    print(f"  CUDA GPU Reference:   {'Disabled' if no_gpu else 'Enabled (run once per seed)'}")
    print("=" * 110 + "\n")

    # -------------------------------------------------------------------------
    # 1. BASELINE PHASE (Sequential and GPU, executed once per seed)
    # -------------------------------------------------------------------------
    baseline_stats = {}
    for sd in seeds:
        seq_time, seq_std = None, 0.0
        gpu_time, gpu_std = None, 0.0
        n_v, n_f, n_p = size, 0, 0

        if not no_seq or not no_gpu:
            print(f">>> [Baseline Phase] Recording reference baselines for seed {sd} (runs={runs}) <<<")
            base_json = RESULTS_DIR / f"cores_base_{mesh_type}_{size}_seed{sd}.json"
            cmd = [
                str(BINARY),
                "--type", mesh_type,
                "--size", str(size),
                "--seed", str(sd),
                "--runs", str(runs),
                "--threads", "1",
                "--output", str(base_json)
            ]
            if no_seq:
                cmd.append("--no-seq")
            if no_gpu:
                cmd.append("--no-gpu")
            if cores_spec:
                cmd.extend(["--cores", cores_spec])

            data = run_cmd(cmd)
            if data:
                if data.get("has_sequential_baseline", False):
                    seq_d = data.get("sequential", {})
                    seq_time = seq_d.get("time_total_algorithm")
                    seq_std = seq_d.get("time_total_algorithm_std", 0.0)

                if data.get("has_gpu", True) and "gpu" in data:
                    gpu_d = data.get("gpu", {})
                    gpu_time = gpu_d.get("time_total_algorithm")
                    gpu_std = gpu_d.get("time_total_algorithm_std", 0.0)
                    n_v = gpu_d.get("n_vertices", size)
                    n_f = gpu_d.get("n_faces", 0)
                    n_p = gpu_d.get("n_polygons", 0)
                else:
                    d_dict = data.get("cpu_data", data.get("cpu", {}))
                    n_v = d_dict.get("n_vertices", size)
                    n_f = d_dict.get("n_faces", 0)
                    n_p = d_dict.get("n_polygons", 0)

            # Clean temporary baseline json
            if base_json.exists():
                try:
                    base_json.unlink()
                except Exception:
                    pass

        baseline_stats[sd] = {
            "seq_time": seq_time, "seq_std": seq_std,
            "gpu_time": gpu_time, "gpu_std": gpu_std,
            "n_vertices": n_v, "n_faces": n_f, "n_polygons": n_p
        }

    # -------------------------------------------------------------------------
    # 2. OPENMP SCALING PHASE (Across all thread counts)
    # -------------------------------------------------------------------------
    raw_records = []
    total_steps = len(threads_list) * len(seeds)
    current_step = 0

    for sd in seeds:
        b_info = baseline_stats[sd]
        for t in threads_list:
            current_step += 1
            print(f"\n[{current_step}/{total_steps}] >>> Testing OpenMP Scaling: threads={t}, seed={sd}, runs={runs} <<<")
            tmp_json = RESULTS_DIR / f"cores_step_{mesh_type}_{size}_t{t}_seed{sd}.json"

            cmd = [
                str(BINARY),
                "--type", mesh_type,
                "--size", str(size),
                "--seed", str(sd),
                "--runs", str(runs),
                "--threads", str(t),
                "--no-seq",
                "--no-gpu",
                "--output", str(tmp_json)
            ]
            if cores_spec:
                cmd.extend(["--cores", cores_spec])

            data = run_cmd(cmd)
            if data:
                has_dcel = data.get("has_omp_dcel", False)
                has_task = data.get("has_omp_task", False)
                dcel_dict = data.get("cpu_dcel", {})
                data_dict = data.get("cpu_data", data.get("cpu_loops", data.get("cpu", {})))
                tasks_dict = data.get("cpu_tasks", {})

                dcel_time = dcel_dict.get("time_total_algorithm") if has_dcel else None
                dcel_std = dcel_dict.get("time_total_algorithm_std", 0.0) if has_dcel else 0.0

                data_time = data_dict.get("time_total_algorithm", 0.0)
                data_std = data_dict.get("time_total_algorithm_std", 0.0)

                task_time = tasks_dict.get("time_total_algorithm") if has_task else None
                task_std = tasks_dict.get("time_total_algorithm_std", 0.0) if has_task else 0.0

                rec = {
                    "threads": t,
                    "requested_size": size,
                    "seed": sd,
                    "runs": runs,
                    "n_vertices": b_info["n_vertices"],
                    "n_faces": b_info["n_faces"],
                    "n_polygons": b_info["n_polygons"],
                    "cpu_cores_affinity": data.get("cpu_cores_affinity", cores_spec or "all"),
                    "seq_time_ms": b_info["seq_time"],
                    "seq_time_std_ms": b_info["seq_std"],
                    "gpu_time_ms": b_info["gpu_time"],
                    "gpu_time_std_ms": b_info["gpu_std"],
                    "omp_dcel_time_ms": dcel_time,
                    "omp_dcel_time_std_ms": dcel_std,
                    "omp_data_time_ms": data_time,
                    "omp_data_time_std_ms": data_std,
                    "omp_loops_time_ms": data_time,
                    "omp_tasks_time_ms": task_time,
                    "omp_tasks_time_std_ms": task_std,
                }
                raw_records.append(rec)

            if tmp_json.exists():
                try:
                    tmp_json.unlink()
                except Exception:
                    pass

    if not raw_records:
        print("[Error] No cores scaling benchmark data collected.")
        return None

    raw_df = pd.DataFrame(raw_records)
    is_multi_seed = len(seeds) > 1
    seeds_slug = "_".join(str(s) for s in seeds)
    seeds_label = ", ".join(str(s) for s in seeds)

    # -------------------------------------------------------------------------
    # 3. AGGREGATE RESULTS & COMPUTE SCALING METRICS
    # -------------------------------------------------------------------------
    if is_multi_seed:
        raw_csv_path = RESULTS_DIR / f"cores_{mesh_type}_{size}_seeds_{seeds_slug}_raw.csv"
        raw_df.to_csv(raw_csv_path, index=False)

        summary_records = []
        for t, group in raw_df.groupby("threads", sort=True):
            mean_seq = group["seq_time_ms"].mean() if group["seq_time_ms"].notnull().all() else None
            std_seq = group["seq_time_ms"].std() if (mean_seq and len(group) > 1) else group["seq_time_std_ms"].mean()

            mean_gpu = group["gpu_time_ms"].mean() if group["gpu_time_ms"].notnull().all() else None
            std_gpu = group["gpu_time_ms"].std() if (mean_gpu and len(group) > 1) else group["gpu_time_std_ms"].mean()

            mean_dcel = group["omp_dcel_time_ms"].mean() if group["omp_dcel_time_ms"].notnull().all() else None
            std_dcel = group["omp_dcel_time_ms"].std() if (mean_dcel and len(group) > 1) else group["omp_dcel_time_std_ms"].mean()

            mean_data = group["omp_data_time_ms"].mean()
            std_data = group["omp_data_time_ms"].std() if len(group) > 1 else group["omp_data_time_std_ms"].mean()

            mean_task = group["omp_tasks_time_ms"].mean() if group["omp_tasks_time_ms"].notnull().all() else None
            std_task = group["omp_tasks_time_ms"].std() if (mean_task and len(group) > 1) else group["omp_tasks_time_std_ms"].mean()

            rec = {
                "threads": t,
                "requested_size": size,
                "n_vertices": int(round(group["n_vertices"].mean())),
                "n_faces": int(round(group["n_faces"].mean())),
                "n_polygons": int(round(group["n_polygons"].mean())),
                "seeds_count": len(seeds),
                "seeds_list": seeds_label,
                "runs_per_seed": runs,
                "runs": runs,
                "cpu_cores_affinity": group["cpu_cores_affinity"].iloc[0],
                "seq_time_ms": mean_seq,
                "seq_time_std_ms": std_seq if pd.notnull(std_seq) else 0.0,
                "gpu_time_ms": mean_gpu,
                "gpu_time_std_ms": std_gpu if pd.notnull(std_gpu) else 0.0,
                "omp_dcel_time_ms": mean_dcel,
                "omp_dcel_time_std_ms": std_dcel if pd.notnull(std_dcel) else 0.0,
                "omp_data_time_ms": mean_data,
                "omp_data_time_std_ms": std_data if pd.notnull(std_data) else 0.0,
                "omp_loops_time_ms": mean_data,
                "omp_tasks_time_ms": mean_task,
                "omp_tasks_time_std_ms": std_task if pd.notnull(std_task) else 0.0,
            }
            summary_records.append(rec)
        df = pd.DataFrame(summary_records)
    else:
        df = raw_df.sort_values(by="threads").copy()

    # Calculate Speedups & Parallel Efficiency
    has_seq = df["seq_time_ms"].notnull().any() and (df["seq_time_ms"] > 0).any()
    seq_val = df["seq_time_ms"].dropna().iloc[0] if has_seq else None

    # Speedups vs Sequential
    if has_seq:
        if "omp_dcel_time_ms" in df.columns:
            df["speedup_dcel_vs_seq"] = seq_val / df["omp_dcel_time_ms"]
        if "omp_data_time_ms" in df.columns:
            df["speedup_data_vs_seq"] = seq_val / df["omp_data_time_ms"]
        if "omp_tasks_time_ms" in df.columns:
            df["speedup_task_vs_seq"] = seq_val / df["omp_tasks_time_ms"]

    # Self-Speedup and Efficiency (vs 1 thread)
    row_1 = df[df["threads"] == 1]
    if not row_1.empty:
        if "omp_dcel_time_ms" in df.columns and row_1["omp_dcel_time_ms"].notnull().any():
            t1_dcel = row_1["omp_dcel_time_ms"].iloc[0]
            df["speedup_dcel_self"] = t1_dcel / df["omp_dcel_time_ms"]
            df["efficiency_dcel"] = (df["speedup_dcel_self"] / df["threads"]) * 100.0

        if "omp_data_time_ms" in df.columns and row_1["omp_data_time_ms"].notnull().any():
            t1_data = row_1["omp_data_time_ms"].iloc[0]
            df["speedup_data_self"] = t1_data / df["omp_data_time_ms"]
            df["efficiency_data"] = (df["speedup_data_self"] / df["threads"]) * 100.0

        if "omp_tasks_time_ms" in df.columns and row_1["omp_tasks_time_ms"].notnull().any():
            t1_task = row_1["omp_tasks_time_ms"].iloc[0]
            df["speedup_task_self"] = t1_task / df["omp_tasks_time_ms"]
            df["efficiency_task"] = (df["speedup_task_self"] / df["threads"]) * 100.0

    # Save CSV files
    if is_multi_seed:
        csv_path = RESULTS_DIR / f"cores_{mesh_type}_{size}_seeds_{seeds_slug}.csv"
        df.to_csv(csv_path, index=False)
        default_csv = RESULTS_DIR / f"cores_{mesh_type}_{size}.csv"
        df.to_csv(default_csv, index=False)
        print(f"\n[Cores] Multi-seed summary saved to: {csv_path}")
    else:
        csv_path = RESULTS_DIR / f"cores_{mesh_type}_{size}_seed{seeds[0]}.csv"
        df.to_csv(csv_path, index=False)
        default_csv = RESULTS_DIR / f"cores_{mesh_type}_{size}.csv"
        df.to_csv(default_csv, index=False)
        print(f"\n[Cores] Summary table saved to: {csv_path}")

    # -------------------------------------------------------------------------
    # 4. PRINT TERMINAL REPORT TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 140)
    title_run_info = f"{runs} RUNS/SAMPLE" if runs > 1 else "1 RUN"
    if is_multi_seed:
        title_hdr = f"OPENMP CORES SCALING SUMMARY ({mesh_type.upper()}, N={size:,}, SEEDS=[{seeds_label}], {title_run_info})"
    else:
        title_hdr = f"OPENMP CORES SCALING SUMMARY ({mesh_type.upper()}, N={size:,}, SEED={seeds[0]}, {title_run_info})"
    print(f"{title_hdr:^140}")
    print("=" * 140)

    header = f"{'Threads':>8} | {'OMP DCEL (ms)':>17} | {'OMP Data (ms)':>17} | {'OMP Task (ms)':>17} | {'DCEL Sp (Seq)':>14} | {'Data Sp (Seq)':>14} | {'Task Sp (Seq)':>14} | {'Data Eff (%)':>13} | {'Task Eff (%)':>13}"
    print(header)
    print("-" * 140)

    for _, row in df.iterrows():
        t = int(row["threads"])
        def fmt_time(val, std):
            if val is None or pd.isna(val): return f"{'-':>17}"
            return f"{val:7.2f} ± {std:5.2f}"

        dcel_s = fmt_time(row.get("omp_dcel_time_ms"), row.get("omp_dcel_time_std_ms", 0.0))
        data_s = fmt_time(row.get("omp_data_time_ms"), row.get("omp_data_time_std_ms", 0.0))
        task_s = fmt_time(row.get("omp_tasks_time_ms"), row.get("omp_tasks_time_std_ms", 0.0))

        def fmt_sp(key):
            v = row.get(key)
            return f"{v:7.2f}x" if (v is not None and pd.notnull(v) and v > 0) else f"{'-':>8}"

        def fmt_eff(key):
            v = row.get(key)
            return f"{v:6.1f}%" if (v is not None and pd.notnull(v) and v > 0) else f"{'-':>7}"

        dcel_sp = fmt_sp("speedup_dcel_vs_seq")
        data_sp = fmt_sp("speedup_data_vs_seq")
        task_sp = fmt_sp("speedup_task_vs_seq")
        data_eff = fmt_eff("efficiency_data")
        task_eff = fmt_eff("efficiency_task")

        print(f"{t:>8} | {dcel_s} | {data_s} | {task_s} | {dcel_sp:>14} | {data_sp:>14} | {task_sp:>14} | {data_eff:>13} | {task_eff:>13}")

    print("=" * 140)

    if has_seq:
        print(f"  * Polylla-Mesh-DCEL (Sequential Baseline): {seq_val:.2f} ms")
    if df["gpu_time_ms"].notnull().any() and (df["gpu_time_ms"] > 0).any():
        g_val = df["gpu_time_ms"].dropna().iloc[0]
        print(f"  * GPolylla (CUDA GPU Reference)         : {g_val:.2f} ms")
    print("=" * 140 + "\n")

    # -------------------------------------------------------------------------
    # 5. GENERATE PLOTS
    # -------------------------------------------------------------------------
    if plot:
        try:
            from plot_cores import plot_cores_data
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from plot_cores import plot_cores_data

        affinity_info = cores_spec or "all"
        # 1. Standard PNG plots with titles in plots/
        print("[Plots] Generating standard PNG visualizations in plots/...")
        plot_cores_data(df, mesh_type=mesh_type, size=size, seed=seeds[0], output_dir=PLOTS_DIR,
                        seeds_info=(seeds_label if is_multi_seed else ""), runs=runs,
                        affinity_info=affinity_info, formats=["png"], show_title=True)

        # 2. Vector PDF plots without titles for paper in plots_pdf/
        if generate_pdf:
            print("[Plots] Generating publication-ready vector PDFs (no title) in plots_pdf/...")
            plot_cores_data(df, mesh_type=mesh_type, size=size, seed=seeds[0], output_dir=PLOTS_PDF_DIR,
                            seeds_info=(seeds_label if is_multi_seed else ""), runs=runs,
                            affinity_info=affinity_info, formats=["pdf"], show_title=False)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="PPolylla OpenMP Cores & Thread Scalability Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Strong scaling benchmark on 1M vertices testing 1, 2, 4, 8, 12, 16, 20 threads:
  python3 scripts/benchmark_cores.py --size 1000000 --threads 1,2,4,8,12,16,20 --runs 3

  # Strong scaling on Performance cores (P-cores 0-7):
  python3 scripts/benchmark_cores.py --cores p-cores --threads 1,2,4,6,8 --runs 3

  # Evaluate on grid mesh with 500K vertices across 2 seeds:
  python3 scripts/benchmark_cores.py --type grid --size 500000 --seeds 42,43 --runs 2
"""
    )
    parser.add_argument("-t", "--type", choices=["grid", "uniform"], default="uniform", help="Mesh type (grid or uniform, default: uniform)")
    parser.add_argument("-s", "--size", type=int, default=1000000, help="Number of vertices for strong scaling (default: 1000000)")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed for single run (default: 42)")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated list of random seeds (e.g. '42,43'). Overrides --seed.")
    parser.add_argument("--runs", "--repeat", type=int, default=3, dest="runs", help="Repetitions per data point for average and std dev (default: 3)")
    parser.add_argument("-j", "--threads", type=str, default="", help="Comma-separated list of thread counts (e.g. '1,2,4,8,12,16,20')")
    parser.add_argument("-c", "--cores", type=str, default="", help="CPU core affinity for OpenMP (e.g. 'p-cores', 'e-cores', '0-7', 'all')")
    parser.add_argument("--cores-sweep", "--threads-sweep", "--scaling", "--cores-benchmark", action="store_true", help="Compatibility flag for runner script")
    parser.add_argument("--no-seq", action="store_true", help="Skip sequential CPU baseline")
    parser.add_argument("--no-gpu", action="store_true", help="Skip CUDA GPU reference run")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-pdf", action="store_true", help="Disable PDF generation")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing cores CSV results")
    parser.add_argument("--csv", type=str, default=None, help="Specific cores CSV file to plot (when using --plot-only)")

    args = parser.parse_args()

    if args.plot_only or args.csv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import plot_cores as plotter
        if args.csv:
            plotter.plot_cores_file(Path(args.csv))
        else:
            plotter.main()
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else [args.seed]
    threads_list = [int(x.strip()) for x in args.threads.split(",") if x.strip()] if args.threads else None

    run_cores_benchmark(
        mesh_type=args.type,
        size=args.size,
        seeds=seeds,
        runs=args.runs,
        threads_list=threads_list,
        cores_spec=args.cores,
        no_seq=args.no_seq,
        no_gpu=args.no_gpu,
        plot=not args.no_plot,
        generate_pdf=not args.no_pdf
    )

if __name__ == "__main__":
    main()
