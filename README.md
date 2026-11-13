# PPolylla: Comparative Benchmark of Parallel Polylla Algorithms

Benchmark suite and performance analysis to compare five implementations of the **Polylla** algorithm:
1. **Sequential CPU (Baseline)**: [ssalinasfe/Polylla-Mesh-DCEL](https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git)
2. **Multi-core CPU (OpenMP DCEL)**: [rcarrascoc/MPolylla-Mesh-DCEL](https://github.com/rcarrascoc/MPolylla-Mesh-DCEL.git)
3. **Multi-core CPU (OpenMP Loops / Data)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`main` branch)
4. **Multi-core CPU (OpenMP Tasks)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`Task` branch)
5. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)

The system generates benchmark datasets (triangle meshes) **directly in shared memory (RAM)** to ensure that all five implementations evaluate the exact same set of vertices and faces, supporting a random seed (`seed`) for 100% reproducible experiments.

---

## 🚀 Key Features

- **Isolated Submodules**: Original repositories are integrated as Git submodules in the `submodules/` directory:
  - `submodules/Polylla-Mesh-DCEL` (Sequential DCEL baseline)
  - `submodules/MPolylla-Mesh-DCEL` (Multi-core OpenMP DCEL)
  - `submodules/MPolylla` (Multi-core OpenMP Loops / Data - `main` branch)
  - `submodules/MPolylla-Task` (Multi-core OpenMP Tasks - `Task` branch)
  - `submodules/GPolylla` (Parallel CUDA GPU)
  Keeping `src/` clean and decoupled through independent shared libraries (`libseq_runner.so`, `libmpolylla_dcel_runner.so`, `libmpolylla_runner.so`, `libmpolylla_task_runner.so`, `libgpolylla_runner.so`).
- **In-Memory Generation (Zero Disk Overhead)**:
  - **Grid (`grid`)**: Regular structured mesh of $k \times k$ vertices with diagonal division identical to the one used in GPolylla publications.
  - **Uniform (`uniform`)**: Uniformly distributed points in $[0, 1] \times [0, 1]$ with boundary snapping (`move_point`) and in-memory Delaunay triangulation completed in milliseconds thanks to `delaunator-cpp`.
- **Exact Reproducibility & Statistical Rigor**:
  - `--seed` parameter to accurately replicate uniform distributions.
  - `--seeds <S1,S2,...>` to evaluate across multiple distributions in a single run.
  - `--runs <N>` repetitions to report mean execution times and standard deviations ($\pm \sigma$).
- **CPU Core Affinity & Scaling**:
  - `-c, --cores <SPEC>`: Pin OpenMP threads to specific cores, core ranges (e.g., `0-7`), or hybrid architectures (`p-cores` / `e-cores`).
  - Strong scaling benchmark mode (`--cores-sweep` / `--scaling`) to measure speedup and parallel efficiency across thread counts.
- **Comprehensive Timer Breakdown**:
  1. Triangulation data structure construction (`time_triangulation_generation`)
  2. Label maximum edges (`d_time_to_label_max_edges`)
  3. Label frontier edges (`d_time_to_label_frontier_edges`)
  4. Label seed edges (`d_time_to_label_seed_edges`)
  5. Repair extra frontier edges (`d_time_to_label_extra_frontier_edge` / `d_time_to_repair`)
  6. Traversal phase 1 (`d_time_to_traversal`)
  7. Traversal phase 2 / Frontier search (`d_time_to_traversal_search_frontier_edge`)
  8. Seed overwriting (`d_time_to_overwrite_seed`)
  9. Prefix scan of seed edges (`d_time_to_label_scan_edges`)
  10. Seed compaction (`d_time_to_label_compaction_edges`)
  - Host-device transfers (`d_time_copy_to_device`, `d_time_to_back_to_host`)
  - **Relative Speedups**:
    - GPU vs Sequential Baseline (with and without memory transfers)
    - OpenMP DCEL vs Sequential Baseline
    - OpenMP Data (Loops) vs Sequential Baseline
    - OpenMP Tasks vs Sequential Baseline
    - OpenMP relative speedups (Tasks vs Data, Tasks vs DCEL, Data vs DCEL)
    - GPU vs OpenMP variants (DCEL, Data, Tasks)
- **Automatic Export to `.txt`, `.json`, and `.csv`**: Generates a human-readable report with exhaustive hardware metadata (CPU, GPU, RAM, OS, CUDA), exact commit hashes of each submodule, and run parameters.
- **Single-Command Execution**: `./run_benchmark.sh` script, CLI `scripts/benchmark.py`, and scalability analyzer `scripts/benchmark_cores.py` supporting automated sweeps and Matplotlib plot generation.

---

## 📁 Project Structure

```
PPolylla/
├── CMakeLists.txt              # CMake build configuration (C++14, CUDA sm_89/native, OpenMP)
├── run_benchmark.sh            # Main benchmark runner script
├── README.md                   # Project documentation
├── submodules/                 # External submodules (remote repositories)
│   ├── Polylla-Mesh-DCEL/      # Sequential CPU baseline
│   ├── MPolylla-Mesh-DCEL/     # CPU implementation (OpenMP DCEL)
│   ├── MPolylla/               # CPU implementation (OpenMP Loops / Data - main branch)
│   ├── MPolylla-Task/          # CPU implementation (OpenMP Tasks - Task branch)
│   └── GPolylla/               # GPU implementation (CUDA)
├── src/                        # Benchmark suite source code
│   ├── common/
│   │   ├── benchmark_types.hpp # Data structures, timers, metadata, TXT and JSON
│   │   └── cpu_affinity.hpp    # CPU core affinity binding and topology detection
│   ├── generators/
│   │   ├── delaunator.hpp      # Ultra-fast 2D Delaunay triangulator in C++
│   │   └── mesh_generator.hpp  # In-memory mesh generators (grid and uniform)
│   ├── runners/
│   │   ├── seq_runner.hpp / .cpp         # Polylla-Mesh-DCEL wrapper in libseq_runner.so
│   │   ├── cpu_dcel_runner.hpp / .cpp    # MPolylla-Mesh-DCEL wrapper in libmpolylla_dcel_runner.so
│   │   ├── cpu_runner.hpp / .cpp         # MPolylla Loops wrapper in libmpolylla_runner.so
│   │   ├── cpu_task_runner.hpp / .cpp    # MPolylla Tasks wrapper in libmpolylla_task_runner.so
│   │   └── gpu_runner.hpp / .cu          # GPolylla wrapper in libgpolylla_runner.so
│   └── main.cpp                # Main C++ benchmark executable
├── scripts/
│   ├── benchmark.py            # Python orchestrator for size sweeps and plotting
│   └── benchmark_cores.py      # Python orchestrator for OpenMP core scaling and efficiency
├── results/                    # Benchmark results in TXT, JSON, and CSV
└── plots/                      # Generated plots (PNG)
```

---

## 🖥️ Quick Start (Single Command)

### 1. Full Single Benchmark (Sequential vs OpenMP DCEL vs OpenMP Data vs OpenMP Tasks vs GPU)

```bash
# Uniform mesh with 100,000 vertices and seed 42
./run_benchmark.sh --type uniform --size 100000 --seed 42

# 3 repetitions to report average execution times and standard deviations
./run_benchmark.sh --type uniform --size 100000 --runs 3

# Grid mesh with 1,000,000 vertices (skipping sequential baseline to speed up execution)
./run_benchmark.sh --type grid --size 1000000 --no-seq

# Run only the CPU variants (skip GPU)
./run_benchmark.sh --type uniform --size 100000 --no-gpu
```

### 2. CPU Core Affinity and Scalability Analysis

```bash
# Pin execution to Performance cores (P-cores)
./run_benchmark.sh --size 100000 --cores p-cores

# Pin execution to 4 threads on specific CPU cores (0 to 3)
./run_benchmark.sh --size 100000 --cores 0-3 --threads 4

# Run OpenMP strong scaling sweep across core counts (1, 2, 4, 8, 16, 32...)
./run_benchmark.sh --cores-sweep --type uniform --size 1000000 --runs 3
```

### 3. Automated Size Sweep and Plotting

To evaluate multiple mesh sizes (e.g., 10k, 50k, 100k vertices):

```bash
./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000 --seed 42
```

Generated output:
- Human-readable individual reports in `results/*.txt` and structured in `results/*.json`
- Consolidated CSV table in `results/sweep_uniform_seed42.csv`
- Speedup plot in `plots/speedup_uniform.png`
- Runtime plot in `plots/time_uniform.png` (log-scale curves)

---

## 📊 Sample Terminal Output

```
==========================================================================================================================================================
                                                              POLYLLA BENCHMARK COMPARISON REPORT                                                         
                                  (Polylla-Mesh-DCEL vs MPolylla DCEL vs MPolylla Data vs MPolylla Task vs GPolylla)                                    
==========================================================================================================================================================
  Mesh Type:       uniform
  Requested Size:  100000
  Random Seed:     42
  Repetitions:     1 run
  CPU Threads:     20
  CPU Cores/Aff:   all
  Mesh Gen Time:   139.476 ms
----------------------------------------------------------------------------------------------------------------------------------------------------------
  Metric / Element             Polylla-Mesh-DCEL    MPolylla DCEL    MPolylla Data    MPolylla Task        GPolylla           Validation            
----------------------------------------------------------------------------------------------------------------------------------------------------------
  Vertices (n_vertices)                   100002           100002           100002           100002          100002        MATCH (All Parallel)
  Faces (n_faces)                         199992           199992           199992           199992          199992        MATCH (All Parallel)
  Half-Edges (n_half_edges)               599986           599986           599986           599986          599986        MATCH (All Parallel)
  Polygons (n_polygons)                    32046            32046            32004            32001           32053            DIFF (Parallel)*
----------------------------------------------------------------------------------------------------------------------------------------------------------
  Phase Timers (ms)            Polylla-Mesh-DCEL    MPolylla DCEL    MPolylla Data    MPolylla Task        GPolylla       Speedup GPolylla/Base 
----------------------------------------------------------------------------------------------------------------------------------------------------------
  1. Triangulation Struct               156.5320         159.2010         161.9130         161.9700        158.3400                     0.99x
  2. Label Max Edges                     43.7306          18.4210           3.9194           2.3554          4.7260                     9.25x
  3. Label Frontier Edges                14.8783           6.1200           1.1791           1.0246          0.3735                    39.83x
  4. Label Seed Edges                    15.9823           7.8450           1.2180           0.9418          0.0000                         -
  5. Repair Extra Frontier                2.1867           1.8900           1.4740           1.1416          0.0442                    49.47x
  6. Traversal Phase 1                   16.7280           8.9400           2.3387           2.2993          0.0435                   384.40x
  7. Traversal Phase 2 (Search)           0.0000           0.0000           0.4919           0.5811          0.0197                         -
  8. Repair Overwrite Seed                0.0000           0.0000           0.7310           0.8704          0.0401                         -
  9. Label Seed Scan                      0.0000           0.0000           0.0000           0.0000          0.0000                         -
  10. Seed Compaction                     0.0000           0.0000           0.4815           0.5413          0.1783                         -
----------------------------------------------------------------------------------------------------------------------------------------------------------
  TOTAL ALGORITHM TIME                   93.5059          43.2160          11.8335           9.7556          5.0995                    18.34x
----------------------------------------------------------------------------------------------------------------------------------------------------------
  GPU Transfer Host->Device:              1.7028 ms
  GPU Transfer Device->Host:              2.0751 ms
  GPU TOTAL (Alg + Transfers):            8.8774 ms
----------------------------------------------------------------------------------------------------------------------------------------------------------
  >>> SPEEDUP GPolylla vs Polylla-Mesh-DCEL (Alg only):         18.34x
  >>> SPEEDUP GPolylla vs Polylla-Mesh-DCEL (With Transfers):   10.53x
  >>> SPEEDUP GPolylla vs MPolylla DCEL (20 cores):              8.47x
  >>> SPEEDUP GPolylla vs MPolylla Data (20 cores):              2.32x
  >>> SPEEDUP GPolylla vs MPolylla Task (20 cores):              1.91x
  >>> SPEEDUP MPolylla DCEL (20 cores) vs Polylla-Mesh-DCEL:     2.16x
  >>> SPEEDUP MPolylla Data (20 cores) vs Polylla-Mesh-DCEL:     7.90x
  >>> SPEEDUP MPolylla Task (20 cores) vs Polylla-Mesh-DCEL:     9.58x
  >>> SPEEDUP MPolylla Task vs MPolylla Data:                    1.21x
  >>> SPEEDUP MPolylla Task vs MPolylla DCEL:                    4.43x
  >>> SPEEDUP MPolylla Data vs MPolylla DCEL:                    3.65x
==========================================================================================================================================================
```

---

## 🔧 Command-Line Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `-t, --type <grid\|uniform>` | Mesh type to generate in memory | `uniform` |
| `-s, --size <N>` | Number of requested vertices | `100000` |
| `-r, --seed <SEED>` | Pseudorandom generator seed | `42` |
| `--seeds <S1,S2,...>` | Comma-separated list of random seeds for multi-seed runs | None |
| `--runs, --repeat <N>` | Number of repetitions per experiment to compute average times and std dev | `1` |
| `-j, --threads <T>` | OpenMP threads for CPU (0 = all detected cores or core count) | `0` |
| `-c, --cores <SPEC>` | CPU core affinity (e.g., `0-7`, `8-19`, `p-cores`, `e-cores`) | `all` |
| `-o, --output <PATH>` | Output JSON file path | `results/benchmark_...json` |
| `-f, --off <FILE>` | Use an existing `.off` file instead of generating in memory | None |
| `--no-seq` | Skip the sequential CPU baseline (`Polylla-Mesh-DCEL`) | `false` |
| `--no-gpu` | Skip the CUDA GPU variant (`GPolylla`), executing CPU variants only | `false` |
| `--save-mesh <PATH>` | Save generated mesh to a persistent `.off` file | None |
| `--sweep` | Enable multi-size sweep mode via Python script | `false` |
| `--sizes <N1,N2,...>` | Comma-separated list of sizes for the sweep | `10k,50k,100k,500k,1M` |
| `--cores-sweep, --scaling` | Run OpenMP thread and core scalability benchmark | `false` |
| `-h, --help` | Display help menu | - |
