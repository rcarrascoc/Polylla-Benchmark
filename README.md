# Polylla: Comparative Benchmark of Parallel Polylla Algorithms

Benchmark suite and performance analysis to compare four implementations of the **Polylla** algorithm:
1. **Sequential CPU (Baseline)**: [ssalinasfe/Polylla-Mesh-DCEL](https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git)
2. **Multi-core CPU (OpenMP Loops)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`main` branch)
3. **Multi-core CPU (OpenMP Tasks)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`Task` branch)
4. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)

The system generates benchmark datasets (triangle meshes) **directly in shared memory (RAM)** to ensure that all four implementations evaluate the exact same set of vertices and faces, supporting a random seed (`seed`) for 100% reproducible experiments.

---

## 🚀 Key Features

- **Isolated Submodules**: Original repositories are integrated as Git submodules in the `submodules/` directory:
  - `submodules/Polylla-Mesh-DCEL` (Sequential baseline)
  - `submodules/MPolylla` (OpenMP Loops parallel - `main` branch)
  - `submodules/MPolylla-Task` (OpenMP Tasks parallel - `Task` branch)
  - `submodules/GPolylla` (CUDA GPU parallel)
  Keeping `src/` clean and decoupled through independent shared libraries (`libseq_runner.so`, `libmpolylla_runner.so`, `libmpolylla_task_runner.so`, `libgpolylla_runner.so`).
- **In-Memory Generation (Zero Disk Overhead)**:
  - **Grid (`grid`)**: Regular structured mesh of $k \times k$ vertices with diagonal division identical to the one used in GPolylla publications.
  - **Uniform (`uniform`)**: Uniformly distributed points in $[0, 1] \times [0, 1]$ with boundary snapping (`move_point`) and in-memory Delaunay triangulation completed in milliseconds thanks to `delaunator-cpp`.
- **Exact Reproducibility**: `--seed` parameter to accurately replicate uniform distributions.
- **Comprehensive Timer Breakdown**:
  1. Triangulation data structure construction (`time_triangulation_generation`)
  2. Label maximum edges (`d_time_to_label_max_edges`)
  3. Label frontier edges (`d_time_to_label_frontier_edges`)
  4. Label seed edges (`d_time_to_label_seed_edges`)
  5. Repair extra frontier edges (`d_time_to_label_extra_frontier_edge`)
  6. Traversal phase 1 (`d_time_to_traversal`)
  7. Traversal phase 2 / Frontier search (`d_time_to_traversal_search_frontier_edge`)
  8. Seed overwriting (`d_time_to_overwrite_seed`)
  9. Prefix scan of seed edges (`d_time_to_label_scan_edges`)
  10. Seed compaction (`d_time_to_label_compaction_edges`)
  - Host-device transfers (`d_time_copy_to_device`, `d_time_to_back_to_host`)
  - **GPU vs Sequential Baseline Speedup** (with and without memory transfers)
  - **OpenMP (Loops) vs Sequential Baseline Speedup**
  - **OpenMP (Tasks) vs Sequential Baseline Speedup**
  - **OpenMP (Tasks) vs OpenMP (Loops) Speedup**
  - **GPU vs OpenMP (Loops and Tasks) Speedup**
- **Automatic Export to `.txt` and `.json`**: Generates a human-readable report with exhaustive hardware metadata (CPU, GPU, RAM, OS, CUDA), exact commit hashes of each submodule, and run parameters.
- **Single-Command Execution**: `./run_benchmark.sh` script and `scripts/benchmark.py` CLI supporting automated parameter sweeps, CSV summary tables, and Matplotlib plot generation.

---

## 📁 Project Structure

```
PPolylla/
├── CMakeLists.txt              # CMake build configuration (C++14, CUDA sm_89/native, OpenMP)
├── run_benchmark.sh            # Main benchmark runner script
├── README.md                   # Project documentation
├── submodules/                 # External submodules (remote repositories)
│   ├── Polylla-Mesh-DCEL/      # Sequential CPU baseline
│   ├── MPolylla/               # CPU implementation (OpenMP Loops - main branch)
│   ├── MPolylla-Task/          # CPU implementation (OpenMP Tasks - Task branch)
│   └── GPolylla/               # GPU implementation (CUDA)
├── src/                        # Benchmark suite source code
│   ├── common/
│   │   └── benchmark_types.hpp # Data structures, timers, metadata, TXT and JSON
│   ├── generators/
│   │   ├── delaunator.hpp         # Ultra-fast 2D Delaunay triangulator in C++
│   │   └── mesh_generator.hpp     # In-memory mesh generators (grid and uniform)
│   ├── runners/
│   │   ├── seq_runner.hpp / .cpp     # Polylla-Mesh-DCEL wrapper in libseq_runner.so
│   │   ├── cpu_runner.hpp / .cpp     # MPolylla Loops wrapper in libmpolylla_runner.so
│   │   ├── cpu_task_runner.hpp / .cpp# MPolylla Tasks wrapper in libmpolylla_task_runner.so
│   │   └── gpu_runner.hpp / .cu    # GPolylla wrapper in libgpolylla_runner.so
│   └── main.cpp                # Main C++ benchmark executable
├── scripts/
│   └── benchmark.py            # Python orchestrator for sweeps and plotting
├── results/                    # Benchmark results in TXT, JSON, and CSV
└── plots/                      # Generated plots (PNG)
```

---

## 🖥️ Quick Start (Single Command)

### 1. Full Single Benchmark (Sequential vs OpenMP Loops vs Tasks vs GPU)

```bash
# Uniform mesh with 100,000 vertices and seed 42
./run_benchmark.sh --type uniform --size 100000 --seed 42

# Grid mesh with 1,000,000 vertices (skipping sequential to speed up run)
./run_benchmark.sh --type grid --size 1000000 --no-seq

# Run only the CPU variants (skip GPU)
./run_benchmark.sh --type uniform --size 100000 --no-gpu
```

### 2. Automated Size Sweep and Plotting

To evaluate multiple sizes (e.g., 10k, 50k, 100k vertices):

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
======================================================================================================================================
                                                  POLYLLA BENCHMARK COMPARISON REPORT                                                 
                              (Sequential Baseline vs OpenMP Loops vs OpenMP Tasks vs CUDA GPU)                                       
======================================================================================================================================
  Mesh Type:       uniform
  Requested Size:  100000
  Random Seed:     42
  CPU Threads:     20
  Mesh Gen Time:   139.476 ms
--------------------------------------------------------------------------------------------------------------------------------------
  Metric / Element             SEQ (Baseline)      OMP (Loops)       OMP (Tasks)       GPU (GPolylla)           Validation            
--------------------------------------------------------------------------------------------------------------------------------------
  Vertices (n_vertices)                100002            100002            100002              100002        MATCH (All Parallel)
  Faces (n_faces)                      199992            199992            199992              199992        MATCH (All Parallel)
  Half-Edges (n_half_edges)            599986            599986            599986              599986        MATCH (All Parallel)
  Polygons (n_polygons)                 32046             32004             32001               32053            DIFF (Parallel)*
--------------------------------------------------------------------------------------------------------------------------------------
  Phase Timers (ms)            SEQ (Baseline)      OMP (Loops)       OMP (Tasks)       GPU (GPolylla)         Speedup GPU/Seq         
--------------------------------------------------------------------------------------------------------------------------------------
  1. Triangulation Struct            156.5320          161.9130          161.9700            158.3400                     0.99x
  2. Label Max Edges                  43.7306            3.9194            2.3554              4.7260                     9.25x
  3. Label Frontier Edges             14.8783            1.1791            1.0246              0.3735                    39.83x
  4. Label Seed Edges                 15.9823            1.2180            0.9418              0.0000                         -
  5. Repair Extra Frontier             2.1867            1.4740            1.1416              0.0442                    49.47x
  6. Traversal Phase 1                16.7280            2.3387            2.2993              0.0435                   384.40x
  7. Traversal Phase 2 (Search)            0.0000            0.4919            0.5811              0.0197                         -
  8. Repair Overwrite Seed             0.0000            0.7310            0.8704              0.0401                         -
  9. Label Seed Scan                   0.0000            0.0000            0.0000              0.0000                         -
  10. Seed Compaction                  0.0000            0.4815            0.5413              0.1783                         -
--------------------------------------------------------------------------------------------------------------------------------------
  TOTAL ALGORITHM TIME                93.5059           11.8335            9.7556              5.0995                    18.34x
--------------------------------------------------------------------------------------------------------------------------------------
  GPU Transfer Host->Device:                1.7028 ms
  GPU Transfer Device->Host:                2.0751 ms
  GPU TOTAL (Alg + Transfers):              8.8774 ms
--------------------------------------------------------------------------------------------------------------------------------------
  >>> SPEEDUP GPU vs Sequential (Alg only):                 18.34x
  >>> SPEEDUP GPU vs Sequential (With Transfers):           10.53x
  >>> SPEEDUP OpenMP (Loops) vs Sequential:                  7.90x
  >>> SPEEDUP OpenMP (Tasks) vs Sequential:                  9.58x
  >>> SPEEDUP GPU vs OpenMP (Loops):                         2.32x
  >>> SPEEDUP GPU vs OpenMP (Tasks):                         1.91x
  >>> SPEEDUP OpenMP (Tasks) vs OpenMP (Loops):              1.21x
======================================================================================================================================
```

---

## 🔧 Command-Line Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `-t, --type <grid\|uniform>` | Mesh type to generate in memory | `uniform` |
| `-s, --size <N>` | Number of requested vertices | `100000` |
| `-r, --seed <SEED>` | Pseudorandom generator seed | `42` |
| `-j, --threads <T>` | OpenMP threads for CPU (0 = all detected cores) | `0` |
| `-o, --output <PATH>` | Output JSON file path | `results/benchmark_...json` |
| `-f, --off <FILE>` | Use an existing `.off` file instead of generating in memory | None |
| `--no-seq` | Skip the sequential CPU baseline | `false` |
| `--no-gpu` | Skip the CUDA GPU variant (runs CPU variants only) | `false` |
| `--save-mesh <PATH>` | Save generated mesh to a persistent `.off` file | None |
| `--sweep` | Enable multi-size sweep mode via Python script | `false` |
| `--sizes <N1,N2,...>` | Comma-separated list of sizes for the sweep | `10k,50k,100k,500k,1M` |
| `-h, --help` | Display help menu | - |
