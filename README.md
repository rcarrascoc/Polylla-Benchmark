# Polylla: Comparative Benchmark of Parallel Polylla Algorithms

A high-performance benchmark suite and analysis framework for evaluating and comparing five implementations of the **Polylla** polygonal meshing algorithm:
1. **Sequential CPU (Baseline)**: [ssalinasfe/Polylla-Mesh-DCEL](https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git)
2. **Multi-core CPU (OpenMP DCEL)**: [rcarrascoc/MPolylla-Mesh-DCEL](https://github.com/rcarrascoc/MPolylla-Mesh-DCEL.git)
3. **Multi-core CPU (OpenMP Loops / Data)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`main` branch)
4. **Multi-core CPU (OpenMP Tasks)**: [rcarrascoc/MPolylla](https://github.com/rcarrascoc/MPolylla.git) (`Task` branch)
5. **GPU (CUDA)**: [ssalinasfe/GPolylla](https://github.com/ssalinasfe/GPolylla.git)

The system generates test meshes **directly in shared memory (RAM)**, guaranteeing that all five implementations evaluate the exact same set of vertices, faces, and half-edges with zero disk I/O overhead. A pseudorandom seed (`seed`) ensures 100% reproducible experiments.

---

## 💻 Prerequisites & System Compatibility

The benchmark suite is designed to compile and run smoothly on **any Linux machine** (x86_64, ARM64/aarch64), whether it has a dedicated NVIDIA GPU or is a CPU-only workstation, laptop, or cluster node.

### System Requirements

- **Operating System**: Linux (Ubuntu, Debian, Fedora, Arch, CentOS/RHEL, etc.)
- **C++ Compiler**: GCC (`g++` >= 7.0) or Clang (`clang++` >= 8.0) supporting C++14 and OpenMP.
- **Build System**: CMake (>= 3.18) and Make.
- **GPU Acceleration (Optional)**: NVIDIA GPU with CUDA Toolkit (`nvcc` >= 11.0 / 12.0) and NVIDIA drivers.
- **Python (Optional, for plotting and automated sweeps)**: Python 3.8+ with `numpy`, `pandas`, `matplotlib`.

### Installing Dependencies

On **Debian / Ubuntu**:
```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip

# Optional: Install Python dependencies for plotting and sweeps
pip3 install -r requirements.txt
```

On **Fedora / RHEL**:
```bash
sudo dnf install -y gcc-c++ cmake make git python3 python3-pip
pip3 install -r requirements.txt
```

> [!NOTE]
> **Universal CPU-Only Fallback**: If `nvcc` or an NVIDIA GPU is not detected, CMake automatically disables CUDA compilation (`-DENABLE_CUDA=OFF`). The project will build and run all sequential and parallel OpenMP CPU implementations seamlessly on any standard computer.

---

## 🛠️ How to Compile

### Option 1: Automatic Build (Recommended)
You do not need to compile manually. Executing the runner script automatically fetches submodules and compiles the benchmark binary if it has not yet been built:
```bash
./run_benchmark.sh --help
```

### Option 2: Manual Build (Full with CUDA GPU Support)
```bash
# 1. Clone repository and initialize all submodules
git clone --recurse-submodules https://github.com/rcarrascoc/Polylla-Benchmark.git
cd Polylla-Benchmark

# (Or if already cloned, initialize submodules)
git submodule update --init --recursive

# 2. Configure and compile
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

### Option 3: Manual Build (CPU-Only / Any Machine without GPU)
To build exclusively for CPU (without requiring CUDA or an NVIDIA compiler):
```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DENABLE_CUDA=OFF ..
make -j$(nproc)
cd ..
```

> [!TIP]
> **Targeting a specific CUDA Architecture**: By default, CMake targets your local GPU (`native`). To cross-compile for a specific architecture, pass `-DCMAKE_CUDA_ARCHITECTURES`:
> ```bash
> cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..   # e.g., Ada Lovelace (RTX 40xx)
> cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..   # e.g., Ampere (A100, RTX 30xx)
> ```

---

## ⚡ Quick Start

### 1. Running on CPU-Only (No GPU Required)

Run all four CPU implementations (**Sequential Baseline**, **OpenMP DCEL**, **OpenMP Data/Loops**, and **OpenMP Tasks**) by passing `--no-gpu`:

```bash
# Default uniform mesh of 100,000 vertices on CPU
./run_benchmark.sh --type uniform --size 100000 --no-gpu

# Run on a specific number of OpenMP CPU threads (e.g., 4 threads)
./run_benchmark.sh --type uniform --size 100000 --no-gpu --threads 4

# Run with 3 repetitions to get statistical averages and standard deviations
./run_benchmark.sh --type uniform --size 100000 --no-gpu --runs 3
```

#### OpenMP CPU Core Scalability & Strong Scaling Benchmark
Evaluate thread scalability and parallel efficiency across core counts (e.g., 1, 2, 4, 8, 16, 32 threads):
```bash
# Run scalability sweep using the dedicated runner
./run_cores_benchmark.sh --type uniform --size 1000000 --runs 3

# Or directly through the main runner:
./run_benchmark.sh --cores-sweep --type uniform --size 1000000 --runs 3
```

---

### 2. Running with GPU (CUDA)

Run the complete comparison across all five variants (Sequential, OpenMP DCEL, OpenMP Data, OpenMP Tasks, and CUDA GPU):

```bash
# Complete 5-variant benchmark on 100,000 vertices (seed 42)
./run_benchmark.sh --type uniform --size 100000 --seed 42

# Structured grid mesh of 500,000 vertices
./run_benchmark.sh --type grid --size 500000
```

#### Skip Slow Sequential Baseline for Large Meshes (`--no-seq`)
For large meshes ($N \ge 1,000,000$), the single-threaded sequential baseline can become very slow. Use `--no-seq` to compare parallel OpenMP variants directly against CUDA GPU:
```bash
./run_benchmark.sh --type uniform --size 1000000 --no-seq
```

---

### 3. CPU Core Affinity (P-cores / E-cores / SMT)

Pin OpenMP execution to specific CPU cores or architectures on hybrid Intel/AMD CPUs:
```bash
# Pin execution to Performance cores (P-cores)
./run_benchmark.sh --size 100000 --cores p-cores

# Pin execution to Efficiency cores (E-cores)
./run_benchmark.sh --size 100000 --cores e-cores

# Pin to specific physical core ranges (e.g., cores 0 to 7)
./run_benchmark.sh --size 100000 --cores 0-7 --threads 8
```

---

### 4. Automated Size Sweeps and Plot Generation

Sweep across multiple vertex counts to generate CSV tables and comparison plots (`plots/*.png`):
```bash
./run_benchmark.sh --sweep --type uniform --sizes 10000,50000,100000,500000 --seed 42
```

Outputs generated:
- Detailed human-readable text logs in `results/*.txt`
- Structured machine-readable metrics in `results/*.json`
- Consolidated CSV dataset in `results/sweep_uniform_seed42.csv`
- Speedup chart in `plots/speedup_uniform.png`
- Execution runtime curves (log scale) in `plots/time_uniform.png`

---

## 🚀 Key Features

- **Isolated Submodules**: Original repositories are integrated as Git submodules in `submodules/`:
  - `submodules/Polylla-Mesh-DCEL` (Sequential DCEL baseline)
  - `submodules/MPolylla-Mesh-DCEL` (Multi-core OpenMP DCEL)
  - `submodules/MPolylla` (Multi-core OpenMP Loops / Data - `main` branch)
  - `submodules/MPolylla-Task` (Multi-core OpenMP Tasks - `Task` branch)
  - `submodules/GPolylla` (Parallel CUDA GPU)
  Each implementation is encapsulated inside an independent shared library (`libseq_runner.so`, `libmpolylla_dcel_runner.so`, `libmpolylla_runner.so`, `libmpolylla_task_runner.so`, `libgpolylla_runner.so`), preventing namespace collisions.
- **In-Memory Generation (Zero Disk Overhead)**:
  - **Grid (`grid`)**: Regular structured mesh of $k \times k$ vertices with diagonal division identical to the one used in GPolylla publications.
  - **Uniform (`uniform`)**: Uniformly distributed points in $[0, 1] \times [0, 1]$ with boundary snapping (`move_point`) and in-memory Delaunay triangulation completed in milliseconds via `delaunator-cpp`.
- **Exact Reproducibility & Statistical Rigor**:
  - `--seed <N>` to reproduce exact random distributions.
  - `--seeds <S1,S2,...>` to evaluate multiple seeds in a single experiment.
  - `--runs <N>` repetitions to report mean execution times and standard deviations ($\pm \sigma$).
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
    - OpenMP relative comparisons (Tasks vs Data, Tasks vs DCEL, Data vs DCEL)
    - GPU vs OpenMP variants (DCEL, Data, Tasks)
- **Automatic Export to `.txt`, `.json`, and `.csv`**: Generates a human-readable report with exhaustive hardware metadata (CPU, GPU, RAM, OS, CUDA), exact commit hashes of each submodule, and run parameters.

---

## 📁 Project Structure

```
PPolylla/
├── CMakeLists.txt              # CMake build configuration (C++14, CUDA, OpenMP)
├── run_benchmark.sh            # Main benchmark runner script (compiles & runs)
├── run_cores_benchmark.sh      # OpenMP CPU core scaling & thread efficiency script
├── requirements.txt            # Python dependencies (numpy, pandas, matplotlib)
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
│   ├── benchmark.py            # Python orchestrator for sweeps and plotting
│   └── benchmark_cores.py      # Python orchestrator for OpenMP core scaling and efficiency
├── results/                    # Benchmark results in TXT, JSON, and CSV
└── plots/                      # Generated plots (PNG and PDF)
```

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
| `-t, --type <grid\|uniform>` | Mesh distribution type to generate in memory | `uniform` |
| `-s, --size <N>` | Target number of vertices to generate | `100000` |
| `-r, --seed <SEED>` | Pseudorandom generator seed for a single run | `42` |
| `--seeds <S1,S2,...>` | Comma-separated list of random seeds for multi-seed evaluation | None |
| `--runs, --repeat <N>` | Number of repetitions per experiment (computes average and $\pm \sigma$) | `1` |
| `-j, --threads <T>` | OpenMP threads count (0 = all detected CPU cores) | `0` |
| `-c, --cores <SPEC>` | OpenMP core affinity (`all`, `p-cores`, `e-cores`, range e.g. `0-7`) | `all` |
| `-o, --output <PATH>` | Custom output path for comparison JSON report | `results/benchmark_...json` |
| `-f, --off <FILE>` | Use an existing `.off` file instead of in-memory generation | None |
| `--no-seq` | Skip the sequential CPU baseline (`Polylla-Mesh-DCEL`) | `false` |
| `--no-gpu` | Skip the CUDA GPU benchmark (`GPolylla`), executing CPU variants only | `false` |
| `--save-mesh <PATH>` | Save generated mesh to a persistent `.off` file | None |
| `--sweep` | Enable multi-size sweep mode via Python orchestrator | `false` |
| `--sizes <N1,N2,...>` | Comma-separated list of vertex sizes for the sweep | `10k,50k,100k,500k,1M` |
| `--cores-sweep, --scaling` | Run OpenMP thread and core scalability benchmark | `false` |
| `-h, --help` | Display command-line help menu | - |
