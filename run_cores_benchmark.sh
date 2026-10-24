#!/usr/bin/env bash
# ==============================================================================
# PPolylla OpenMP Cores & Thread Scalability Benchmark Runner
# Measures strong scaling across thread counts and CPU core types (P-cores / E-cores)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/ppolylla_benchmark"

# Check submodules
if [ ! -f "${SCRIPT_DIR}/submodules/GPolylla/CMakeLists.txt" ] || [ ! -f "${SCRIPT_DIR}/submodules/MPolylla/CMakeLists.txt" ] || [ ! -f "${SCRIPT_DIR}/submodules/MPolylla-Mesh-DCEL/CMakeLists.txt" ]; then
    echo "[PPolylla] Initializing and updating git submodules..."
    git submodule update --init --recursive
fi

# Build project if binary doesn't exist
if [ ! -f "${BINARY}" ]; then
    echo "[PPolylla] Binary not found. Building project with CMake..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
    make -j$(nproc)
    cd "${SCRIPT_DIR}"
    echo "[PPolylla] Build successful!"
fi

# Execute cores scalability benchmark
python3 "${SCRIPT_DIR}/scripts/benchmark_cores.py" "$@"
