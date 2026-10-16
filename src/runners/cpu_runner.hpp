#pragma once

#include "src/common/benchmark_types.hpp"

struct MemoryMesh;

#ifdef __cplusplus
extern "C" {
#endif

// Runs MPolylla (OpenMP) on the input OFF file and populates out_stats.
// If num_threads <= 0, uses omp_get_max_threads().
// Returns true on success, false otherwise.
bool run_mpolylla_from_off(const char* off_path, const char* out_json_path, int num_threads, PolyllaStats* out_stats);

// Runs MPolylla (OpenMP) directly from an in-memory mesh.
bool run_mpolylla_from_memory(const MemoryMesh* mesh, const char* out_json_path, int num_threads, PolyllaStats* out_stats);

#ifdef __cplusplus
}
#endif
