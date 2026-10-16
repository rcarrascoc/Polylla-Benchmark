#pragma once

#include "src/common/benchmark_types.hpp"

struct MemoryMesh;

#ifdef __cplusplus
extern "C" {
#endif

// Runs GPolylla on the input OFF file and populates out_stats.
// Returns true on success, false otherwise.
bool run_gpolylla_from_off(const char* off_path, const char* out_json_path, PolyllaStats* out_stats);

// Runs GPolylla directly from an in-memory mesh.
bool run_gpolylla_from_memory(const MemoryMesh* mesh, const char* out_json_path, PolyllaStats* out_stats);

#ifdef __cplusplus
}
#endif
