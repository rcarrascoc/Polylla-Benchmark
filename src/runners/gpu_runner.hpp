#pragma once

#include "src/common/benchmark_types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Runs GPolylla on the input OFF file and populates out_stats.
// Returns true on success, false otherwise.
bool run_gpolylla_from_off(const char* off_path, const char* out_json_path, PolyllaStats* out_stats);

#ifdef __cplusplus
}
#endif
