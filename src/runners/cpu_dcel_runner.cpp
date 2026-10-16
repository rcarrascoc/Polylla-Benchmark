#include "cpu_dcel_runner.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>
#include <omp.h>

// Include MPolylla-Mesh-DCEL core implementation
#include "polylla.hpp"

extern "C" __attribute__((visibility("default")))
bool run_mpolylla_dcel_from_off(const char* off_path, const char* out_json_path, int num_threads, PolyllaStats* out_stats) {
    if (!off_path || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_mpolylla_dcel_from_off";
        }
        return false;
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    try {
        std::cout << "[CPU DCEL Runner] Executing MPolylla-Mesh-DCEL (OpenMP with " 
                  << omp_get_max_threads() << " threads) on: " << off_path << std::endl;
        std::string off_str(off_path);
        std::string json_str(out_json_path);
        Polylla mesh(off_str);
        mesh.print_stats(json_str);

        bool parsed = out_stats->parse_json_file(out_json_path);
        out_stats->backend = "CPU DCEL";
        out_stats->parallel = 1;
        out_stats->success = parsed;
        return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[CPU DCEL Runner Error] Exception: " << e.what() << std::endl;
        out_stats->success = false;
        out_stats->error_message = e.what();
        return false;
    } catch (...) {
        std::cerr << "[CPU DCEL Runner Error] Unknown exception in MPolylla-Mesh-DCEL" << std::endl;
        out_stats->success = false;
        out_stats->error_message = "Unknown exception in MPolylla-Mesh-DCEL";
        return false;
    }
}
