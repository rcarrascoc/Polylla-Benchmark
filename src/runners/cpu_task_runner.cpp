#include "cpu_task_runner.hpp"
#include <iostream>
#include <string>
#include <cstdio>
#include <omp.h>

// Include MPolylla Task core implementation
#include "mpolylla_omp.h"

extern "C" __attribute__((visibility("default")))
bool run_mpolylla_task_from_off(const char* off_path, const char* out_json_path, int num_threads, PolyllaStats* out_stats) {
    if (!off_path || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_mpolylla_task_from_off";
        }
        return false;
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    try {
        std::cout << "[CPU Task Runner] Executing MPolylla-Task (OpenMP Tasks with " 
                  << omp_get_max_threads() << " threads) on: " << off_path << std::endl;
        std::string off_str(off_path);
        std::string json_str(out_json_path);
        Polylla mesh(off_str);
        mesh.print_stats(json_str);

        bool parsed = out_stats->parse_json_file(out_json_path);
        out_stats->backend = "OMP-Task";
        out_stats->success = parsed;
        return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[CPU Task Runner Error] Exception: " << e.what() << std::endl;
        out_stats->success = false;
        out_stats->error_message = e.what();
        return false;
    } catch (...) {
        std::cerr << "[CPU Task Runner Error] Unknown exception in MPolylla-Task" << std::endl;
        out_stats->success = false;
        out_stats->error_message = "Unknown exception in MPolylla-Task";
        return false;
    }
}
