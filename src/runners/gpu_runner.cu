#include "gpu_runner.hpp"
#include <iostream>
#include <string>
#include <cstdio>

// Include GPolylla core implementation
#include "polylla.cu"

extern "C" __attribute__((visibility("default")))
bool run_gpolylla_from_off(const char* off_path, const char* out_json_path, PolyllaStats* out_stats) {
    if (!off_path || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_gpolylla_from_off";
        }
        return false;
    }

    try {
        std::cout << "[GPU Runner] Executing GPolylla on: " << off_path << std::endl;
        std::string off_str(off_path);
        std::string json_str(out_json_path);
        Polylla mesh(off_str);
        mesh.print_stats(json_str);

        bool parsed = out_stats->parse_json_file(out_json_path);
        out_stats->backend = "GPU";
        out_stats->success = parsed;
        return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[GPU Runner Error] Exception: " << e.what() << std::endl;
        out_stats->success = false;
        out_stats->error_message = e.what();
        return false;
    } catch (...) {
        std::cerr << "[GPU Runner Error] Unknown exception in GPolylla" << std::endl;
        out_stats->success = false;
        out_stats->error_message = "Unknown exception in GPolylla";
        return false;
    }
}
