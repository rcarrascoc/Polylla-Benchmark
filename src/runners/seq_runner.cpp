#include "seq_runner.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>

// Include sequential Polylla-Mesh-DCEL core implementation
#include "polylla.hpp"

extern "C" __attribute__((visibility("default")))
bool run_seq_polylla_from_off(const char* off_path, const char* out_json_path, PolyllaStats* out_stats) {
    if (!off_path || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_seq_polylla_from_off";
        }
        return false;
    }

    try {
        std::cout << "[Sequential Runner] Executing Polylla-Mesh-DCEL on: " << off_path << std::endl;
        std::string off_str(off_path);
        std::string json_str(out_json_path);
        Polylla mesh(off_str);
        mesh.print_stats(json_str);

        bool parsed = out_stats->parse_json_file(out_json_path);
        out_stats->backend = "Sequential";
        out_stats->parallel = 0;
        out_stats->success = parsed;
        return parsed;
    } catch (const std::exception& e) {
        std::cerr << "[Sequential Runner Error] Exception: " << e.what() << std::endl;
        out_stats->success = false;
        out_stats->error_message = e.what();
        return false;
    } catch (...) {
        std::cerr << "[Sequential Runner Error] Unknown exception in Polylla-Mesh-DCEL" << std::endl;
        out_stats->success = false;
        out_stats->error_message = "Unknown exception in Polylla-Mesh-DCEL";
        return false;
    }
}
