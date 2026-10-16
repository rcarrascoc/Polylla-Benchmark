#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cstdint>
#include <ctime>
#include <sys/utsname.h>
#include <unistd.h>

inline std::string exec_cmd_safe(const char* cmd) {
    char buffer[256];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "N/A";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
        result.pop_back();
    }
    return result.empty() ? "N/A" : result;
}

struct SystemMetadata {
    std::string timestamp;
    std::string command_line;
    std::string hostname;
    std::string os_info;
    std::string cpu_model;
    int cpu_cores = 0;
    std::string gpu_info;
    std::string cuda_version;
    std::string commit_main;
    std::string commit_gpolylla;
    std::string commit_mpolylla;
    std::string commit_mpolylla_task;
    std::string commit_mpolylla_dcel;
    std::string commit_seq;

    void detect(const std::string& cmd_line = "") {
        command_line = cmd_line;

        // Timestamp
        std::time_t now = std::time(nullptr);
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S %Z", std::localtime(&now));
        timestamp = time_buf;

        // OS and Host
        utsname u;
        if (uname(&u) == 0) {
            hostname = u.nodename;
            os_info = std::string(u.sysname) + " " + u.release + " (" + u.machine + ")";
        }

        // CPU
        cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
        cpu_model = exec_cmd_safe("lscpu | grep 'Model name:' | sed 's/Model name:[ \t]*//'");

        // GPU
        gpu_info = exec_cmd_safe("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader");
        cuda_version = exec_cmd_safe("nvcc --version | grep 'release' | sed 's/.*release //;s/,.*//'");

        // Git commits
        commit_main = exec_cmd_safe("git rev-parse HEAD");
        commit_gpolylla = exec_cmd_safe("git -C submodules/GPolylla rev-parse HEAD");
        commit_mpolylla = exec_cmd_safe("git -C submodules/MPolylla rev-parse HEAD");
        commit_mpolylla_task = exec_cmd_safe("git -C submodules/MPolylla-Task rev-parse HEAD");
        commit_mpolylla_dcel = exec_cmd_safe("git -C submodules/MPolylla-Mesh-DCEL rev-parse HEAD");
        commit_seq = exec_cmd_safe("git -C submodules/Polylla-Mesh-DCEL rev-parse HEAD");
    }
};

struct PolyllaStats {
    std::string backend; // "Sequential", "CPU", "GPU"
    int parallel = 0;
    int n_polygons = 0;
    int n_half_edges = 0;
    int n_faces = 0;
    int n_vertices = 0;
    int n_frontier_edges = 0;
    int n_barrier_edge_tips = 0;
    int n_polygons_to_repair = 0;
    int n_polygons_added_after_repair = 0;

    // Timers in milliseconds (ms)
    double time_triangulation_generation = 0.0;
    double d_time_copy_to_device = 0.0;
    double d_time_to_label_max_edges = 0.0;
    double d_time_to_label_frontier_edges = 0.0;
    double d_time_to_label_seed_edges = 0.0;
    double d_time_to_label_extra_frontier_edge = 0.0;
    double d_time_to_traversal = 0.0;
    double d_time_to_traversal_search_frontier_edge = 0.0;
    double d_time_to_overwrite_seed = 0.0;
    double d_time_to_label_scan_edges = 0.0;
    double d_time_to_label_compaction_edges = 0.0;
    double d_time_to_label_total = 0.0;
    double d_time_to_traversal_and_repair = 0.0;
    double d_time_to_back_to_host = 0.0;
    double d_time_to_repair = 0.0;
    double d_time_to_generate_polygonal_mesh = 0.0; // Total algorithm time

    // Total time with memory copies (GPU)
    double total_with_copies = 0.0;

    // Memory usage (bytes)
    long long memory_total = 0;
    long long memory_total_gpu = 0;

    bool success = false;
    std::string error_message = "";

    void compute_derived() {
        if (d_time_to_label_total == 0.0) {
            d_time_to_label_total = d_time_to_label_max_edges + d_time_to_label_frontier_edges + d_time_to_label_seed_edges;
        }
        total_with_copies = d_time_to_generate_polygonal_mesh + d_time_copy_to_device + d_time_to_back_to_host;
    }

    bool parse_json_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            error_message = "Cannot open JSON file: " + filename;
            success = false;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            size_t last = line.find_last_not_of(" \t\r\n,");
            line = line.substr(first, (last - first + 1));

            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;

            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);

            auto clean_str = [](std::string& s) {
                size_t q1 = s.find('"');
                while (q1 != std::string::npos) {
                    s.erase(q1, 1);
                    q1 = s.find('"');
                }
                size_t c = s.find_first_not_of(" \t");
                if (c != std::string::npos) s = s.substr(c);
                c = s.find_last_not_of(" \t,");
                if (c != std::string::npos) s = s.substr(0, c + 1);
            };

            clean_str(key);
            clean_str(val);

            try {
                if (key == "parallel") parallel = std::stoi(val);
                else if (key == "n_polygons") n_polygons = std::stoi(val);
                else if (key == "n_frontier_edges") n_frontier_edges = std::stoi(val);
                else if (key == "n_barrier_edge_tips") n_barrier_edge_tips = std::stoi(val);
                else if (key == "n_half_edges") n_half_edges = std::stoi(val);
                else if (key == "n_faces") n_faces = std::stoi(val);
                else if (key == "n_vertices") n_vertices = std::stoi(val);
                else if (key == "n_polygons_to_repair") n_polygons_to_repair = std::stoi(val);
                else if (key == "n_polygons_added_after_repair") n_polygons_added_after_repair = std::stoi(val);
                else if (key == "time_triangulation_generation") time_triangulation_generation = std::stod(val);
                else if (key == "d_time_copy_to_device") d_time_copy_to_device = std::stod(val);
                else if (key == "d_time_to_label_max_edges" || key == "time_to_label_max_edges") d_time_to_label_max_edges = std::stod(val);
                else if (key == "d_time_to_label_frontier_edges" || key == "time_to_label_frontier_edges") d_time_to_label_frontier_edges = std::stod(val);
                else if (key == "d_time_to_label_seed_edges" || key == "time_to_label_seed_edges") d_time_to_label_seed_edges = std::stod(val);
                else if (key == "d_time_to_label_extra_frontier_edge") d_time_to_label_extra_frontier_edge = std::stod(val);
                else if (key == "d_time_to_label_scan_edges") d_time_to_label_scan_edges = std::stod(val);
                else if (key == "d_time_to_label_compaction_edges") d_time_to_label_compaction_edges = std::stod(val);
                else if (key == "d_time_to_label_total" || key == "time_to_label_total") d_time_to_label_total = std::stod(val);
                else if (key == "d_time_to_traversal_and_repair" || key == "time_to_traversal_and_repair") d_time_to_traversal_and_repair = std::stod(val);
                else if (key == "d_time_to_traversal" || key == "time_to_traversal") d_time_to_traversal = std::stod(val);
                else if (key == "d_time_to_traversal_search_frontier_edge") d_time_to_traversal_search_frontier_edge = std::stod(val);
                else if (key == "d_time_to_back_to_host") d_time_to_back_to_host = std::stod(val);
                else if (key == "d_time_to_overwrite_seed") d_time_to_overwrite_seed = std::stod(val);
                else if (key == "d_time_to_repair" || key == "time_to_repair") d_time_to_repair = std::stod(val);
                else if (key == "d_time_to_generate_polygonal_mesh" || key == "time_to_generate_polygonal_mesh") d_time_to_generate_polygonal_mesh = std::stod(val);
                else if (key == "memory_total") memory_total = std::stoll(val);
                else if (key == "memory_total_gpu") memory_total_gpu = std::stoll(val);
            } catch (...) {
            }
        }
        file.close();
        compute_derived();
        success = true;
        return true;
    }
};

struct BenchmarkComparison {
    std::string mesh_type;
    int requested_size = 0;
    uint64_t seed = 0;
    int cpu_threads = 0;
    std::string cpu_cores_affinity = "all";
    std::vector<int> cpu_cores_list;

    double mesh_gen_time_ms = 0.0;
    double off_write_time_ms = 0.0;

    bool has_seq = false;
    PolyllaStats seq_stats;
    bool has_dcel = false;
    PolyllaStats dcel_stats;      // OpenMP DCEL (rcarrascoc/MPolylla-Mesh-DCEL)
    PolyllaStats cpu_stats;       // OpenMP Data (branch main)
    bool has_task = false;
    PolyllaStats task_stats;      // OpenMP Tasks (branch Task)
    PolyllaStats gpu_stats;       // CUDA GPU

    SystemMetadata meta;

    double speedup_gpu_vs_seq = 0.0;             // Seq time / GPU time
    double speedup_dcel_vs_seq = 0.0;            // Seq time / OMP DCEL time
    double speedup_omp_vs_seq = 0.0;             // Seq time / OMP data time
    double speedup_task_vs_seq = 0.0;            // Seq time / OMP task time
    double speedup_gpu_vs_dcel = 0.0;            // OMP DCEL time / GPU time
    double speedup_gpu_vs_omp = 0.0;             // OMP data time / GPU time
    double speedup_gpu_vs_task = 0.0;            // OMP task time / GPU time
    double speedup_dcel_vs_omp = 0.0;            // OMP data time / OMP DCEL time
    double speedup_task_vs_omp = 0.0;            // OMP data time / OMP task time
    double speedup_task_vs_dcel = 0.0;           // OMP DCEL time / OMP task time
    double speedup_gpu_with_copies_vs_seq = 0.0; // Seq time / (GPU time + copies)

    void compute_comparison() {
        if (has_seq && seq_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            if (gpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_gpu_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (has_dcel && dcel_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_dcel_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / dcel_stats.d_time_to_generate_polygonal_mesh;
            }
            if (cpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_omp_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / cpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (has_task && task_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_task_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / task_stats.d_time_to_generate_polygonal_mesh;
            }
            if (gpu_stats.total_with_copies > 0.0) {
                speedup_gpu_with_copies_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / gpu_stats.total_with_copies;
            }
        }
        if (gpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            if (has_dcel && dcel_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_gpu_vs_dcel = dcel_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (cpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_gpu_vs_omp = cpu_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (has_task && task_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_gpu_vs_task = task_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
            }
        }
        if (has_dcel && cpu_stats.d_time_to_generate_polygonal_mesh > 0.0 && dcel_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            speedup_dcel_vs_omp = cpu_stats.d_time_to_generate_polygonal_mesh / dcel_stats.d_time_to_generate_polygonal_mesh;
        }
        if (has_task && cpu_stats.d_time_to_generate_polygonal_mesh > 0.0 && task_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            speedup_task_vs_omp = cpu_stats.d_time_to_generate_polygonal_mesh / task_stats.d_time_to_generate_polygonal_mesh;
        }
        if (has_task && has_dcel && dcel_stats.d_time_to_generate_polygonal_mesh > 0.0 && task_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            speedup_task_vs_dcel = dcel_stats.d_time_to_generate_polygonal_mesh / task_stats.d_time_to_generate_polygonal_mesh;
        }
    }

    void print_table(std::ostream& os = std::cout) const {
        os << "\n"
           << "==========================================================================================================================================================\n"
           << "                                                              POLYLLA BENCHMARK COMPARISON REPORT                                                         \n"
           << "                                  (Polylla-Mesh-DCEL vs MPolylla DCEL vs MPolylla Data vs MPolylla Task vs GPolylla)                                    \n"
           << "==========================================================================================================================================================\n";
        os << "  Mesh Type:       " << mesh_type << "\n"
           << "  Requested Size:  " << requested_size << "\n"
           << "  Random Seed:     " << seed << "\n"
           << "  CPU Threads:     " << cpu_threads << "\n"
           << "  CPU Cores/Aff:   " << cpu_cores_affinity << "\n"
           << "  Mesh Gen Time:   " << std::fixed << std::setprecision(3) << mesh_gen_time_ms << " ms\n"
           << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n"
           << "  Metric / Element             Polylla-Mesh-DCEL    MPolylla DCEL    MPolylla Data    MPolylla Task        GPolylla           Validation            \n"
           << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";

        auto print_elem = [&](const char* name, long long seq, long long dcel, long long omp, long long task, long long gpu) {
            bool all_parallel_match = (omp == gpu && (!has_task || task == gpu) && (!has_dcel || dcel == gpu));
            std::string status = all_parallel_match ? "MATCH (All Parallel)" : "DIFF (Parallel)";
            if (has_seq && (seq != omp || (has_task && seq != task) || (has_dcel && seq != dcel))) status += "*";
            os << "  " << std::left << std::setw(25) << name
               << "  " << std::right << std::setw(17) << (has_seq ? std::to_string(seq) : "-")
               << "  " << std::right << std::setw(15) << (has_dcel ? std::to_string(dcel) : "-")
               << "  " << std::right << std::setw(15) << omp
               << "  " << std::right << std::setw(15) << (has_task ? std::to_string(task) : "-")
               << "  " << std::right << std::setw(14) << gpu
               << "  " << std::right << std::setw(23) << status
               << "\n";
        };

        print_elem("Vertices (n_vertices)", seq_stats.n_vertices, dcel_stats.n_vertices, cpu_stats.n_vertices, task_stats.n_vertices, gpu_stats.n_vertices);
        print_elem("Faces (n_faces)", seq_stats.n_faces, dcel_stats.n_faces, cpu_stats.n_faces, task_stats.n_faces, gpu_stats.n_faces);
        print_elem("Half-Edges (n_half_edges)", seq_stats.n_half_edges, dcel_stats.n_half_edges, cpu_stats.n_half_edges, task_stats.n_half_edges, gpu_stats.n_half_edges);
        print_elem("Polygons (n_polygons)", seq_stats.n_polygons, dcel_stats.n_polygons, cpu_stats.n_polygons, task_stats.n_polygons, gpu_stats.n_polygons);

        os << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n"
           << "  Phase Timers (ms)            Polylla-Mesh-DCEL    MPolylla DCEL    MPolylla Data    MPolylla Task        GPolylla       Speedup GPolylla/Base       \n"
           << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";

        auto print_timer = [&](const char* name, double seq_t, double dcel_t, double omp_t, double task_t, double gpu_t) {
            double sp_gpu_seq = (has_seq && gpu_t > 0.0) ? (seq_t / gpu_t) : 0.0;
            os << "  " << std::left << std::setw(25) << name
               << "  " << std::right << std::setw(17) << std::fixed << std::setprecision(4) << (has_seq ? seq_t : 0.0)
               << "  " << std::right << std::setw(15) << std::fixed << std::setprecision(4) << (has_dcel ? dcel_t : 0.0)
               << "  " << std::right << std::setw(15) << std::fixed << std::setprecision(4) << omp_t
               << "  " << std::right << std::setw(15) << std::fixed << std::setprecision(4) << (has_task ? task_t : 0.0)
               << "  " << std::right << std::setw(14) << std::fixed << std::setprecision(4) << gpu_t;
            if (sp_gpu_seq > 0.0) {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << sp_gpu_seq << "x";
                os << "  " << std::right << std::setw(24) << ss.str() << "\n";
            } else {
                os << "  " << std::right << std::setw(25) << "-\n";
            }
        };

        print_timer("1. Triangulation Struct", seq_stats.time_triangulation_generation, dcel_stats.time_triangulation_generation, cpu_stats.time_triangulation_generation, task_stats.time_triangulation_generation, gpu_stats.time_triangulation_generation);
        print_timer("2. Label Max Edges", seq_stats.d_time_to_label_max_edges, dcel_stats.d_time_to_label_max_edges, cpu_stats.d_time_to_label_max_edges, task_stats.d_time_to_label_max_edges, gpu_stats.d_time_to_label_max_edges);
        print_timer("3. Label Frontier Edges", seq_stats.d_time_to_label_frontier_edges, dcel_stats.d_time_to_label_frontier_edges, cpu_stats.d_time_to_label_frontier_edges, task_stats.d_time_to_label_frontier_edges, gpu_stats.d_time_to_label_frontier_edges);
        print_timer("4. Label Seed Edges", seq_stats.d_time_to_label_seed_edges, dcel_stats.d_time_to_label_seed_edges, cpu_stats.d_time_to_label_seed_edges, task_stats.d_time_to_label_seed_edges, gpu_stats.d_time_to_label_seed_edges);
        print_timer("5. Repair Extra Frontier", seq_stats.d_time_to_repair, dcel_stats.d_time_to_repair, cpu_stats.d_time_to_label_extra_frontier_edge, task_stats.d_time_to_label_extra_frontier_edge, gpu_stats.d_time_to_label_extra_frontier_edge);
        print_timer("6. Traversal Phase 1", seq_stats.d_time_to_traversal, dcel_stats.d_time_to_traversal, cpu_stats.d_time_to_traversal, task_stats.d_time_to_traversal, gpu_stats.d_time_to_traversal);
        print_timer("7. Traversal Phase 2 (Search)", 0.0, 0.0, cpu_stats.d_time_to_traversal_search_frontier_edge, task_stats.d_time_to_traversal_search_frontier_edge, gpu_stats.d_time_to_traversal_search_frontier_edge);
        print_timer("8. Repair Overwrite Seed", 0.0, 0.0, cpu_stats.d_time_to_overwrite_seed, task_stats.d_time_to_overwrite_seed, gpu_stats.d_time_to_overwrite_seed);
        print_timer("9. Label Seed Scan", 0.0, 0.0, cpu_stats.d_time_to_label_scan_edges, task_stats.d_time_to_label_scan_edges, gpu_stats.d_time_to_label_scan_edges);
        print_timer("10. Seed Compaction", 0.0, 0.0, cpu_stats.d_time_to_label_compaction_edges, task_stats.d_time_to_label_compaction_edges, gpu_stats.d_time_to_label_compaction_edges);

        os << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
        print_timer("TOTAL ALGORITHM TIME", seq_stats.d_time_to_generate_polygonal_mesh, dcel_stats.d_time_to_generate_polygonal_mesh, cpu_stats.d_time_to_generate_polygonal_mesh, task_stats.d_time_to_generate_polygonal_mesh, gpu_stats.d_time_to_generate_polygonal_mesh);
        os << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
        os << "  GPU Transfer Host->Device:      " << std::right << std::setw(16) << std::fixed << std::setprecision(4) << gpu_stats.d_time_copy_to_device << " ms\n";
        os << "  GPU Transfer Device->Host:      " << std::right << std::setw(16) << std::fixed << std::setprecision(4) << gpu_stats.d_time_to_back_to_host << " ms\n";
        os << "  GPU TOTAL (Alg + Transfers):    " << std::right << std::setw(16) << std::fixed << std::setprecision(4) << gpu_stats.total_with_copies << " ms\n";
        os << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
        std::string c_unit = (cpu_threads == 1 ? "core" : "cores");
        std::string omp_c_str = "(" + std::to_string(cpu_threads) + " " + c_unit + ")";
        if (has_seq) {
            os << "  >>> SPEEDUP GPolylla vs Polylla-Mesh-DCEL (Alg only):         " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_seq << "x\n";
            os << "  >>> SPEEDUP GPolylla vs Polylla-Mesh-DCEL (With Transfers):   " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_with_copies_vs_seq << "x\n";
            if (has_dcel) {
                os << "  >>> SPEEDUP MPolylla DCEL " << std::left << std::setw(11) << omp_c_str << " vs Polylla-Mesh-DCEL:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_dcel_vs_seq << "x\n";
            }
            os << "  >>> SPEEDUP MPolylla Data " << std::left << std::setw(11) << omp_c_str << " vs Polylla-Mesh-DCEL:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_omp_vs_seq << "x\n";
            if (has_task) {
                os << "  >>> SPEEDUP MPolylla Task " << std::left << std::setw(11) << omp_c_str << " vs Polylla-Mesh-DCEL:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_task_vs_seq << "x\n";
            }
        }
        if (has_dcel) {
            os << "  >>> SPEEDUP GPolylla vs MPolylla DCEL " << std::left << std::setw(11) << omp_c_str << ":                " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_dcel << "x\n";
        }
        os << "  >>> SPEEDUP GPolylla vs MPolylla Data " << std::left << std::setw(11) << omp_c_str << ":                " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_omp << "x\n";
        if (has_task) {
            os << "  >>> SPEEDUP GPolylla vs MPolylla Task " << std::left << std::setw(11) << omp_c_str << ":                " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_task << "x\n";
            os << "  >>> SPEEDUP MPolylla Task vs MPolylla Data " << std::left << std::setw(11) << omp_c_str << ":           " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_task_vs_omp << "x\n";
        }
        os << "==========================================================================================================================================================\n\n";
    }

    void save_txt(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "Warning: Could not save TXT report to " << filepath << std::endl;
            return;
        }

        out << "======================================================================================================================================\n"
            << "                                                  POLYLLA BENCHMARK EXECUTION REPORT\n"
            << "======================================================================================================================================\n\n"
            << "[1] EXECUTION METADATA\n"
            << "--------------------------------------------------------------------------------------------------------------------------------------\n"
            << "  Timestamp:            " << meta.timestamp << "\n"
            << "  Invocation Command:   " << meta.command_line << "\n"
            << "  Hostname:             " << meta.hostname << "\n"
            << "  OS / Kernel:          " << meta.os_info << "\n"
            << "  CPU Model:            " << meta.cpu_model << " (" << meta.cpu_cores << " logical threads)\n"
            << "  GPU Hardware:         " << meta.gpu_info << "\n"
            << "  CUDA Version:         " << meta.cuda_version << "\n\n"
            << "[2] SUBMODULE VERSIONS & COMMITS\n"
            << "--------------------------------------------------------------------------------------------------------------------------------------\n"
            << "  Project (PPolylla):         " << meta.commit_main << "\n"
            << "  Baseline (Sequential):      https://github.com/ssalinasfe/Polylla-Mesh-DCEL.git\n"
            << "                              Commit: " << meta.commit_seq << "\n"
            << "  OpenMP DCEL (MPolylla):     https://github.com/rcarrascoc/MPolylla-Mesh-DCEL.git\n"
            << "                              Commit: " << meta.commit_mpolylla_dcel << "\n"
            << "  OpenMP Data (MPolylla):     https://github.com/rcarrascoc/MPolylla.git (branch main)\n"
            << "                              Commit: " << meta.commit_mpolylla << "\n"
            << "  OpenMP Tasks (MPolylla):    https://github.com/rcarrascoc/MPolylla.git (branch Task)\n"
            << "                              Commit: " << meta.commit_mpolylla_task << "\n"
            << "  GPU (GPolylla):             https://github.com/ssalinasfe/GPolylla.git\n"
            << "                              Commit: " << meta.commit_gpolylla << "\n\n"
            << "[3] EXPERIMENT PARAMETERS\n"
            << "--------------------------------------------------------------------------------------------------------------------------------------\n"
            << "  Mesh Type:            " << mesh_type << "\n"
            << "  Requested Vertices:   " << requested_size << "\n"
            << "  Random Seed:          " << seed << "\n"
            << "  CPU OpenMP Threads:   " << cpu_threads << "\n"
            << "  CPU Core Affinity:    " << cpu_cores_affinity << "\n"
            << "  Mesh Gen Time (RAM):  " << std::fixed << std::setprecision(3) << mesh_gen_time_ms << " ms\n"
            << "  RAM Buffer Write:     " << std::fixed << std::setprecision(3) << off_write_time_ms << " ms\n\n";

        // Call print_table on out stream
        print_table(out);

        out << "[4] DETAILED MEMORY FOOTPRINT\n"
            << "--------------------------------------------------------------------------------------------------------------------------------------\n"
            << "  GPU VRAM Usage (cudaMemGetInfo):  " << gpu_stats.memory_total_gpu << " bytes ("
            << std::fixed << std::setprecision(2) << (gpu_stats.memory_total_gpu / (1024.0 * 1024.0)) << " MB)\n"
            << "  GPU Data Structures Estimated:    " << gpu_stats.memory_total << " bytes ("
            << std::fixed << std::setprecision(2) << (gpu_stats.memory_total / (1024.0 * 1024.0)) << " MB)\n"
            << "  Sequential CPU Memory Estimated:  " << seq_stats.memory_total << " bytes ("
            << std::fixed << std::setprecision(2) << (seq_stats.memory_total / (1024.0 * 1024.0)) << " MB)\n\n"
            << "[5] REPRODUCIBILITY INSTRUCTIONS\n"
            << "--------------------------------------------------------------------------------------------------------------------------------------\n"
            << "  To reproduce this exact experiment with the same input mesh, execute:\n"
            << "  ./run_benchmark.sh --type " << mesh_type << " --size " << requested_size << " --seed " << seed;
        if (cpu_threads > 0) out << " --threads " << cpu_threads;
        if (!cpu_cores_affinity.empty() && cpu_cores_affinity.find("all") == std::string::npos) {
            out << " --cores \"" << cpu_cores_affinity << "\"";
        }
        out << "\n"
            << "======================================================================================================================================\n";

        out.close();
        std::cout << "[Report] Saved TXT benchmark report to:  " << filepath << std::endl;
    }

    void save_json(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "Warning: Could not save JSON to " << filepath << std::endl;
            return;
        }
        out << "{\n"
            << "  \"metadata\": {\n"
            << "    \"timestamp\": \"" << meta.timestamp << "\",\n"
            << "    \"command_line\": \"" << meta.command_line << "\",\n"
            << "    \"hostname\": \"" << meta.hostname << "\",\n"
            << "    \"os_info\": \"" << meta.os_info << "\",\n"
            << "    \"cpu_model\": \"" << meta.cpu_model << "\",\n"
            << "    \"cpu_cores\": " << meta.cpu_cores << ",\n"
            << "    \"gpu_info\": \"" << meta.gpu_info << "\",\n"
            << "    \"cuda_version\": \"" << meta.cuda_version << "\",\n"
            << "    \"commit_main\": \"" << meta.commit_main << "\",\n"
            << "    \"commit_gpolylla\": \"" << meta.commit_gpolylla << "\",\n"
            << "    \"commit_mpolylla\": \"" << meta.commit_mpolylla << "\",\n"
            << "    \"commit_mpolylla_task\": \"" << meta.commit_mpolylla_task << "\",\n"
            << "    \"commit_mpolylla_dcel\": \"" << meta.commit_mpolylla_dcel << "\",\n"
            << "    \"commit_seq\": \"" << meta.commit_seq << "\"\n"
            << "  },\n"
            << "  \"mesh_type\": \"" << mesh_type << "\",\n"
            << "  \"requested_size\": " << requested_size << ",\n"
            << "  \"seed\": " << seed << ",\n"
            << "  \"cpu_threads\": " << cpu_threads << ",\n"
            << "  \"cpu_cores_affinity\": \"" << cpu_cores_affinity << "\",\n"
            << "  \"mesh_gen_time_ms\": " << mesh_gen_time_ms << ",\n"
            << "  \"off_write_time_ms\": " << off_write_time_ms << ",\n"
            << "  \"has_sequential_baseline\": " << (has_seq ? "true" : "false") << ",\n"
            << "  \"has_omp_dcel\": " << (has_dcel ? "true" : "false") << ",\n"
            << "  \"has_omp_task\": " << (has_task ? "true" : "false") << ",\n"
            << "  \"speedup_gpu_vs_sequential\": " << speedup_gpu_vs_seq << ",\n"
            << "  \"speedup_gpu_with_copies_vs_sequential\": " << speedup_gpu_with_copies_vs_seq << ",\n"
            << "  \"speedup_omp_dcel_vs_sequential\": " << speedup_dcel_vs_seq << ",\n"
            << "  \"speedup_omp_data_vs_sequential\": " << speedup_omp_vs_seq << ",\n"
            << "  \"speedup_omp_vs_sequential\": " << speedup_omp_vs_seq << ",\n"
            << "  \"speedup_omp_task_vs_sequential\": " << speedup_task_vs_seq << ",\n"
            << "  \"speedup_gpu_vs_omp_dcel\": " << speedup_gpu_vs_dcel << ",\n"
            << "  \"speedup_gpu_vs_omp_data\": " << speedup_gpu_vs_omp << ",\n"
            << "  \"speedup_gpu_vs_omp\": " << speedup_gpu_vs_omp << ",\n"
            << "  \"speedup_gpu_vs_omp_task\": " << speedup_gpu_vs_task << ",\n"
            << "  \"speedup_omp_task_vs_omp_data\": " << speedup_task_vs_omp << ",\n"
            << "  \"speedup_omp_task_vs_omp\": " << speedup_task_vs_omp << ",\n";

        if (has_seq) {
            out << "  \"sequential\": {\n"
                << "    \"n_vertices\": " << seq_stats.n_vertices << ",\n"
                << "    \"n_faces\": " << seq_stats.n_faces << ",\n"
                << "    \"n_half_edges\": " << seq_stats.n_half_edges << ",\n"
                << "    \"n_polygons\": " << seq_stats.n_polygons << ",\n"
                << "    \"time_triangulation_generation\": " << seq_stats.time_triangulation_generation << ",\n"
                << "    \"time_label_max_edges\": " << seq_stats.d_time_to_label_max_edges << ",\n"
                << "    \"time_label_frontier_edges\": " << seq_stats.d_time_to_label_frontier_edges << ",\n"
                << "    \"time_label_seed_edges\": " << seq_stats.d_time_to_label_seed_edges << ",\n"
                << "    \"time_traversal\": " << seq_stats.d_time_to_traversal << ",\n"
                << "    \"time_repair\": " << seq_stats.d_time_to_repair << ",\n"
                << "    \"time_total_algorithm\": " << seq_stats.d_time_to_generate_polygonal_mesh << "\n"
                << "  },\n";
        }

        if (has_dcel) {
            out << "  \"cpu_dcel\": {\n"
                << "    \"n_vertices\": " << dcel_stats.n_vertices << ",\n"
                << "    \"n_faces\": " << dcel_stats.n_faces << ",\n"
                << "    \"n_half_edges\": " << dcel_stats.n_half_edges << ",\n"
                << "    \"n_polygons\": " << dcel_stats.n_polygons << ",\n"
                << "    \"time_triangulation_generation\": " << dcel_stats.time_triangulation_generation << ",\n"
                << "    \"time_label_max_edges\": " << dcel_stats.d_time_to_label_max_edges << ",\n"
                << "    \"time_label_frontier_edges\": " << dcel_stats.d_time_to_label_frontier_edges << ",\n"
                << "    \"time_label_seed_edges\": " << dcel_stats.d_time_to_label_seed_edges << ",\n"
                << "    \"time_traversal\": " << dcel_stats.d_time_to_traversal << ",\n"
                << "    \"time_repair\": " << dcel_stats.d_time_to_repair << ",\n"
                << "    \"time_total_algorithm\": " << dcel_stats.d_time_to_generate_polygonal_mesh << "\n"
                << "  },\n";
        }

        out << "  \"cpu_data\": {\n"
            << "    \"n_vertices\": " << cpu_stats.n_vertices << ",\n"
            << "    \"n_faces\": " << cpu_stats.n_faces << ",\n"
            << "    \"n_half_edges\": " << cpu_stats.n_half_edges << ",\n"
            << "    \"n_polygons\": " << cpu_stats.n_polygons << ",\n"
            << "    \"time_triangulation_generation\": " << cpu_stats.time_triangulation_generation << ",\n"
            << "    \"time_label_max_edges\": " << cpu_stats.d_time_to_label_max_edges << ",\n"
            << "    \"time_label_frontier_edges\": " << cpu_stats.d_time_to_label_frontier_edges << ",\n"
            << "    \"time_label_seed_edges\": " << cpu_stats.d_time_to_label_seed_edges << ",\n"
            << "    \"time_label_extra_frontier_edge\": " << cpu_stats.d_time_to_label_extra_frontier_edge << ",\n"
            << "    \"time_traversal_1\": " << cpu_stats.d_time_to_traversal << ",\n"
            << "    \"time_traversal_2\": " << cpu_stats.d_time_to_traversal_search_frontier_edge << ",\n"
            << "    \"time_overwrite_seed\": " << cpu_stats.d_time_to_overwrite_seed << ",\n"
            << "    \"time_label_scan_edges\": " << cpu_stats.d_time_to_label_scan_edges << ",\n"
            << "    \"time_label_compaction_edges\": " << cpu_stats.d_time_to_label_compaction_edges << ",\n"
            << "    \"time_total_algorithm\": " << cpu_stats.d_time_to_generate_polygonal_mesh << "\n"
            << "  },\n";

        if (has_task) {
            out << "  \"cpu_tasks\": {\n"
                << "    \"n_vertices\": " << task_stats.n_vertices << ",\n"
                << "    \"n_faces\": " << task_stats.n_faces << ",\n"
                << "    \"n_half_edges\": " << task_stats.n_half_edges << ",\n"
                << "    \"n_polygons\": " << task_stats.n_polygons << ",\n"
                << "    \"time_triangulation_generation\": " << task_stats.time_triangulation_generation << ",\n"
                << "    \"time_label_max_edges\": " << task_stats.d_time_to_label_max_edges << ",\n"
                << "    \"time_label_frontier_edges\": " << task_stats.d_time_to_label_frontier_edges << ",\n"
                << "    \"time_label_seed_edges\": " << task_stats.d_time_to_label_seed_edges << ",\n"
                << "    \"time_label_extra_frontier_edge\": " << task_stats.d_time_to_label_extra_frontier_edge << ",\n"
                << "    \"time_traversal_1\": " << task_stats.d_time_to_traversal << ",\n"
                << "    \"time_traversal_2\": " << task_stats.d_time_to_traversal_search_frontier_edge << ",\n"
                << "    \"time_overwrite_seed\": " << task_stats.d_time_to_overwrite_seed << ",\n"
                << "    \"time_label_scan_edges\": " << task_stats.d_time_to_label_scan_edges << ",\n"
                << "    \"time_label_compaction_edges\": " << task_stats.d_time_to_label_compaction_edges << ",\n"
                << "    \"time_total_algorithm\": " << task_stats.d_time_to_generate_polygonal_mesh << "\n"
                << "  },\n";
        }

        out << "  \"gpu\": {\n"
            << "    \"n_vertices\": " << gpu_stats.n_vertices << ",\n"
            << "    \"n_faces\": " << gpu_stats.n_faces << ",\n"
            << "    \"n_half_edges\": " << gpu_stats.n_half_edges << ",\n"
            << "    \"n_polygons\": " << gpu_stats.n_polygons << ",\n"
            << "    \"time_triangulation_generation\": " << gpu_stats.time_triangulation_generation << ",\n"
            << "    \"time_copy_to_device\": " << gpu_stats.d_time_copy_to_device << ",\n"
            << "    \"time_label_max_edges\": " << gpu_stats.d_time_to_label_max_edges << ",\n"
            << "    \"time_label_frontier_edges\": " << gpu_stats.d_time_to_label_frontier_edges << ",\n"
            << "    \"time_label_seed_edges\": " << gpu_stats.d_time_to_label_seed_edges << ",\n"
            << "    \"time_label_extra_frontier_edge\": " << gpu_stats.d_time_to_label_extra_frontier_edge << ",\n"
            << "    \"time_traversal_1\": " << gpu_stats.d_time_to_traversal << ",\n"
            << "    \"time_traversal_2\": " << gpu_stats.d_time_to_traversal_search_frontier_edge << ",\n"
            << "    \"time_overwrite_seed\": " << gpu_stats.d_time_to_overwrite_seed << ",\n"
            << "    \"time_label_scan_edges\": " << gpu_stats.d_time_to_label_scan_edges << ",\n"
            << "    \"time_label_compaction_edges\": " << gpu_stats.d_time_to_label_compaction_edges << ",\n"
            << "    \"time_back_to_host\": " << gpu_stats.d_time_to_back_to_host << ",\n"
            << "    \"time_total_algorithm\": " << gpu_stats.d_time_to_generate_polygonal_mesh << ",\n"
            << "    \"time_total_with_copies\": " << gpu_stats.total_with_copies << ",\n"
            << "    \"memory_total_gpu\": " << gpu_stats.memory_total_gpu << "\n"
            << "  }\n"
            << "}\n";
        out.close();
        std::cout << "[Report] Saved JSON benchmark results to: " << filepath << std::endl;
    }
};
