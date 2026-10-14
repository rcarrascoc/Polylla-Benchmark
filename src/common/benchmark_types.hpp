#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cstdint>

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
                // Handle both with and without d_ prefix
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

    double mesh_gen_time_ms = 0.0;
    double off_write_time_ms = 0.0;

    bool has_seq = false;
    PolyllaStats seq_stats;
    PolyllaStats cpu_stats;
    PolyllaStats gpu_stats;

    double speedup_gpu_vs_seq = 0.0;         // Seq time / GPU time
    double speedup_omp_vs_seq = 0.0;         // Seq time / OMP time
    double speedup_gpu_vs_omp = 0.0;         // OMP time / GPU time
    double speedup_gpu_with_copies_vs_seq = 0.0; // Seq time / (GPU time + copies)

    void compute_comparison() {
        if (has_seq && seq_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            if (gpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_gpu_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (cpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
                speedup_omp_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / cpu_stats.d_time_to_generate_polygonal_mesh;
            }
            if (gpu_stats.total_with_copies > 0.0) {
                speedup_gpu_with_copies_vs_seq = seq_stats.d_time_to_generate_polygonal_mesh / gpu_stats.total_with_copies;
            }
        }
        if (gpu_stats.d_time_to_generate_polygonal_mesh > 0.0 && cpu_stats.d_time_to_generate_polygonal_mesh > 0.0) {
            speedup_gpu_vs_omp = cpu_stats.d_time_to_generate_polygonal_mesh / gpu_stats.d_time_to_generate_polygonal_mesh;
        }
    }

    void print_table(std::ostream& os = std::cout) const {
        os << "\n"
           << "===================================================================================================================\n"
           << "                                 POLYLLA BENCHMARK COMPARISON REPORT                                               \n"
           << "                 (Sequential Baseline vs OpenMP Multi-Core CPU vs CUDA GPU)                                       \n"
           << "===================================================================================================================\n";
        os << "  Mesh Type:       " << mesh_type << "\n"
           << "  Requested Size:  " << requested_size << "\n"
           << "  Random Seed:     " << seed << "\n"
           << "  CPU Threads:     " << cpu_threads << "\n"
           << "  Mesh Gen Time:   " << std::fixed << std::setprecision(3) << mesh_gen_time_ms << " ms\n"
           << "-------------------------------------------------------------------------------------------------------------------\n"
           << "  Metric / Element             SEQ (Baseline)       CPU (MPolylla)       GPU (GPolylla)       Validation           \n"
           << "-------------------------------------------------------------------------------------------------------------------\n";

        auto print_elem = [&](const char* name, long long seq, long long cpu, long long gpu) {
            std::string status = (cpu == gpu) ? "MATCH (CPU=GPU)" : "DIFF (CPU!=GPU)";
            if (has_seq && seq != cpu) status += "*";
            os << "  " << std::left << std::setw(25) << name
               << "  " << std::right << std::setw(17) << (has_seq ? std::to_string(seq) : "-")
               << "  " << std::right << std::setw(19) << cpu
               << "  " << std::right << std::setw(19) << gpu
               << "  " << std::right << std::setw(23) << status
               << "\n";
        };

        print_elem("Vertices (n_vertices)", seq_stats.n_vertices, cpu_stats.n_vertices, gpu_stats.n_vertices);
        print_elem("Faces (n_faces)", seq_stats.n_faces, cpu_stats.n_faces, gpu_stats.n_faces);
        print_elem("Half-Edges (n_half_edges)", seq_stats.n_half_edges, cpu_stats.n_half_edges, gpu_stats.n_half_edges);
        print_elem("Polygons (n_polygons)", seq_stats.n_polygons, cpu_stats.n_polygons, gpu_stats.n_polygons);

        os << "-------------------------------------------------------------------------------------------------------------------\n"
           << "  Phase Timers (ms)            SEQ (Baseline)       CPU (MPolylla)       GPU (GPolylla)      Speedup GPU/Seq       \n"
           << "-------------------------------------------------------------------------------------------------------------------\n";

        auto print_timer = [&](const char* name, double seq_t, double cpu_t, double gpu_t) {
            double sp_gpu_seq = (has_seq && gpu_t > 0.0) ? (seq_t / gpu_t) : 0.0;
            os << "  " << std::left << std::setw(25) << name
               << "  " << std::right << std::setw(17) << std::fixed << std::setprecision(4) << (has_seq ? seq_t : 0.0)
               << "  " << std::right << std::setw(19) << std::fixed << std::setprecision(4) << cpu_t
               << "  " << std::right << std::setw(19) << std::fixed << std::setprecision(4) << gpu_t;
            if (sp_gpu_seq > 0.0) {
                os << "  " << std::right << std::setw(20) << std::fixed << std::setprecision(2) << sp_gpu_seq << "x\n";
            } else {
                os << "  " << std::right << std::setw(21) << "-\n";
            }
        };

        print_timer("1. Triangulation Struct", seq_stats.time_triangulation_generation, cpu_stats.time_triangulation_generation, gpu_stats.time_triangulation_generation);
        print_timer("2. Label Max Edges", seq_stats.d_time_to_label_max_edges, cpu_stats.d_time_to_label_max_edges, gpu_stats.d_time_to_label_max_edges);
        print_timer("3. Label Frontier Edges", seq_stats.d_time_to_label_frontier_edges, cpu_stats.d_time_to_label_frontier_edges, gpu_stats.d_time_to_label_frontier_edges);
        print_timer("4. Label Seed Edges", seq_stats.d_time_to_label_seed_edges, cpu_stats.d_time_to_label_seed_edges, gpu_stats.d_time_to_label_seed_edges);
        print_timer("5. Repair Extra Frontier", seq_stats.d_time_to_repair, cpu_stats.d_time_to_label_extra_frontier_edge, gpu_stats.d_time_to_label_extra_frontier_edge);
        print_timer("6. Traversal Phase 1", seq_stats.d_time_to_traversal, cpu_stats.d_time_to_traversal, gpu_stats.d_time_to_traversal);
        print_timer("7. Traversal Phase 2 (Search)", 0.0, cpu_stats.d_time_to_traversal_search_frontier_edge, gpu_stats.d_time_to_traversal_search_frontier_edge);
        print_timer("8. Repair Overwrite Seed", 0.0, cpu_stats.d_time_to_overwrite_seed, gpu_stats.d_time_to_overwrite_seed);
        print_timer("9. Label Seed Scan", 0.0, cpu_stats.d_time_to_label_scan_edges, gpu_stats.d_time_to_label_scan_edges);
        print_timer("10. Seed Compaction", 0.0, cpu_stats.d_time_to_label_compaction_edges, gpu_stats.d_time_to_label_compaction_edges);

        os << "-------------------------------------------------------------------------------------------------------------------\n";
        print_timer("TOTAL ALGORITHM TIME", seq_stats.d_time_to_generate_polygonal_mesh, cpu_stats.d_time_to_generate_polygonal_mesh, gpu_stats.d_time_to_generate_polygonal_mesh);
        os << "-------------------------------------------------------------------------------------------------------------------\n";
        os << "  GPU Transfer Host->Device:      " << std::right << std::setw(17) << std::fixed << std::setprecision(4) << gpu_stats.d_time_copy_to_device << " ms\n";
        os << "  GPU Transfer Device->Host:      " << std::right << std::setw(17) << std::fixed << std::setprecision(4) << gpu_stats.d_time_to_back_to_host << " ms\n";
        os << "  GPU TOTAL (Alg + Transfers):    " << std::right << std::setw(17) << std::fixed << std::setprecision(4) << gpu_stats.total_with_copies << " ms\n";
        os << "-------------------------------------------------------------------------------------------------------------------\n";
        if (has_seq) {
            os << "  >>> SPEEDUP GPU vs Sequential (Alg only):        " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_seq << "x\n";
            os << "  >>> SPEEDUP GPU vs Sequential (With Transfers):  " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_with_copies_vs_seq << "x\n";
            os << "  >>> SPEEDUP OpenMP vs Sequential:                " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_omp_vs_seq << "x\n";
        }
        os << "  >>> SPEEDUP GPU vs OpenMP (Alg only):            " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << speedup_gpu_vs_omp << "x\n";
        os << "===================================================================================================================\n\n";
    }

    void save_json(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "Warning: Could not save JSON to " << filepath << std::endl;
            return;
        }
        out << "{\n"
            << "  \"mesh_type\": \"" << mesh_type << "\",\n"
            << "  \"requested_size\": " << requested_size << ",\n"
            << "  \"seed\": " << seed << ",\n"
            << "  \"cpu_threads\": " << cpu_threads << ",\n"
            << "  \"mesh_gen_time_ms\": " << mesh_gen_time_ms << ",\n"
            << "  \"has_sequential_baseline\": " << (has_seq ? "true" : "false") << ",\n"
            << "  \"speedup_gpu_vs_sequential\": " << speedup_gpu_vs_seq << ",\n"
            << "  \"speedup_gpu_with_copies_vs_sequential\": " << speedup_gpu_with_copies_vs_seq << ",\n"
            << "  \"speedup_omp_vs_sequential\": " << speedup_omp_vs_seq << ",\n"
            << "  \"speedup_gpu_vs_omp\": " << speedup_gpu_vs_omp << ",\n";

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

        out << "  \"cpu\": {\n"
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
            << "  },\n"
            << "  \"gpu\": {\n"
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
