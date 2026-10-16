#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sstream>

#include "src/common/benchmark_types.hpp"
#include "src/common/cpu_affinity.hpp"
#include "src/generators/mesh_generator.hpp"
#include "src/runners/seq_runner.hpp"
#include "src/runners/gpu_runner.hpp"
#include "src/runners/cpu_dcel_runner.hpp"
#include "src/runners/cpu_runner.hpp"
#include "src/runners/cpu_task_runner.hpp"

void print_help(const char* prog_name) {
    std::cout << "\n========================================================================================\n"
              << "       PPolylla: Parallel Polylla Benchmark (Sequential vs OpenMP vs CUDA GPU)          \n"
              << "========================================================================================\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  -t, --type <grid|uniform>    Mesh distribution type (default: uniform)\n"
              << "  -s, --size <N>               Target number of vertices (default: 100000)\n"
              << "  -r, --seed <SEED>            Random seed for single run (default: 42)\n"
              << "  --seeds <S1,S2,...>          Comma-separated list of random seeds (e.g. '42,43,44')\n"
              << "  --runs, --repeat <N>         Number of repetitions per experiment for average time (default: 1)\n"
              << "  -j, --threads <T>            OpenMP threads count (default: 0 = all cores or count of --cores)\n"
              << "  -c, --cores <SPEC>           CPU core affinity for OpenMP (e.g. '0-7', '8-19', 'p-cores', 'e-cores')\n"
              << "  -o, --output <FILE>          Path to save comparison JSON (default: results/...)\n"
              << "  -f, --off <FILE>             Run benchmark on an existing .off file\n"
              << "  --no-seq                     Skip sequential CPU baseline (useful for very large meshes)\n"
              << "  --save-mesh <FILE>           Save the generated mesh to a persistent .off file\n"
              << "  -h, --help                   Show this help message\n\n"
              << "Robust Benchmark Examples:\n"
              << "  # Run with 3 repetitions to get average times and standard deviations:\n"
              << "  " << prog_name << " --size 100000 --runs 3\n\n"
              << "  # Run across multiple seeds (e.g. 42, 43, 44) with 3 repetitions each:\n"
              << "  " << prog_name << " --size 100000 --seeds 42,43,44 --runs 3\n\n"
              << "Core Selection Examples:\n"
              << "  # Run on all 8 Performance cores (P-cores 0-7):\n"
              << "  " << prog_name << " --cores p-cores\n\n"
              << "  # Run on 4 threads pinned to cores 0-3:\n"
              << "  " << prog_name << " --cores 0-3 --threads 4\n"
              << "========================================================================================\n\n";
}

BenchmarkComparison run_benchmark_instance(
    const std::string& mesh_type,
    int requested_size,
    uint64_t seed,
    int num_runs,
    const std::string& input_off,
    const std::string& save_mesh_path,
    bool run_sequential,
    int cpu_threads,
    const std::string& affinity_desc,
    const std::vector<int>& affinity_cores,
    const SystemMetadata& meta)
{
    BenchmarkComparison comp;
    comp.mesh_type = mesh_type;
    comp.requested_size = requested_size;
    comp.seed = seed;
    comp.runs = num_runs;
    comp.cpu_threads = cpu_threads;
    comp.cpu_cores_affinity = affinity_desc;
    comp.cpu_cores_list = affinity_cores;
    comp.meta = meta;

    std::string test_off_file;
    bool is_temp_file = false;

    if (!input_off.empty()) {
        test_off_file = input_off;
        comp.mesh_type = "file (" + input_off + ")";
        std::cout << "[Benchmark] Using provided input mesh file: " << test_off_file << std::endl;
    } else {
        std::cout << "\n========================================================================\n"
                  << "  GENERATING INPUT DATASET IN MEMORY (" << mesh_type << ", size=" << requested_size << ", seed=" << seed << ")\n"
                  << "========================================================================\n";

        MemoryMesh mesh;
        if (mesh_type == "grid") {
            mesh = MeshGenerator::generate_grid(requested_size);
        } else if (mesh_type == "uniform") {
            mesh = MeshGenerator::generate_uniform(requested_size, seed);
        } else {
            std::cerr << "Error: Unknown mesh type '" << mesh_type << "'. Use 'grid' or 'uniform'." << std::endl;
            return comp;
        }

        comp.mesh_gen_time_ms = mesh.gen_time_ms;
        std::cout << "[MeshGen] Generated " << mesh.n_vertices << " vertices and " 
                  << mesh.n_faces << " triangles in " << mesh.gen_time_ms << " ms\n";

        std::string shm_dir = "/dev/shm";
        if (access(shm_dir.c_str(), W_OK) != 0) {
            shm_dir = "/tmp";
        }
        test_off_file = shm_dir + "/ppolylla_mesh_" + std::to_string(getpid()) + "_s" + std::to_string(seed) + ".off";
        is_temp_file = true;

        auto t_write_0 = std::chrono::high_resolution_clock::now();
        if (!mesh.write_off_file(test_off_file)) {
            std::cerr << "Error: Failed to write in-memory mesh to " << test_off_file << std::endl;
            return comp;
        }
        auto t_write_1 = std::chrono::high_resolution_clock::now();
        comp.off_write_time_ms = std::chrono::duration<double, std::milli>(t_write_1 - t_write_0).count();
        std::cout << "[MeshGen] In-memory mesh written to RAM buffer (" << test_off_file 
                  << ") in " << comp.off_write_time_ms << " ms\n";

        if (!save_mesh_path.empty()) {
            mesh.write_off_file(save_mesh_path);
            std::cout << "[MeshGen] Persistent mesh saved to: " << save_mesh_path << "\n";
        }
    }

    // 1. Sequential Baseline
    if (run_sequential) {
        std::cout << "\n------------------------------------------------------------------------\n"
                  << "  [1/5] RUNNING SEQUENTIAL BASELINE (ssalinasfe/Polylla-Mesh-DCEL"
                  << (num_runs > 1 ? (", " + std::to_string(num_runs) + " runs") : "") << ")\n"
                  << "------------------------------------------------------------------------\n";
        std::vector<PolyllaStats> seq_runs;
        for (int r = 0; r < num_runs; ++r) {
            std::string tmp_seq_json = "/dev/shm/seq_stats_" + std::to_string(getpid()) + "_" + std::to_string(r) + ".json";
            PolyllaStats s;
            bool ok = run_seq_polylla_from_off(test_off_file.c_str(), tmp_seq_json.c_str(), &s);
            std::remove(tmp_seq_json.c_str());
            if (ok) {
                if (num_runs > 1) {
                    std::cout << "    Run " << (r + 1) << "/" << num_runs << ": " 
                              << std::fixed << std::setprecision(3) << s.d_time_to_generate_polygonal_mesh << " ms\n";
                }
                seq_runs.push_back(s);
            } else {
                std::cerr << "Warning: Sequential benchmark run " << (r + 1) << " failed! Message: " << s.error_message << std::endl;
            }
        }
        comp.has_seq = !seq_runs.empty();
        if (comp.has_seq) {
            comp.seq_stats.aggregate_runs(seq_runs);
            if (num_runs > 1) {
                std::cout << "    --> Polylla-Mesh-DCEL Mean: " << std::fixed << std::setprecision(3) 
                          << comp.seq_stats.d_time_to_generate_polygonal_mesh << " ± " 
                          << comp.seq_stats.time_total_algorithm_std << " ms\n";
            }
        }
    } else {
        std::cout << "\n[Info] Skipping sequential baseline (--no-seq requested)\n";
        comp.has_seq = false;
    }

    // 2. OpenMP CPU (DCEL - rcarrascoc/MPolylla-Mesh-DCEL)
    std::cout << "\n------------------------------------------------------------------------\n"
              << "  [2/5] RUNNING CPU MULTI-CORE DCEL (rcarrascoc/MPolylla-Mesh-DCEL"
              << (num_runs > 1 ? (", " + std::to_string(num_runs) + " runs") : "") << ")\n"
              << "------------------------------------------------------------------------\n";
    std::vector<PolyllaStats> dcel_runs;
    {
        ScopedCpuAffinity affinity_guard(affinity_cores, comp.cpu_threads);
        for (int r = 0; r < num_runs; ++r) {
            std::string tmp_dcel_json = "/dev/shm/dcel_stats_" + std::to_string(getpid()) + "_" + std::to_string(r) + ".json";
            PolyllaStats s;
            bool ok = run_mpolylla_dcel_from_off(test_off_file.c_str(), tmp_dcel_json.c_str(), comp.cpu_threads, &s);
            std::remove(tmp_dcel_json.c_str());
            if (ok) {
                if (num_runs > 1) {
                    std::cout << "    Run " << (r + 1) << "/" << num_runs << ": " 
                              << std::fixed << std::setprecision(3) << s.d_time_to_generate_polygonal_mesh << " ms\n";
                }
                dcel_runs.push_back(s);
            } else {
                std::cerr << "Warning: CPU DCEL benchmark run " << (r + 1) << " failed! Message: " << s.error_message << std::endl;
            }
        }
    }
    comp.has_dcel = !dcel_runs.empty();
    if (comp.has_dcel) {
        comp.dcel_stats.aggregate_runs(dcel_runs);
        if (num_runs > 1) {
            std::cout << "    --> MPolylla DCEL Mean: " << std::fixed << std::setprecision(3) 
                      << comp.dcel_stats.d_time_to_generate_polygonal_mesh << " ± " 
                      << comp.dcel_stats.time_total_algorithm_std << " ms\n";
        }
    }

    // 3. OpenMP CPU (Data - main branch)
    std::cout << "\n------------------------------------------------------------------------\n"
              << "  [3/5] RUNNING CPU MULTI-CORE DATA (rcarrascoc/MPolylla - main"
              << (num_runs > 1 ? (", " + std::to_string(num_runs) + " runs") : "") << ")\n"
              << "------------------------------------------------------------------------\n";
    std::vector<PolyllaStats> cpu_runs;
    {
        ScopedCpuAffinity affinity_guard(affinity_cores, comp.cpu_threads);
        for (int r = 0; r < num_runs; ++r) {
            std::string tmp_cpu_json = "/dev/shm/cpu_stats_" + std::to_string(getpid()) + "_" + std::to_string(r) + ".json";
            PolyllaStats s;
            bool ok = run_mpolylla_from_off(test_off_file.c_str(), tmp_cpu_json.c_str(), comp.cpu_threads, &s);
            std::remove(tmp_cpu_json.c_str());
            if (ok) {
                if (num_runs > 1) {
                    std::cout << "    Run " << (r + 1) << "/" << num_runs << ": " 
                              << std::fixed << std::setprecision(3) << s.d_time_to_generate_polygonal_mesh << " ms\n";
                }
                cpu_runs.push_back(s);
            } else {
                std::cerr << "Warning: CPU Data benchmark run " << (r + 1) << " failed! Message: " << s.error_message << std::endl;
            }
        }
    }
    if (!cpu_runs.empty()) {
        comp.cpu_stats.aggregate_runs(cpu_runs);
        if (num_runs > 1) {
            std::cout << "    --> MPolylla Data Mean: " << std::fixed << std::setprecision(3) 
                      << comp.cpu_stats.d_time_to_generate_polygonal_mesh << " ± " 
                      << comp.cpu_stats.time_total_algorithm_std << " ms\n";
        }
    }

    // 4. OpenMP CPU (Tasks - Task branch)
    std::cout << "\n------------------------------------------------------------------------\n"
              << "  [4/5] RUNNING CPU MULTI-CORE TASKS (rcarrascoc/MPolylla - Task"
              << (num_runs > 1 ? (", " + std::to_string(num_runs) + " runs") : "") << ")\n"
              << "------------------------------------------------------------------------\n";
    std::vector<PolyllaStats> task_runs;
    {
        ScopedCpuAffinity affinity_guard(affinity_cores, comp.cpu_threads);
        for (int r = 0; r < num_runs; ++r) {
            std::string tmp_task_json = "/dev/shm/task_stats_" + std::to_string(getpid()) + "_" + std::to_string(r) + ".json";
            PolyllaStats s;
            bool ok = run_mpolylla_task_from_off(test_off_file.c_str(), tmp_task_json.c_str(), comp.cpu_threads, &s);
            std::remove(tmp_task_json.c_str());
            if (ok) {
                if (num_runs > 1) {
                    std::cout << "    Run " << (r + 1) << "/" << num_runs << ": " 
                              << std::fixed << std::setprecision(3) << s.d_time_to_generate_polygonal_mesh << " ms\n";
                }
                task_runs.push_back(s);
            } else {
                std::cerr << "Warning: CPU Tasks benchmark run " << (r + 1) << " failed! Message: " << s.error_message << std::endl;
            }
        }
    }
    comp.has_task = !task_runs.empty();
    if (comp.has_task) {
        comp.task_stats.aggregate_runs(task_runs);
        if (num_runs > 1) {
            std::cout << "    --> MPolylla Task Mean: " << std::fixed << std::setprecision(3) 
                      << comp.task_stats.d_time_to_generate_polygonal_mesh << " ± " 
                      << comp.task_stats.time_total_algorithm_std << " ms\n";
        }
    }

    // 5. GPU CUDA
    std::cout << "\n------------------------------------------------------------------------\n"
              << "  [5/5] RUNNING GPU BENCHMARK (ssalinasfe/GPolylla - CUDA"
              << (num_runs > 1 ? (", " + std::to_string(num_runs) + " runs") : "") << ")\n"
              << "------------------------------------------------------------------------\n";
    std::vector<PolyllaStats> gpu_runs;
    for (int r = 0; r < num_runs; ++r) {
        std::string tmp_gpu_json = "/dev/shm/gpu_stats_" + std::to_string(getpid()) + "_" + std::to_string(r) + ".json";
        PolyllaStats s;
        bool ok = run_gpolylla_from_off(test_off_file.c_str(), tmp_gpu_json.c_str(), &s);
        std::remove(tmp_gpu_json.c_str());
        if (ok) {
            if (num_runs > 1) {
                std::cout << "    Run " << (r + 1) << "/" << num_runs << ": " 
                          << std::fixed << std::setprecision(3) << s.d_time_to_generate_polygonal_mesh << " ms\n";
            }
            gpu_runs.push_back(s);
        } else {
            std::cerr << "Warning: GPU benchmark run " << (r + 1) << " failed! Message: " << s.error_message << std::endl;
        }
    }
    if (!gpu_runs.empty()) {
        comp.gpu_stats.aggregate_runs(gpu_runs);
        if (num_runs > 1) {
            std::cout << "    --> GPolylla Mean: " << std::fixed << std::setprecision(3) 
                      << comp.gpu_stats.d_time_to_generate_polygonal_mesh << " ± " 
                      << comp.gpu_stats.time_total_algorithm_std << " ms\n";
        }
    }

    // Clean up temporary mesh file
    if (is_temp_file) {
        std::remove(test_off_file.c_str());
    }

    comp.compute_comparison();
    return comp;
}

int main(int argc, char* argv[]) {
    std::string mesh_type = "uniform";
    int requested_size = 100000;
    uint64_t seed = 42;
    std::vector<uint64_t> seeds_list;
    int num_runs = 1;
    int cpu_threads = 0;
    std::string cores_spec = "";
    std::string output_json = "";
    std::string input_off = "";
    std::string save_mesh_path = "";
    bool run_sequential = true;

    static struct option long_options[] = {
        {"type",      required_argument, 0, 't'},
        {"size",      required_argument, 0, 's'},
        {"seed",      required_argument, 0, 'r'},
        {"seeds",     required_argument, 0, 1003},
        {"runs",      required_argument, 0, 1004},
        {"repeat",    required_argument, 0, 1004},
        {"threads",   required_argument, 0, 'j'},
        {"cores",     required_argument, 0, 'c'},
        {"output",    required_argument, 0, 'o'},
        {"off",       required_argument, 0, 'f'},
        {"no-seq",    no_argument,       0, 1002},
        {"save-mesh", required_argument, 0, 1001},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:s:r:j:c:o:f:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't': mesh_type = optarg; break;
            case 's': requested_size = std::atoi(optarg); break;
            case 'r': seed = std::strtoull(optarg, nullptr, 10); break;
            case 1003: {
                seeds_list.clear();
                std::stringstream ss(optarg);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    size_t first = item.find_first_not_of(" \t");
                    if (first != std::string::npos) {
                        size_t last = item.find_last_not_of(" \t");
                        seeds_list.push_back(std::strtoull(item.substr(first, last - first + 1).c_str(), nullptr, 10));
                    }
                }
                break;
            }
            case 1004: num_runs = std::max(1, std::atoi(optarg)); break;
            case 'j': cpu_threads = std::atoi(optarg); break;
            case 'c': cores_spec = optarg; break;
            case 'o': output_json = optarg; break;
            case 'f': input_off = optarg; break;
            case 1001: save_mesh_path = optarg; break;
            case 1002: run_sequential = false; break;
            case 'h':
            default:
                print_help(argv[0]);
                return 0;
        }
    }

    if (seeds_list.empty()) {
        seeds_list.push_back(seed);
    }

    // Ensure results directory exists
    struct stat st = {0};
    if (stat("results", &st) == -1) {
        mkdir("results", 0775);
    }

    // Detect CPU topology and configure affinity
    CpuTopologyInfo topo = CpuAffinityManager::get_topology();
    std::string affinity_desc = "all";
    std::vector<int> affinity_cores;
    if (!cores_spec.empty()) {
        affinity_cores = CpuAffinityManager::parse_cores_spec(cores_spec, topo, affinity_desc);
    } else {
        for (int i = 0; i < topo.total_logical_cores; i++) affinity_cores.push_back(i);
        affinity_desc = "all (" + std::to_string(topo.total_logical_cores) + " cores: 0-" + std::to_string(topo.total_logical_cores - 1) + ")";
    }

    if (cpu_threads <= 0) {
        cpu_threads = static_cast<int>(affinity_cores.size());
    }

    SystemMetadata meta;
    std::string full_cmd_line = "";
    for (int i = 0; i < argc; i++) {
        full_cmd_line += argv[i];
        if (i < argc - 1) full_cmd_line += " ";
    }
    meta.detect(full_cmd_line);

    std::cout << "[Affinity] Selected OpenMP cores: " << affinity_desc 
              << " | Active threads: " << cpu_threads 
              << " | Repetitions/algo: " << num_runs << std::endl;

    if (seeds_list.size() == 1) {
        uint64_t current_seed = seeds_list[0];
        BenchmarkComparison comp = run_benchmark_instance(
            mesh_type, requested_size, current_seed, num_runs, input_off, save_mesh_path,
            run_sequential, cpu_threads, affinity_desc, affinity_cores, meta);

        comp.print_table(std::cout);

        std::string base_path = output_json;
        if (base_path.empty()) {
            std::string safe_type = (mesh_type == "grid" || mesh_type == "uniform") ? mesh_type : "custom";
            base_path = "results/benchmark_" + safe_type + "_" + std::to_string(comp.requested_size) + 
                        "_seed" + std::to_string(comp.seed);
        } else {
            if (base_path.size() >= 5 && base_path.substr(base_path.size() - 5) == ".json") {
                base_path = base_path.substr(0, base_path.size() - 5);
            } else if (base_path.size() >= 4 && base_path.substr(base_path.size() - 4) == ".txt") {
                base_path = base_path.substr(0, base_path.size() - 4);
            }
        }

        std::string json_file = base_path + ".json";
        std::string txt_file = base_path + ".txt";

        comp.save_json(json_file);
        comp.save_txt(txt_file);

        return (comp.cpu_stats.success && comp.gpu_stats.success) ? 0 : 1;
    } else {
        // Multi-seed execution
        MultiSeedSummary summary;
        std::cout << "\n========================================================================\n"
                  << "  STARTING MULTI-SEED BENCHMARK (" << seeds_list.size() << " seeds, " << num_runs << " runs per algorithm)\n"
                  << "========================================================================\n";

        for (size_t i = 0; i < seeds_list.size(); ++i) {
            uint64_t current_seed = seeds_list[i];
            std::cout << "\n>>> [Seed " << (i + 1) << "/" << seeds_list.size() << ": " << current_seed << "] <<<\n";
            BenchmarkComparison comp = run_benchmark_instance(
                mesh_type, requested_size, current_seed, num_runs, input_off, save_mesh_path,
                run_sequential, cpu_threads, affinity_desc, affinity_cores, meta);

            std::string safe_type = (mesh_type == "grid" || mesh_type == "uniform") ? mesh_type : "custom";
            std::string seed_base = "results/benchmark_" + safe_type + "_" + std::to_string(requested_size) + "_seed" + std::to_string(current_seed);
            comp.save_json(seed_base + ".json");
            comp.save_txt(seed_base + ".txt");

            summary.comparisons.push_back(comp);
        }

        summary.compute();
        summary.print_table(std::cout);

        std::string base_path = output_json;
        if (base_path.empty()) {
            std::string safe_type = (mesh_type == "grid" || mesh_type == "uniform") ? mesh_type : "custom";
            base_path = "results/benchmark_" + safe_type + "_" + std::to_string(requested_size) + "_multiseed";
        } else {
            if (base_path.size() >= 5 && base_path.substr(base_path.size() - 5) == ".json") {
                base_path = base_path.substr(0, base_path.size() - 5);
            } else if (base_path.size() >= 4 && base_path.substr(base_path.size() - 4) == ".txt") {
                base_path = base_path.substr(0, base_path.size() - 4);
            }
        }

        summary.save_json(base_path + ".json");
        summary.save_txt(base_path + ".txt");

        return 0;
    }
}
