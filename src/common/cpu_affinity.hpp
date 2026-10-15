#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <omp.h>
#include <dirent.h>

struct CpuTopologyInfo {
    bool is_hybrid = false;
    std::vector<int> p_cores;
    std::vector<int> e_cores;
    std::string p_cores_str = "";
    std::string e_cores_str = "";
    int total_logical_cores = 0;

    static std::string trim_str(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static std::string read_first_line(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return "";
        std::string line;
        std::getline(file, line);
        return trim_str(line);
    }

    static std::vector<int> parse_ranges(const std::string& spec) {
        std::vector<int> cores;
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim_str(token);
            if (token.empty()) continue;
            auto dash_pos = token.find('-');
            if (dash_pos != std::string::npos) {
                try {
                    int start = std::stoi(trim_str(token.substr(0, dash_pos)));
                    int end = std::stoi(trim_str(token.substr(dash_pos + 1)));
                    if (start <= end) {
                        for (int c = start; c <= end; c++) cores.push_back(c);
                    } else {
                        for (int c = start; c >= end; c--) cores.push_back(c);
                    }
                } catch (...) {}
            } else {
                try {
                    cores.push_back(std::stoi(token));
                } catch (...) {}
            }
        }
        std::sort(cores.begin(), cores.end());
        cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
        return cores;
    }

    void detect() {
        total_logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (total_logical_cores <= 0) total_logical_cores = 1;

        // 1. Check standard Linux hybrid sysfs PMU paths (/sys/devices/cpu_core & /sys/devices/cpu_atom)
        std::string p_sys = read_first_line("/sys/devices/cpu_core/cpus");
        std::string e_sys = read_first_line("/sys/devices/cpu_atom/cpus");

        if (!p_sys.empty() && !e_sys.empty()) {
            is_hybrid = true;
            p_cores_str = p_sys;
            e_cores_str = e_sys;
            p_cores = parse_ranges(p_sys);
            e_cores = parse_ranges(e_sys);
            return;
        }

        // 2. Fallback: check maximum CPU frequencies across all online cores
        std::vector<long long> max_freqs(total_logical_cores, 0);
        long long highest_freq = 0;
        long long lowest_freq = -1;
        bool has_freq_info = false;

        for (int i = 0; i < total_logical_cores; i++) {
            std::string freq_path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";
            std::string freq_str = read_first_line(freq_path);
            if (!freq_str.empty()) {
                try {
                    long long freq = std::stoll(freq_str);
                    max_freqs[i] = freq;
                    has_freq_info = true;
                    if (freq > highest_freq) highest_freq = freq;
                    if (lowest_freq < 0 || freq < lowest_freq) lowest_freq = freq;
                } catch (...) {}
            }
        }

        if (has_freq_info && highest_freq > 0 && lowest_freq > 0 && highest_freq != lowest_freq) {
            is_hybrid = true;
            for (int i = 0; i < total_logical_cores; i++) {
                if (max_freqs[i] == highest_freq) {
                    p_cores.push_back(i);
                } else {
                    e_cores.push_back(i);
                }
            }
            p_cores_str = format_core_list(p_cores);
            e_cores_str = format_core_list(e_cores);
            return;
        }

        // Homogeneous CPU (all cores identical)
        is_hybrid = false;
        for (int i = 0; i < total_logical_cores; i++) {
            p_cores.push_back(i);
        }
        p_cores_str = "0-" + std::to_string(total_logical_cores - 1);
        e_cores_str = "";
    }

    static std::string format_core_list(const std::vector<int>& cores) {
        if (cores.empty()) return "";
        std::ostringstream ss;
        size_t i = 0;
        while (i < cores.size()) {
            size_t j = i;
            while (j + 1 < cores.size() && cores[j + 1] == cores[j] + 1) {
                j++;
            }
            if (i > 0) ss << ",";
            if (j > i) {
                ss << cores[i] << "-" << cores[j];
            } else {
                ss << cores[i];
            }
            i = j + 1;
        }
        return ss.str();
    }
};

class CpuAffinityManager {
public:
    static CpuTopologyInfo get_topology() {
        CpuTopologyInfo topo;
        topo.detect();
        return topo;
    }

    static std::vector<int> parse_cores_spec(const std::string& input_spec,
                                             const CpuTopologyInfo& topo,
                                             std::string& out_description) {
        std::string spec = CpuTopologyInfo::trim_str(input_spec);
        std::string lower_spec = spec;
        std::transform(lower_spec.begin(), lower_spec.end(), lower_spec.begin(), ::tolower);

        std::vector<int> result;

        if (lower_spec.empty() || lower_spec == "all" || lower_spec == "any") {
            for (int i = 0; i < topo.total_logical_cores; i++) {
                result.push_back(i);
            }
            out_description = "all (" + std::to_string(result.size()) + " cores: 0-" + std::to_string(topo.total_logical_cores - 1) + ")";
            return result;
        }

        if (lower_spec == "p" || lower_spec == "p-core" || lower_spec == "p-cores" ||
            lower_spec == "perf" || lower_spec == "performance") {
            result = topo.p_cores;
            out_description = "P-cores (" + topo.p_cores_str + ", " + std::to_string(result.size()) + " cores)";
            return result;
        }

        if (lower_spec == "e" || lower_spec == "e-core" || lower_spec == "e-cores" ||
            lower_spec == "eff" || lower_spec == "efficient" || lower_spec == "efficiency") {
            if (!topo.is_hybrid || topo.e_cores.empty()) {
                std::cerr << "[Warning] System does not have distinct E-cores. Falling back to all cores." << std::endl;
                for (int i = 0; i < topo.total_logical_cores; i++) result.push_back(i);
                out_description = "all (fallback, no E-cores detected)";
                return result;
            }
            result = topo.e_cores;
            out_description = "E-cores (" + topo.e_cores_str + ", " + std::to_string(result.size()) + " cores)";
            return result;
        }

        // Custom range or list (e.g. "0-7", "8-19", "0,2,4,6")
        result = CpuTopologyInfo::parse_ranges(spec);
        // Filter valid online core indices
        std::vector<int> valid_cores;
        for (int c : result) {
            if (c >= 0 && c < topo.total_logical_cores) {
                valid_cores.push_back(c);
            } else {
                std::cerr << "[Warning] Core index " << c << " is out of range [0, " 
                          << topo.total_logical_cores - 1 << "]. Skipping." << std::endl;
            }
        }
        result = valid_cores;

        if (result.empty()) {
            std::cerr << "[Warning] No valid cores found in spec '" << input_spec 
                      << "'. Falling back to all cores." << std::endl;
            for (int i = 0; i < topo.total_logical_cores; i++) result.push_back(i);
            out_description = "all (fallback, invalid spec)";
            return result;
        }

        std::string formatted = CpuTopologyInfo::format_core_list(result);
        out_description = formatted + " (" + std::to_string(result.size()) + " core" + (result.size() > 1 ? "s" : "") + ")";
        return result;
    }

    static bool apply_process_affinity(const std::vector<int>& cores, cpu_set_t* out_prev_mask) {
        if (cores.empty()) return false;

        if (out_prev_mask) {
            if (sched_getaffinity(0, sizeof(cpu_set_t), out_prev_mask) != 0) {
                perror("[CPU Affinity] sched_getaffinity failed");
                return false;
            }
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int c : cores) {
            CPU_SET(c, &cpuset);
        }

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("[CPU Affinity] sched_setaffinity failed");
            return false;
        }

        return true;
    }

    static void restore_process_affinity(const cpu_set_t& prev_mask) {
        sched_setaffinity(0, sizeof(cpu_set_t), &prev_mask);
    }

    static void bind_openmp_threads(const std::vector<int>& cores, int num_threads) {
        if (cores.empty() || num_threads <= 0) return;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int target_core = cores[tid % cores.size()];
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(target_core, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
    }
};

class ScopedCpuAffinity {
private:
    bool active = false;
    cpu_set_t prev_mask;

public:
    ScopedCpuAffinity(const std::vector<int>& cores, int num_threads = 0) {
        if (!cores.empty()) {
            active = CpuAffinityManager::apply_process_affinity(cores, &prev_mask);
            if (active && num_threads > 0) {
                CpuAffinityManager::bind_openmp_threads(cores, num_threads);
            }
        }
    }

    ~ScopedCpuAffinity() {
        if (active) {
            CpuAffinityManager::restore_process_affinity(prev_mask);
        }
    }
};
