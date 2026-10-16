#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>
#include <cstdio>
#include <unistd.h>
#include <sstream>
#include "delaunator.hpp"

struct MemoryMesh {
    std::string name;
    std::string type; // "grid" or "uniform" or "off_file"
    int n_vertices = 0;
    int n_faces = 0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<int> faces; // Flattened 3*n_faces indices
    double gen_time_ms = 0.0;

    bool write_off_file(const std::string& filepath) const {
        FILE* fp = fopen(filepath.c_str(), "w");
        if (!fp) {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return false;
        }

        // Use a large write buffer for fast in-memory writing (64KB)
        char buffer[64 * 1024];
        setvbuf(fp, buffer, _IOFBF, sizeof(buffer));

        fprintf(fp, "OFF\n");
        fprintf(fp, "%d %d 0\n", n_vertices, n_faces);

        // Write points
        for (int i = 0; i < n_vertices; i++) {
            fprintf(fp, "%.10f %.10f 0.0\n", x[i], y[i]);
        }

        // Write faces
        for (int i = 0; i < n_faces; i++) {
            fprintf(fp, "3 %d %d %d\n", faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
        }

        fclose(fp);
        return true;
    }

    bool load_off_file(const std::string& filepath) {
        auto t_start = std::chrono::high_resolution_clock::now();
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open OFF file for reading: " << filepath << std::endl;
            return false;
        }

        std::string line;
        std::string header;
        bool found_header = false;
        int nv = 0, nf = 0, ne = 0;

        while (std::getline(file, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream iss(line.substr(first));
            if (!found_header) {
                iss >> header;
                if (header.find("OFF") == 0) {
                    found_header = true;
                    std::string rest = header.substr(3);
                    if (!rest.empty()) {
                        std::istringstream r_iss(rest);
                        if (r_iss >> nv) {
                            iss >> nf >> ne;
                            break;
                        }
                    }
                    if (iss >> nv >> nf >> ne) {
                        break;
                    }
                } else {
                    std::cerr << "Error: File is not a valid OFF file: " << filepath << std::endl;
                    return false;
                }
            } else {
                if (iss >> nv >> nf >> ne) {
                    break;
                }
            }
        }

        if (nv <= 0 || nf <= 0) {
            std::cerr << "Error: Invalid vertex or face count in OFF file: " << filepath << std::endl;
            return false;
        }

        n_vertices = nv;
        n_faces = nf;
        x.clear();
        y.clear();
        faces.clear();
        x.reserve(n_vertices);
        y.reserve(n_vertices);
        faces.reserve(3 * n_faces);

        int v_read = 0;
        while (v_read < n_vertices && std::getline(file, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream iss(line.substr(first));
            double px, py, pz;
            if (iss >> px >> py >> pz) {
                x.push_back(px);
                y.push_back(py);
                v_read++;
            }
        }

        int f_read = 0;
        while (f_read < n_faces && std::getline(file, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;
            std::istringstream iss(line.substr(first));
            int k, v0, v1, v2;
            if (iss >> k >> v0 >> v1 >> v2) {
                if (k != 3) {
                    std::cerr << "Warning: Face with " << k << " vertices in OFF file." << std::endl;
                }
                faces.push_back(v0);
                faces.push_back(v1);
                faces.push_back(v2);
                f_read++;
            }
        }

        file.close();
        auto t_end = std::chrono::high_resolution_clock::now();
        gen_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        name = filepath;
        type = "off_file";
        return (v_read == n_vertices && f_read == n_faces);
    }
};

class MeshGenerator {
public:
    static void move_point(double max_val, double& px, double& py, double tolerance, std::mt19937_64& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        if (r > 0.5) {
            if (px >= max_val * (1.0 - tolerance)) px = max_val;
            if (py >= max_val * (1.0 - tolerance)) py = max_val;
            if (px <= max_val * tolerance) px = 0.0;
            if (py <= max_val * tolerance) py = 0.0;
        } else {
            if (px <= max_val * tolerance) px = 0.0;
            if (py <= max_val * tolerance) py = 0.0;
            if (px >= max_val * (1.0 - tolerance)) px = max_val;
            if (py >= max_val * (1.0 - tolerance)) py = max_val;
        }
    }

    static MemoryMesh generate_grid(int requested_size) {
        auto t_start = std::chrono::high_resolution_clock::now();

        MemoryMesh mesh;
        mesh.type = "grid";

        int sqrt_n = static_cast<int>(std::sqrt(requested_size));
        if (sqrt_n < 2) sqrt_n = 2;
        int n = sqrt_n * sqrt_n;

        mesh.n_vertices = n;
        mesh.x.reserve(n);
        mesh.y.reserve(n);

        // Grid coordinates matching GPolylla Triangulation(size)
        for (int i = 0; i < sqrt_n; i++) {
            for (int j = 0; j < sqrt_n; j++) {
                mesh.x.push_back(static_cast<double>(i));
                mesh.y.push_back(static_cast<double>(j));
            }
        }

        // Triangles matching GPolylla:
        // Square cell (i, j) with vertices i*sqrt_n + j, etc.
        int num_triangles_est = 2 * (sqrt_n - 1) * (sqrt_n - 1);
        mesh.faces.reserve(3 * num_triangles_est);

        for (int i = 0; i < n - sqrt_n; i++) {
            if (i % sqrt_n != sqrt_n - 1) {
                // Triangle 1
                mesh.faces.push_back(i);
                mesh.faces.push_back(i + 1);
                mesh.faces.push_back(i + sqrt_n + 1);

                // Triangle 2
                mesh.faces.push_back(i);
                mesh.faces.push_back(i + sqrt_n + 1);
                mesh.faces.push_back(i + sqrt_n);
            }
        }

        mesh.n_faces = mesh.faces.size() / 3;

        auto t_end = std::chrono::high_resolution_clock::now();
        mesh.gen_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        mesh.name = "grid_" + std::to_string(n);

        return mesh;
    }

    static MemoryMesh generate_uniform(int requested_size, uint64_t seed) {
        auto t_start = std::chrono::high_resolution_clock::now();

        MemoryMesh mesh;
        mesh.type = "uniform";

        if (requested_size < 10) requested_size = 10;
        double tolerance = 1.0 / static_cast<double>(requested_size);

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int n_inner = requested_size - 2;
        std::vector<double> coords;
        coords.reserve(2 * (requested_size + 4));

        for (int i = 0; i < n_inner; i++) {
            double px = dist(rng);
            double py = dist(rng);
            move_point(1.0, px, py, tolerance, rng);
            coords.push_back(px);
            coords.push_back(py);
        }

        // Bounding box corners as in GPolylla experiments
        coords.push_back(0.0); coords.push_back(0.0);
        coords.push_back(1.0); coords.push_back(1.0);
        coords.push_back(0.0); coords.push_back(1.0);
        coords.push_back(1.0); coords.push_back(0.0);

        int total_points = coords.size() / 2;
        mesh.n_vertices = total_points;
        mesh.x.reserve(total_points);
        mesh.y.reserve(total_points);

        for (int i = 0; i < total_points; i++) {
            mesh.x.push_back(coords[2 * i]);
            mesh.y.push_back(coords[2 * i + 1]);
        }

        // Fast Delaunay triangulation in memory
        delaunator::Delaunator d(coords);

        int num_tri = d.triangles.size() / 3;
        mesh.n_faces = num_tri;
        mesh.faces.reserve(3 * num_tri);

        for (size_t i = 0; i < d.triangles.size(); i++) {
            mesh.faces.push_back(static_cast<int>(d.triangles[i]));
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        mesh.gen_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        mesh.name = "uniform_" + std::to_string(total_points) + "_s" + std::to_string(seed);

        return mesh;
    }
};
