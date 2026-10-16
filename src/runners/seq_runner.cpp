#include "seq_runner.hpp"
#include "src/generators/mesh_generator.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <chrono>

#define private public
#define protected public
#include "polylla.hpp"
#undef private
#undef protected

static Triangulation* create_seq_triangulation(const MemoryMesh* mesh) {
    Triangulation* tri = new Triangulation();
    tri->n_vertices = mesh->n_vertices;
    tri->n_faces = mesh->n_faces;
    tri->Vertices.reserve(mesh->n_vertices);
    for (int i = 0; i < mesh->n_vertices; i++) {
        vertex v;
        v.x = mesh->x[i];
        v.y = mesh->y[i];
        tri->Vertices.push_back(v);
    }
    auto t_start = std::chrono::high_resolution_clock::now();
    tri->HalfEdges.reserve(3 * mesh->n_vertices);
    std::vector<int> faces_copy = mesh->faces;
    tri->construct_interior_halfEdges_from_faces(faces_copy);
    tri->construct_exterior_halfEdges();
    auto t_end = std::chrono::high_resolution_clock::now();
    tri->t_triangulation_generation = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return tri;
}

extern "C" __attribute__((visibility("default")))
bool run_seq_polylla_from_memory(const MemoryMesh* mesh, const char* out_json_path, PolyllaStats* out_stats) {
    if (!mesh || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_seq_polylla_from_memory";
        }
        return false;
    }

    try {
        Triangulation* tri = create_seq_triangulation(mesh);
        std::string json_str(out_json_path);
        Polylla poly_mesh;
        poly_mesh.mesh_input = tri;
        poly_mesh.mesh_output = new Triangulation(*tri);
        poly_mesh.construct_Polylla();
        poly_mesh.print_stats(json_str);

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
