#include "gpu_runner.hpp"
#include "src/generators/mesh_generator.hpp"
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <chrono>

#define private public
#define protected public
#include "polylla.cu"
#undef private
#undef protected

static Triangulation* create_gpu_triangulation(const MemoryMesh* mesh) {
    Triangulation* tri = new Triangulation();
    tri->n_vertices = mesh->n_vertices;
    tri->n_faces = mesh->n_faces;
    tri->v_x.reserve(mesh->n_vertices);
    tri->v_y.reserve(mesh->n_vertices);
    tri->v_is_border.reserve(mesh->n_vertices);
    tri->v_incident_halfedge.reserve(mesh->n_vertices);

    for (int i = 0; i < mesh->n_vertices; i++) {
        tri->v_x.push_back(mesh->x[i]);
        tri->v_y.push_back(mesh->y[i]);
        tri->v_is_border.push_back(false);
        tri->v_incident_halfedge.push_back(-1);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<int> faces_copy = mesh->faces;
    tri->construct_interior_halfEdges_from_faces(faces_copy);
    tri->construct_exterior_halfEdges();
    auto t_end = std::chrono::high_resolution_clock::now();
    tri->t_triangulation_generation = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return tri;
}

extern "C" __attribute__((visibility("default")))
bool run_gpolylla_from_memory(const MemoryMesh* mesh, const char* out_json_path, PolyllaStats* out_stats) {
    if (!mesh || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_gpolylla_from_memory";
        }
        return false;
    }

    try {
        Triangulation* tri = create_gpu_triangulation(mesh);
        std::string json_str(out_json_path);

        Polylla poly_mesh;
        poly_mesh.mesh_input = tri;
        poly_mesh.mesh_output = new Triangulation(*tri);
        poly_mesh.construct_Polylla();
        poly_mesh.print_stats(json_str);

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
