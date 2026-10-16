#include "cpu_task_runner.hpp"
#include "src/generators/mesh_generator.hpp"
#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>
#include <omp.h>

#define private public
#define protected public
#include "mpolylla_omp.h"
#undef private
#undef protected

static Triangulation* create_mpolylla_triangulation(const MemoryMesh* mesh) {
    Triangulation* tri = new Triangulation();
    tri->n_vertices = mesh->n_vertices;
    tri->n_faces = mesh->n_faces;
    tri->Vertices.reserve(mesh->n_vertices);
    for (int i = 0; i < mesh->n_vertices; i++) {
        vertex ve;
        ve.x = mesh->x[i];
        ve.y = mesh->y[i];
        tri->Vertices.push_back(ve);
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
bool run_mpolylla_task_from_memory(const MemoryMesh* mesh, const char* out_json_path, int num_threads, PolyllaStats* out_stats) {
    if (!mesh || !out_json_path || !out_stats) {
        if (out_stats) {
            out_stats->success = false;
            out_stats->error_message = "Null argument passed to run_mpolylla_task_from_memory";
        }
        return false;
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    try {
        Triangulation* tri = create_mpolylla_triangulation(mesh);
        std::string json_str(out_json_path);

        Polylla poly_mesh;
        poly_mesh.mesh_input = tri;
        poly_mesh.mesh_output = new Triangulation(*tri);
        poly_mesh.n_vertices = tri->n_vertices;
        poly_mesh.n_halfedges = tri->n_halfedges;
        poly_mesh.n_faces = tri->n_faces;
        poly_mesh.m_polygons = 0;

        poly_mesh.h_max_edges = new bit_vector_d[poly_mesh.n_halfedges];
        poly_mesh.h_frontier_edges = new bit_vector_d[poly_mesh.n_halfedges];
        poly_mesh.h_seed_edges = new bit_vector_d[poly_mesh.n_halfedges];
        poly_mesh.h_scan = new int[poly_mesh.n_halfedges];
        poly_mesh.h_seed_edges_comp = new int[poly_mesh.n_halfedges];

        memset(poly_mesh.h_max_edges, 0, sizeof(bit_vector_d) * poly_mesh.n_halfedges);
        memset(poly_mesh.h_frontier_edges, 0, sizeof(bit_vector_d) * poly_mesh.n_halfedges);
        memset(poly_mesh.h_seed_edges, 0, sizeof(bit_vector_d) * poly_mesh.n_halfedges);
        memset(poly_mesh.h_scan, 0, sizeof(int) * poly_mesh.n_halfedges);
        memset(poly_mesh.h_seed_edges_comp, 0, sizeof(int) * poly_mesh.n_halfedges);

        poly_mesh.construct_Polylla();
        poly_mesh.print_stats(json_str);

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
