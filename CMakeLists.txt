cmake_minimum_required(VERSION 3.18)

project(PPolylla LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect or configure CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Common include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 1. Build malloccountfiles static library with position independent code
add_library(malloccountfiles STATIC
    submodules/GPolylla/external/malloc_count-0.7.1/malloc_count.c
    submodules/GPolylla/external/malloc_count-0.7.1/stack_count.c
)
set_target_properties(malloccountfiles PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 2. Build Sequential runner shared library (ssalinasfe/Polylla-Mesh-DCEL - Baseline)
add_library(seq_runner SHARED
    src/runners/seq_runner.cpp
)
target_include_directories(seq_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/Polylla-Mesh-DCEL/src
)
set_target_properties(seq_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# 3. Build GPU runner shared library (ssalinasfe/GPolylla - CUDA)
add_library(gpolylla_runner SHARED
    src/runners/gpu_runner.cu
)
target_include_directories(gpolylla_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/GPolylla/src
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/GPolylla/external
)
set_target_properties(gpolylla_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(gpolylla_runner PRIVATE
    malloccountfiles
    nvidia-ml
    pthread
)

# 4. Build CPU runner shared library (rcarrascoc/MPolylla - OpenMP Loops)
add_library(mpolylla_runner SHARED
    src/runners/cpu_runner.cpp
)
target_include_directories(mpolylla_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPolylla/src
)
set_target_properties(mpolylla_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(mpolylla_runner PRIVATE
    OpenMP::OpenMP_CXX
)

# 5. Build CPU Task runner shared library (rcarrascoc/MPolylla - OpenMP Tasks branch)
add_library(mpolylla_task_runner SHARED
    src/runners/cpu_task_runner.cpp
)
target_include_directories(mpolylla_task_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPolylla-Task/src
)
set_target_properties(mpolylla_task_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(mpolylla_task_runner PRIVATE
    OpenMP::OpenMP_CXX
)

# 6. Build CPU DCEL runner shared library (rcarrascoc/MPolylla-Mesh-DCEL)
add_library(mpolylla_dcel_runner SHARED
    src/runners/cpu_dcel_runner.cpp
)
target_include_directories(mpolylla_dcel_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPolylla-Mesh-DCEL/src
)
set_target_properties(mpolylla_dcel_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(mpolylla_dcel_runner PRIVATE
    OpenMP::OpenMP_CXX
)

# 7. Build main benchmark executable
add_executable(ppolylla_benchmark
    src/main.cpp
)
target_link_libraries(ppolylla_benchmark PRIVATE
    seq_runner
    mpolylla_dcel_runner
    mpolylla_runner
    mpolylla_task_runner
    gpolylla_runner
    OpenMP::OpenMP_CXX
    pthread
)

# Set runtime path so the executable finds its runner libraries
set_target_properties(ppolylla_benchmark PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
