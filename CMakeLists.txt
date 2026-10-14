cmake_minimum_required(VERSION 3.18)

project(PPolylla LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect or configure CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to modern architecture or native (RTX 4090 is sm_89, RTX 3090 is sm_86, A100 is sm_80)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Common include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/GPolylla/src
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/GPolylla/external
    ${CMAKE_CURRENT_SOURCE_DIR}/submodules/MPolylla/src
)

# 1. Build malloccountfiles static library with position independent code
add_library(malloccountfiles STATIC
    submodules/GPolylla/external/malloc_count-0.7.1/malloc_count.c
    submodules/GPolylla/external/malloc_count-0.7.1/stack_count.c
)
set_target_properties(malloccountfiles PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 2. Build GPU runner shared library (ssalinasfe/GPolylla)
add_library(gpolylla_runner SHARED
    src/runners/gpu_runner.cu
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

# 3. Build CPU runner shared library (rcarrascoc/MPolylla - OpenMP)
add_library(mpolylla_runner SHARED
    src/runners/cpu_runner.cpp
)
set_target_properties(mpolylla_runner PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(mpolylla_runner PRIVATE
    OpenMP::OpenMP_CXX
)

# 4. Build main benchmark executable
add_executable(ppolylla_benchmark
    src/main.cpp
)
target_link_libraries(ppolylla_benchmark PRIVATE
    gpolylla_runner
    mpolylla_runner
)

# Set runtime path so the executable finds its runner libraries
set_target_properties(ppolylla_benchmark PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
