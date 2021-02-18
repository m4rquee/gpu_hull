cmake_minimum_required(VERSION 3.19)
project(gpu_hull LANGUAGES CUDA CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

# GPU Hull library:
add_library(gpu_hull STATIC include/gpu_hull/gpu_vector.cuh)

set_property(TARGET gpu_hull PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu_hull PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_hull PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(gpu_hull m)

# Test:
add_executable(gpu_hull_test src/test.cu include/gpu_hull/gpu_vector.cuh)

set_property(TARGET gpu_hull_test PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(gpu_hull_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_hull_test PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(gpu_hull_test m)

add_test("Vector memset and memcpy"  gpu_hull_test)
