cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES 89)
project(cuda_notes VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_subdirectory(third_party)