set(CUTLASS_PATH "/home/shiyi/cuda-gemm/third_party/cutlass/include")
set(CUTLASS_UTIL_PATH "/home/shiyi/cuda-gemm/third_party/cutlass/tools/util/include")
set(CUTLASS_NVCC_ARCHS 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(sgemm_cute sgemm_cute.cu)
target_link_libraries(sgemm_cute PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )

target_include_directories(sgemm_cute PRIVATE ${CUTLASS_PATH} ${CUTLASS_UTIL_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G>>)
else()
    target_compile_options(sgemm_cute PRIVATE -lineinfo)
endif()

add_executable(vectorAdd_cute vectorAdd_cute.cu)
target_link_libraries(vectorAdd_cute PRIVATE 
    CUDA::cudart 
    CUDA::cublas
    )

target_include_directories(vectorAdd_cute PRIVATE ${CUTLASS_PATH} ${CUTLASS_UTIL_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vectorAdd_cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G>>)
else()
    target_compile_options(vectorAdd_cute PRIVATE -lineinfo)
endif()
