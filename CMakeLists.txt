cmake_minimum_required(VERSION 3.18)

project(nawah LANGUAGES CXX CUDA)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Using ccache to speed up compilation")
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
endif()

message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES "75;86")

add_subdirectory(vendor/pybind11)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")



pybind11_add_module(nawah bindings/bindings.cpp ${SOURCES})


target_include_directories(nawah PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(nawah PRIVATE
    pybind11::module
    CUDA::toolkit
    CUDA::cublas
    OpenMP::OpenMP_CXX
)

target_compile_options(nawah PRIVATE
  -mavx2
  -mfma
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

