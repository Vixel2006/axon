cmake_minimum_required(VERSION 3.18)

project(nawah LANGUAGES CXX CUDA)

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

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

pybind11_add_module(neura bindings/bindings.cpp ${SOURCES})

target_include_directories(neura PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(neura PRIVATE
    pybind11::module
    CUDA::toolkit
)