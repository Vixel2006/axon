#pragma once

#include "allocator.h"
#include <stdexcept>
#include <cuda_runtime.h>

struct CUDAAllocator: public Allocator {
    void* allocate(size_t bytes) override {
        void *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);

        if (err != cudaSuccess) throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (ptr) cudaFree(ptr);
    }
};