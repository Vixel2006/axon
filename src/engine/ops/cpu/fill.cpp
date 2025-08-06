#include "engine/ops.h"
#include <random>
#include <cstring>
#include "tensor.h"
#include <omp.h>

void CpuOps::fill_zeros(Tensor& t) const {
    std::memset(t.raw_ptr(), 0, t.numel() * sizeof(float));
}

void CpuOps::fill_ones(Tensor& t) const {
    float* data = static_cast<float*>(t.raw_ptr());
    size_t numel = t.numel();
    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        data[i] = 1.0f;
    }
}

void CpuOps::fill_uniform(Tensor& t) const {
    float* data = static_cast<float*>(t.raw_ptr());
    size_t numel = t.numel();
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        #pragma omp for
        for (size_t i = 0; i < numel; ++i) {
            data[i] = dis(gen);
        }
    }
}

void CpuOps::fill_randn(Tensor& t) const {
    float* data = static_cast<float*>(t.raw_ptr());
    size_t numel = t.numel();
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 1.0f);
        #pragma omp for
        for (size_t i = 0; i < numel; ++i) {
            data[i] = dis(gen);
        }
    }
}

