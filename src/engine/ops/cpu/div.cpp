#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <omp.h>
#include <vector>
#include <memory>
#include <stdexcept>

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

Tensor CpuOps::div(const Tensor &numerator, float denominator) {
    const size_t num_elements = numerator.numel();

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for div output tensor.");
    }

    const float* a_data = static_cast<const float*>(numerator.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    const float inv_denominator = 1.0f / denominator;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = a_data[i] * inv_denominator;
    }

    bool c_requires_grad = numerator.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(numerator.shape(), numerator.strides(), numerator.dtype(), numerator.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
    
    if (c_requires_grad) {
      t.set_ctx({numerator}, CpuAutograd::div);
    }

    return t;
}


Tensor CpuOps::div(const Tensor &numerator, const Tensor &denominator) {
    if (numerator.shape() != denominator.shape()) {
        throw std::runtime_error("Tensor shapes must be identical for element-wise division.");
    }
    const size_t num_elements = numerator.numel();

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for div output tensor.");
    }

    const float* a_data = static_cast<const float*>(numerator.data_ptr().get());
    const float* b_data = static_cast<const float*>(denominator.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = a_data[i] / b_data[i];
    }

    bool c_requires_grad = numerator.requires_grad() || denominator.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(numerator.shape(), numerator.strides(), numerator.dtype(), numerator.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({numerator, denominator}, CpuAutograd::div);
    }

    return t;
}

