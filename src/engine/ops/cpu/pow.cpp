#include "tensor.h"
#include "engine/ops.h"
#include "autograd/ops.h"
#include "helpers.h"
#include <cmath>
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

Tensor CpuOps::pow(const Tensor &base, float exponent) {
    const size_t num_elements = base.numel();

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for pow output tensor.");
    }

    const float* a_data = static_cast<const float*>(base.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = powf(a_data[i], exponent);
    }

    bool c_requires_grad = base.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(base.shape(), base.strides(), base.dtype(), base.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({base}, CpuAutograd::pow);
    }

    return t;
}

Tensor CpuOps::pow(const Tensor &base, const Tensor &exponent) {
    if (base.shape() != exponent.shape()) {
        throw std::runtime_error("Tensor shapes must be identical for element-wise pow.");
    }
    const size_t num_elements = base.numel();

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for pow output tensor.");
    }

    const float* a_data = static_cast<const float*>(base.data_ptr().get());
    const float* b_data = static_cast<const float*>(exponent.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = powf(a_data[i], b_data[i]);
    }

    bool c_requires_grad = base.requires_grad() || exponent.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t =  Tensor(base.shape(), base.strides(), base.dtype(), base.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
    
    if (c_requires_grad) {
      t.set_ctx({base, exponent}, CpuAutograd::pow);
    }

    return t;
}

