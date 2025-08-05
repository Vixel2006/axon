#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <cmath>
#include <omp.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <limits>

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

Tensor CpuOps::softmax(const Tensor &a) {
    const size_t num_elements = a.numel();

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for softmax output tensor.");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    float max_val = -std::numeric_limits<float>::infinity();
    #pragma omp parallel for reduction(max:max_val)
    for (int64_t i = 0; i < num_elements; ++i) {
        if (a_data[i] > max_val) {
            max_val = a_data[i];
        }
    }

    float sum_exp = 0.0f;
    #pragma omp parallel for reduction(+:sum_exp)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = expf(a_data[i] - max_val);
        sum_exp += c_data[i];
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] /= sum_exp;
    }

    bool c_requires_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
