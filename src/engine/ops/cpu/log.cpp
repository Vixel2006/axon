#include "tensor.h"
#include "engine/ops.h"
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

Tensor CpuOps::log(const Tensor &a) {
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
        throw std::runtime_error("Failed to allocate aligned memory for log output tensor.");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; ++i) {
        c_data[i] = logf(a_data[i]);
    }

    bool c_requires_grad = a.requires_grad();

    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
        t.set_ctx({a}, CpuAutograd::log);
    }

    return t;
}


