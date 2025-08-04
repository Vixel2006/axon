#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <immintrin.h> // For AVX intrinsics and aligned memory allocation
#include <omp.h>       // For OpenMP
#include <stdexcept>
#include <vector>
#include <memory>

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

Tensor CpuOps::sub(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes are mismatched for subtraction.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return Tensor({}, {}, a.dtype(), a.device(), nullptr, 0, false, nullptr, std::nullopt);
    }

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for the output tensor.");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    const float* b_data = static_cast<const float*>(b.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < num_elements; ++i) {
        c_data[i] = a_data[i] - b_data[i];
    }

    std::vector<int64_t> c_shape = a.shape();
    std::vector<int64_t> c_strides = compute_strides_(c_shape);
    bool c_requires_grad = a.requires_grad() || b.requires_grad();

    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});

    return Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
