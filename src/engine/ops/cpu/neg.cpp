#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

Tensor CpuOps::neg(const Tensor &a) {
    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return Tensor({}, {}, a.dtype(), a.device(), nullptr, 0, false, nullptr, std::nullopt);
    }

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32); // 32-byte alignment for AVX
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        throw std::runtime_error("Failed to allocate aligned memory for the output tensor.");
    }

    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < num_elements; ++i) {
        c_data[i] = -1 * a_data[i];
    }

    std::vector<int64_t> c_shape = a.shape();
    std::vector<int64_t> c_strides = compute_strides_(c_shape);
    bool c_requires_grad = a.requires_grad();

    std::shared_ptr<void> data(c_data, AlignedDeleter{});

    return Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
