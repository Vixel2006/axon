#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include "strided_indexer.h"
#include <immintrin.h>
#include <omp.h>
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

Tensor CpuOps::mul(const Tensor &a, float scalar) {
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
    float* c_data = static_cast<float*>(c_data_raw);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < num_elements; ++i) {
        c_data[i] = a_data[i] * scalar;
    }

    std::vector<int64_t> c_shape = a.shape();
    std::vector<int64_t> c_strides = compute_strides_(c_shape);
    bool c_requires_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({a}, CpuAutograd::mul);
    }

    return t;
}

Tensor CpuOps::mul(const Tensor &a, const Tensor &b) {
    std::vector<int64_t> c_shape = compute_broadcast_shape(a.shape(), b.shape());

    size_t num_elements = 1;
    for(int64_t dim : c_shape) {
        num_elements *= dim;
    }

    if (num_elements == 0) {
        return Tensor({}, {}, a.dtype(), a.device(), nullptr, 0, false, nullptr, std::nullopt);
    }

    Tensor a_broad = a.broadcast(c_shape);
    Tensor b_broad = b.broadcast(c_shape);

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

    const float* a_data = static_cast<const float*>(a_broad.data_ptr().get());
    const float* b_data = static_cast<const float*>(b_broad.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    StridedIndexer indexer_a(a_broad.shape(), a_broad.strides());
    StridedIndexer indexer_b(b_broad.shape(), b_broad.strides());

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_elements; ++i) {
        size_t offset_a = indexer_a.get_offset(i);
        size_t offset_b = indexer_b.get_offset(i);
        c_data[i] = a_data[offset_a] * b_data[offset_b];
    }

    std::vector<int64_t> c_strides = compute_strides_(c_shape);
    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    std::shared_ptr<void> data(c_data, AlignedDeleter{});
    Tensor t = Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({a, b}, CpuAutograd::mul);
    }

    return t;
}

