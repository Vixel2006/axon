#include "tensor.h"
#include "engine/ops.h"
#include "autograd/ops.h"
#include "helpers.h"
#include <immintrin.h>
#include <omp.h>
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

Tensor CpuOps::relu(const Tensor &a) {
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
        throw std::runtime_error("Failed to allocate aligned memory for ReLU output tensor.");
    }

    const float* a_data = static_cast<const float*>(a.data_ptr().get());
    float* c_data = static_cast<float*>(c_data_raw);

    const __m256 zeros = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_elements; i += 8) {
        if (i + 8 <= num_elements) {
            __m256 a_vec = _mm256_loadu_ps(a_data + i);
            __m256 result_vec = _mm256_max_ps(zeros, a_vec);
            _mm256_store_ps(c_data + i, result_vec);
        } else {
            for (int64_t j = i; j < num_elements; ++j) {
                c_data[j] = a_data[j] > 0.0f ? a_data[j] : 0.0f;
            }
        }
    }

    bool c_requires_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});

    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    /*
    if (c_requries_grad) {
        t.set_ctx({a}, CpuAutograd::relu);
    }
    */

    return t;
}

