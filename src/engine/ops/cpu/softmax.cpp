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
    if (a.shape().size() != 2) {
        throw std::runtime_error("CpuOps::softmax currently only supports 2D tensors.");
    }
    
    const size_t num_elements = a.numel();
    const int rows = a.shape()[0];
    const int cols = a.shape()[1];

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

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float* row_start = a_data + i * cols;
        float* out_row_start = c_data + i * cols;

        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < cols; ++j) {
            if (row_start[j] > max_val) {
                max_val = row_start[j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float val = expf(row_start[j] - max_val);
            out_row_start[j] = val;
            sum_exp += val;
        }

        for (int j = 0; j < cols; ++j) {
            out_row_start[j] /= sum_exp;
        }
    }

    bool c_requires_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);


    return t;
}
