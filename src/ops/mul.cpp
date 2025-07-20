#include "tensor.h"
#include "ops/mul.h"
#include "helpers.h"

Tensor mul_cpu(const Tensor &a, float b) {
    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* c_data = new float[a.numel()];

    #pragma omp simd
    for (int i = 0; i < a.numel(); ++i) {
        c_data[i] = a_data[i] * b;
    }

    std::vector<__int64_t> c_shape = a.shape();
    std::vector<__int64_t> c_strides = compute_strides_(c_shape);
    bool c_requries_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data, [](void* ptr) {
        delete[] static_cast<float*>(ptr);
    });

    return Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requries_grad, nullptr);
}
