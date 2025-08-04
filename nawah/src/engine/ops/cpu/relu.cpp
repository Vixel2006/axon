#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"

Tensor CpuOps::relu(const Tensor& t, float leakage) {
    float* t_data = static_cast<float*>(t.data_ptr().get());
    float* c_data = new float[t.numel()];

    //#pragma omp simd
    for (int i = 0; i < t.numel(); ++i) {
        if (t_data[i] < 0) {
          c_data[i] = leakage;
        } else {
          c_data[i] = t_data[i];
        }
    }

    std::vector<__int64_t> c_shape = t.shape();
    std::vector<__int64_t> c_strides = compute_strides_(c_shape);
    bool c_requries_grad = t.requires_grad();
    std::shared_ptr<void> data(c_data, [](void* ptr) {
        delete[] static_cast<float*>(ptr);
    });
    return Tensor(c_shape, c_strides, t.dtype(), t.device(), data, 0, c_requries_grad, nullptr, std::nullopt);
}

