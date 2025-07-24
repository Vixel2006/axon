#include "tensor.h"
#include "engine/ops/impl/add.h"
#include "helpers.h"

Tensor add_cpu(const Tensor &a, const Tensor &b) {
    if (a.shape().size() != b.shape().size()) { throw std::runtime_error("the ndim of first tensor is not the same for the second one"); }
    
    for (size_t i = 0; i < a.shape().size(); ++i) {
        if (a.shape()[i] != b.shape()[i]) { throw std::runtime_error("the tensor shapes are mismatched."); }
    }
    
    float* a_data = static_cast<float*>(a.data_ptr().get());
    float* b_data = static_cast<float*>(b.data_ptr().get());
    float* c_data = new float[a.numel()];

    #pragma omp simd
    for (int i = 0; i < a.numel(); ++i) {
        c_data[i] = a_data[i] + b_data[i];
    }

    std::vector<__int64_t> c_shape = a.shape();
    std::vector<__int64_t> c_strides = compute_strides_(c_shape);
    bool c_requries_grad = a.requires_grad() || b.requires_grad();
    std::shared_ptr<void> data(c_data, [](void* ptr) {
        delete[] static_cast<float*>(ptr);
    });

    return Tensor(c_shape, c_strides, a.dtype(), a.device(), data, 0, c_requries_grad, nullptr);
}