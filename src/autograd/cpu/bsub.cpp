#include "tensor.h"
#include "autograd/ops.h"
#include <stdexcept>

#define PARALLEL_THRESHOLD 4096

void CpuAutograd::sub(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    const size_t num_elements = a.numel();

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A gradient pointer is null in 'sub' backward pass.");
    }

    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    if (a_req_grad && b_req_grad) {
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            const float grad_out = out_grad_p[i];
            a_grad_p[i] += grad_out;
            b_grad_p[i] -= grad_out;
        }
    } else if (a_req_grad) {
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            a_grad_p[i] += out_grad_p[i];
        }
    } else if (b_req_grad) {
        #pragma omp parallel for simd schedule(static) if(num_elements > PARALLEL_THRESHOLD)
        for (size_t i = 0; i < num_elements; ++i) {
            b_grad_p[i] -= out_grad_p[i];
        }
    }
}
