#include "autograd/ops.h"
#include "tensor.h"
#include <stdexcept>
#include <vector>

#define PARALLEL_THRESHOLD 4096

void CpuAutograd::matmul(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    Tensor out_grad = out;

    const int M = a.shape()[0];
    const int K = a.shape()[1];
    const int N = b.shape()[1];

    const float* a_data_p = static_cast<const float*>(a.data_ptr().get());
    const float* b_data_p = static_cast<const float*>(b.data_ptr().get());
    const float* out_grad_p = static_cast<const float*>(out_grad.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    if (!a_req_grad && !b_req_grad) {
        return;
    }

    if (!out_grad_p || (a_req_grad && !a_grad_p) || (b_req_grad && !b_grad_p)) {
        throw std::runtime_error("A required pointer is null in 'matmul' backward pass (CPU).");
    }

    if (a_req_grad) {
        #pragma omp parallel for schedule(static) collapse(2) if(M * K > PARALLEL_THRESHOLD)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < N; ++l) {
                    sum += out_grad_p[i * N + l] * b_data_p[j * N + l];
                }
                a_grad_p[i * K + j] += sum;
            }
        }
    }

    if (b_req_grad) {
        #pragma omp parallel for schedule(static) if(K * N > PARALLEL_THRESHOLD)
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < M; ++l) {
                    sum += a_data_p[l * K + i] * out_grad_p[l * N + j];
                }
                b_grad_p[i * N + j] += sum;
            }
        }
    }
}
