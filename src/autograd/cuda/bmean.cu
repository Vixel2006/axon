#include "autograd/ops.h"
#include "tensor.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

__global__ void mean_backward_kernel(
    const float* grad_out_data,
    float* grad_a_data,
    const int64_t* d_in_strides,
    const int64_t* d_out_strides,
    const int64_t* d_out_shape,
    int ndim,
    size_t num_elements,
    float scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        int64_t out_idx = 0;
        int64_t temp_i = i;

        for (int dim_idx = 0; dim_idx < ndim; ++dim_idx) {
            const int64_t coord = temp_i / d_in_strides[dim_idx];
            temp_i %= d_in_strides[dim_idx];

            if (dim_idx < 32 && d_out_shape[dim_idx] > 1) {
                out_idx += coord * d_out_strides[dim_idx];
            }
        }

        grad_a_data[i] += grad_out_data[out_idx] * scale;
    }
}


void CudaAutograd::mean(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];

    if (!a.requires_grad()) {
        return;
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return;
    }

    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());

    if (!out_grad_p || !a_grad_p) {
        throw std::runtime_error("A gradient pointer is null in 'mean' backward pass (CUDA).");
    }

    const int64_t reduction_size = a.numel() / out.numel();
    if (reduction_size == 0) {
        return;
    }
    const float scale = 1.0f / static_cast<float>(reduction_size);

    const auto& in_strides_vec = a.strides();
    const auto& out_shape_vec = out.shape();
    const auto& out_strides_vec = out.strides();
    const int ndim = a.ndim();

    int64_t* d_in_strides;
    int64_t* d_out_shape;
    int64_t* d_out_strides;

    CUDA_CHECK(cudaMalloc(&d_in_strides, in_strides_vec.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_shape, out_shape_vec.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_strides, out_strides_vec.size() * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_in_strides, in_strides_vec.data(), in_strides_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_shape, out_shape_vec.data(), out_shape_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides_vec.data(), out_strides_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    mean_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_grad_p,
        a_grad_p,
        d_in_strides,
        d_out_strides,
        d_out_shape,
        ndim,
        num_elements,
        scale
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_in_strides));
    CUDA_CHECK(cudaFree(d_out_shape));
    CUDA_CHECK(cudaFree(d_out_strides));
}
