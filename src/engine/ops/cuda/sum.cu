#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

<<<<<<< Updated upstream
=======
__global__ void full_reduction_sum_kernel(
    const float* in_data,
    float* out_data,
    size_t num_elements
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = index; i < num_elements; i += stride) {
        local_sum += in_data[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_data, sdata[0]);
    }
}

__global__ void sum_reduction_kernel(
    const float* in_data,
    float* out_data,
    const int64_t* d_in_strides,
    const int64_t* d_out_strides,
    const int64_t* d_in_shape,
    int ndim,
    int reduction_dim,
    bool keepdim,
    size_t num_output_elements
) {
    extern __shared__ float sdata[];

    int64_t output_idx = blockIdx.x;
    if (output_idx >= num_output_elements) {
        return;
    }

    const int64_t reduction_size = d_in_shape[reduction_dim];
    const int64_t reduction_stride = d_in_strides[reduction_dim];

    int64_t start_in_offset = 0;
    int64_t temp_i = output_idx;
    int out_dim_idx = 0;
    for (int in_dim_idx = 0; in_dim_idx < ndim; ++in_dim_idx) {
        if (in_dim_idx == reduction_dim) {
            if (keepdim) {
                out_dim_idx++;
            }
            continue;
        }

        const int64_t coord = temp_i / d_out_strides[out_dim_idx];
        start_in_offset += coord * d_in_strides[in_dim_idx];
        temp_i %= d_out_strides[out_dim_idx];
        out_dim_idx++;
    }

    float sum = 0.0f;
    for (int64_t j = threadIdx.x; j < reduction_size; j += blockDim.x) {
        sum += in_data[start_in_offset + j * reduction_stride];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_data[output_idx] = sdata[0];
    }
}
>>>>>>> Stashed changes

Tensor CudaOps::sum(const Tensor &a) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::sum must be on the CUDA device.");
    }

    // The result of a full reduction is a scalar tensor.
    std::vector<int64_t> new_shape = {1};
    bool result_requires_grad = a.requires_grad();
    Tensor result(new_shape, a.dtype(), deviceToString(a.device()), result_requires_grad);

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        float* d_result = static_cast<float*>(result.raw_ptr());
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        return result;
    }

    const float* d_a = static_cast<const float*>(a.raw_ptr());
    float* d_result = static_cast<float*>(result.raw_ptr());

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = std::min(
        (int)((num_elements + threadsPerBlock - 1) / threadsPerBlock),
        4096
    );
    const size_t shmem_size = threadsPerBlock * sizeof(float);

    full_reduction_sum_kernel<<<blocksPerGrid, threadsPerBlock, shmem_size>>>(
        d_a, d_result, num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    if (result_requires_grad) {
      result.set_ctx({a}, CudaAutograd::sum);
    }

    return result;
}

Tensor CudaOps::sum(const Tensor &a, int dim, bool keepdim) {
}
