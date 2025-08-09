#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include <cuda_runtime.h>
#include <stdexcept>

<<<<<<< Updated upstream
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

__global__ void sum_reduce_kernel_for_mean(const float* input_data, float* output_data, size_t num_elements) {
=======
__global__ void full_reduction_mean_kernel(
    const float* in_data,
    float* out_data,
    unsigned int* d_finished_blocks,
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

        unsigned int finished_count = atomicInc(d_finished_blocks, gridDim.x);

        if (finished_count == gridDim.x - 1) {
            out_data[0] /= static_cast<float>(num_elements);
        }
    }
}


__global__ void mean_reduction_kernel(
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
>>>>>>> Stashed changes
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    float my_sum = 0.0f;
    for (size_t j = i; j < num_elements; j += stride) {
        my_sum += input_data[j];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_data[blockIdx.x] = sdata[0];
    }
}

<<<<<<< Updated upstream
Tensor CudaOps::mean(const Tensor &a, int axis, bool keep_dims) {
=======
Tensor CudaOps::mean(const Tensor &a) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device.");
    }
    std::vector<int64_t> new_shape = {1};
    Tensor result(new_shape, a.dtype(), deviceToString(a.device()), a.requires_grad());

    const size_t num_elements = a.numel();
    if (num_elements == 0) { return result; }

    const float* d_a = static_cast<const float*>(a.raw_ptr());
    float* d_result = static_cast<float*>(result.raw_ptr());

    unsigned int* d_finished_blocks;
    CUDA_CHECK(cudaMalloc(&d_finished_blocks, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_finished_blocks, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = std::min((int)((num_elements + threadsPerBlock - 1) / threadsPerBlock), 4096);
    const size_t shmem_size = threadsPerBlock * sizeof(float);

    full_reduction_mean_kernel<<<blocksPerGrid, threadsPerBlock, shmem_size>>>(
        d_a,
        d_result,
        d_finished_blocks,
        num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_finished_blocks));

    if (a.requires_grad()) {
        result.set_ctx({a}, CudaAutograd::mean);
    }

    return result;
}

Tensor CudaOps::mean(const Tensor &a, int dim, bool keepdim) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::mean must be on the CUDA device.");
    }
    int ndim = a.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Reduction dimension is out of bounds.");
    }

    std::vector<int64_t> new_shape = reduce_shape(a.shape(), dim, keepdim);
    bool result_requires_grad = a.requires_grad();
    Tensor result(new_shape, a.dtype(), deviceToString(a.device()), result_requires_grad);

    if (a.numel() == 0) {
        return result;
    }
    const size_t num_output_elements = result.numel();

    const float* d_a = static_cast<const float*>(a.raw_ptr());
    float* d_result = static_cast<float*>(result.raw_ptr());

    const auto& in_shape_vec = a.shape();
    const auto& in_strides_vec = a.strides();
    const auto& out_strides_vec = result.strides();

    int64_t* d_in_shape;
    int64_t* d_in_strides;
    int64_t* d_out_strides;

    CUDA_CHECK(cudaMalloc(&d_in_shape, in_shape_vec.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_in_strides, in_strides_vec.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_strides, out_strides_vec.size() * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_in_shape, in_shape_vec.data(), in_shape_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_strides, in_strides_vec.data(), in_strides_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides_vec.data(), out_strides_vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = num_output_elements;
    const size_t shmem_size = threadsPerBlock * sizeof(float);

    mean_reduction_kernel<<<blocksPerGrid, threadsPerBlock, shmem_size>>>(
        d_a,
        d_result,
        d_in_strides,
        d_out_strides,
        d_in_shape,
        ndim,
        dim,
        keepdim,
        num_output_elements
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_in_shape));
    CUDA_CHECK(cudaFree(d_in_strides));
    CUDA_CHECK(cudaFree(d_out_strides));

    if (result_requires_grad) {
      result.set_ctx({a}, CudaAutograd::mean);
    }

    return result;
>>>>>>> Stashed changes
}

