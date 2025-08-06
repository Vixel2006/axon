#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"
#include "allocator/allocatorFactory.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void reduce_rows_kernel(const float* input_data, float* output_data, int rows, int cols, bool do_max) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float identity = do_max ? -__int_as_float(0x7F800000) : 0.0f;
    float result = identity;

    for (int i = tid; i < cols; i += blockDim.x) {
        result = do_max ? fmaxf(result, input_data[row * cols + i]) : result + input_data[row * cols + i];
    }
    sdata[tid] = result;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = do_max ? fmaxf(sdata[tid], sdata[tid + s]) : sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_data[row] = sdata[0];
    }
}


__global__ void broadcast_sub_kernel(const float* a_data, const float* b_vec, float* c_data, size_t num_elements, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        int row = i / cols;
        c_data[i] = a_data[i] - b_vec[row];
    }
}


__global__ void broadcast_div_kernel(const float* a_data, const float* b_vec, float* c_data, size_t num_elements, int cols) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        int row = i / cols;
        c_data[i] = a_data[i] / (b_vec[row] + 1e-9f);
    }
}

Tensor CudaOps::softmax(const Tensor &a) {
    if (a.shape().size() != 2) {
        throw std::runtime_error("CudaOps::softmax currently only supports 2D tensors.");
    }
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::softmax must be on the CUDA device.");
    }
    
    const int rows = a.shape()[0];
    const int cols = a.shape()[1];
    const size_t num_elements = a.numel();

    if (num_elements == 0) {
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }

    auto allocator = AllocatorFactory::get(a.device());
    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };

    void* d_max_vals_raw = allocator->allocate(rows * sizeof(float));
    float* d_max_vals = static_cast<float*>(d_max_vals_raw);

    const int threadsPerBlock = 256;
    dim3 reduceGrid(rows);
    size_t sharedMem = threadsPerBlock * sizeof(float);
    
    reduce_rows_kernel<<<reduceGrid, threadsPerBlock, sharedMem>>>(
        static_cast<const float*>(a.raw_ptr()), d_max_vals, rows, cols, true
    );
    CUDA_CHECK(cudaGetLastError());
    
    // --- Step 2: Subtract the max value from each element in the row ---
    void* d_shifted_raw = allocator->allocate(num_elements * sizeof(float));
    float* d_shifted = static_cast<float*>(d_shifted_raw);
    dim3 elementwiseGrid((num_elements + threadsPerBlock - 1) / threadsPerBlock);

    broadcast_sub_kernel<<<elementwiseGrid, threadsPerBlock>>>(
        static_cast<const float*>(a.raw_ptr()), d_max_vals, d_shifted, num_elements, cols
    );
    CUDA_CHECK(cudaGetLastError());
    allocator->deallocate(d_max_vals_raw);

    Tensor shifted_tensor = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), 
                                   std::shared_ptr<void>(d_shifted_raw, deleter), 0, false, nullptr, std::nullopt);

    Tensor exps_tensor = CudaOps::exp(shifted_tensor);

    void* d_sum_exps_raw = allocator->allocate(rows * sizeof(float));
    float* d_sum_exps = static_cast<float*>(d_sum_exps_raw);
    
    reduce_rows_kernel<<<reduceGrid, threadsPerBlock, sharedMem>>>(
        static_cast<const float*>(exps_tensor.raw_ptr()), d_sum_exps, rows, cols, false // false for sum
    );
    CUDA_CHECK(cudaGetLastError());

    void* d_out_raw = allocator->allocate(num_elements * sizeof(float));
    float* d_out = static_cast<float*>(d_out_raw);

    broadcast_div_kernel<<<elementwiseGrid, threadsPerBlock>>>(
        static_cast<const float*>(exps_tensor.raw_ptr()), d_sum_exps, d_out, num_elements, cols
    );
    CUDA_CHECK(cudaGetLastError());
    allocator->deallocate(d_sum_exps_raw);

    std::shared_ptr<void> data(d_out_raw, deleter);
    bool out_requires_grad = a.requires_grad();
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, out_requires_grad, nullptr, std::nullopt);

    return t;
}
