#include "engine/ops.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error: ") +           \
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

__global__ void softmax_kernel(const float* a, float* c, int rows, int cols) {
    // Each block processes one row of the input tensor.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Use shared memory for efficient parallel reduction within a block.
    extern __shared__ float sdata[];

    // --- Step 1: Find the maximum value in the row for numerical stability ---
    // Use the IEEE 754 bit representation for negative infinity.
    // This is the correct and portable way to do this in a CUDA kernel.
    float max_val = -__int_as_float(0x7F800000);
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, a[row * cols + i]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Perform reduction in shared memory to find the absolute max for the row
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0]; // The max value for the entire row

    // --- Step 2: Compute exponentials and their sum ---
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(a[row * cols + i] - max_val);
        c[row * cols + i] = val; // Store intermediate exp value
        sum_exp += val;
    }
    sdata[tid] = sum_exp;
    __syncthreads();

    // Perform reduction in shared memory to get the total sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_exp = sdata[0]; // The total sum of exponentials for the row

    // --- Step 3: Normalize to get the final softmax values ---
    for (int i = tid; i < cols; i += blockDim.x) {
        c[row * cols + i] /= sum_exp;
    }
}


Tensor CudaOps::softmax(const Tensor &a) {
    // Assuming a 2D tensor for softmax
    if (a.shape().size() != 2) {
        throw std::runtime_error("Softmax currently only supports 2D tensors.");
    }
    const size_t num_elements = a.numel();
    const int rows = a.shape()[0];
    const int cols = a.shape()[1];

    // Allocate device memory
    float *d_a, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, num_elements * sizeof(float)));

    // Copy input tensor from host to device
    CUDA_CHECK(cudaMemcpy(d_a, a.data_ptr().get(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    dim3 blocksPerGrid(rows);

    // The amount of shared memory is dynamic, specified at launch time.
    size_t shared_mem_size = threadsPerBlock * sizeof(float);

    // Launch the softmax kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_a, d_c, rows, cols);
    CUDA_CHECK(cudaGetLastError()); // Check for any errors during kernel launch

    // Allocate host memory for the result
    void* c_data_raw;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(num_elements * sizeof(float), 32);
    #else
    if (posix_memalign(&c_data_raw, 32, num_elements * sizeof(float)) != 0) {
        c_data_raw = nullptr;
    }
    #endif
    if (!c_data_raw) {
        cudaFree(d_a);
        cudaFree(d_c);
        throw std::runtime_error("Failed to allocate aligned memory for the output tensor.");
    }

    // Copy result back from device to host
    CUDA_CHECK(cudaMemcpy(c_data_raw, d_c, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_c);

    bool c_requires_grad = a.requires_grad();
    auto data = std::shared_ptr<void>(c_data_raw, AlignedDeleter{});

    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}

