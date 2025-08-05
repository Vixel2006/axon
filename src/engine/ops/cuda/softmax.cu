#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "allocator/allocatorFactory.h" // Include the factory
#include <cuda_runtime.h>
#include <stdexcept>

// Consistent error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

/**
 * @brief CUDA kernel for row-wise softmax on a 2D tensor.
 *
 * This kernel is designed for efficiency by assigning one block to compute the
 * softmax for one entire row. It uses shared memory to perform two parallel
 * reductions: one to find the maximum value for numerical stability, and a
 * second to find the sum of exponentials.
 */
__global__ void softmax_kernel(const float* input_data, float* output_data, int rows, int cols) {
    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Use shared memory for efficient parallel reduction within a block.
    extern __shared__ float sdata[];

    // --- Step 1: Find the maximum value in the row for numerical stability ---
    float max_val = -__int_as_float(0x7F800000); // Negative infinity
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, input_data[row * cols + i]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Perform reduction in shared memory to find the row's true max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // --- Step 2: Compute sum of exponentials ---
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        // Store intermediate exp(x-max) in output and accumulate sum
        float val = expf(input_data[row * cols + i] - max_val);
        output_data[row * cols + i] = val;
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
    sum_exp = sdata[0];

    // --- Step 3: Normalize to get the final softmax values ---
    for (int i = tid; i < cols; i += blockDim.x) {
        output_data[row * cols + i] /= sum_exp;
    }
}


/**
 * @brief Performs a row-wise softmax on a 2D tensor that is already on the CUDA device.
 *
 * This function follows a "Device-In, Device-Out" paradigm. It allocates new GPU memory
 * for the output tensor using the framework's custom allocator and returns a new tensor
 * that also resides on the GPU.
 */
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
    const size_t data_size = num_elements * sizeof(float);

    if (num_elements == 0) {
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }

    // 1. Get the device pointer directly from the input tensor.
    const float* d_a = static_cast<const float*>(a.raw_ptr());

    // 2. Allocate memory for the output tensor ON THE DEVICE using the AllocatorFactory.
    auto allocator = AllocatorFactory::get(a.device());
    void* d_c_raw = allocator->allocate(data_size);
    if (!d_c_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_c = static_cast<float*>(d_c_raw);

    // 3. Define kernel launch parameters
    const int threadsPerBlock = 256;
    dim3 blocksPerGrid(rows);
    const size_t shared_mem_size = threadsPerBlock * sizeof(float);

    // 4. Launch the softmax kernel
    softmax_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_a, d_c, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    // 5. Create the output Tensor object, wrapping the new DEVICE pointer.
    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad();

    // The new Tensor is created with the new shape, strides, and the device-side data pointer.
    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
