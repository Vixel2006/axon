#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "allocator/allocatorFactory.h" // Include the factory
#include <cuda_runtime.h>
#include <stdexcept>

// Macro for robust CUDA error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)


/**
 * @brief CUDA kernel for element-wise addition of two tensors (a + b).
 *
 * This kernel uses a grid-stride loop to ensure that all elements are processed.
 */
__global__ void add_kernel(const float* a_data, const float* b_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = a_data[i] + b_data[i];
    }
}

/**
 * @brief Performs element-wise addition on two tensors that are already on the CUDA device.
 *
 * This function follows a "Device-In, Device-Out" paradigm. It assumes the input tensors'
 * data pointers point to GPU memory. It allocates new GPU memory for the output tensor
 * using the framework's custom allocator and returns a new tensor that also resides on the GPU.
 */
Tensor CudaOps::add(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes are mismatched for addition.");
    }
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::add must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        // Return an empty tensor with correct device properties
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

    // --- The Core Fix ---
    // 1. Get the device pointers directly from the input tensors.
    // NO cudaMemcpy from host to device needed here.
    const float* d_a = static_cast<const float*>(a.raw_ptr());
    const float* d_b = static_cast<const float*>(b.raw_ptr());

    // 2. Allocate memory for the output tensor ON THE DEVICE using the AllocatorFactory.
    auto allocator = AllocatorFactory::get(a.device());
    void* d_c_raw = allocator->allocate(data_size);
    if (!d_c_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_c = static_cast<float*>(d_c_raw);

    // 3. Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the kernel on the device
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel execution

    // 5. Create the output Tensor object, wrapping the new DEVICE pointer.
    // The shared_ptr now uses the factory's custom deleter, which knows how to free GPU memory.
    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    
    // The new Tensor is created with the new shape, strides, and the device-side data pointer.
    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
