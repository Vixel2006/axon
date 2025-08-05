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
 * @brief CUDA kernel for element-wise Rectified Linear Unit (ReLU) activation.
 *
 * This kernel uses a grid-stride loop to ensure all elements are processed. It applies
 * the function fmaxf(0.0f, x) to each element of the input tensor.
 */
__global__ void relu_kernel(const float* a_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = fmaxf(0.0f, a_data[i]); // Use fmaxf for single-precision max(0, x)
    }
}


/**
 * @brief Performs an element-wise ReLU on a tensor that is already on the CUDA device.
 *
 * This function follows a "Device-In, Device-Out" paradigm. It assumes the input tensor's
 * data pointer points to GPU memory. It allocates new GPU memory for the output tensor
 * using the framework's custom allocator and returns a new tensor that also resides on the GPU.
 */
Tensor CudaOps::relu(const Tensor &a) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::relu must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        // Return an empty tensor with the correct device properties
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

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
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the kernel on the device
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel execution

    // 5. Create the output Tensor object, wrapping the new DEVICE pointer.
    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad();

    // The new Tensor is created with the new shape, strides, and the device-side data pointer.
    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
