#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h" // For AlignedDeleter
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

// A custom deleter for host memory allocated with posix_memalign or _aligned_malloc
// It's good practice to define this in a central header (like helpers.h) to avoid re-definition.
struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};


/**
 * @brief CUDA kernel for element-wise Rectified Linear Unit (ReLU) activation.
 *
 * This kernel uses a grid-stride loop to ensure all elements are processed. It applies
 * the function fmaxf(0.0f, x) to each element of the input tensor.
 *
 * @param a_data Pointer to the device memory of the input tensor.
 * @param c_data Pointer to the device memory of the output tensor.
 * @param num_elements The total number of elements in the tensor.
 */
__global__ void relu_kernel(const float* a_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = fmaxf(0.0f, a_data[i]); // Use fmaxf for single-precision max(0, x)
    }
}

Tensor CudaOps::relu(const Tensor &a) {
    // Ensure the tensor is on the correct device
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::relu must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        // Return an empty tensor with the same properties
        return Tensor({}, {}, a.dtype(), a.device(), nullptr, 0, false, nullptr, std::nullopt);
    }
    const size_t data_size = num_elements * sizeof(float);

    // 1. Allocate memory on the CUDA device for the input and output
    float *d_a, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, data_size));
    CUDA_CHECK(cudaMalloc(&d_c, data_size));

    // 2. Copy input tensor data from host RAM to device VRAM
    CUDA_CHECK(cudaMemcpy(d_a, a.data_ptr().get(), data_size, cudaMemcpyHostToDevice));

    // 3. Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the kernel on the device
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel execution

    // 5. Allocate aligned memory on the host for the result
    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(data_size, 32);
    #else
    if (posix_memalign(&c_data_raw, 32, data_size) != 0) {
        c_data_raw = nullptr;
    }
    #endif

    if (!c_data_raw) {
        // Free device memory before throwing to prevent leaks
        cudaFree(d_a);
        cudaFree(d_c);
        throw std::runtime_error("Failed to allocate aligned host memory for the output tensor.");
    }

    // 6. Copy the result from device VRAM back to host RAM
    CUDA_CHECK(cudaMemcpy(c_data_raw, d_c, data_size, cudaMemcpyDeviceToHost));

    // 7. Free the allocated device memory
    cudaFree(d_a);
    cudaFree(d_c);

    // 8. Create the output Tensor object
    bool c_requires_grad = a.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});

    // The output tensor has the same shape and strides as the input
    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
