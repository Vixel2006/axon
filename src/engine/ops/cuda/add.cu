#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h" // For AlignedDeleter if needed, though often it's in the same file as Tensor
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
 * @brief CUDA kernel for element-wise addition of two tensors (a + b).
 *
 * This kernel uses a grid-stride loop to ensure that all elements are processed,
 * regardless of the tensor size or the number of threads launched. This makes the
 * kernel robust and efficient.
 *
 * @param a_data Pointer to the device memory of the first input tensor.
 * @param b_data Pointer to the device memory of the second input tensor.
 * @param c_data Pointer to the device memory of the output tensor.
 * @param num_elements The total number of elements in the tensors.
 */
__global__ void add_kernel(const float* a_data, const float* b_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = a_data[i] + b_data[i];
    }
}

Tensor CudaOps::add(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes are mismatched for addition.");
    }
    // Ensure tensors are on the correct device, or handle appropriately
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::add must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return Tensor({}, {}, a.dtype(), a.device(), nullptr, 0, false, nullptr, std::nullopt);
    }
    const size_t data_size = num_elements * sizeof(float);

    // 1. Allocate memory on the CUDA device for inputs and output
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, data_size));
    CUDA_CHECK(cudaMalloc(&d_b, data_size));
    CUDA_CHECK(cudaMalloc(&d_c, data_size));

    // 2. Copy input tensor data from host RAM to device VRAM
    // Note: a.data_ptr() and b.data_ptr() must point to host-accessible memory.
    // If they could already be on the device, this logic would need adjustment.
    CUDA_CHECK(cudaMemcpy(d_a, a.data_ptr().get(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data_ptr().get(), data_size, cudaMemcpyHostToDevice));

    // 3. Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the kernel on the device
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
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
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("Failed to allocate aligned host memory for the output tensor.");
    }

    // 6. Copy the result from device VRAM back to host RAM
    CUDA_CHECK(cudaMemcpy(c_data_raw, d_c, data_size, cudaMemcpyDeviceToHost));

    // 7. Free the allocated device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 8. Create the output Tensor object
    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});

    return Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
}
