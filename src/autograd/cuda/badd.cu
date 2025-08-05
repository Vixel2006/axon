#include "autograd/ops.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>

// Macro for robust CUDA error checking, essential for debugging GPU code.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

/**
 * @brief CUDA kernel for the backward pass of the addition operation.
 *
 * This kernel performs the gradient accumulation for the two input tensors 'a' and 'b'.
 * It uses a grid-stride loop to ensure that it can process tensors of any size,
 * regardless of the launch configuration.
 *
 * @param out_grad_p Pointer to the gradient of the output tensor.
 * @param a_grad_p Pointer to the gradient of the first input tensor ('a').
 * @param b_grad_p Pointer to the gradient of the second input tensor ('b').
 * @param a_req_grad Boolean flag indicating if 'a' requires a gradient.
 * @param b_req_grad Boolean flag indicating if 'b' requires a gradient.
 * @param num_elements The total number of elements in the tensors.
 */
__global__ void badd_kernel(const float* out_grad_p,
                            float* a_grad_p,
                            float* b_grad_p,
                            bool a_req_grad,
                            bool b_req_grad,
                            size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        // The gradient flowing from the output is the same for both inputs.
        const float grad = out_grad_p[i];
        
        // Accumulate the gradient for tensor 'a' if needed.
        if (a_req_grad) {
            a_grad_p[i] += grad;
        }

        // Accumulate the gradient for tensor 'b' if needed.
        if (b_req_grad) {
            b_grad_p[i] += grad;
        }
    }
}


/**
 * @brief Performs the backward pass for the 'add' operation on the CUDA device.
 *
 * This function launches a CUDA kernel to compute the gradients for the input tensors.
 * It assumes all tensor data and gradients are already located on the GPU.
 */
void CudaAutograd::add(const Tensor& out, std::vector<Tensor>& prev) {
    Tensor t = out;
    Tensor& a = prev[0];
    Tensor& b = prev[1];
    
    // Hoist the gradient requirement checks to the host side.
    const bool a_req_grad = a.requires_grad();
    const bool b_req_grad = b.requires_grad();

    // If neither input tensor requires a gradient, there's no work to do.
    if (!a_req_grad && !b_req_grad) {
        return;
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return;
    }

    // Get the raw device pointers from the Tensor objects.
    const float* out_grad_p = static_cast<const float*>(t.grad_ptr().get());
    float* a_grad_p = static_cast<float*>(a.grad_ptr().get());
    float* b_grad_p = static_cast<float*>(b.grad_ptr().get());

    if (!out_grad_p || !a_grad_p || !b_grad_p) {
        throw std::runtime_error("A gradient pointer is null in 'add' backward pass (CUDA).");
    }

    // Define kernel launch parameters.
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the single, unified kernel to update gradients.
    badd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        out_grad_p,
        a_grad_p,
        b_grad_p,
        a_req_grad,
        b_req_grad,
        num_elements
    );

    // Check for any errors that occurred during kernel execution.
    CUDA_CHECK(cudaGetLastError());
}
