#include "engine/ops.h"
#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "allocator/allocatorFactory.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void element_wise_add_kernel(const float* a_data, const float* b_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = a_data[i] + b_data[i];
    }
}

__global__ void scalar_add_kernel(const float* a_data, float scalar, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = a_data[i] + scalar;
    }
}

Tensor CudaOps::add(const Tensor &a, float scalar) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::add must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

    const float* d_a = static_cast<const float*>(a.raw_ptr());

    auto allocator = AllocatorFactory::get(a.device());
    void* d_c_raw = allocator->allocate(data_size);
    if (!d_c_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_c = static_cast<float*>(d_c_raw);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    scalar_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad();
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
   
    if (c_requires_grad) {
      t.set_ctx({a}, CudaAutograd::add);
    }

    return t;
}

Tensor CudaOps::add(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Tensor shapes are mismatched for addition.");
    }
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::add must be on the CUDA device.");
    }

    const size_t num_elements = a.numel();
    if (num_elements == 0) {
        return Tensor(a.shape(), a.dtype(), deviceToString(a.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

    const float* d_a = static_cast<const float*>(a.raw_ptr());
    const float* d_b = static_cast<const float*>(b.raw_ptr());

    auto allocator = AllocatorFactory::get(a.device());
    void* d_c_raw = allocator->allocate(data_size);
    if (!d_c_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_c = static_cast<float*>(d_c_raw);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    element_wise_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
   
    if (c_requires_grad) {
      t.set_ctx({a, b}, CudaAutograd::add);
    }

    return t;
}
