#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "allocator/allocatorFactory.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void div_kernel_tt(const float* __restrict__ numerator_data,
                              const float* __restrict__ denominator_data,
                              float* __restrict__ out_data,
                              size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        out_data[i] = numerator_data[i] / denominator_data[i];
    }
}

__global__ void div_kernel_ts(const float* __restrict__ numerator_data,
                              float denominator_scalar,
                              float* __restrict__ out_data,
                              size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
        out_data[i] = numerator_data[i] / denominator_scalar;
    }
}


Tensor CudaOps::div(const Tensor& numerator, const Tensor& denominator) {
    if (numerator.shape() != denominator.shape()) {
        throw std::runtime_error("Tensor shapes must be identical for element-wise division.");
    }
    if (numerator.device().type != DeviceType::CUDA || denominator.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CudaOps::div can only operate on CUDA tensors.");
    }

    const size_t num_elements = numerator.numel();
    if (num_elements == 0) {
        return Tensor(numerator.shape(), numerator.dtype(), deviceToString(numerator.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

    const float* d_numerator = static_cast<const float*>(numerator.raw_ptr());
    const float* d_denominator = static_cast<const float*>(denominator.raw_ptr());

    auto allocator = AllocatorFactory::get(numerator.device());
    void* d_out_raw = allocator->allocate(data_size);
    if (!d_out_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_out = static_cast<float*>(d_out_raw);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    div_kernel_tt<<<blocksPerGrid, threadsPerBlock>>>(d_numerator, d_denominator, d_out, num_elements);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_out_raw, deleter);

    bool out_requires_grad = numerator.requires_grad() || denominator.requires_grad();
    Tensor t = Tensor(numerator.shape(), numerator.strides(), numerator.dtype(), numerator.device(), data, 0, out_requires_grad, nullptr, std::nullopt);

    if (out_requires_grad) {
      t.set_ctx({numerator, denominator}, CudaAutograd::div);
    }

    return t;
}


Tensor CudaOps::div(const Tensor& numerator, float denominator) {
    if (numerator.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CudaOps::div can only operate on a CUDA tensor.");
    }
    if (denominator == 0.0f) {
        throw std::runtime_error("Scalar division by zero is not allowed.");
    }

    const size_t num_elements = numerator.numel();
    if (num_elements == 0) {
        return Tensor(numerator.shape(), numerator.dtype(), deviceToString(numerator.device()), false);
    }
    const size_t data_size = num_elements * sizeof(float);

    const float* d_numerator = static_cast<const float*>(numerator.raw_ptr());

    auto allocator = AllocatorFactory::get(numerator.device());
    void* d_out_raw = allocator->allocate(data_size);
    if (!d_out_raw) {
        throw std::runtime_error("Failed to allocate CUDA memory for output tensor via AllocatorFactory.");
    }
    float* d_out = static_cast<float*>(d_out_raw);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    div_kernel_ts<<<blocksPerGrid, threadsPerBlock>>>(d_numerator, denominator, d_out, num_elements);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_out_raw, deleter);

    bool out_requires_grad = numerator.requires_grad();
    Tensor t = Tensor(numerator.shape(), numerator.strides(), numerator.dtype(), numerator.device(), data, 0, out_requires_grad, nullptr, std::nullopt);

    if (out_requires_grad) {
      t.set_ctx({numerator}, CudaAutograd::div);
    }

    return t;
}
