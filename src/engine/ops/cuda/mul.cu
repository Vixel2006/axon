#include "engine/ops.h"
#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "allocator/allocatorFactory.h"
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void broadcast_mul_kernel(
    const float* a_data,
    const int64_t* a_strides,
    const float* b_data,
    const int64_t* b_strides,
    float* c_data,
    const int64_t* c_shape,
    int c_ndim,
    size_t num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) {
        return;
    }

    int64_t temp_i = i;
    size_t a_offset = 0;
    size_t b_offset = 0;

    for (int d = c_ndim - 1; d >= 0; --d) {
        int64_t coord = temp_i % c_shape[d];
        temp_i /= c_shape[d];

        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    c_data[i] = a_data[a_offset] * b_data[b_offset];
}

__global__ void scalar_mul_kernel(const float* __restrict__ a_data, float scalar, float* __restrict__ c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t i = index; i < num_elements; i += stride) {
      c_data[i] = a_data[i] * scalar;
    }
}

Tensor CudaOps::mul(const Tensor& a, float scalar) {
    if (a.device().type != DeviceType::CUDA) {
        throw std::runtime_error("CudaOps::mul can only operate on CUDA tensors.");
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

    scalar_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError());

    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad();
    Tensor t = Tensor(a.shape(), a.strides(), a.dtype(), a.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({a}, CudaAutograd::mul);
    }

    return t;
}

Tensor CudaOps::mul(const Tensor& a, const Tensor& b) {
    if (a.device().type != DeviceType::CUDA || b.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::mul must be on the CUDA device.");
    }

    std::vector<int64_t> c_shape = compute_broadcast_shape(a.shape(), b.shape());
    size_t num_elements = std::accumulate(c_shape.begin(), c_shape.end(), 1, std::multiplies<int64_t>());
    int c_ndim = c_shape.size();

    if (num_elements == 0) {
        return Tensor(c_shape, a.dtype(), deviceToString(a.device()), false);
    }

    Tensor a_broad = a.broadcast(c_shape);
    Tensor b_broad = b.broadcast(c_shape);

    const float* d_a = static_cast<const float*>(a_broad.raw_ptr());
    const float* d_b = static_cast<const float*>(b_broad.raw_ptr());

    auto allocator = AllocatorFactory::get(a.device());
    void* d_c_raw = allocator->allocate(num_elements * sizeof(float));
    float* d_c = static_cast<float*>(d_c_raw);

    int64_t *d_a_strides, *d_b_strides, *d_c_shape;
    CUDA_CHECK(cudaMalloc(&d_a_strides, c_ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_b_strides, c_ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_c_shape, c_ndim * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_a_strides, a_broad.strides().data(), c_ndim * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_strides, b_broad.strides().data(), c_ndim * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_shape, c_shape.data(), c_ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    broadcast_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_a, d_a_strides,
        d_b, d_b_strides,
        d_c, d_c_shape, c_ndim, num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_a_strides));
    CUDA_CHECK(cudaFree(d_b_strides));
    CUDA_CHECK(cudaFree(d_c_shape));

    auto deleter = [allocator](void* ptr) { allocator->deallocate(ptr); };
    std::shared_ptr<void> data(d_c_raw, deleter);

    bool c_requires_grad = a.requires_grad() || b.requires_grad();
    Tensor t = Tensor(c_shape, a.dtype(), deviceToString(a.device()), c_requires_grad);
    t.set_data_ptr(data);

    if (c_requires_grad) {
      t.set_ctx({a, b}, CudaAutograd::mul);
    }

    return t;
}

