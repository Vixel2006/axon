#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

struct AlignedDeleter {
    void operator()(void* ptr) const {
        #ifdef _MSC_VER
        _aligned_free(ptr);
        #else
        free(ptr);
        #endif
    }
};

__global__ void pow_tensor_scalar_kernel(const float* base_data, const float exponent, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = powf(base_data[i], exponent);
    }
}

__global__ void pow_tensor_tensor_kernel(const float* base_data, const float* exp_data, float* c_data, size_t num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < num_elements; i += stride) {
        c_data[i] = powf(base_data[i], exp_data[i]);
    }
}


Tensor CudaOps::pow(const Tensor &base, float exponent) {
    if (base.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensor for CudaOps::pow must be on the CUDA device.");
    }

    const size_t num_elements = base.numel();
    if (num_elements == 0) {
        return Tensor({}, {}, base.dtype(), base.device(), nullptr, 0, false, nullptr, std::nullopt);
    }
    const size_t data_size = num_elements * sizeof(float);

    float *d_base, *d_c;
    CUDA_CHECK(cudaMalloc(&d_base, data_size));
    CUDA_CHECK(cudaMalloc(&d_c, data_size));

    CUDA_CHECK(cudaMemcpy(d_base, base.data_ptr().get(), data_size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    pow_tensor_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_base, exponent, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError());

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(data_size, 32);
    #else
    if (posix_memalign(&c_data_raw, 32, data_size) != 0) c_data_raw = nullptr;
    #endif

    if (!c_data_raw) {
        cudaFree(d_base);
        cudaFree(d_c);
        throw std::runtime_error("Failed to allocate aligned host memory for the output tensor.");
    }

    CUDA_CHECK(cudaMemcpy(c_data_raw, d_c, data_size, cudaMemcpyDeviceToHost));
    cudaFree(d_base);
    cudaFree(d_c);

    bool c_requires_grad = base.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(base.shape(), base.strides(), base.dtype(), base.device(), data, 0, c_requires_grad, nullptr, std::nullopt);
    
    if (c_requires_grad) {
      t.set_ctx({base}, CudaAutograd::pow);
    }

    return t;
}


Tensor CudaOps::pow(const Tensor &base, const Tensor &exponent) {
    if (base.shape() != exponent.shape()) {
        throw std::runtime_error("Tensor shapes are mismatched for pow operation.");
    }
    if (base.device().type != DeviceType::CUDA || exponent.device().type != DeviceType::CUDA) {
        throw std::runtime_error("Input tensors for CudaOps::pow must be on the CUDA device.");
    }

    const size_t num_elements = base.numel();
    if (num_elements == 0) {
        return Tensor({}, {}, base.dtype(), base.device(), nullptr, 0, false, nullptr, std::nullopt);
    }
    const size_t data_size = num_elements * sizeof(float);

    float *d_base, *d_exp, *d_c;
    CUDA_CHECK(cudaMalloc(&d_base, data_size));
    CUDA_CHECK(cudaMalloc(&d_exp, data_size));
    CUDA_CHECK(cudaMalloc(&d_c, data_size));

    CUDA_CHECK(cudaMemcpy(d_base, base.data_ptr().get(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_exp, exponent.data_ptr().get(), data_size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    pow_tensor_tensor_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_base, d_exp, d_c, num_elements);
    CUDA_CHECK(cudaGetLastError());

    void* c_data_raw = nullptr;
    #ifdef _MSC_VER
    c_data_raw = _aligned_malloc(data_size, 32);
    #else
    if (posix_memalign(&c_data_raw, 32, data_size) != 0) c_data_raw = nullptr;
    #endif

    if (!c_data_raw) {
        cudaFree(d_base);
        cudaFree(d_exp);
        cudaFree(d_c);
        throw std::runtime_error("Failed to allocate aligned host memory for the output tensor.");
    }

    CUDA_CHECK(cudaMemcpy(c_data_raw, d_c, data_size, cudaMemcpyDeviceToHost));
    cudaFree(d_base);
    cudaFree(d_exp);
    cudaFree(d_c);

    bool c_requires_grad = base.requires_grad() || exponent.requires_grad();
    std::shared_ptr<void> data(c_data_raw, AlignedDeleter{});
    Tensor t = Tensor(base.shape(), base.strides(), base.dtype(), base.device(), data, 0, c_requires_grad, nullptr, std::nullopt);

    if (c_requires_grad) {
      t.set_ctx({base, exponent}, CudaAutograd::pow);
    }

    return t;
}
