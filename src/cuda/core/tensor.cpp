#include <numeric>
#include <functional>
#include <stdexcept>
#include "tensor.h"
#include "dtype.h"
#include "device.h"
#include <vector>
#include <cuda_runtime.h>

std::vector<__int64_t> compute_strides_(const std::vector<__int64_t>& shape) {
    __int64_t acc = 1;
    std::vector<__int64_t> strides(shape.size());

    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= shape[i];
    }

    return strides;
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    __int64_t acc = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != acc) {
            return false;
        }
        acc *= shape_[i];
    }
    return true;
}

Tensor::Tensor(const std::vector<__int64_t>& shape, DType dtype, Device device)
    : shape_(shape),
      strides_(compute_strides_(shape)),
      dtype_(dtype),
      device_(device)
{
    size_t num_elements = this->numel();
    if (num_elements == 0) {
        return;
    }
    
    size_t size_in_bytes = num_elements * DtypeToSize(dtype_);
    void* raw_ptr = nullptr;

    auto deleter = [device = this->device_](void* ptr) {
        if (!ptr) return;
        if (device.type == DeviceType::CUDA) {
            cudaFree(ptr);
        } else {
            free(ptr);
        }
    };

    if (this->device_.type == DeviceType::CUDA) {
        cudaError_t err = cudaMalloc(&raw_ptr, size_in_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
        }
    } else if (this->device_.type == DeviceType::CPU) {
        raw_ptr = malloc(size_in_bytes);
        if (!raw_ptr) {
            throw std::runtime_error("CPU malloc failed.");
        }
    } else {
        throw std::runtime_error("Unsupported device type.");
    }

    data_ptr_ = std::shared_ptr<void>(raw_ptr, deleter);
}

Tensor::Tensor(const std::vector<__int64_t>& shape, const std::vector<__int64_t>& strides, DType dtype, Device device, std::shared_ptr<void> data_ptr): shape_(shape), strides_(strides), dtype_(dtype), device_(device), data_ptr_(data_ptr) {
    if (this->strides_.size() != this->shape_.size()) {
        throw std::runtime_error("Shape and stride dimensions mismatch in Tensor constructor.");
    }
}

size_t Tensor::numel() const {
    if (this->shape_.empty()) {
        return 0;
    }
    
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<__int64_t>());
}

Tensor Tensor::view(const std::vector<__int64_t>& new_shape) const {
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<__int64_t>());

    if (new_numel != this->numel()) {
        throw std::runtime_error(
            "view() failed: cannot reshape tensor of total size " + std::to_string(this->numel()) +
            " into shape with " + std::to_string(new_numel) + " elements."
        );
    }

    if (!this->is_contiguous()) {
        throw std::runtime_error("view() can only be called on a contiguous tensor.");
    }

    std::vector<__int64_t> new_strides = compute_strides_(new_shape);

    return Tensor(new_shape, new_strides, this->dtype_, this->device_, this->data_ptr_);
}
