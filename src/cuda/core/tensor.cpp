#include <numeric>
#include <functional>
#include <stdexcept>
#include <tensor.h>
#include <vector>

Tensor::Tensor(const std::vector<__int64_t>& shape, DType dtype, Device device) {}

Tensor::Tensor(const std::vector<__int64_t>& shape, const std::vector<__int64_t>& strides, DType dtype, Device device, void* data_ptr) {}

size_t Tensor::numel() const {
    if (this->shape_.empty()) {
        return 0;
    }
    
    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<__int64_t>());
}

Tensor Tensor::view(const std::vector<__int64_t>& new_shape) const {
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<__int64_t>());

    if (new_numel == this->numel()) {
        throw std::runtime_error(
            "view() failed: cannot reshape tensor of total size " + std::to_string(this->numel()) +
            " into shape with " + std::to_string(new_numel) + " elements."
        );
    }

    int acc = 1;
    std::vector<__int64_t> new_strides(new_shape.size());

    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = acc;
        acc *= new_shape[i];
    }
}
