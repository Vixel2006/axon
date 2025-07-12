#pragma once

#include <vector>
#include <memory>
#include <device.h>
#include <dtype.h>

class Tensor {
public:
    // Tensor Constructors
    Tensor(const std::vector<__int64_t>& shape, DType dtype, Device device);
    Tensor(const std::vector<__int64_t>& shape, const std::vector<__int64_t>& strides, DType dtype, Device device, void* data_ptr);
    
    // Tensor Destructor
    ~Tensor();

    // Preventing Copying
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Getters
    std::shared_ptr<void> data() const { return data_ptr_; }
    const std::vector<__int64_t>& shape() const { return shape_; }
    const std::vector<__int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t numel() const;
    bool is_contiguous() const;

    // View method
    Tensor view(const std::vector<__int64_t>& new_shape) const;

private:
    // Tensor data
    std::shared_ptr<void> data_ptr_;
    std::vector<__int64_t> shape_;
    std::vector<__int64_t> strides_;
    DType dtype_;
    Device device_;

    bool owns_data_;
};
