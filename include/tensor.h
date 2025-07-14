#pragma once

#include <vector>
#include <memory>
#include "device.h"
#include "dtype.h"
#include "indexing.h"

class Tensor
{
public:
    // Tensor Constructors
    Tensor(const std::vector<__int64_t> &shape, DType dtype, Device device);
    Tensor(const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, DType dtype, Device device, std::shared_ptr<void> data_ptr, __int64_t offset);

    void* raw_ptr() const;

    // Tensor Destructor
    ~Tensor();

    // Copy
    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;

    // Move
    Tensor(Tensor &&other) noexcept = default;
    Tensor &operator=(Tensor &&other) noexcept = default;

    // Getters
    std::shared_ptr<void> data() const { return data_ptr_; }
    const std::vector<__int64_t> &shape() const { return shape_; }
    const std::vector<__int64_t> &strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t numel() const;
    bool is_contiguous() const;

    Tensor get_item(const std::vector<std::shared_ptr<IndexStrategy>> &strategies) const;

    Tensor view(std::vector<__int64_t> &new_shape) const;
    Tensor squeeze(int dim);
    Tensor unsqueeze(int dim);
    Tensor permute(const std::vector<int> &order);
    Tensor transpose(int n, int m) const;
    Tensor expand(const std::vector<__int64_t> &new_shape) const;
    Tensor broadcast(const std::vector<__int64_t> &new_shape) const;
    Tensor flatten(int start, int end) const;

private:
    // Tensor data
    std::shared_ptr<void> data_ptr_;
    std::vector<__int64_t> shape_;
    std::vector<__int64_t> strides_;
    DType dtype_;
    Device device_;
    __int64_t offset_;
};
