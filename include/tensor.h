#pragma once

#include <vector>
#include <memory>
#include "device.h"
#include "dtype.h"
#include "indexing.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

class Tensor
{
public:
    Tensor(const std::vector<__int64_t> &shape, DType dtype, const std::string &device_str = "cpu", bool requires_grad = false);
    Tensor(const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, DType dtype, Device device, std::shared_ptr<void> data_ptr, __int64_t offset, bool requires_grad, std::shared_ptr<void> grad);
    Tensor(const py::list &data, DType dtype = DType::float32, const std::string &device_str = "cpu", bool requires_grad = false);

    void *raw_ptr() const;

    ~Tensor();

    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;

    Tensor(Tensor &&other) noexcept = default;
    Tensor &operator=(Tensor &&other) noexcept = default;

    py::list data() const;
    const std::vector<__int64_t> &shape() const { return shape_; }
    const std::vector<__int64_t> &strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }
    std::shared_ptr<void> grad() const { return grad_; }
    size_t numel() const;
    bool is_contiguous() const;

    void fill_helper(py::list &output, size_t depth, std::vector<size_t> &indices) const;
    void fill(py::list &output) const;

    void fill_ptr_helper(const py::list &list, size_t depth, std::vector<size_t> &indices);
    void fill_ptr(const py::list &output);

    Tensor get_item(const std::vector<std::shared_ptr<IndexStrategy>> &strategies) const;

    Tensor view(std::vector<__int64_t> &new_shape) const;
    Tensor squeeze(int dim);
    Tensor unsqueeze(int dim);
    Tensor permute(const std::vector<int> &order);
    Tensor transpose(int n, int m) const;
    Tensor expand(const std::vector<__int64_t> &new_shape) const;
    Tensor broadcast(const std::vector<__int64_t> &new_shape) const;
    Tensor flatten(int start, int end) const;

    template <typename T>
    void flatten_list(const py::list &data, T *ptr);
    void get_shape(const py::list &data, std::vector<__int64_t> &shape, size_t depth);

private:
    std::shared_ptr<void> data_ptr_;
    std::vector<__int64_t> shape_;
    std::vector<__int64_t> strides_;
    DType dtype_;
    Device device_;
    __int64_t offset_;
    bool requires_grad_;
    std::shared_ptr<void> grad_;
};
