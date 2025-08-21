#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include "autograd/ops.h"
#include "autograd/tape.h"
#include "device.h"
#include "dtype.h"
#include "engine/ops.h"
#include "indexing.h"
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

class Tensor {
public:
  Tensor(const std::vector<__int64_t> &shape, DType dtype,
         const std::string &device_str = "cpu", bool requires_grad = false);
  Tensor(const std::vector<__int64_t> &shape,
         const std::vector<__int64_t> &strides, DType dtype, Device device,
         std::shared_ptr<void> data_ptr, __int64_t offset, bool requires_grad,
         std::shared_ptr<void> grad, std::optional<Tape> ctx);
  Tensor(const py::list &data, DType dtype = DType::float32,
         const std::string &device_str = "cpu", bool requires_grad = false);

  void *raw_ptr() const;

  ~Tensor();

  Tensor(const Tensor &other) = default;
  Tensor &operator=(const Tensor &other) = default;

  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(Tensor &&other) noexcept = default;
  bool operator==(const Tensor &other) const { return this == &other; }

  void zero_grad();
  void seed_gradient();

  py::list data() const;
  py::list grad() const;
  const std::vector<__int64_t> &shape() const { return shape_; }
  const std::vector<__int64_t> &strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  std::shared_ptr<void> data_ptr() const { return data_ptr_; }
  std::shared_ptr<void> grad_ptr() { return grad_; }
  bool requires_grad() const { return requires_grad_; }
  __int64_t offset() const { return offset_; }
  size_t ndim() const { return shape_.size(); }
  const std::optional<Tape> &ctx() { return ctx_; }

  void to(std::string device);

  void
  set_ctx(const std::vector<Tensor> &prev,
          std::function<void(Tensor &, std::vector<Tensor> &)> backward_fn) {
    if (requires_grad_ && !ctx_.has_value()) {
      ctx_ = Tape();
    }
    if (ctx_.has_value()) {
      ctx_->prev = prev;
      ctx_->backward_fn = backward_fn;
    }
  }

  void set_data_ptr(std::shared_ptr<void> data) { data_ptr_ = data; }

  size_t numel() const;
  bool is_contiguous() const;

  void fill_helper(py::list &output, size_t depth,
                   std::vector<size_t> &indices) const;
  void fill(py::list &output) const;

  void fill_grad_helper(py::list &output, size_t depth,
                        std::vector<size_t> &indices) const;
  void fill_grad(py::list &output) const;

  void fill_ptr_helper(const py::list &list, size_t depth,
                       std::vector<size_t> &indices);
  void fill_ptr(const py::list &output);

  Tensor
  get_item(const std::vector<std::shared_ptr<IndexStrategy>> &strategies) const;

  Tensor view(std::vector<__int64_t> &new_shape) const;
  Tensor squeeze(int dim);
  Tensor unsqueeze(int dim);
  Tensor permute(const std::vector<int> &order);
  Tensor transpose(int n, int m) const;
  Tensor expand(const std::vector<__int64_t> &new_shape) const;
  Tensor broadcast(const std::vector<__int64_t> &new_shape) const;
  Tensor flatten(int start, int end) const;
  static Tensor cat(const std::vector<Tensor> &tensors, int dim);

  void flatten_list(const py::list &data, float *ptr);
  void get_shape(const py::list &data, std::vector<__int64_t> &shape,
                 size_t depth);

  Tensor neg() const;
  Tensor add(const Tensor &other) const;
  Tensor add(float scalar) const;
  Tensor sub(const Tensor &other) const;
  Tensor sub(float scalar) const;
  Tensor mul(const Tensor &other) const;
  Tensor mul(float scalar) const;
  Tensor div(const Tensor &other) const;
  Tensor div(float other) const;
  Tensor matmul(const Tensor &other) const;
  Tensor sum() const;
  Tensor sum(int dim, bool keepdim = false) const;
  Tensor mean() const;
  Tensor mean(int dim, bool keepdim = false) const;

  std::vector<Tensor> build_topo() const;
  void backward();

private:
  std::shared_ptr<void> data_ptr_;
  std::vector<__int64_t> shape_;
  std::vector<__int64_t> strides_;
  DType dtype_;
  Device device_;
  __int64_t offset_;
  bool requires_grad_;
  std::shared_ptr<void> grad_;
  std::optional<Tape> ctx_;
  Ops *ops_;
};

#endif
