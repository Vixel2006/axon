#include "tensor.h"

#include <cuda_runtime.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "allocator/allocatorFactory.h"
#include "helpers.h"
#include "engine/ops.h"
#include "autograd/ops.h"
#include "backend_registery.h"

bool Tensor::is_contiguous() const {
    int64_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (strides_[i] != stride) {
            return false;
        }
        stride *= shape_[i];
    }
    return true;
}

Tensor::Tensor(const std::vector<__int64_t> &shape, DType dtype,
               const std::string &device_str, bool requires_grad)
    : shape_(shape),
      strides_(compute_strides_(shape)),
      dtype_(dtype),
      device_(parse_device(device_str)),
      offset_(0),
      requires_grad_(requires_grad),
      grad_(nullptr),
      ctx_(std::nullopt) {
  size_t num_elements = this->numel();
  if (num_elements == 0) {
    data_ptr_ = nullptr;
    return;
  }

  size_t size_in_bytes = num_elements * DtypeToSize(dtype_);
  auto allocator = AllocatorFactory::get(device_);
  auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };

  void *raw_data_ptr = allocator->allocate(size_in_bytes);
  if (raw_data_ptr == nullptr) {
    throw std::runtime_error("Memory allocation failed for tensor on device " + device_str);
  }
  data_ptr_ = std::shared_ptr<void>(raw_data_ptr, deleter);

  if (device_.type == DeviceType::CPU) { ops_ = get_cpu_ops(); }
  else if (device_.type == DeviceType::CUDA) { ops_ = get_gpu_ops(); }

  if (requires_grad_) {
    void *raw_grad_ptr = allocator->allocate(size_in_bytes);
    if (raw_grad_ptr == nullptr) {
      throw std::runtime_error("Memory allocation failed for gradient on device " + device_str);
    }

    // Zero out the allocated gradient memory
    if (device_.type == DeviceType::CPU) {
      std::memset(raw_grad_ptr, 0, size_in_bytes);
    } else if (device_.type == DeviceType::CUDA) {
      cudaError_t err = cudaMemset(raw_grad_ptr, 0, size_in_bytes);
      if (err != cudaSuccess) {
        allocator->deallocate(raw_grad_ptr); 
        throw std::runtime_error("Failed to zero out gradient tensor on CUDA device: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    grad_ = std::shared_ptr<void>(raw_grad_ptr, deleter);
  }
}


Tensor::Tensor(const std::vector<__int64_t> &shape,
               const std::vector<__int64_t> &strides, DType dtype,
               Device device, std::shared_ptr<void> data_ptr, __int64_t offset,
               bool requires_grad, std::shared_ptr<void> grad, std::optional<Tape> ctx)
    : shape_(shape),
      strides_(strides),
      dtype_(dtype),
      device_(device),
      data_ptr_(data_ptr),
      offset_(offset),
      requires_grad_(requires_grad),
      grad_(grad),
      ctx_(ctx) {
  if (this->strides_.size() != this->shape_.size()) {
    throw std::runtime_error(
        "Shape and stride dimensions mismatch in Tensor constructor.");
  }


  if (device_.type == DeviceType::CPU) { ops_ = get_cpu_ops(); }
  else if (device_.type == DeviceType::CUDA) { ops_ = get_gpu_ops(); }

  if (requires_grad_ && grad_ == nullptr) {
    size_t num_elements = this->numel();
    size_t size_in_bytes = num_elements * DtypeToSize(dtype_);
    auto allocator = AllocatorFactory::get(device_);
    auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };

    void *raw_grad_ptr = allocator->allocate(size_in_bytes);
    if (raw_grad_ptr == nullptr) {
      throw std::runtime_error("Memory allocation failed for gradient on device " + deviceToString(device_));
    }

    // Zero out the allocated gradient memory
    if (device_.type == DeviceType::CPU) {
      std::memset(raw_grad_ptr, 0, size_in_bytes);
    } else if (device_.type == DeviceType::CUDA) {
      cudaError_t err = cudaMemset(raw_grad_ptr, 0, size_in_bytes);
      if (err != cudaSuccess) {
        allocator->deallocate(raw_grad_ptr); 
        throw std::runtime_error("Failed to zero out gradient tensor on CUDA device: " +
                                std::string(cudaGetErrorString(err)));
      }
    }

    grad_ = std::shared_ptr<void>(raw_grad_ptr, deleter);
  }
}

void Tensor::get_shape(const py::list &data, std::vector<__int64_t> &shape,
                       size_t depth = 0) {
  __int64_t len = data.size();
  if (len == 0) {
    shape.clear();
    return;
  }

  if (depth == shape.size()) {
    shape.push_back(len);
  } else if (shape[depth] != len) {
    throw std::runtime_error("Inconsistent tensor dimensions");
  }

  if (py::isinstance<py::list>(data[0])) {
    for (const auto &item : data) {
      if (!py::isinstance<py::list>(item)) {
        throw std::runtime_error("Mixed types in tensor list");
      }
      get_shape(item.cast<py::list>(), shape, depth + 1);
    }
  } else {
    for (const auto &item : data) {
      if (!py::isinstance<py::float_>(item) &&
          !py::isinstance<py::int_>(item)) {
        throw std::runtime_error("Tensor elements must be numbers");
      }
    }
  }
}

Tensor::Tensor(const py::list &data, DType dtype, const std::string &device_str,
               bool requires_grad)
    : dtype_(dtype),
      device_(parse_device(device_str)),
      offset_(0),
      requires_grad_(requires_grad),
      grad_(nullptr),
      ctx_(std::nullopt) {
  get_shape(data, shape_);
  size_t total_size = numel();

  if (total_size == 0) {
    if (!data.empty()) {
      strides_ = compute_strides_(shape_);
    }
    return;
  }

  strides_ = compute_strides_(shape_);

  auto allocator = AllocatorFactory::get(device_);
  auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
  
  size_t size_in_bytes = total_size * DtypeToSize(dtype_);

  void *raw_data_ptr = allocator->allocate(size_in_bytes);
  if (!raw_data_ptr) {
    throw std::runtime_error("Memory allocation failed for tensor data.");
  }
  data_ptr_ = std::shared_ptr<void>(raw_data_ptr, deleter);

  
  if (device_.type == DeviceType::CPU) { ops_ = get_cpu_ops(); }
  else if (device_.type == DeviceType::CUDA) { ops_ = get_gpu_ops(); }

  if (requires_grad_) {
    void *raw_grad_ptr = allocator->allocate(size_in_bytes);
    if (!raw_grad_ptr) {
      throw std::runtime_error("Memory allocation failed for gradient.");
    }

    if (device_.type == DeviceType::CPU) {
      std::memset(raw_grad_ptr, 0, size_in_bytes);
    } else if (device_.type == DeviceType::CUDA) {
      cudaError_t err = cudaMemset(raw_grad_ptr, 0, size_in_bytes);
      if (err != cudaSuccess) {
        allocator->deallocate(raw_grad_ptr);
        throw std::runtime_error("Failed to zero out gradient tensor on CUDA device: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }

    grad_ = std::shared_ptr<void>(raw_grad_ptr, deleter);
  }

  if (dtype_ == DType::float32) {
    std::vector<float> temp_host_buffer(total_size);
    this->flatten_list(data, temp_host_buffer.data());

    if (device_.type == DeviceType::CPU) {
      std::memcpy(data_ptr_.get(), temp_host_buffer.data(), size_in_bytes);
    } else if (device_.type == DeviceType::CUDA) {
      cudaError_t err = cudaMemcpy(data_ptr_.get(), temp_host_buffer.data(),
                                   size_in_bytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
    }
  } else {
    throw std::runtime_error(
        "Unsupported DType for Python list initialization.");
  }
}

void Tensor::seed_gradient() {
    if (!requires_grad_ || grad_ == nullptr) {
        return;
    }

    size_t num_elements = this->numel();
    if (num_elements == 0) {
        return;
    }
    size_t size_in_bytes = num_elements * DtypeToSize(dtype_);
    void* grad_data_ptr = grad_.get();

    if (device_.type == DeviceType::CPU) {
        float* grad_ptr = static_cast<float*>(grad_data_ptr);
        for (size_t i = 0; i < num_elements; ++i) {
            grad_ptr[i] = 1.0f;
        }
    } else if (device_.type == DeviceType::CUDA) {
        std::vector<char> host_buffer(size_in_bytes);

        float* host_ptr = reinterpret_cast<float*>(host_buffer.data());
        for (size_t i = 0; i < num_elements; ++i) {
            host_ptr[i] = 1.0f;
        }

        cudaError_t err = cudaMemcpy(grad_data_ptr, host_buffer.data(), size_in_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy seed gradient to CUDA device: " + 
                                     std::string(cudaGetErrorString(err)));
        }
    }
}

void *Tensor::raw_ptr() const {
  return static_cast<void *>(static_cast<char *>(data_ptr_.get()) +
                             offset_ * DtypeToSize(dtype_));
}

size_t Tensor::numel() const {
  if (this->shape_.empty()) {
    return 1;
  }

  return std::accumulate(shape_.begin(), shape_.end(), 1LL,
                         std::multiplies<__int64_t>());
}

void Tensor::fill_helper(py::list &output, size_t depth,
                         std::vector<size_t> &indices) const {
  float *data = static_cast<float *>(this->raw_ptr());

  if (depth == shape_.size() - 1) {
    for (size_t i = 0; i < shape_[depth]; ++i) {
      indices[depth] = i;
      size_t data_idx = 0;
      for (size_t d = 0; d < shape_.size(); ++d) {
        data_idx += indices[d] * strides_[d];
      }
      output.append(data[data_idx]);
    }
  }
  else {
    for (size_t i = 0; i < shape_[depth]; ++i) {
      py::list nested_list;
      indices[depth] = i;
      fill_helper(nested_list, depth + 1, indices);
      output.append(nested_list);
    }
  }
}

void Tensor::fill(py::list &output) const {
    if (shape_.empty()) {
        if (numel() == 1) {
            float value;
            if (device_.type == DeviceType::CUDA) {
                cudaMemcpy(&value, this->raw_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                value = *(static_cast<float *>(this->raw_ptr()));
            }
            output.append(value);
        }
        return;
    }

    if (device_.type == DeviceType::CUDA) {
        size_t total_elements = this->numel();
        std::vector<float> host_data(total_elements);

        cudaError_t err = cudaMemcpy(host_data.data(), data_ptr_.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor data from device to host for display: " + std::string(cudaGetErrorString(err)));
        }

        Tensor cpu_view(shape_, strides_, dtype_, Device{DeviceType::CPU, 0}, 
                        std::shared_ptr<void>(host_data.data(), [](void*){}),
                        0, false, nullptr, std::nullopt);
        
        std::vector<size_t> indices(shape_.size(), 0);
        cpu_view.fill_helper(output, 0, indices);

    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        fill_helper(output, 0, indices);
    }
}

void Tensor::fill_ptr_helper(const py::list &list, size_t depth,
                             std::vector<size_t> &indices) {
  float *data = static_cast<float *>(this->raw_ptr());

  if (depth == shape_.size() - 1) {
    if (static_cast<size_t>(shape_[depth]) != list.size()) {
      throw std::runtime_error("List size does not match shape at depth " +
                               std::to_string(depth));
    }
    for (size_t i = 0; i < shape_[depth]; ++i) {
      indices[depth] = i;
      size_t data_idx = 0;
      for (size_t d = 0; d < shape_.size(); ++d) {
        data_idx += indices[d] * strides_[d];
      }
      try {
        data[data_idx] = py::cast<float>(list[i]);
      } catch (const py::cast_error &e) {
        throw std::runtime_error("Element at index " + std::to_string(i) +
                                 " is not convertible to float at depth " +
                                 std::to_string(depth));
      }
    }
  }
  else {
    if (static_cast<size_t>(shape_[depth]) != list.size()) {
      throw std::runtime_error("List size does not match shape at depth " +
                               std::to_string(depth));
    }
    for (size_t i = 0; i < shape_[depth]; ++i) {
      if (!py::isinstance<py::list>(list[i])) {
        throw std::runtime_error("Expected nested list at index " +
                                 std::to_string(i) + " at depth " +
                                 std::to_string(depth));
      }
      indices[depth] = i;
      fill_ptr_helper(py::cast<py::list>(list[i]), depth + 1, indices);
    }
  }
}


void Tensor::fill_grad_helper(py::list &output, size_t depth,
                              std::vector<size_t> &indices) const {
    float *grad_data = static_cast<float *>(grad_.get());

    if (depth == shape_.size() - 1) {
        for (size_t i = 0; i < shape_[depth]; ++i) {
            indices[depth] = i;
            size_t data_idx = 0;
            for (size_t d = 0; d < shape_.size(); ++d) {
                data_idx += indices[d] * strides_[d];
            }
            output.append(grad_data[data_idx]);
        }
    } else {
        for (size_t i = 0; i < shape_[depth]; ++i) {
            py::list nested_list;
            indices[depth] = i;
            fill_grad_helper(nested_list, depth + 1, indices);
            output.append(nested_list);
        }
    }
}

void Tensor::fill_grad(py::list &output) const {
    if (!grad_) {
        return;
    }

    if (shape_.empty()) {
        if (numel() == 1) {
            float value;
            if (device_.type == DeviceType::CUDA) {
                cudaMemcpy(&value, grad_.get(), sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                value = *(static_cast<float *>(grad_.get()));
            }
            output.append(value);
        }
        return;
    }

    if (device_.type == DeviceType::CUDA) {
        size_t total_elements = this->numel();
        std::vector<float> host_grad_data(total_elements);

        cudaError_t err = cudaMemcpy(host_grad_data.data(), grad_.get(), total_elements * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy gradient data from device to host for display: " + std::string(cudaGetErrorString(err)));
        }

        Tensor cpu_view(shape_, strides_, dtype_, Device{DeviceType::CPU, 0}, 
                        nullptr,
                        0, true, 
                        std::shared_ptr<void>(host_grad_data.data(), [](void*){}),
                        std::nullopt);

        std::vector<size_t> indices(shape_.size(), 0);
        cpu_view.fill_grad_helper(output, 0, indices);

    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        fill_grad_helper(output, 0, indices);
    }
}

py::list Tensor::grad() const {
    py::list output;
    fill_grad(output);
    return output;
}


void Tensor::fill_ptr(const py::list &list) {
  if (shape_.empty()) {
    if (list.size() != 1) {
      throw std::runtime_error(
          "Expected a list with one element for a 0D tensor, but got size " +
          std::to_string(list.size()));
    }
    try {
      float *data = static_cast<float *>(this->raw_ptr());
      data[0] = py::cast<float>(list[0]);
    } catch (const py::cast_error &e) {
      throw std::runtime_error("Scalar element is not convertible to float");
    }
  }
  else {
    std::vector<size_t> indices(shape_.size(), 0);
    fill_ptr_helper(list, 0, indices);
  }
}

void flatten_list_recursive(const py::list &list, float *&ptr) {
  for (const auto &item : list) {
    if (py::isinstance<py::list>(item)) {
      flatten_list_recursive(item.cast<py::list>(), ptr);
    } else {
      *ptr = item.cast<float>();
      ptr++;
    }
  }
}

void Tensor::flatten_list(const py::list &data, float *ptr) {
  flatten_list_recursive(data, ptr);
}

py::list Tensor::data() const {
  py::list output;
  fill(output);
  return output;
}

Tensor Tensor::get_item(
    const std::vector<std::shared_ptr<IndexStrategy>> &strategies) const {
  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_strides;
  int64_t offset = offset_;

  int ellipsis_pos = -1;
  int num_new_axes = 0;

  for (int i = 0; i < strategies.size(); ++i) {
    if (dynamic_cast<EllipsisIndex *>(strategies[i].get())) {
      if (ellipsis_pos != -1) {
        throw std::runtime_error("an index can only have one ellipsis ('...')");
      }
      ellipsis_pos = i;
    } else if (dynamic_cast<NewAxisIndex *>(strategies[i].get())) {
      num_new_axes++;
    }
  }

  int num_ellipsis_dims = 0;
  if (ellipsis_pos != -1) {
    int non_special_strategies = strategies.size() - 1 - num_new_axes;
    if (non_special_strategies > shape_.size()) {
      throw std::out_of_range("Too many indices for tensor");
    }
    num_ellipsis_dims = shape_.size() - non_special_strategies;
  } else {
    if (strategies.size() - num_new_axes > shape_.size()) {
      throw std::out_of_range("Too many indices for tensor");
    }
  }

  size_t dim_idx = 0;
  for (const auto &strategy : strategies) {
    if (auto p = dynamic_cast<NewAxisIndex *>(strategy.get())) {
      p->apply(0, shape_, strides_, offset, new_shape, new_strides);
    } else if (auto p = dynamic_cast<EllipsisIndex *>(strategy.get())) {
      FullSlice full_slice_strategy;
      for (int k = 0; k < num_ellipsis_dims; ++k) {
        if (dim_idx >= shape_.size())
          break;
        full_slice_strategy.apply(dim_idx, shape_, strides_, offset, new_shape,
                                  new_strides);
        dim_idx++;
      }
    } else {
      if (dim_idx >= shape_.size()) {
        throw std::out_of_range("Too many indices for tensor");
      }
      strategy->apply(dim_idx, shape_, strides_, offset, new_shape,
                      new_strides);
      dim_idx++;
    }
  }

  while (dim_idx < shape_.size()) {
    new_shape.push_back(shape_[dim_idx]);
    new_strides.push_back(strides_[dim_idx]);
    dim_idx++;
  }

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::view(std::vector<__int64_t> &new_shape) const {
  if (!this->is_contiguous()) {
    throw std::runtime_error(
        "view(): can only be called on a contiguous tensor.");
  }
  __int64_t known_product = 1;
  __int64_t inferred_index = -1;

  for (size_t i = 0; i < new_shape.size(); ++i) {
    __int64_t dim = new_shape[i];

    if (dim == -1) {
      if (inferred_index != -1) {
        throw std::invalid_argument(
            "view(): only one dimension can be inferred (-1), but got another "
            "at index " +
            std::to_string(i));
      }
      inferred_index = i;
    } else if (dim <= 0) {
      throw std::invalid_argument(
          "view(): shape dimension at index " + std::to_string(i) +
          " must be > 0 or -1 for inference, but got " + std::to_string(dim));
    } else {
      known_product *= dim;
    }
  }

  __int64_t total = this->numel();

  if (inferred_index != -1) {
    if (total % known_product != 0) {
      throw std::invalid_argument(
          "view(): cannot infer missing dimension at index " +
          std::to_string(inferred_index) +
          " — product of known dims = " + std::to_string(known_product) +
          " does not divide total elements = " + std::to_string(total));
    }

    new_shape[inferred_index] = total / known_product;
  }

  __int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL,
                                        std::multiplies<__int64_t>());
  if (new_numel != total) {
    throw std::invalid_argument(
        "view(): mismatch — original numel = " + std::to_string(total) +
        ", new shape produces = " + std::to_string(new_numel));
  }

  if (!this->is_contiguous()) {
    throw std::runtime_error(
        "view(): tensor must be contiguous to be reshaped.");
  }

  std::vector<__int64_t> new_strides = compute_strides_(new_shape);
  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::squeeze(int dim) {
  if (dim == -1) dim = shape_.size() - 1;

  if (dim < -1 || dim >= static_cast<int>(shape_.size())) {
    throw std::out_of_range("squeeze(): Dimension " + std::to_string(dim) +
                            " is out of bounds for tensor with " +
                            std::to_string(shape_.size()) + " dimensions.");
  }

  if (shape_[dim] != 1) {
    throw std::runtime_error("squeeze(): Cannot squeeze dimension " +
                             std::to_string(dim) + " with size " +
                             std::to_string(shape_[dim]) +
                             ". Only dimensions of size 1 can be squeezed.");
  }

  std::vector<__int64_t> new_shape = shape_;
  std::vector<__int64_t> new_strides = strides_;

  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::unsqueeze(int dim) {
  const int ndim = shape_.size();

  if (dim < 0) {
    dim = ndim + 1 + dim;
  }

  if (dim < 0 || dim > ndim) {
    throw std::out_of_range(
        "unsqueeze(): Dimension out of range. Got " + std::to_string(dim) +
        " but expected to be in range [-" + std::to_string(ndim + 1) + ", " +
        std::to_string(ndim) + "].");
  }

  std::vector<int64_t> new_shape = shape_;

  new_shape.insert(new_shape.begin() + dim, 1);

  std::vector<int64_t> new_strides = compute_strides_(new_shape);

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::permute(const std::vector<int> &order) {
  if (order.size() != shape_.size()) {
    throw std::invalid_argument(
        "permute(): `order` must have the same number of dimensions as tensor " "shape. "
        "Expected " +
        std::to_string(shape_.size()) + ", got " +
        std::to_string(order.size()) + ".");
  }

  std::vector<bool> seen(order.size(), false);
  for (int i : order) {
    if (i < 0 || i > shape_.size()) {
      throw std::out_of_range(
          "permute(): each index in `order` must be in range [0, " +
          std::to_string(shape_.size() - 1) + "], but got " +
          std::to_string(i) + ".");
    }
    if (seen[i]) {
      throw std::invalid_argument("permute(): duplicate index " +
                                  std::to_string(i) + " in `order`.");
    }
    seen[i] = true;
  }

  std::vector<int64_t> new_shape(shape_.size());
  std::vector<int64_t> new_strides(shape_.size());

  for (size_t i = 0; i < order.size(); ++i) {
    new_shape[i] = shape_[order[i]];
    new_strides[i] = strides_[order[i]];
  }

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::transpose(int n, int m) const {
  const size_t rank = shape_.size();

  if (n < 0) n += rank;
  if (m < 0) m += rank;

  if (n < 0 || n >= rank) {
    throw std::out_of_range(
        "transpose(): dimension `n` is out of bounds. Got " +
        std::to_string(n) + ", but tensor has rank " + std::to_string(rank) +
        ".");
  }
  if (m < 0 || m >= rank) {
    throw std::out_of_range(
        "transpose(): dimension `m` is out of bounds. Got " +
        std::to_string(m) + ", but tensor has rank " + std::to_string(rank) +
        ".");
  }

  if (n == m) {
    throw std::invalid_argument(
        "transpose(): dimensions `n` and `m` must be different, but both are " +
        std::to_string(n) + ".");
  }

  std::vector<__int64_t> new_shape = shape_;
  std::vector<__int64_t> new_strides = strides_;

  std::swap(new_shape[n], new_shape[m]);
  std::swap(new_strides[n], new_strides[m]);

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::expand(const std::vector<__int64_t> &new_shape) const {
  if (new_shape.size() != shape_.size()) {
    throw std::runtime_error(
        "expand() error: Dimensionality mismatch. "
        "Tried to expand from shape " +
        shapeToString(shape_) + " to shape " + shapeToString(new_shape) +
        ". "
        "Both shapes must have the same number of dimensions (" +
        std::to_string(shape_.size()) + " expected, got " +
        std::to_string(new_shape.size()) + ").");
  }

  std::vector<__int64_t> new_strides = strides_;

  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] != shape_[i] && shape_[i] != 1) {
      throw std::runtime_error("expand() error: Cannot expand dimension " +
                               std::to_string(i) + " from size " +
                               std::to_string(shape_[i]) + " to " +
                               std::to_string(new_shape[i]) +
                               ". "
                               "Only dimensions of size 1 can be expanded.");
    } else if (new_shape[i] != shape_[i]) {
      new_strides[i] = 0;
    }
  }

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::broadcast(const std::vector<__int64_t> &new_shape) const {
  const size_t ndim = shape_.size();
  const size_t new_ndim = new_shape.size();

  if (ndim > new_ndim) {
    throw std::runtime_error(
        "Cannot broadcast: source tensor has higher rank than target shape.");
  }

  std::vector<__int64_t> reshaped_shape = shape_;
  std::vector<__int64_t> reshaped_strides = strides_;

  size_t diff = new_ndim - ndim;
  reshaped_shape.insert(reshaped_shape.begin(), diff, 1);
  reshaped_strides.insert(reshaped_strides.begin(), diff, 0);

  std::vector<__int64_t> final_strides(new_ndim);

  for (size_t i = 0; i < new_ndim; ++i) {
    if (reshaped_shape[i] == new_shape[i]) {
      final_strides[i] = reshaped_strides[i];
    } else if (reshaped_shape[i] == 1) {
      final_strides[i] = 0;
    } else {
      throw std::runtime_error("broadcast() error: cannot broadcast dim " +
                               std::to_string(i) + " from " +
                               std::to_string(reshaped_shape[i]) + " to " +
                               std::to_string(new_shape[i]) +
                               ". Only dims of size 1 can be broadcast.");
    }
  }

  return Tensor(new_shape, final_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::flatten(int start, int end) const {
  if (!this->is_contiguous()) {
    throw std::runtime_error(
        "flatten(): can only be called on a contiguous tensor.");
  }

  int ndim = shape_.size();

  if (start < 0) start += ndim;
  if (end < 0) end += ndim;

  if (start < 0 || start >= ndim) {
    throw std::out_of_range("flatten() error: 'start' dimension " +
                            std::to_string(start) +
                            " is out of bounds for tensor with " +
                            std::to_string(ndim) + " dimensions.");
  }

  if (end < 0 || end >= ndim) {
    throw std::out_of_range("flatten() error: 'end' dimension " +
                            std::to_string(end) +
                            " is out of bounds for tensor with " +
                            std::to_string(ndim) + " dimensions.");
  }

  if (start > end) {
    throw std::invalid_argument(
        "flatten() error: 'start' index (" + std::to_string(start) +
        ") cannot be greater than 'end' index (" + std::to_string(end) + ").");
  }

  std::vector<__int64_t> new_shape;

  for (int i = 0; i < start; ++i) {
    new_shape.push_back(shape_[i]);
  }

  __int64_t flattened_dim = 1;
  for (int i = start; i <= end; ++i) {
    flattened_dim *= shape_[i];
  }
  new_shape.push_back(flattened_dim);

  for (int i = end + 1; i < ndim; ++i) {
    new_shape.push_back(shape_[i]);
  }

  std::vector<__int64_t> new_strides = compute_strides_(new_shape);

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset_,
                requires_grad_, grad_, ctx_);
}

Tensor Tensor::neg() const {
  return this->ops_->neg(*this);
}

Tensor Tensor::add(const Tensor& other) const {
  return this->ops_->add(*this, other);
}

Tensor Tensor::add(float scalar) const {
  return this->ops_->add(*this, scalar);
}

Tensor Tensor::sub(const Tensor& other) const {
  return this->ops_->sub(*this, other);
}

Tensor Tensor::sub(float scalar) const {
  return this->ops_->sub(*this, scalar);
}

Tensor Tensor::mul(const Tensor& other) const {
  return this->ops_->mul(*this, other);
}

Tensor Tensor::mul(float scalar) const {
  return this->ops_->mul(*this, scalar);
}

Tensor Tensor::div(const Tensor& other) const {
  return this->ops_->div(*this, other);
}

Tensor Tensor::div(float other) const {
  return this->ops_->div(*this, other);
}

Tensor Tensor::matmul(const Tensor& other) const {
  return this->ops_->matmul(*this, other);
}

Tensor Tensor::sum() const {
  return this->ops_->sum(*this);
}

Tensor Tensor::sum(int dim, bool keepdim) const {
  return this->ops_->sum(*this, dim, keepdim);
}

Tensor Tensor::mean() const {
  return this->ops_->mean(*this);
}

Tensor Tensor::mean(int dim, bool keepdim) const {
  return this->ops_->mean(*this, dim, keepdim);
}

std::vector<Tensor> Tensor::build_topo() const {
  if (!requires_grad_) { throw std::runtime_error("Can't do backward prop on when requires_grad=False"); }

  std::vector<Tensor> topo;
  std::vector<Tensor> visited;

  std::function<void(const Tensor&)> _visit = [&](const Tensor& t) {
        if (std::find(visited.begin(), visited.end(), t) == visited.end()) {
            visited.push_back(t);
            if (t.ctx_.has_value()) {
                for (const auto& parent : t.ctx_->prev) {
                    _visit(parent);
                }
                topo.push_back(t);
            }
        }
    };

    _visit(*this);


  return topo;
}

void Tensor::backward() {
  if (!requires_grad_ || !ctx_.has_value()) { return ; }

  seed_gradient();
  std::vector<Tensor> topo_sorted_graph = build_topo();

  for (auto it = topo_sorted_graph.rbegin(); it != topo_sorted_graph.rend(); ++it) {
    Tensor& t = *it;

    if (t.ctx().has_value()) {
      auto& tape = t.ctx();
      std::vector<Tensor> inputs = tape->prev;

      tape->backward_fn(t, inputs);
    }
  }
}

Tensor::~Tensor() {}

