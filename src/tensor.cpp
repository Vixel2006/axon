#include "tensor.h"

#include <cuda_runtime.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "allocator/allocatorFactory.h"
#include "helpers.h"
#include "ops/add.h"
#include "ops/matmul.h"
#include "ops/mul.h"
#include "ops/sub.h"
#include "ops/
bool Tensor::is_contiguous() const {
  if (shape_.size() <= 1) return true;
  __int64_t expected_stride = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    if (shape_[i] == 1) continue;
    if (strides_[i] != expected_stride) return false;
    expected_stride *= shape_[i];
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
      grad_(nullptr) {
  size_t num_elements = this->numel();
  if (num_elements == 0) {
    data_ptr_ = nullptr;
    return;
  }

  size_t size_in_bytes = num_elements * DtypeToSize(dtype_);

  auto allocator = AllocatorFactory::get(device_);
  void *raw_data_ptr = allocator->allocate(size_in_bytes);

  if (raw_data_ptr == nullptr) {
    throw std::runtime_error("Memory allocation failed for tensor on device " +
                             device_str +
                             ". The device might be out of memory.");
  }

  auto deleter = [allocator](void *ptr) { allocator->deallocate(ptr); };
  data_ptr_ = std::shared_ptr<void>(raw_data_ptr, deleter);

  if (requires_grad_) {
    void *raw_grad_ptr = allocator->allocate(size_in_bytes);
    if (raw_grad_ptr == nullptr) {
      throw std::runtime_error(
          "Memory allocation failed for gradient on device " + device_str);
    }
    grad_ = std::shared_ptr<void>(raw_grad_ptr, deleter);
  }
}

Tensor::Tensor(const std::vector<__int64_t> &shape,
               const std::vector<__int64_t> &strides, DType dtype,
               Device device, std::shared_ptr<void> data_ptr, __int64_t offset,
               bool requires_grad, std::shared_ptr<void> grad)
    : shape_(shape),
      strides_(strides),
      dtype_(dtype),
      device_(device),
      data_ptr_(data_ptr),
      offset_(offset),
      requires_grad_(requires_grad),
      grad_(grad) {
  if (this->strides_.size() != this->shape_.size()) {
    throw std::runtime_error(
        "Shape and stride dimensions mismatch in Tensor constructor.");
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
      grad_(nullptr) {
  // 1. Determine the tensor's shape and total number of elements from the
  // nested list.
  get_shape(data, shape_);
  size_t total_size = numel();

  // Handle creation of an empty tensor (e.g., from an empty list).
  if (total_size == 0) {
    if (!data.empty()) {
      // This case happens for lists like [[]] or [[], []]
      strides_ = compute_strides_(shape_);
    }
    return;  // Nothing more to do for an empty tensor.
  }

  // 2. Compute strides for a new, contiguous tensor.
  strides_ = compute_strides_(shape_);

  // 3. Get the appropriate memory allocator for the target device.
  std::shared_ptr<Allocator> allocator = AllocatorFactory::get(device_);
  size_t size_in_bytes = total_size * DtypeToSize(dtype_);

  // 4. Allocate the final memory for the tensor's data on the target device.
  void *raw_data_ptr = allocator->allocate(size_in_bytes);
  if (!raw_data_ptr) {
    throw std::runtime_error("Memory allocation failed for tensor data.");
  }
  data_ptr_ = std::shared_ptr<void>(
      raw_data_ptr, [allocator](void *ptr) { allocator->deallocate(ptr); });

  // 5. Allocate memory for the gradient if required.
  if (requires_grad_) {
    void *raw_grad_ptr = allocator->allocate(size_in_bytes);
    if (!raw_grad_ptr) {
      throw std::runtime_error("Memory allocation failed for gradient.");
    }
    grad_ = std::shared_ptr<void>(
        raw_grad_ptr, [allocator](void *ptr) { allocator->deallocate(ptr); });
  }

  // 6. Populate the tensor with data from the Python list.
  if (dtype_ == DType::float32) {
    // Step A: Create a temporary buffer on the HOST (CPU) memory.
    std::vector<float> temp_host_buffer(total_size);

    // Step B: Fill this host buffer by "flattening" the nested Python list.
    // This requires the `flatten_list` helper function.
    this->flatten_list(data, temp_host_buffer.data());

    // Step C: Copy the data from the prepared host buffer to the final
    // destination.
    if (device_.type == DeviceType::CPU) {
      // Destination is CPU memory, so use memcpy.
      std::memcpy(data_ptr_.get(), temp_host_buffer.data(), size_in_bytes);
    } else if (device_.type == DeviceType::CUDA) {
      // Destination is GPU memory, so use cudaMemcpy.
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

void *Tensor::raw_ptr() const {
  return static_cast<void *>(static_cast<char *>(data_ptr_.get()) +
                             offset_ * DtypeToSize(dtype_));
}

size_t Tensor::numel() const {
  // A tensor with an empty shape is a scalar, which has 1 element.
  if (this->shape_.empty()) {
    return 1;
  }

  // For non-scalar tensors, multiply the dimensions.
  // This correctly returns 0 if any dimension is 0.
  return std::accumulate(shape_.begin(), shape_.end(), 1LL,
                         std::multiplies<__int64_t>());
}

void Tensor::fill_helper(py::list &output, size_t depth,
                         std::vector<size_t> &indices) const {
  // Use raw_ptr() which correctly accounts for the tensor's offset.
  // All calculated indices will be relative to this pointer.
  float *data = static_cast<float *>(this->raw_ptr());

  // Base case: We are at the last dimension of the tensor.
  // We should now fill the 'output' list with the actual numbers.
  if (depth == shape_.size() - 1) {
    for (size_t i = 0; i < shape_[depth]; ++i) {
      indices[depth] = i;
      size_t data_idx = 0;
      // Calculate the final linear index using the full multidimensional index
      // and strides.
      for (size_t d = 0; d < shape_.size(); ++d) {
        data_idx += indices[d] * strides_[d];
      }
      output.append(data[data_idx]);
    }
  }
  // Recursive case: We are not at the last dimension yet.
  // We need to create a nested list for the next dimension.
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
  // Handle 0-dimensional (scalar) tensor as a special case.
  if (shape_.empty()) {
    if (numel() == 1) {
      float *data = static_cast<float *>(this->raw_ptr());
      output.append(data[0]);
    }
    return;
  }

  // For other tensors, start the recursion.
  std::vector<size_t> indices(shape_.size(), 0);
  fill_helper(output, 0, indices);
}

void Tensor::fill_ptr_helper(const py::list &list, size_t depth,
                             std::vector<size_t> &indices) {
  // Use raw_ptr() which correctly accounts for the tensor's offset.
  float *data = static_cast<float *>(this->raw_ptr());

  // Base case: We are at the last dimension.
  // The 'list' parameter should be a flat list of numbers.
  if (depth == shape_.size() - 1) {
    if (static_cast<size_t>(shape_[depth]) != list.size()) {
      throw std::runtime_error("List size does not match shape at depth " +
                               std::to_string(depth));
    }
    for (size_t i = 0; i < shape_[depth]; ++i) {
      indices[depth] = i;
      size_t data_idx = 0;
      // Calculate the final linear index from the full multidimensional index
      // and strides.
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
  // Recursive case: Traverse the nested lists.
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

void Tensor::fill_ptr(const py::list &list) {
  // Handle 0-dimensional (scalar) tensor as a special case.
  if (shape_.empty()) {
    if (list.size() != 1) {
      // For a scalar, we expect the input to be a list with a single element,
      // e.g., [5.0]
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
  // For other tensors, start the recursion.
  else {
    std::vector<size_t> indices(shape_.size(), 0);
    fill_ptr_helper(list, 0, indices);
  }
}

template <typename T>
void flatten_list_recursive(const py::list &list, T *&ptr) {
  for (const auto &item : list) {
    if (py::isinstance<py::list>(item)) {
      flatten_list_recursive(item.cast<py::list>(), ptr);
    } else {
      // Cast, write to the current pointer location, and then advance the
      // pointer
      *ptr = item.cast<T>();
      ptr++;
    }
  }
}

template <typename T>
void Tensor::flatten_list(const py::list &data, T *ptr) {
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

  // --- Pre-computation Step ---
  // First, find the ellipsis and count new axes
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

  // Number of dimensions the Ellipsis needs to expand into
  int num_ellipsis_dims = 0;
  if (ellipsis_pos != -1) {
    // The number of strategies that are NOT Ellipsis or NewAxis must match
    // the number of dimensions they are applied to.
    int non_special_strategies = strategies.size() - 1 - num_new_axes;
    if (non_special_strategies > shape_.size()) {
      throw std::out_of_range("Too many indices for tensor");
    }
    num_ellipsis_dims = shape_.size() - non_special_strategies;
  } else {
    // No ellipsis, so number of "real" indices must be <= number of dims
    if (strategies.size() - num_new_axes > shape_.size()) {
      throw std::out_of_range("Too many indices for tensor");
    }
  }

  // --- Main Application Loop ---
  size_t dim_idx = 0;  // Tracks the current dimension of the *original* tensor
  for (const auto &strategy : strategies) {
    if (auto p = dynamic_cast<NewAxisIndex *>(strategy.get())) {
      // NewAxis adds a dimension without consuming an original one.
      p->apply(0, shape_, strides_, offset, new_shape, new_strides);
    } else if (auto p = dynamic_cast<EllipsisIndex *>(strategy.get())) {
      // Ellipsis expands into N full slices.
      FullSlice full_slice_strategy;
      for (int k = 0; k < num_ellipsis_dims; ++k) {
        if (dim_idx >= shape_.size())
          break;  // Should not happen with correct logic
        full_slice_strategy.apply(dim_idx, shape_, strides_, offset, new_shape,
                                  new_strides);
        dim_idx++;
      }
    } else {
      // Regular index (Integer, Slice) that consumes a dimension.
      if (dim_idx >= shape_.size()) {
        throw std::out_of_range("Too many indices for tensor");
      }
      strategy->apply(dim_idx, shape_, strides_, offset, new_shape,
                      new_strides);
      dim_idx++;
    }
  }

  // Handle any remaining, un-indexed dimensions (implicit trailing full slices)
  while (dim_idx < shape_.size()) {
    new_shape.push_back(shape_[dim_idx]);
    new_strides.push_back(strides_[dim_idx]);
    dim_idx++;
  }

  return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_, offset,
                requires_grad_, grad_);
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
}

Tensor Tensor::permute(const std::vector<int> &order) {
  if (order.size() != shape_.size()) {
    throw std::invalid_argument(
        "permute(): `order` must have the same number of dimensions as tensor "
        "shape. "
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
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
                requires_grad_, grad_);
}

Tensor Tensor::add(const Tensor &other) const {
  // Here we use the add based on the backend used
  const Tensor &t = *this;
  if (device_.type == DeviceType::CPU) {
    return add_cpu(t, other);
  } else {
    return add_gpu(t, other);
  }
}

Tensor Tensor::sub(const Tensor &other) const {
  const Tensor &t = *this;
  if (device_.type == DeviceType::CPU) {
    return sub_cpu(t, other);
  } else {
    return sub_gpu(t, other);
  }
}

Tensor Tensor::mul(float b) const {
  const Tensor &t = *this;
  if (device_.type == DeviceType::CPU) {
    return mul_cpu(t, b);
  } else {
    return mul_gpu(t, b);
  }
}

Tensor Tensor::matmul(const Tensor &other) const {
  const Tensor &t = *this;
  if (device_.type == DeviceType::CPU) {
    return matmul_cpu(t, other);
  } else {
    return matmul_gpu(t, other);
  }
}

template<typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    Union<T> a(*this);
    Union<T> b(other);
    return AddTrait<T>::apply(a, b);
}

template<typename T>
Tensor<T> Tensor<T>::sub(const Tensor<T>& other) const {
    Union<T> a(*this);
    Union<T> b(other);
    return SubTrait<T>::apply(a, b);
}

template<typename T>
Tensor<T> Tensor<T>::mul(T scalar) const {
    Union<T> a(*this);
    Union<T> b(scalar);
    return MulTrait<T>::apply(a, b);
}

template<typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    Union<T> a(*this);
    Union<T> b(other);
    return MatMulTrait<T>::apply(a, b);
}


Tensor::~Tensor() {}
