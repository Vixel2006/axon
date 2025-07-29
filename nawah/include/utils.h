#ifndef NAWAH_UTILS_H
#define NAWAH_UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

class Tensor;

inline std::vector<int64_t> compute_broadcast_matmul_shape(const Tensor& a, const Tensor& b) {
    // The matrix multiply dimensions
    const int64_t M = a.shape()[a.shape().size() - 2];
    const int64_t N = b.shape()[b.shape().size() - 1];

    // The batch dimensions (all dimensions except the last two)
    std::vector<int64_t> a_batch_shape(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch_shape(b.shape().begin(), b.shape().end() - 2);
    
    // Use a standard broadcasting algorithm for the batch dimensions
    const size_t max_len = std::max(a_batch_shape.size(), b_batch_shape.size());
    std::vector<int64_t> c_batch_shape(max_len);

    for (size_t i = 0; i < max_len; ++i) {
        int64_t dim_a = (i < a_batch_shape.size()) ? a_batch_shape[a_batch_shape.size() - 1 - i] : 1;
        int64_t dim_b = (i < b_batch_shape.size()) ? b_batch_shape[b_batch_shape.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Tensors are not broadcastable for matmul.");
        }
        c_batch_shape[max_len - 1 - i] = std::max(dim_a, dim_b);
    }
    
    // Append the matrix dimensions to the broadcasted batch dimensions
    c_batch_shape.push_back(M);
    c_batch_shape.push_back(N);
    
    return c_batch_shape;
}

inline float* get_data_ptr_for_batch(const Tensor& tensor, int64_t batch_idx) {
  const auto& shape = tensor.shape();
  const auto& strides = tensor.strides();
  const int dims = tensor.ndim();

  const int batch_dims = dims - 2;

  if (batch_dims <= 0) {
    // If there are no batch dims, the offset is always 0 relative to raw_ptr
    return static_cast<float*>(tensor.raw_ptr());
  }

  int64_t offset = 0;
  int64_t remaining_idx = batch_idx;

  // This loop converts the flat `batch_idx` into a multi-dimensional index
  // and calculates the corresponding offset using strides.
  for (int i = 0; i < batch_dims; ++i) {
    // Calculate how many elements are in the dimensions to the right of the
    // current one
    int64_t stride_for_coord_calc = 1;
    
    for (int j = i + 1; j < batch_dims; ++j) {
      stride_for_coord_calc *= shape[j];
    }
    
    // Calculate the coordinate for the current dimension 'i'
    int64_t coord = remaining_idx / stride_for_coord_calc;
    remaining_idx %= stride_for_coord_calc;

    // Add the contribution of this dimension to the total offset
    offset += coord * strides[i];
  }

  // Return the base pointer plus the calculated offset in elements
  return static_cast<float*>(tensor.raw_ptr()) + offset;
}

#endif
