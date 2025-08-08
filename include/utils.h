#ifndef NAWAH_UTILS_H
#define NAWAH_UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

class Tensor;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA Error in " #call " : ") + \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

inline std::vector<int64_t> compute_broadcast_matmul_shape(const Tensor& a, const Tensor& b) {
    const int64_t M = a.shape()[a.shape().size() - 2];
    const int64_t N = b.shape()[b.shape().size() - 1];

    std::vector<int64_t> a_batch_shape(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch_shape(b.shape().begin(), b.shape().end() - 2);
    
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
    return static_cast<float*>(tensor.raw_ptr());
  }

  int64_t offset = 0;
  int64_t remaining_idx = batch_idx;

  for (int i = 0; i < batch_dims; ++i) {
    int64_t stride_for_coord_calc = 1;
    
    for (int j = i + 1; j < batch_dims; ++j) {
      stride_for_coord_calc *= shape[j];
    }
    
    int64_t coord = remaining_idx / stride_for_coord_calc;
    remaining_idx %= stride_for_coord_calc;

    offset += coord * strides[i];
  }

  return static_cast<float*>(tensor.raw_ptr()) + offset;
}

inline std::shared_ptr<int64_t> copy_strides_to_device(const std::vector<int64_t>& strides) {
    if (strides.empty()) {
        return nullptr;
    }

    int64_t* d_ptr;
    size_t size_bytes = strides.size() * sizeof(int64_t);

    CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
    if (!d_ptr) {
        throw std::runtime_error("Failed to allocate device memory for strides.");
    }

    CUDA_CHECK(cudaMemcpy(d_ptr, strides.data(), size_bytes, cudaMemcpyHostToDevice));

    auto deleter = [](int64_t* ptr) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    };

    return std::shared_ptr<int64_t>(d_ptr, deleter);
}

#endif
