#include "ops/matmul.h"

#include "helpers.h"
#include "tensor.h"

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 32

// Given a flat batch index, calculate the offset to the start of that 2D matrix
// slice.
float* get_data_ptr_for_batch(const Tensor& tensor, int64_t batch_idx) {
  const auto& shape = tensor.shape();
  const auto& strides = tensor.strides();
  const int dims = tensor.ndim();

  // We only iterate over batch dimensions (all except the last two)
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

void matmul_2d_kernel(const float* a, const float* b, float* c, int64_t M,
                      int64_t N, int64_t K, int64_t a_stride_m,
                      int64_t a_stride_k, int64_t b_stride_k,
                      int64_t b_stride_n, int64_t c_stride_m,
                      int64_t c_stride_n) {
  for (int64_t i0 = 0; i0 < M; i0 += BLOCK_SIZE_M) {
    for (int64_t j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
      for (int64_t k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
        for (int64_t i = i0; i < std::min(i0 + BLOCK_SIZE_M, M); ++i) {
          for (int64_t j = j0; j < std::min(j0 + BLOCK_SIZE_N, N); ++j) {
            float sum = 0.0f;

#pragma omp simd reduction(+ : sum)
            for (int64_t k = k0; k < std::min(k0 + BLOCK_SIZE_K, K); ++k) {
              sum += a[i * a_stride_m + k * a_stride_k] *
                     b[k * b_stride_k + j * b_stride_n];
            }

            if (k0 == 0) {
              c[i * c_stride_m + j * c_stride_n] = sum;
            } else {
              c[i * c_stride_m + j * c_stride_n] += sum;
            }
          }
        }
      }
    }
  }
}

Tensor matmul_cpu(const Tensor& a, const Tensor& b) {
  // Here we do the matmul for cpu
  // First of all we need to do check
  // We get the two final dims from both tensors, if a.shape_[-1] !=
  // b.shape_[-2] then we can't do matmul we also need to broadcast if needed
  // (here we try to broadcast the tensors if anyone needs). then we do the
  // matmul, we do row * col so we can do the #pragma opm simd to do the simd
  // calculation for [[1,3,4], [1,3,4]] @ [[1, 2], [1,2], [1,2]] we loop for i,
  // j, k -> k is the common axis but now we need to know how to do this using
  // shape_, strides_, offset_ we can loop for i, j, k and say A[strides[0] * i
  // + strides[1] * j + offset] * B[strides[0] * j + strides[1] * k + offset] =
  // C[i * strides[0], k * strides[1]] we need to generalize for any number of
  // dimensions so we need to do the above calc for each other dim. So how to
  // loop over the other dims (in the most effiecent way possible to use in the
  // very high Performance DL library we're working on.)
  if (a.ndim() < 2 || b.ndim() < 2) {
    throw std::runtime_error("Matmul requires at least 2 dimensions.");
  }
  const int64_t K_a = a.shape().back();
  const int64_t K_b = b.shape()[b.ndim() - 2];
  if (K_a != K_b) {
    throw std::runtime_error("Inner dimensions for matmul do not match.");
  }

  // 2. Compute Output Shape via Broadcasting
  std::vector<int64_t> c_shape = compute_broadcast_matmul_shape(a, b);
  Tensor c = Tensor(c_shape, a.dtype());

  const int c_dims = c.shape().size();
  const int64_t M = c.shape()[c.ndim() - 2];
  const int64_t N = c.shape()[c.ndim() - 1];
  const int64_t K = K_a;

  std::vector<int64_t> batch_shape(c_shape.begin(), c_shape.end() - 2);

  // Target shape for 'a' is (batch_dims..., M, K)
  std::vector<int64_t> a_target_shape = batch_shape;
  a_target_shape.push_back(M);
  a_target_shape.push_back(K);

  // Target shape for 'b' is (batch_dims..., K, N)
  std::vector<int64_t> b_target_shape = batch_shape;
  b_target_shape.push_back(K);
  b_target_shape.push_back(N);

  // Use the .broadcast() method which is designed for this.
  Tensor a_exp = a.broadcast(a_target_shape);
  Tensor b_exp = b.broadcast(b_target_shape);

  int64_t batch_count = 1;
  for (size_t i = 0; i < c_dims - 2; ++i) {
    batch_count *= c.shape()[i];
  }

#pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
    float* a_ptr = get_data_ptr_for_batch(a_exp, batch_idx);
    float* b_ptr = get_data_ptr_for_batch(b_exp, batch_idx);
    float* c_ptr = get_data_ptr_for_batch(c, batch_idx);

    // Get strides for the actual 2D matrices
    const int64_t a_stride_m = a_exp.strides()[a_exp.ndim() - 2];
    const int64_t a_stride_k = a_exp.strides()[a_exp.ndim() - 1];
    const int64_t b_stride_k = b_exp.strides()[b_exp.ndim() - 2];
    const int64_t b_stride_n = b_exp.strides()[b_exp.ndim() - 1];
    const int64_t c_stride_m = c.strides()[c.ndim() - 2];
    const int64_t c_stride_n = c.strides()[c.ndim() - 1];

    // 5. Call the optimized 2D kernel
    matmul_2d_kernel(a_ptr, b_ptr, c_ptr, M, N, K, a_stride_m, a_stride_k,
                     b_stride_k, b_stride_n, c_stride_m, c_stride_n);
  }

  return c;
}