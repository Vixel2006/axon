#include "ops/matmul.h"
#include "tensor.h"
#include "utils.h"

#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 32



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

  // Parallelize the loop over batches
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