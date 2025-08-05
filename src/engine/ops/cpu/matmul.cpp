#include "engine/ops.h"
#include "helpers.h"
#include "utils.h"
#include "tensor.h"

#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <numeric>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 256
#define BLOCK_SIZE_K 64
#define REG_MR 6
#define REG_NR 16

void pack_a_tile(const float* a, float* pack_buffer, int M_tile, int K_tile, int64_t a_stride_m, int64_t a_stride_k) {
    for (int i = 0; i < M_tile; ++i) {
        for (int k = 0; k < K_tile; ++k) {
            pack_buffer[i * K_tile + k] = a[i * a_stride_m + k * a_stride_k];
        }
    }
}
void pack_b_tile(const float* b, float* pack_buffer, int K_tile, int N_tile, int64_t b_stride_k, int64_t b_stride_n) {
    for (int k = 0; k < K_tile; ++k) {
        for (int j = 0; j < N_tile; ++j) {
            pack_buffer[k * N_tile + j] = b[k * b_stride_k + j * b_stride_n];
        }
    }
}

void micro_kernel_6x16(const float* pack_a, const float* pack_b, float* c, int K, int64_t c_stride_m) {
    __m256 c_ymm[6][2];
    for (int i = 0; i < REG_MR; ++i) {
        c_ymm[i][0] = _mm256_setzero_ps();
        c_ymm[i][1] = _mm256_setzero_ps();
    }
    for (int k = 0; k < K; ++k) {
        const __m256 b_vec0 = _mm256_loadu_ps(pack_b + k * REG_NR + 0);
        const __m256 b_vec1 = _mm256_loadu_ps(pack_b + k * REG_NR + 8);
        for (int i = 0; i < REG_MR; ++i) {
            const __m256 a_broadcast = _mm256_broadcast_ss(pack_a + i * K + k);
            c_ymm[i][0] = _mm256_fmadd_ps(a_broadcast, b_vec0, c_ymm[i][0]);
            c_ymm[i][1] = _mm256_fmadd_ps(a_broadcast, b_vec1, c_ymm[i][1]);
        }
    }
    for (int i = 0; i < REG_MR; ++i) {
        _mm256_storeu_ps(&c[i * c_stride_m + 0], _mm256_add_ps(_mm256_loadu_ps(&c[i * c_stride_m + 0]), c_ymm[i][0]));
        _mm256_storeu_ps(&c[i * c_stride_m + 8], _mm256_add_ps(_mm256_loadu_ps(&c[i * c_stride_m + 8]), c_ymm[i][1]));
    }
}

void general_kernel(const float* a, const float* b, float* c,
                    int M, int N, int K,
                    int64_t a_stride_k_pack, int64_t b_stride_k_pack, int64_t c_stride_m) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * a_stride_k_pack + k] * b[k * b_stride_k_pack + j];
            }
            c[i * c_stride_m + j] += sum;
        }
    }
}


void matmul_2d_kernel_optimized(const float* a, const float* b, float* c, int64_t M,
                                int64_t N, int64_t K, int64_t a_stride_m,
                                int64_t a_stride_k, int64_t b_stride_k,
                                int64_t b_stride_n, int64_t c_stride_m,
                                int64_t c_stride_n) {
    if (c_stride_n != 1) {
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i * c_stride_m + j * c_stride_n] = 0.0f;
        }
    }

    #pragma omp parallel
    {
        std::vector<float> pack_a_buffer(BLOCK_SIZE_M * BLOCK_SIZE_K);
        std::vector<float> pack_b_buffer(BLOCK_SIZE_K * BLOCK_SIZE_N);
        std::vector<float> micro_b_panel(BLOCK_SIZE_K * REG_NR);

        #pragma omp for schedule(dynamic)
        for (int64_t i0 = 0; i0 < M; i0 += BLOCK_SIZE_M) {
            int64_t M_block = std::min(M, i0 + BLOCK_SIZE_M) - i0;

            for (int64_t k0 = 0; k0 < K; k0 += BLOCK_SIZE_K) {
                int64_t K_block = std::min(K, k0 + BLOCK_SIZE_K) - k0;
                
                pack_a_tile(&a[i0 * a_stride_m + k0 * a_stride_k], pack_a_buffer.data(), M_block, K_block, a_stride_m, a_stride_k);
                
                for (int64_t j0 = 0; j0 < N; j0 += BLOCK_SIZE_N) {
                    int64_t N_block = std::min(N, j0 + BLOCK_SIZE_N) - j0;

                    pack_b_tile(&b[k0 * b_stride_k + j0 * b_stride_n], pack_b_buffer.data(), K_block, N_block, b_stride_k, b_stride_n);

                    for (int64_t j_r = 0; j_r < N_block; j_r += REG_NR) {
                        int64_t N_rem = std::min((int64_t)REG_NR, N_block - j_r);

                        for (int64_t i_r = 0; i_r < M_block; i_r += REG_MR) {
                            int64_t M_rem = std::min((int64_t)REG_MR, M_block - i_r);
                            
                            // Pointer to the top-left of the current C tile
                            float* c_tile_ptr = &c[(i0 + i_r) * c_stride_m + (j0 + j_r)];
                            // Pointer to the top-left of the current A panel
                            const float* a_panel_ptr = &pack_a_buffer[i_r * K_block];
                            
                            if (M_rem == REG_MR && N_rem == REG_NR) {
                                for(int k_micro = 0; k_micro < K_block; ++k_micro) {
                                    for (int j_micro = 0; j_micro < REG_NR; ++j_micro) {
                                        micro_b_panel[k_micro * REG_NR + j_micro] = pack_b_buffer[k_micro * N_block + j_r + j_micro];
                                    }
                                }
                                micro_kernel_6x16(a_panel_ptr, micro_b_panel.data(), c_tile_ptr, K_block, c_stride_m);
                            } else {
                                general_kernel(a_panel_ptr, &pack_b_buffer[j_r], c_tile_ptr,
                                               M_rem, N_rem, K_block,
                                               K_block, N_block, c_stride_m);
                            }
                        }
                    }
                }
            }
        }
    }
}

Tensor CpuOps::matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) throw std::runtime_error("Matmul requires at least 2 dimensions.");
    const int64_t K_a = a.shape().back();
    const int64_t K_b = b.shape()[b.ndim() - 2];
    if (K_a != K_b) throw std::runtime_error("Inner dimensions for matmul do not match.");

    std::vector<int64_t> c_shape = compute_broadcast_matmul_shape(a, b);
    std::vector<int64_t> batch_shape(c_shape.begin(), c_shape.end() - 2);

    std::vector<int64_t> a_bcast_shape = batch_shape;
    a_bcast_shape.push_back(a.shape()[a.ndim() - 2]);
    a_bcast_shape.push_back(a.shape()[a.ndim() - 1]);

    std::vector<int64_t> b_bcast_shape = batch_shape;
    b_bcast_shape.push_back(b.shape()[b.ndim() - 2]);
    b_bcast_shape.push_back(b.shape()[b.ndim() - 1]);

    Tensor c = Tensor(c_shape, a.dtype());
    Tensor a_exp = a.broadcast(a_bcast_shape);
    Tensor b_exp = b.broadcast(b_bcast_shape);

    const int c_dims = c.shape().size();
    const int64_t M = c.shape()[c_dims - 2];
    const int64_t N = c.shape()[c_dims - 1];
    const int64_t K = K_a;

    int64_t batch_count = 1;
    for (size_t i = 0; i < c_dims - 2; ++i) {
        batch_count *= c.shape()[i];
    }

    for (int64_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        float* a_ptr = get_data_ptr_for_batch(a_exp, batch_idx);
        float* b_ptr = get_data_ptr_for_batch(b_exp, batch_idx);
        float* c_ptr = get_data_ptr_for_batch(c, batch_idx);

        const int64_t a_stride_m = a_exp.strides()[a_exp.ndim() - 2];
        const int64_t a_stride_k = a_exp.strides()[a_exp.ndim() - 1];
        const int64_t b_stride_k = b_exp.strides()[b_exp.ndim() - 2];
        const int64_t b_stride_n = b_exp.strides()[b_exp.ndim() - 1];
        const int64_t c_stride_m = c.strides()[c.ndim() - 2];
        const int64_t c_stride_n = c.strides()[c.ndim() - 1];

        matmul_2d_kernel_optimized(a_ptr, b_ptr, c_ptr, M, N, K, a_stride_m, a_stride_k,
                                   b_stride_k, b_stride_n, c_stride_m, c_stride_n);
    }

    return c;
}
