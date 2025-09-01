#include "utils.h"

int get_num_batches(const int *shape, int ndim) {
  int batch_nums = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    batch_nums *= shape[i];
  }

  return batch_nums;
}

void im2row(const float *im, float *row, int N, int C, int H, int W, int Kh,
            int Kw, int Sh, int Sw, int Hout, int Wout, int padding) {
  for (int n = 0; n < N; ++n) {
    for (int oh = 0; oh < Hout; ++oh) {
      for (int ow = 0; ow < Wout; ++ow) {
        int row_start_idx =
            n * Hout * Wout * C * Kh * Kw + (oh * Wout + ow) * C * Kh * Kw;

        int col_offset = 0;
        for (int c = 0; c < C; ++c) {
          for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
              int im_h = oh * Sh + kh - padding;
              int im_w = ow * Sw + kw - padding;

              if (im_h >= 0 && im_h < H && im_w >= 0 && im_w < W) {
                int im_index = n * C * H * W + c * H * W + im_h * W + im_w;
                row[row_start_idx + col_offset] = im[im_index];
              } else {
                row[row_start_idx + col_offset] = 0.0f;
              }
              ++col_offset;
            }
          }
        }
      }
    }
  }
}
