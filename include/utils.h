#ifndef NAWAH_UTILS_H
#define NAWAH_UTILS_H

#include "tensor.h"

int get_num_batches(const int *shape, int ndim);

void im2row(const float *im, float *row, int N, int C, int H, int W, int Kh,
            int Kw, int Sh, int Sw, int Hout, int Wout, int padding);

void col2im(const float *col, float *im, int N, int C, int H, int W, int Kh,
            int Kw, int Sh, int Sw, int Hout, int Wout, int padding);

#endif
