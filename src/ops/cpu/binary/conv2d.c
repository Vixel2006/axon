#include "ops/cpu/binary.h"

void conv2d_op_cpu(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                   const int* stride, const int padding)
{
    LOG_INFO("conv2d_op_cpu: Entering function");

    if (!check_tensors(in, kernel, out, "conv2d_op")) return;

    int Cin = kernel_size[0];
    int Cout = kernel_size[1];
    int Kh = kernel_size[2];
    int Kw = kernel_size[3];
    int Sh = stride[0];
    int Sw = stride[1];
    int H = in->shape[in->ndim - 2];
    int W = in->shape[in->ndim - 1];
    int Hout = (H + 2 * padding - Kh) / Sh + 1;
    int Wout = (W + 2 * padding - Kw) / Sw + 1;
    int N = in->shape[0];

    int out_size = N * Cout * Hout * Wout;
    float* data = alloc_tensor_data(out_size, "conv2d_op");
    if (!data) return;
    memset(data, 0, sizeof(float) * out_size);

    const int TILE_H = 16;
    const int TILE_W = 16;

    for (int n = 0; n < N; ++n)
    {
        for (int ic = 0; ic < Cin; ++ic)
        {
            for (int kh = 0; kh < Kh; ++kh)
            {
                for (int kw = 0; kw < Kw; ++kw)
                {
                    for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H)
                    {
                        int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

                        for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W)
                        {
                            int ow_end = (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

                            for (int oh = oh_start; oh < oh_end; ++oh)
                            {
                                for (int ow = ow_start; ow < ow_end; ++ow)
                                {
                                    int ih = oh * Sh - padding + kh;
                                    int iw = ow * Sw - padding + kw;

                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    {
                                        int in_idx = n * Cin * H * W + ic * H * W + ih * W + iw;
                                        float in_val = in->data->data[in_idx];

                                        for (int oc = 0; oc < Cout; ++oc)
                                        {
                                            int kernel_idx =
                                                oc * Cin * Kh * Kw + ic * Kh * Kw + kh * Kw + kw;
                                            int out_idx = n * Cout * Hout * Wout +
                                                          oc * Hout * Wout + oh * Wout + ow;
                                            data[out_idx] +=
                                                in_val * kernel->data->data[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    from_data(out, data);
    SAFE_FREE(&data, free);
}
