#include "autograd/cpu/binary/common.h"

void conv2d_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras)
{
    LOG_INFO("conv2d_grad_op_cpu: Entering function with n_prev=%d", n_prev);

    Tensor* in = prev[0];
    Tensor* kernel = prev[1];

    BackwardConvExtras* conv_extras = (BackwardConvExtras*) extras;

    int N = in->shape[0];
    int Cin = in->shape[1];
    int Hin = conv_extras->H_in;
    int Win = conv_extras->W_in;
    int Cout = out->shape[1];
    int Kh = conv_extras->Kh;
    int Kw = conv_extras->Kw;
    int Sh = conv_extras->Sh;
    int Sw = conv_extras->Sw;
    int Hout = conv_extras->Hout;
    int Wout = conv_extras->Wout;
    int padding = conv_extras->padding;

    const int TILE_H = 16;
    const int TILE_W = 16;

    if (kernel->requires_grad)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H)
            {
                int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

                for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W)
                {
                    int ow_end = (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

                    for (int kh = 0; kh < Kh; ++kh)
                    {
                        for (int kw = 0; kw < Kw; ++kw)
                        {
                            for (int oh = oh_start; oh < oh_end; ++oh)
                            {
                                for (int ow = ow_start; ow < ow_end; ++ow)
                                {
                                    int ih = oh * Sh - padding + kh;
                                    int iw = ow * Sw - padding + kw;

                                    if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win)
                                    {
                                        for (int cout = 0; cout < Cout; ++cout)
                                        {
                                            float out_grad_val =
                                                out->grad->data
                                                    ->data[n * Cout * Hout * Wout +
                                                           cout * Hout * Wout + oh * Wout + ow];

                                            for (int cin = 0; cin < Cin; ++cin)
                                            {
                                                int in_idx = n * Cin * Hin * Win + cin * Hin * Win +
                                                             ih * Win + iw;
                                                int kernel_grad_idx = cout * Cin * Kh * Kw +
                                                                      cin * Kh * Kw + kh * Kw + kw;

                                                kernel->grad->data->data[kernel_grad_idx] +=
                                                    in->data->data[in_idx] * out_grad_val;
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
    }

    if (in->requires_grad)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int cout = 0; cout < Cout; ++cout)
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

                                        if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win)
                                        {
                                            float out_grad_val =
                                                out->grad->data
                                                    ->data[n * Cout * Hout * Wout +
                                                           cout * Hout * Wout + oh * Wout + ow];

                                            for (int cin = 0; cin < Cin; ++cin)
                                            {
                                                int kernel_idx = cout * Cin * Kh * Kw +
                                                                 cin * Kh * Kw + kh * Kw + kw;
                                                int in_grad_idx = n * Cin * Hin * Win +
                                                                  cin * Hin * Win + ih * Win + iw;

                                                in->grad->data->data[in_grad_idx] +=
                                                    kernel->data->data[kernel_idx] * out_grad_val;
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
    }

    LOG_INFO("conv2d_grad_op_cpu: Exiting function.");
}
