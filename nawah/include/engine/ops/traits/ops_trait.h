#ifndef NAWAH_ENGINE_OPS_TRAITS_OPS_TRAIT_H
#define NAWAH_ENGINE_OPS_TRAITS_OPS_TRAIT_H

#include "tensor.h"


template<typename OpImpl>
struct OpTrait {
    static Tensor forward(const Tensor& a, const Tensor& b) {
        if (a.device().type == DeviceType::CPU) {
            return OpImpl::forward_cpu(a, b);
        } else if (a.device().type == DeviceType::CUDA) {
            return OpImpl::forward_gpu(a, b);
        }
    }
    static Tensor backward(const Tensor& a, const Tensor& b) {
        if (a.device().type == DeviceType::CPU) {
            return OpImpl::backward_cpu(a, b);
        } else if (a.device().type == DeviceType::CUDA) {
            return OpImpl::backward_gpu(a, b);
        }
    }
};

template<typename OpImpl>
struct ReductionTrait {
  static Tensor forward(const Tensor& a, int dim, bool keepdim) {
    if (a.device().type == DeviceType::CPU) {
      return OpImpl::forward_cpu(a, dim, keepdim);
    } else if (a.device().type == DeviceType::CUDA) {
      return OpImpl::forward_gpu(a, dim, keepdim);
    }
  }
};

#endif
