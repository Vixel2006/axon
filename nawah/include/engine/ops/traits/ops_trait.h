#ifndef NAWAH_ENGINE_OPS_TRAITS_OPS_TRAIT_H
#define NAWAH_ENGINE_OPS_TRAITS_OPS_TRAIT_H

#include "tensor.h"


template<typename OpImpl>
struct OpTrait {
    static Tensor operation(const Tensor& a, const Tensor& b) {
        if (a.device().type == DeviceType::CPU) {
            return OpImpl::cpu(a, b);
        } else if (a.device().type == DeviceType::CUDA) {
            return OpImpl::gpu(a, b);
        }
    }
};

template<typename OpImpl>
struct ReductionTrait {
  static Tensor operation(const Tensor& a, int dim, bool keepdim) {
    if (a.device().type == DeviceType::CPU) {
      return OpImpl::cpu(a, dim, keepdim);
    } else if (a.device().type == DeviceType::CUDA) {
      return OpImpl::gpu(a, dim, keepdim);
    }
  }
};

#endif
