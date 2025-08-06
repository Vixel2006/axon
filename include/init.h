#pragma once

#ifndef INIT_H
#define INIT_H
#include "tensor.h"
#include "helpers.h"
#include "backend_registery.h"
#include "tensor.h"
#include "engine/ops.h"
#include "helpers.h"
#include <string>

Ops* get_ops_for_device(const std::string& device_str) {
    Device dev = parse_device(device_str);
    if (dev.type == DeviceType::CPU) {
        return get_cpu_ops();
    } else if (dev.type == DeviceType::CUDA) {
        return get_gpu_ops();
    }
    throw std::runtime_error("Unsupported device for factory function.");
}


Tensor zeros(const std::vector<__int64_t> &shape, const std::string& device = "cpu", bool requires_grad = false)
{
    Tensor t(shape, DType::float32, device, requires_grad);
    
    Ops* ops = get_ops_for_device(device);

    ops->fill_zeros(t);

    return t;
}


Tensor ones(const std::vector<__int64_t> &shape, const std::string& device = "cpu", bool requires_grad = false)
{
    Tensor t(shape, DType::float32, device, requires_grad);
    Ops* ops = get_ops_for_device(device);
    ops->fill_ones(t);
    return t;
}

Tensor randn(const std::vector<__int64_t> &shape, const std::string& device = "cpu", bool requires_grad = false)
{
    Tensor t(shape, DType::float32, device, requires_grad);
    Ops* ops = get_ops_for_device(device);
    ops->fill_randn(t);
    return t;
}

Tensor uniform(const std::vector<__int64_t> &shape, const std::string& device = "cpu", bool requires_grad = false)
{
    Tensor t(shape, DType::float32, device, requires_grad);
    Ops* ops = get_ops_for_device(device);
    ops->fill_uniform(t);
    return t;
}

Tensor zeros_like(const Tensor& other) {
    return zeros(other.shape(), deviceToString(other.device()), other.requires_grad());
}


Tensor ones_like(const Tensor& other) {
    return ones(other.shape(), deviceToString(other.device()), other.requires_grad());
}
#endif
