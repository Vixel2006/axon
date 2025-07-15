#pragma once

#ifndef INIT_H
#define INIT_H
#include "tensor.h"
#include "helpers.h"

Tensor zeros(const std::vector<__int64_t> &shape)
{
    std::vector<__int64_t> strides = compute_strides_(shape);
}

Tensor zeros(const std::vector<__int64_t> &shape, bool requires_grad)
{
    std::vector<__int64_t> strides = compute_strides_(shape);
}

Tensor ones(const std::vector<__int64_t> &shape)
{
    std::vector<__int64_t> strides = compute_strides_(shape);
}

Tensor randn(const std::vector<__int64_t> &shape)
{
    std::vector<__int64_t> strides = compute_strides_(shape);
}

Tensor uniform(const std::vector<__int64_t> &shape)
{
    std::vector<__int64_t> strides = compute_strides_(shape);
}

#endif