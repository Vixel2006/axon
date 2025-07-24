#pragma once

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "tensor.h"


inline std::vector<__int64_t> compute_strides_(const std::vector<__int64_t> &shape)
{
    __int64_t acc = 1;
    std::vector<__int64_t> strides(shape.size());

    for (int i = shape.size() - 1; i >= 0; --i)
    {
        strides[i] = acc;
        acc *= shape[i];
    }

    return strides;
}

inline std::string shapeToString(const std::vector<__int64_t> &shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        out += std::to_string(shape[i]);
        if (i != shape.size() - 1)
            out += ", ";
    }
    out += "]";
    return out;
}

inline std::string dtypeToString(DType dtype)
{
    switch (dtype)
    {
    case DType::float16:
        return "float16";
    case DType::float32:
        return "float32";
    case DType::int8:
        return "int8";
    case DType::int32:
        return "int32";
    case DType::uint8:
        return "uint8";
    default:
        throw std::runtime_error("Unknown DType provided to dtypeToString");
    }
}

inline std::string deviceToString(const Device &device)
{
    switch (device.type)
    {
    case DeviceType::CPU:
        return "cpu";
    case DeviceType::CUDA:
        return "cuda:" + std::to_string(device.index);
    default:
        throw std::runtime_error("Unknown DeviceType provided to deviceToString");
    }
}

inline Device parse_device(const std::string &device_str)
{
    if (device_str == "cpu")
    {
        return {DeviceType::CPU, 0};
    }
    if (device_str.rfind("cuda:", 0) == 0)
    {
        try
        {
            std::string index_str = device_str.substr(5);
            if (index_str.empty())
            {
                throw std::invalid_argument("Device index is missing in '" + device_str + "'");
            }
            int index = std::stoi(index_str);
            return {DeviceType::CUDA, index};
        }
        catch (const std::invalid_argument &e)
        {
            throw std::invalid_argument("Invalid CUDA device format: '" + device_str + "'. Expected 'cuda:N'.");
        }
        catch (const std::out_of_range &e)
        {
            throw std::out_of_range("Device index out of range in '" + device_str + "'");
        }
    }

    throw std::invalid_argument("Unsupported device string: '" + device_str + "'. Use 'cpu' or 'cuda:N'.");
}

inline void cuda_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


#endif