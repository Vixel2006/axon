#pragma once

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <complex>
#include <omp.h>
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

inline std::vector<__int64_t> reduce_shape(const std::vector<__int64_t> &shape, int dim, bool keepdim) {
  if (dim < 0 || dim >= static_cast<int>(shape.size())) {
    std::ostringstream oss;
    oss << "Invalid dimension " << dim << " for tensor of rank " << shape.size() << ".";
    throw std::runtime_error(oss.str());
  }

  std::vector<__int64_t> new_shape = shape;

  if (keepdim) {
    new_shape[dim] = 1;
  } else {
    new_shape.erase(new_shape.begin() + dim);
  }

  return new_shape;
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


inline int64_t next_power_of_2(int64_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

inline void fft(std::vector<std::complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return;
    std::vector<std::complex<double>> even(N/2), odd(N/2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i*2];
        odd[i] = x[i*2 + 1];
    }
    fft(even);
    fft(odd);
    for (int k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

inline void ifft(std::vector<std::complex<double>>& x) {
    int N = x.size();
    for (auto& val : x) val = std::conj(val);
    fft(x);
    for (auto& val : x) val = std::conj(val) / static_cast<double>(N);
}

inline void fft2d(std::vector<std::complex<double>>& data, int H, int W) {
    std::vector<std::complex<double>> row(W);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) row[j] = data[i * W + j];
        fft(row);
        for (int j = 0; j < W; ++j) data[i * W + j] = row[j];
    }
    std::vector<std::complex<double>> col(H);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) col[i] = data[i * W + j];
        fft(col);
        for (int i = 0; i < H; ++i) data[i * W + j] = col[i];
    }
}

inline void ifft2d(std::vector<std::complex<double>>& data, int H, int W) {
    std::vector<std::complex<double>> row(W);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) row[j] = data[i * W + j];
        ifft(row);
        for (int j = 0; j < W; ++j) data[i * W + j] = row[j];
    }
    std::vector<std::complex<double>> col(H);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) col[i] = data[i * W + j];
        ifft(col);
        for (int i = 0; i < H; ++i) data[i * W + j] = col[i];
    }
}


#endif
