#pragma once

#include "cpuAllocator.h"
#include "cudaAllocator.h"
#include "device.h"
#include "allocator.h"

struct AllocatorFactory
{
    static std::shared_ptr<Allocator> get(const Device &device)
    {
        switch (device.type)
        {
        case DeviceType::CPU:
            static std::shared_ptr<Allocator> cpu_alloc = std::make_shared<CPUAllocator>();
            return cpu_alloc;
        case DeviceType::CUDA:
            static std::shared_ptr<Allocator> cuda_alloc = std::make_shared<CUDAAllocator>();
            return cuda_alloc;
        default:
            throw std::runtime_error("Unsupported device type.");
        };
    }
};
