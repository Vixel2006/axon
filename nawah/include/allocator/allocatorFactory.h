#pragma once

#include "cpuAllocator.h"
#include "cudaAllocator.h"
#include "device.h"
#include "allocator.h"
#include <memory>
#include <stdexcept>

struct AllocatorFactory
{
    static std::shared_ptr<Allocator> get(const Device &device);
};
