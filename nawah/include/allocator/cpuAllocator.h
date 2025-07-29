#pragma once

#include "allocator.h"
#include <stdexcept>

struct CPUAllocator : public Allocator
{
    void *allocate(size_t bytes) override
    {
        void *ptr = malloc(bytes);
        if (!ptr)
            throw std::runtime_error("CPU Malloc Failed");
        return ptr;
    }

    void deallocate(void *ptr) override
    {
        free(ptr);
    }
};