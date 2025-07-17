#pragma once

struct Allocator
{
    virtual void *allocate(size_t bytes) = 0;
    virtual void deallocate(void *ptr) = 0;
    virtual ~Allocator() = default;
};
