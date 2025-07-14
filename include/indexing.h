#pragma once

#ifndef INDEXING_H
#define INDEXING_H

#include <memory>
#include <vector>
#include <stdexcept>

struct Slice
{
    __int64_t start;
    __int64_t end;
    __int64_t step;
};

struct Index
{
    enum Type
    {
        INT,
        SLICE
    } type;

    union
    {
        Slice slice_index;
        __int64_t int_index;
    };

    static Index from_int(__int64_t i)
    {
        Index idx;
        idx.type = INT;
        idx.int_index = i;
        return idx;
    }

    static Index from_slice(__int64_t start, __int64_t end, __int64_t step)
    {
        Index idx;
        idx.type = SLICE;
        idx.slice_index = {start, end, step};
        return idx;
    }
};

class IndexStrategy
{
public:
    virtual ~IndexStrategy() = default;
    virtual void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const = 0;
};

class IntegerIndex : public IndexStrategy
{
public:
    explicit IntegerIndex(__int64_t index) : index_(index) {}

    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        __int64_t idx = index_;

        if (idx < 0)
            idx += shape[dim];
        if (idx < 0 || idx >= shape[dim])
            throw std::out_of_range("Index out of bounds");
        offset += strides[dim] * idx;
    }

private:
    __int64_t index_;
};

class SliceIndex : public IndexStrategy
{
public:
    SliceIndex(__int64_t start, __int64_t end, __int64_t step = 1)
        : start_(start), end_(end), step_(step) {}

    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        __int64_t size = shape[dim];

        __int64_t start = start_ < 0 ? start_ + size : start_;
        __int64_t end = end_ < 0 ? end_ + size : end_;

        if (start < 0)
            start = 0;
        if (end > size)
            end = size;

        __int64_t len = (end - start + step_ - 1) / step_;
        new_shape.push_back(len);
        new_strides.push_back(strides[dim] * step_);
        offset += strides[dim] * start;
    }

private:
    __int64_t start_, end_, step_;
};

class FullSlice : public IndexStrategy
{
public:
    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        new_shape.push_back(shape[dim]);
        new_strides.push_back(strides[dim]);
    }
};

#endif