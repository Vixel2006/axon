#ifndef STRIDED_INDEXER_H
#define STRIDED_INDEXER_H

#include <vector>
#include <numeric>
#include <stdexcept>

class StridedIndexer {
public:
   StridedIndexer(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides)
        : shape_(shape), strides_(strides) {
        
        if (shape.size() != strides.size()) {
            throw std::invalid_argument("Shape and strides must have the same number of dimensions.");
        }

        if (!shape.empty()) {
            place_values_.resize(shape.size());
            place_values_.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i) {
                place_values_[i] = place_values_[i + 1] * shape[i + 1];
            }
        }
    }

    size_t get_offset(size_t linear_index) const {
        size_t offset = 0;
        size_t remaining_index = linear_index;

        for (size_t i = 0; i < shape_.size(); ++i) {
            size_t coordinate = remaining_index / place_values_[i];
            
            remaining_index %= place_values_[i];
            
            offset += coordinate * strides_[i];
        }
        return offset;
    }

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> place_values_;
};

#endif 
