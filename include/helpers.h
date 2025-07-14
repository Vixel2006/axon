#pragma once

#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include<vector>

std::string shapeToString(const std::vector<__int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += std::to_string(shape[i]);
        if (i != shape.size() - 1) out += ", ";
    }
    out += "]";
    return out;
}

#endif