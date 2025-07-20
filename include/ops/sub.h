#ifndef NAWAH_SUB_H
#define NAWAH_SUB_H
#include <stdexcept>
#include <vector>

class Tensor;

Tensor sub_cpu(const Tensor& a, const Tensor& b);


#endif