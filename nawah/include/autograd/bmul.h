#include <vector>

#ifndef NAWAH_BACKWARD_MUL_H
#define NAWAH_BACKWARD_MUL_H

class Tensor;

void backward_mul_cpu(const Tensor& out, std::vector<Tensor>& prev);
void backward_mul_gpu(const Tensor& out, std::vector<Tensor>& prev);

#endif
