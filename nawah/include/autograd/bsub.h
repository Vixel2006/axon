#include <vector>

#ifndef NAWAH_BACKWARD_SUB_H
#define NAWAH_BACKWARD_SUB_H

class Tensor;

void backward_sub_cpu(const Tensor& out, std::vector<Tensor>& prev);
void backward_sub_gpu(const Tensor& out, std::vector<Tensor>& prev);

#endif

