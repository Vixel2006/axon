#include <vector>

#ifndef NAWAH_BACKWARD_ADD_H
#define NAWAH_BACKWARD_ADD_H

class Tensor;

void backward_add_cpu(const Tensor& out, std::vector<Tensor>& prev);
void backward_add_gpu(const Tensor& out, std::vector<Tensor>& prev);

#endif
