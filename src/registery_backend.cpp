#include "autograd/ops.h"
#include "backend_registery.h"
#include "engine/ops.h"

Ops *get_cpu_ops() {
  static CpuOps instance;
  return &instance;
}

Ops *get_gpu_ops() {
  static CudaOps instance;
  return &instance;
}
