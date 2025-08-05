#pragma once
#ifndef NAWAH_BACKEND_REGISTERY_H
#define NAWAH_BACKEND_REGISTERY_H
#include "engine/ops.h"
#include "autograd/ops.h"

Ops* get_cpu_ops();
Ops* get_gpu_ops();

#endif
