#ifndef AXON_CUDA_UTILS_H
#define AXON_CUDA_UTILS_H

#include "logger.h"
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(msg)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA Error: %s: %s", msg, cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#endif // AXON_CUDA_UTILS_H
