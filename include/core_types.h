#ifndef AXON_CORE_TYPES_H
#define AXON_CORE_TYPES_H

#include <stdbool.h>

typedef enum
{
    CPU,
    CUDA
} DeviceType;

typedef struct
{
    float* data;
    int size;
    int counter;
} Storage;

typedef struct
{
    DeviceType type;
    int index;
    int counter;
} Device;

#endif // AXON_CORE_TYPES_H
