#ifndef NAWAH_SHARED_PTR
#define NAWAH_SHARED_PTR

typedef enum { FLOAT32, FLOAT64, INT32 } Dtype;

typedef enum { CPU, CUDA } Device;

typedef struct {
    void* elems;
    unsigned int ref_counter;
    Dtype dtype;
    Device device;
} shared_ptr;

shared_ptr* palloc(void* data, int size, Dtype dtype, Device device);
void pfree(shared_ptr* ptr);

#endif
