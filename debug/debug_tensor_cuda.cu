#include "tensor.h"
#include "device_management.h"
#include <stdio.h>

int main() {
    // Initialize CUDA device
    if (axon_init_cuda_device() != 0) {
        fprintf(stderr, "Failed to initialize CUDA device.\n");
        return 1;
    }
    printf("CUDA device initialized successfully.\n");

    // Create a tensor on CUDA device
    int shape[] = {2, 3};
    int ndim = 2;
    axon_tensor_t* tensor = axon_tensor_create(shape, ndim, AXON_DTYPE_FLOAT32, AXON_DEVICE_CUDA);

    if (tensor == NULL) {
        fprintf(stderr, "Failed to create tensor on CUDA device.\n");
        axon_shutdown_cuda_device();
        return 1;
    }

    printf("Tensor created on CUDA device.\n");
    printf("Tensor address: %p\n", (void*)tensor);
    printf("Tensor data address: %p\n", (void*)tensor->data);
    printf("Tensor ndim: %d\n", tensor->ndim);
    printf("Tensor shape: [");
    for (int i = 0; i < tensor->ndim; ++i) {
        printf("%d%s", tensor->shape[i], (i == tensor->ndim - 1) ? "" : ", ");
    }
    printf("]\n");
    printf("Tensor dtype: %d (AXON_DTYPE_FLOAT32)\n", tensor->dtype);
    printf("Tensor device: %d (AXON_DEVICE_CUDA)\n", tensor->device);

    // Clean up
    axon_tensor_free(tensor);
    printf("Tensor freed.\n");

    axon_shutdown_cuda_device();
    printf("CUDA device shut down.\n");

    return 0;
}
