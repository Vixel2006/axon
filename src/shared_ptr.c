#include "shared_ptr.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h> // Added for memcpy and memset

shared_ptr* palloc(void* data, int size, Dtype dtype, Device device) {
    shared_ptr* ptr = malloc(sizeof(shared_ptr));
    ptr->device = device;
    ptr->dtype = dtype;
    ptr->ref_counter = 1;

    size_t element_size = 0; // Declared once here
    void* allocated_elems = NULL;
    // Allocate memory for elems based on dtype and size
    if (device == CPU) {
        switch (dtype) {
            case FLOAT32: element_size = sizeof(float); allocated_elems = malloc(element_size * size); break;
            case FLOAT64: element_size = sizeof(double); allocated_elems = malloc(element_size * size); break;
            case INT32: element_size = sizeof(int); allocated_elems = malloc(element_size * size); break;
        }
    } else if (device == CUDA) {
        LOG_ERROR("CUDA not supported in this build.");
    }

    if (allocated_elems) {
        ptr->elems = allocated_elems;
        if (data != NULL) {
            // Copy data if provided
            memcpy(ptr->elems, data, size * element_size);
            LOG_INFO("Copied provided data to shared_ptr elems.");
        } else {
            // Initialize to zero if no data provided
            memset(ptr->elems, 0, size * element_size);
            LOG_INFO("Initialized shared_ptr elems to zero.");
        }
        LOG_INFO("Allocated CPU memory: %zu bytes, ptr->elems = %p", size * element_size, ptr->elems);
    } else {
        LOG_ERROR("Memory allocation failed for shared_ptr elems.");
        ptr->elems = NULL;
        ptr->ref_counter = 0;
    }

    return ptr;
}

void pfree(shared_ptr* ptr) {
    if (ptr == NULL) {
        return;
    }

    if (ptr->ref_counter > 0) { // Only decrement if > 0
        ptr->ref_counter--;
    }

    if (ptr->ref_counter == 0) { // Only free if ref_counter is 0
        LOG_INFO("Decremented ref_counter to %d, freeing memory", ptr->ref_counter);
        if (ptr->device == CPU) {
            if (ptr->elems) {
                free(ptr->elems);
                LOG_INFO("Freed CPU memory");
            }
        } else if (ptr->device == CUDA) {
            if (ptr->elems) {
                // cudaFree(ptr->>elems); // Removed CUDA specific free
                LOG_INFO("CUDA memory free not supported in this build.");
            }
        }
        LOG_INFO("Freed shared_ptr struct at %p", (void*)ptr);
        free(ptr); // Free the shared_ptr struct itself
    } else {
        LOG_INFO("Decremented ref_counter to %d, not freeing memory yet", ptr->ref_counter);
    }
}