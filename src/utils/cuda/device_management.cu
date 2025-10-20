#include "device_management.h"
#include "logger.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("cuda-runtime error at %s %d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
        }                                                                                          \
    } while (0)

int count_devices()
{
    int num_devices;
    cudaError_t err = cudaGetDeviceCount(&num_devices);
    CHECK_CUDA(err);
    return num_devices;
}

bool is_cuda_available()
{
    if (count_devices() == 0) return false;
    return true;
}

void print_device_props()
{
    int runtime_version = 0;
    int driver_version = 0;

    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    printf("CUDA Runtime Version: %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
    printf("CUDA Driver Version: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);

    int num_devices = count_devices();

    printf("CUDA devices:\n\n");

    for (int i = 0; i < num_devices; ++i)
    {
        cudaDeviceProp props;
        cudaError_t err = cudaGetDeviceProperties(&props, i);
        CHECK_CUDA(err);

        float device_memory_gb = (props.totalGlobalMem + props.sharedMemPerBlock +
                                  props.totalConstMem + props.regsPerBlock) /
                                 (1024.0f * 1024.0f * 1024.0f);

        printf("device %d: %s: %d.%d, total usable memory: %.2fGB\n", i, props.name, props.major,
               props.minor, device_memory_gb);
    }
}

void print_cuda_device_info(int index)
{
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, index);
    CHECK_CUDA(err);

    printf("\n=== CUDA Device %d: %s ===\n", index, props.name);

    auto get_attr = [&](cudaDeviceAttr attr)
    {
        int value;
        cudaDeviceGetAttribute(&value, attr, index);
        return value;
    };

    printf("Compute Capability: %d.%d\n", props.major, props.minor);

    float total_mem_gb = props.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    printf("Total Global Memory: %.2f GB\n", total_mem_gb);

    printf("Warp Size: %d\n", get_attr(cudaDevAttrWarpSize));
    printf("Max Threads per Block: %d\n", get_attr(cudaDevAttrMaxThreadsPerBlock));
    printf("MultiProcessor Count: %d\n", get_attr(cudaDevAttrMultiProcessorCount));

    // More detailed info using cudaDeviceAttr
    int l2_cache = get_attr(cudaDevAttrL2CacheSize);
    int bus_width = get_attr(cudaDevAttrGlobalMemoryBusWidth);
    int mem_clock = get_attr(cudaDevAttrMemoryClockRate);
    int clock_rate = get_attr(cudaDevAttrClockRate);

    printf("Core Clock Rate: %.2f GHz\n", clock_rate / 1e6f);
    printf("Memory Clock Rate: %.2f GHz\n", mem_clock / 1e6f);
    printf("Memory Bus Width: %d bits\n", bus_width);
    printf("L2 Cache Size: %.2f MB\n", l2_cache / (1024.0f * 1024.0f));

    float bandwidth = 2.0f * (mem_clock * 1e3f) * (bus_width / 8.0f) / 1e9f;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

    printf("Max Threads per Multiprocessor: %d\n",
           get_attr(cudaDevAttrMaxThreadsPerMultiProcessor));
    printf("Async Engine Count: %d\n", get_attr(cudaDevAttrAsyncEngineCount));
    printf("Unified Addressing: %s\n", get_attr(cudaDevAttrUnifiedAddressing) ? "Yes" : "No");

    printf("===============================\n\n");
}

void get_cuda_memory_info(int device_id)
{
    cudaError_t err = cudaSetDevice(device_id);
    CHECK_CUDA(err);

    size_t free_byte;
    size_t total_byte;
    err = cudaMemGetInfo(&free_byte, &total_byte);
    CHECK_CUDA(err);

    printf("Device %d: Free Memory: %.2f GB, Total Memory: %.2f GB\n", device_id,
           (double) free_byte / (1024.0 * 1024.0 * 1024.0),
           (double) total_byte / (1024.0 * 1024.0 * 1024.0));
}
