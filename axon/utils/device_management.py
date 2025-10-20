from axon.axon_bindings.c_wrapper_functions import (
    c_count_devices,
    c_is_cuda_available,
    c_print_device_props,
    c_print_cuda_device_info,
    c_get_cuda_memory_info,
)

def cuda_device_count():
    return c_count_devices()

def is_cuda_available():
    return c_is_cuda_available()

def list_cuda_devices():
    c_print_device_props()

def cuda_device_info(device):
    c_print_cuda_device_info(device.index)

def cuda_memory_info(device):
    return c_get_cuda_memory_info(device.index)

