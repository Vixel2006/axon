#ifndef DEVICE_H
#define DEVICE_H

enum class DeviceType { CPU, CUDA };

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};

#endif
