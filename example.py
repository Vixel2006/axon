import neura

t = neura.Tensor([1, 2, 3], neura.DType.float32, neura.Device(neura.DeviceType.CUDA, 0))
print(f"Shape before expand: {t.shape}")
print(f"Strides before expand: {t.strides}")
t = t.unsqueeze(1)
t = t.expand([1, 4, 2, 3])
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")
