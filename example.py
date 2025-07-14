import neura

t = neura.Tensor([2, 3, 9], neura.DType.float32, neura.Device(neura.DeviceType.CUDA, 0))
print(f"Shape before broadcast: {t.shape}")
print(f"Strides before broadcast: {t.strides}")
t = t.flatten(start=0, end=-2)
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")
