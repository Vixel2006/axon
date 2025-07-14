import nawah

t = nawah.Tensor([2, 3, 9], nawah.DType.float32, nawah.Device(nawah.DeviceType.CUDA, 0))
print(f"Shape before broadcast: {t.shape}")
print(f"Strides before broadcast: {t.strides}")
t = t.flatten(start=0, end=-2)
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")
print(t[:, 1, 1:2])
