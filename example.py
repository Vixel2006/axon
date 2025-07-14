import neura

t = neura.Tensor([1, 2, 3], neura.DType.float32, neura.Device(neura.DeviceType.CUDA, 0))
print(t.shape)
t = t.transpose(-2, -1)
print(t.shape)
