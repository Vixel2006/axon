import neura

t = neura.Tensor([2, 3], neura.DType.float32, neura.Device(neura.DeviceType.CUDA, 0))
print(t.shape)
t = t.unsqueeze(32)
print(t.shape)
