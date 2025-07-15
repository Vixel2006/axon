import nawah

t = nawah.Tensor([2, 3, 9], nawah.DType.float32, device='cpu', requires_grad=True)
print(f"Shape before broadcast: {t.shape}")
print(f"Strides before broadcast: {t.strides}")
print(t[:, 1, 1:2])
t = t.flatten(start=0, end=-2)
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")
