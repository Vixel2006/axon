import nawah

t = nawah.Tensor([2, 3, 9], nawah.DType.float32, device="cpu", requires_grad=True)
print(f"Shape before broadcast: {t.shape}")
print(f"Strides before broadcast: {t.strides}")
print(t[1, 1, 1])
t = t.unsqueeze(dim=1).expand([2, 5, 3, 9])
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")

t1 = nawah.Tensor(
    [
        [
            [[3, 4, 5, 4], [3, 4, 5, 4]],
            [[1, 2, 3, 4], [3, 4, 5, 4]],
        ]
    ],
    device="cuda:0",
)
print(t1)
