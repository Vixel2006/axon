import nawah

t = nawah.Tensor(data=[[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cpu")
print(t.view([3, 2]))
print(f"Shape before broadcast: {t.shape}")
print(f"Strides before broadcast: {t.strides}")
print(f"Shape after expand: {t.shape}")
print(f"Strides after expand: {t.strides}")

t1 = nawah.Tensor(
    [
        [[3, 4, 5, 4], [3, 4, 5, 4], [1, 2, 3, 4], [3, 4, 5, 4]],
    ],
    device="cpu",
)

print(t1)

t2 = nawah.Tensor(data=[[1.0], [2.0], [3.0]])  # shape (3, 1)
print(t2)
