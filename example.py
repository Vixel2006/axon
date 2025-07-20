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

print(t1[0, 1:3, :])

tensor_2d = nawah.Tensor(data=[[1, 2, 3], [4, 5, 6]])
tensor1 = nawah.Tensor(data=[[1, 23, 4], [1, 3, 4]])
tensor2 = nawah.Tensor(data=[[1, 3, 4], [2, 3, 54]])
print(tensor1.shape)
print(tensor2.shape)
tensor3 = tensor1 + tensor2
print(tensor3)
tensor4 = tensor1 - tensor2
print(tensor4)
print(tensor1 * 5)
