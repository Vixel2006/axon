import nawah_api as nw
import time

t = nw.Tensor([[1,3,4], [2,3,4]], requires_grad=True, device="cuda:0")
n = nw.Tensor([[1,3,4], [2,3,4]], requires_grad=True, device="cuda:0")

# Input tensor (shape 1×1×3×3)
inp = nw.Tensor(
    [[[ [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9] ]]],
    requires_grad=True,
    device="cpu"
)

# Kernel tensor (shape 1×1×2×2)
kernel = nw.Tensor(
    [[[ [1, 0],
        [0, -1] ]]],
    requires_grad=True,
    device="cpu"
)

out = nw.conv2d(inp, kernel)

out.backward()

print(inp.grad)
print(kernel.grad)

print(t.shape)

t1 = (t+n).mean()

t1.backward()

print(n.grad)
print(t.grad)
print(t1)

net = nw.Net()


net.add("conv2d_first", nw.layers.conv2d(in_channels=1, out_channels=3, kernel_size=(2,2))) # 1 * 3 * 2 * 2
net.add("relu1", nw.activations.relu())
net.add("Flatten", nw.layers.flatten())
net.add("fc2", nw.layers.linear(12, 1))

print(net(inp))
net.summary([1, 1, 3, 3])

print("Registered params")
print(net.params.keys())


