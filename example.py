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

print(nw.conv2d(inp, kernel))


print(t.shape)

t1 = (t+n).mean()

t1.backward()

print(n.grad)
print(t.grad)
print(t1)

net = nw.Net()

x = nw.Tensor([1, 128])

net.add("fc1", nw.layers.linear(128, 256))
net.add("relu1", nw.activations.relu())
net.add("fc2", nw.layers.linear(256, 512))

print(net)
net.summary([1, 128])

print("Registered params")
print(net.params.keys())

