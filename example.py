import nawah_api as nw
import time

a = nw.Tensor([[[1,3,4], [3,4,5], [3,4,5]]], device="cuda:0", requires_grad=True)
b = nw.Tensor([[[1,3,4]]], device="cuda:0", requires_grad=True)

c = a * b
print("---------------------")
print(c)
print("---------------------")
c.backward()

print('----------------------')
print(a.grad)
print('----------------------')
print('----------------------')
print(b.grad)
print('----------------------')

inp = nw.Tensor(
    [[[ [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9] ]]],
    requires_grad=True,
    device="cpu"
)

net = nw.Net()

net.add("conv2d_first", nw.layers.conv2d(in_channels=1, out_channels=3, kernel_size=(2,2)))
net.add("relu1", nw.activations.relu())
net.add("Flatten", nw.layers.flatten())
net.add("fc1", nw.layers.linear(12, 1))

pred = net(inp)

print("Prediction:")
print(pred)

net.summary([3, 1, 3, 3])


print("Registered params")

print(net.params)

truth = nw.ones([1,1], requires_grad=True)


loss = truth - pred

loss.backward()

nw.SGD(net.params.values(), lr=0.1)
print("-------------------------------------------------")
print(net.params)

