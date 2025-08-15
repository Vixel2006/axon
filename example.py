import nawah_api as nw
import time

a = nw.Tensor([[[1,3,4], [3,4,5], [3,4,5]]], device="cuda:0", requires_grad=True)
b = nw.Tensor([[[1,3,4]]], device="cpu", requires_grad=True)

b.to("cuda:0")

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
    device="cuda:0"
)

inpt = nw.Tensor([[1,3,4], [1,3,4]], requires_grad=True, device="cuda:0")

net = nw.Sequential()

net.add("conv2d_first", nw.layers.linear(3, 12))
net.add("relu1", nw.activations.relu())
net.add("fc1", nw.layers.linear(12, 1))

net.to("cuda:0")

net.summary([2,3])

pred = net(inpt)

print("Prediction:")
print(pred)



print("Registered params")

truth = nw.ones([1,1], requires_grad=True, device="cuda:0")


loss = truth - pred

loss.backward()

nw.SGD(net.params.values(), lr=0.1)
print("-------------------------------------------------")
print(net.params)

net1 = nw.Sequential({
    "Conv2D": nw.layers.conv2d(in_channels=1, out_channels=3, kernel_size=(2,2)),
    "ReLU": nw.activations.relu(),
    "Flatten": nw.layers.flatten(),
    "FC1": nw.layers.linear(12, 1)
})

net1.to("cuda:0")

net1.summary([3,1,3,3])

