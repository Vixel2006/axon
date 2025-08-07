import nawah_api as nw
import time

net = nw.Net()

net.add("fc1", nw.layers.linear(128, 256))
net.add("relu1", nw.activations.relu())
net.add("fc2", nw.layers.linear(256, 512))

print(net)
net.summary([1, 128])

print("Registered params")
print(net.params.keys())
