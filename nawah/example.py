import nawah_api as nw
import time

t = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cuda:0")
t1 = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], device="cuda:0")
t3 = nw.Tensor([[6.0 ,7.0 ,8.0], [6.0, 7.0, 7.0]], device="cpu")
print(t + t1)

@nw.pipe
def add_minus(x, n):
    return x - t1 + t3 - n

@nw.pipe
def add_minus_with_no_arguments(x):
    return x - t1 + t3

t2 = t >> add_minus_with_no_arguments()
t4 = t >> add_minus(t3)

print(t2)
print(t4)


tensor = nw.Tensor([[3,4,-5], [3,4,5]], requires_grad=True)
tensor1 = nw.Tensor([[4,5,6], [4,5,6]], requires_grad=True)

t8 = tensor1 * tensor

t8.backward()
print(tensor1.grad)
print(tensor.grad)

t5 = tensor >> nw.relu

print(t5)

t6 = tensor >> nw.log
print(t6)

t7 = tensor >> nw.exp
print(t7)

ten = t / 3
print(ten)


t_cpu = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cpu")
t_cuda = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cuda:0")
start = time.perf_counter()
n = t_cpu >> nw.softmax
end = time.perf_counter()
print(f"CPU: {end - start}")

start = time.perf_counter()
n = t_cuda >> nw.softmax
end = time.perf_counter()
print(f"CUDA: {end - start}")


print(tensor.relu())

t5 = t.view([3,2])
print(t5)
