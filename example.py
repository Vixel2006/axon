import nawah_api as nw
import time

t = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")
t1 = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")
t3 = nw.Tensor([[6.0 ,7.0 ,8.0], [6.0, 7.0, 7.0]], device="cpu")


t_cpu = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
t1_cpu = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
t4 = t_cpu * t1_cpu

print(t4)

t_cpu = nw.Tensor([[3.0, 4.0, -5.0], [-3.0, 4.0, 5.0]], device="cpu")
t_cuda = nw.Tensor([[3.0, 4.0, -5.0], [-3.0, 4.0, 5.0]], device="cuda:0")

start = time.perf_counter()
n = t_cpu >> nw.softmax
print(n)
end = time.perf_counter()
print(f"CPU: {end - start}")

start = time.perf_counter()
n = t_cuda >> nw.softmax
print(n)
end = time.perf_counter()
print(f"CUDA: {end - start}")

