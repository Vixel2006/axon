import nawah_api as nw
import time

t = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")
t1 = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")
t3 = nw.Tensor([[6.0 ,7.0 ,8.0], [6.0, 7.0, 7.0]], device="cpu")
t5 = nw.Tensor([[3,4], [3,4], [3,4]], requires_grad=True, device="cuda:0")

t4 = t @ t5
print(t4)


t4.backward()
print(t.grad)
print(t5.grad)

t_cpu = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
t1_cpu = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
t4 = t_cpu * t1_cpu

print(t4)

t_cpu = nw.Tensor([[3.0, 4.0, -5.0], [-3.0, 4.0, 5.0]], requires_grad=True, device="cpu")
t_cuda = nw.Tensor([[3.0, 4.0, -5.0], [-3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")

start = time.perf_counter()
n = t_cpu >> nw.log
n.backward()
print(t_cpu.grad)
end = time.perf_counter()
print(f"CPU: {end - start}")

start = time.perf_counter()
n = t_cuda >> nw.log
n.backward()
print(t_cuda.grad)
print(n)
end = time.perf_counter()
print(f"CUDA: {end - start}")

