import nawah_api as nw
import time

t = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], requires_grad=True, device="cuda:0")
t1 = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], device="cuda:0")
t3 = nw.Tensor([[6.0 ,7.0 ,8.0], [6.0, 7.0, 7.0]], device="cpu")

t2 = t1 + t

print(t2)

t_cpu = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cpu")
t_cuda = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cuda:0")

start = time.perf_counter()
n = t_cpu >> nw.pow(2)
end = time.perf_counter()
print(f"CPU: {end - start}")

start = time.perf_counter()
n = t_cuda >> nw.pow(2)
end = time.perf_counter()
print(f"CUDA: {end - start}")

"""
@nw.pipe
def add_minus(x, n):
    return x - t1 + t3 - n

def add_minus_with_no_arguments(x):
    return x - t1 + t3

t2 = t >> add_minus_with_no_arguments
t4 = t >> add_minus(t3)
"""




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

