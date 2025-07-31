import nawah
import time

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
tensor1 = nawah.Tensor(data=[[1, 23, 4], [1, 3, 4]], requires_grad=True)
print(tensor1.grad)

tensor2 = nawah.Tensor(data=[[1, 3, 4], [2, 3, 54]], requires_grad=True)
tensor3 = tensor1 + tensor2
print(tensor3.ctx)
print(tensor3.ctx.op)
print(tensor3.ctx.prev[1])
print(tensor3.requires_grad)
print(tensor3.grad)
tensor4 = tensor1 - tensor2
print(tensor1 * tensor2)

t1 = nawah.Tensor([1000, 1000], device="cuda:0")
t2 = nawah.Tensor([1000, 1000], device="cuda:0")


def benchmark_addition(shape=(1000, 1000), runs=100):
    t_cpu1 = nawah.Tensor(shape, device="cpu")
    t_cpu2 = nawah.Tensor(shape, device="cpu")

    t_gpu1 = nawah.Tensor(shape, device="cuda:0")
    t_gpu2 = nawah.Tensor(shape, device="cuda:0")

    # Warm-up (important for fair timing)
    _ = t_cpu1 + t_cpu2
    _ = t_gpu1 + t_gpu2

    # CPU Benchmark
    start_cpu = time.perf_counter()
    for _ in range(runs):
        _ = t_cpu1 + t_cpu2
    end_cpu = time.perf_counter()

    # CUDA Benchmark (sync first!)
    nawah.cuda_synchronize()
    start_gpu = time.perf_counter()
    for _ in range(runs):
        _ = t_gpu1 - t_gpu2
    nawah.cuda_synchronize()
    end_gpu = time.perf_counter()

    print(f"CPU avg time (sub): {(end_cpu - start_cpu) / runs * 1000:.4f} ms")
    print(f"CUDA avg time (sub): {(end_gpu - start_gpu) / runs * 1000:.4f} ms")


benchmark_addition(shape=[1000, 1000])


def benchmark_mul(shape=(1000, 1000), runs=100):
    t_cpu1 = nawah.Tensor(shape, device="cpu")
    t_cpu2 = nawah.Tensor(shape, device="cpu")

    t_gpu1 = nawah.Tensor(shape, device="cuda:0")
    t_gpu2 = nawah.Tensor(shape, device="cuda:0")

    # Warm-up (important for fair timing)
    _ = t_cpu1 * t_cpu2
    _ = t_gpu1 * t_gpu2

    # CPU Benchmark
    start_cpu = time.perf_counter()
    for _ in range(runs):
        _ = t_cpu1 * t_cpu2
    end_cpu = time.perf_counter()

    # CUDA Benchmark (sync first!)
    nawah.cuda_synchronize()
    start_gpu = time.perf_counter()
    for _ in range(runs):
        _ = t_gpu1 * t_gpu2
    nawah.cuda_synchronize()
    end_gpu = time.perf_counter()

    print(f"CPU avg time (element-wise multplication): {(end_cpu - start_cpu) / runs * 1000:.4f} ms")
    print(f"CUDA avg time (element-wise multiplication): {(end_gpu - start_gpu) / runs * 1000:.4f} ms")


benchmark_mul()

"""
def benchmark_matmul(shape=(1000, 1000), runs=100):
    t_cpu1 = nawah.Tensor(shape, device="cpu")
    t_cpu2 = nawah.Tensor(shape, device="cpu")

    t_gpu1 = nawah.Tensor(shape, device="cuda:0")
    t_gpu2 = nawah.Tensor(shape, device="cuda:0")

    # Warm-up (important for fair timing)
    _ = t_cpu1 @ t_cpu2
    _ = t_gpu1 @ t_gpu2

    # CPU Benchmark
    start_cpu = time.perf_counter()
    for _ in range(runs):
        _ = t_cpu1 @ t_cpu2
    end_cpu = time.perf_counter()

    # CUDA Benchmark (sync first!)
    nawah.cuda_synchronize()
    start_gpu = time.perf_counter()
    for _ in range(runs):
        _ = t_gpu1 @ t_gpu2
    nawah.cuda_synchronize()
    end_gpu = time.perf_counter()

    print(f"CPU avg time (matmul): {(end_cpu - start_cpu) / runs * 1000:.4f} ms")
    print(f"CUDA avg time (matmul): {(end_gpu - start_gpu) / runs * 1000:.4f} ms")


benchmark_matmul(shape=[100, 100])
"""

n = nawah.Tensor(data=[[1, 2, 3]], device="cpu")
m = nawah.Tensor(data=[[1], [2], [3]], device="cpu")
print(n @ m)

"""
n = nawah.Tensor(data=[[1, 2, 3]], device="cuda:0")
m = nawah.Tensor(data=[[1], [2], [3]], device="cuda:0")
print(n @ m)
"""

k = nawah.Tensor(data=[[[1,3], [1,3]], [[3,4], [4,5]]])

print(k.sum(dim=-1, keepdim=False))


k = nawah.Tensor(data=[[[1,3], [1,3]], [[3,4], [4,5]]])

print(k.mean(dim=-1, keepdim=False))
