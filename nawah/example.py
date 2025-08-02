import nawah as nw

t = nw.Tensor([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]], device="cpu")
t1 = nw.Tensor([[3.0 ,4.0, 5.0], [3.0, 4.0, 5.0]], device="cpu")
t3 = nw.Tensor([[6.0 ,7.0 ,8.0], [6.0, 7.0, 7.0]], device="cpu")

@nw.pipe
def add_minus(x, n):
    return x - t1 + t3 - n

def add_minus_with_no_arguments(x):
    return x - t1 + t3

t2 = t >> add_minus_with_no_arguments
t4 = t >> add_minus(t3)


print(t2)
print(t4)
