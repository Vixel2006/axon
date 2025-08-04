from ..core import Tensor
from ..api import pipe


def relu(x):
    return x.relu()

@pipe
def lrelu(x, leak):
    return x.relu(leak)
