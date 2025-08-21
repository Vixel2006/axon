class Pipe:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _execute(self, first_arg):
        return self.func(first_arg, *self.args, **self.kwargs)


class Pipeable:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return Pipe(self.func, *args, **kwargs)


def pipe(func):
    return Pipeable(func)
