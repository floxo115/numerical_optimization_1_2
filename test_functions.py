import numpy as np
import torch


class TestFunction:
    def __init__(self, f):
        self.function = f

    def get_gradient(self, x:np.ndarray):
        raise NotImplementedError()

    def get_hessian(self, x:np.ndarray):
        raise NotImplementedError()


class TestFunctionTorch(TestFunction):
    def __init__(self, f):
        super().__init__(f)

    def get_function(self):
        return self.function

    def get_gradient(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return torch.func.grad(self.function)(x).numpy()

    def get_hessian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return torch.func.hessian(self.function)(x).numpy()


# https://en.wikipedia.org/wiki/Rosenbrock_function
rosenbrock_func =\
    test_func_1 = TestFunctionTorch(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
test_func_2 = TestFunctionTorch(lambda x: 150*(x[0]*x[1]) ** 2 + (0.5*x[0] + 2*x[1] - 2)**2)

