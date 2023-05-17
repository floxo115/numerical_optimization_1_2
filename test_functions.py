import numpy as np
import torch

from derivative_approximation import cdf_grad, cdf_hessian

START_POINTS = {
    "test_function_1": [np.array([1.2, 1.2]), np.array([-1.2, 1]), np.array([0.2, 0.8])],
    "test_function_2": [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])]
}


class TestFunction:
    def __init__(self, f):
        self.function = f

    def __call__(self, x: np.ndarray):
        raise NotImplementedError()

    def get_gradient(self, x: np.ndarray):
        raise NotImplementedError()

    def get_hessian(self, x: np.ndarray):
        raise NotImplementedError()


class TestFunctionTorch(TestFunction):
    def __init__(self, f):
        super().__init__(f)

    def __call__(self, x: np.ndarray):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return self.function(x).detach().numpy()

    def get_gradient(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return torch.func.grad(self.function)(x).numpy()

    def get_hessian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return torch.func.hessian(self.function)(x).numpy()


class TestFunctionApprox(TestFunction):
    def __init__(self, f):
        super().__init__(f)

    def __call__(self, x: np.ndarray):
        return self.function(x)

    def get_gradient(self, x):
        return cdf_grad(self.function, x)

    def get_hessian(self, x):
        return cdf_hessian(self.function, x)


# https://en.wikipedia.org/wiki/Rosenbrock_function
rosenbrock_func = \
    test_func_1_torch = TestFunctionTorch(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
test_func_2_torch = TestFunctionTorch(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2)

test_func_1_approx = TestFunctionApprox(lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
test_func_2_approx = TestFunctionApprox(lambda x: 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2)
