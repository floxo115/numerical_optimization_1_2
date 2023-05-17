import line_search
from test_functions import TestFunction
from utils import MaxIterationsError
import numpy as np


class ConjugateGradientFR():
    def __init__(self, f: TestFunction, alpha: float):
        self.f = f
        self.alpha = alpha
        self.last_nabla_x = None
        self.last_d = None

    def step(self, x):
        nabla_x = self.f.get_gradient(x)

        if self.last_d is not None:
            beta = np.dot(nabla_x, nabla_x) / np.dot(self.last_nabla_x, self.last_nabla_x)
            d = -nabla_x + beta * self.last_d
        else:
            d = -nabla_x

        self.last_nabla_x = nabla_x
        self.last_d = d

        alpha = line_search.backtracking_line_search(self.f, x, d, self.alpha)
        return x + alpha * d

    def run_optim(self, start: np.ndarray, stop_grad=10 ** -6, max_iterations=1000):
        k = 0
        x = start
        xs = [x]
        while True:
            if np.linalg.norm(self.f.get_gradient(x)) < stop_grad:
                break
            if k > max_iterations:
                raise MaxIterationsError()

            x = self.step(x)
            xs.append(x)

            k += 1

        xs.append(x)
        return xs

    def reset(self):
        self.last_d = None
        self.last_nabla_x = None


class ConjugateGradientPR():
    def __init__(self, f: TestFunction, alpha: float):
        self.f = f
        self.alpha = alpha
        self.last_nabla_x = None
        self.last_d = None

    def step(self, x):
        nabla_x = self.f.get_gradient(x)

        if self.last_d is not None:
            comp_beta = np.dot(nabla_x, nabla_x - self.last_nabla_x) / np.dot(self.last_nabla_x, self.last_nabla_x)
            beta = comp_beta
            d = -nabla_x + beta * self.last_d
        else:
            d = -nabla_x

        self.last_nabla_x = nabla_x
        self.last_d = d

        alpha = line_search.backtracking_line_search(self.f, x, d, self.alpha)
        return x + alpha * d

    def run_optim(self, start: np.ndarray, stop_grad=10 ** -6, max_iterations=1000):
        k = 0
        x = start
        xs = [x]
        while True:
            if np.linalg.norm(self.f.get_gradient(x)) < stop_grad:
                break
            if k > max_iterations:
                raise MaxIterationsError()

            x = self.step(x)
            xs.append(x)

            k += 1

        xs.append(x)
        return xs

    def reset(self):
        self.last_d = None
        self.last_nabla_x = None