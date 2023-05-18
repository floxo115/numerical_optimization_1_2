from test_functions import TestFunction
import numpy as np
from utils import MaxIterationsError

class OptimizerBase:
    def __init__(self, f: TestFunction):
        self.f = f

    def step(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

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

    def resed(self):
        raise NotImplementedError()