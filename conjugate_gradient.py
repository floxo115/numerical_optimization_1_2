import line_search
from optimizer_base import OptimizerBase
from test_functions import TestFunction
from utils import MaxIterationsError
import numpy as np


class ConjugateGradientNonLinear(OptimizerBase):
    def __init__(self, f: TestFunction, alpha: float):
        super().__init__(f)
        self.alpha = alpha
        self.last_nabla_x = None
        self.last_d = None

    def step(self, x):
        raise NotImplementedError()

    def reset(self):
        self.last_d = None
        self.last_nabla_x = None


class ConjugateGradientFR(ConjugateGradientNonLinear):

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


class ConjugateGradientPR(ConjugateGradientNonLinear):
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
