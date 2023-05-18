from line_search import backtracking_line_search
from utils import MaxIterationsError
import numpy as np
from test_functions import TestFunction


class QuasiNewtonBFGS:
    def __init__(self, f: TestFunction, size_x: int, alpha):
        self.f = f
        self.size_x = size_x
        self.Q = np.eye(size_x)
        self.alpha = alpha

    # algorithm 3.2
    def step(self, x):
        Q = self.Q.copy()
        g = self.f.get_gradient(x)
        r = -Q @ g

        alpha = backtracking_line_search(self.f, x, r, self.alpha)
        x_cur = x + alpha * r
        g_cur = self.f.get_gradient(x_cur)

        s = x_cur - x
        y = g_cur - g

        I = np.eye(self.size_x)
        r = 1 / np.dot(y, s)
        le = (I - r * (np.outer(s, y)))
        ri = (I - r * (np.outer(y, s)))
        update = r * (np.outer(s, s))
        Q = le @ Q @ ri + update

        self.Q = Q

        return x_cur

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
        self.Q = np.eye(self.size_x)


class QuasiNewtonSR1:
    def __init__(self, f: TestFunction, size_x: int, alpha):
        self.f = f
        self.size_x = size_x
        self.H = np.eye(size_x)
        self.alpha = alpha

    def step(self, x):
        H = self.H.copy()
        g = self.f.get_gradient(x)
        r = -H @ g

        alpha = backtracking_line_search(self.f, x, r, self.alpha)
        x_cur = x + alpha * r
        g_cur = self.f.get_gradient(x_cur)

        s = x_cur - x
        y = g_cur - g

        # Reset is the only thing that worked when the denominator gets to small
        # checks from the book did not work
        if np.dot(s - H @ y, y) != 0.0:
            H += np.outer(s - H @ y, s - H @ y) / np.dot(s - H @ y, y)
            self.H = H
        else:
            self.reset()

        return x_cur

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
        self.H = np.eye(self.size_x)
