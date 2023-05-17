# Based on Algorithm 3.3 in Nocedal p. 51
import numpy as np

from line_search import backtracking_line_search
from test_functions import TestFunction
from utils import MaxIterationsError


class NewtonMethodWithModifiedHessian():
    def __init__(self, f: TestFunction):
        self.f = f

    # algorithm 3.2
    def step(self, x):
        grad_f = self.f.get_gradient(x)
        hessian_f = self.f.get_hessian(x)
        hessian_f_pos_def = self.make_pos_def(hessian_f)

        d = np.linalg.solve(hessian_f_pos_def, -grad_f)
        alpha = backtracking_line_search(self.f, x, d, 1, rho=0.5, beta=0.0001)
        return x + alpha * d

    @staticmethod
    # algorithm 3.3
    def make_pos_def(B: np.ndarray, beta=10 ** -5, factor=2):
        # if B is already pos def, do nothing
        if np.all(np.linalg.eigvals(B) > 0):
            return B

        diag = np.diag(B)

        if np.min(diag) > 0:
            tau_0 = 0
        else:
            tau_0 = -np.min(diag) + beta

        tau = tau_0
        while True:
            try:
                L = np.linalg.cholesky(B + np.eye(B.shape[0]) * tau)
            except np.linalg.LinAlgError:
                tau = max(factor * tau, beta)
            else:
                break

        return L @ L.T

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
        pass
