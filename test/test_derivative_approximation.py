import numpy as np
import pytest
from derivative_approximation import cdf_grad, cdf_hessian
from test_functions import test_func_1, test_func_2

np.set_printoptions(precision=15)

def test_cdf_grad_test_at_minimum():
    expected = test_func_1.get_gradient(np.array([1., 1.]))
    actual = cdf_grad(test_func_1.function, np.array([1., 1.]))

    assert np.all(np.isclose(actual, expected)), f"expected {expected}, but got {actual}"


def test_cdf_grad_test_at_random_points():
    for i in range(100):
        rand_inp = np.random.random(2) * 10
        expected = (lambda x: np.array([2 * (x[0] - 1) - 4 * 100 * x[0] * (x[1] - x[0] ** 2), 2 * 100*(x[1]-x[0]**2)]))(rand_inp)
        actual = cdf_grad(test_func_1.function, rand_inp)
        assert np.all(np.abs(actual - expected)/np.abs(expected) < 10**-4)


def test_cdf_hessian_test_at_random_points():
    for i in range(10):
        rand_inp = np.random.random(2) * 10 + np.ones(2, dtype=np.float128)
        expected = (lambda x: np.array([[2 - 4*100*(x[1] - 3*x[0]**2), -4*100*x[0]],[-4*100 * x[0], 2*100]]))(rand_inp)
        actual = cdf_hessian(test_func_1.function, rand_inp,epsilon=10**-7)
        assert np.all(np.linalg.norm(actual - expected)/np.linalg.norm(expected) < 10**-3)
