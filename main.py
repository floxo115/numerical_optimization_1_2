import numpy as np
import torch.func

import utils
from derivative_approximation import cdf_grad
import test_functions

print(test_functions.rosenbrock_func.get_gradient(np.array([1., 1.])))
plot = utils.plot_2d_contour(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
                             test_functions.test_func_2.get_function())
plot.show()

print(cdf_grad(lambda x: x ** 2, np.array([8]), 1))
print(test_functions.TestFunctionTorch(lambda x: x[0]**2 + x[1]**2).get_gradient(np.array([8.,8.])))
