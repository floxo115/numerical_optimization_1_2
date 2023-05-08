import numpy as np
import utils

import test_functions

print(test_functions.rosenbrock_func.get_gradient(np.array([1., 1.])))
plot = utils.plot_2d_contour(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
                             test_functions.test_func_2.get_function())
plot.show()
