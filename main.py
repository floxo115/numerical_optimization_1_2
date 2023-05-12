import numpy as np

from test_functions import test_func_1, test_func_2, START_POINTS
from newton_method_mod import NewtonMethodWithModifiedHessian
from utils import plot_2d_contour
from matplotlib import pyplot as plt

f = test_func_1
nm = NewtonMethodWithModifiedHessian(f)
for start_point in START_POINTS["test_function_1"]:
    print("----------------------------------------")
    print("optimizing with newton method with hessian modifications")
    print(f"test function 1; starting point ({start_point[0], start_point[1]})")

    xs = nm.run_optim(start_point)

    print(f"needed {len(xs)} iterations")

    x_lims = [-4, 4]
    y_lims = [-4, 4]

    plot_2d_contour(
        np.linspace(x_lims[0], x_lims[1], 100),
        np.linspace(x_lims[0], x_lims[1], 100),
        test_func_1.function
    )

    plt.scatter([x[0] for x in xs], [x[1] for x in xs])
    plt.show()

    print(f"solution found at ({xs[-1][0]}, {xs[-1][1]})")
    print(f"solution value {f.function(xs[-1])}")

f = test_func_2
nm = NewtonMethodWithModifiedHessian(f)
for start_point in START_POINTS["test_function_2"]:
    print("----------------------------------------")
    print("optimizing with newton method with hessian modifications")
    print(f"test function 2; starting point ({start_point[0], start_point[1]})")

    xs = nm.run_optim(start_point)

    print(f"needed {len(xs)} iterations")

    x_lims = [-6, 6]
    y_lims = [-6, 6]

    plot_2d_contour(
        np.linspace(x_lims[0], x_lims[1], 100),
        np.linspace(x_lims[0], x_lims[1], 100),
        test_func_2.function
    )

    plt.scatter([x[0] for x in xs], [x[1] for x in xs])
    plt.show()
    print(f"solution found at ({xs[-1][0]}, {xs[-1][1]})")
    print(f"solution value {f.function(xs[-1])}")
