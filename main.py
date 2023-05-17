import numpy as np

from test_functions import test_func_1_torch, test_func_2_torch, START_POINTS, test_func_1_approx, test_func_2_approx
from newton_method_mod import NewtonMethodWithModifiedHessian
from utils import plot_2d_contour
from matplotlib import pyplot as plt


def plot_test_run(f, xs, x_lims, y_lims, desc=""):
    plot_2d_contour(
        np.linspace(x_lims[0], x_lims[1], 100),
        np.linspace(y_lims[0], y_lims[1], 100),
        f
    )

    plt.scatter([x[0] for x in xs], [x[1] for x in xs], c=[i for i in range(len(xs))], cmap="plasma")
    if desc != "":
        plt.title(desc)
    plt.show()


def run_test(description, f, start_points, optim, plot=False):
    for start_point in start_points:
        print("----------------------------------------")
        print("desc")
        print(f"test function 1; starting point ({start_point[0], start_point[1]})")

        xs = optim.run_optim(start_point)

        print(f"needed {len(xs)} iterations")

        print(f"solution found at ({xs[-1][0]:.5}, {xs[-1][1]:.5})")
        print(f"solution value {f(xs[-1]):.5}")

        if plot:
            plot_test_run(f, xs, [-6, 6], [-6, 6], description)


f = test_func_1_torch
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modificatoin for Rosenbrock w. autog.", f, START_POINTS["test_function_1"], nm, plot=True)

f = test_func_1_approx
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modificatoin for Rosenbrock w. approx.", f, START_POINTS["test_function_1"], nm, plot=True)


f = test_func_2_torch
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modificatoin for function 2 w. autog.", f, START_POINTS["test_function_2"], nm, plot=True)

f = test_func_2_approx
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modificatoin for function 2 w. approx.", f, START_POINTS["test_function_2"], nm, plot=True)

