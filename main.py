import numpy as np

from conjugate_gradient import ConjugateGradientFR, ConjugateGradientPR
from quasinewton import QuasiNewtonBFGS, QuasiNewtonSR1
from test_functions import test_func_1_torch, test_func_2_torch, START_POINTS, MINIMA_POINTS, test_func_1_approx, \
    test_func_2_approx
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


def run_test(description, f, start_points, minima, optim, plot=False):
    for start_point in start_points:
        print("----------------------------------------")
        print(description)
        print(f"starting point ({start_point[0], start_point[1]})")

        xs = optim.run_optim(start_point)

        print(f"needed {len(xs)} iterations")

        print(f"solution found at ({xs[-1][0]:.5}, {xs[-1][1]:.5})")
        print(
            f"distance to real minimum: {np.min(np.apply_along_axis(lambda x: np.linalg.norm(xs[-1] - x), 0, minima))}")
        print(f"solution value {f(xs[-1]):.5}")

        optim.reset()

        if plot:
            plot_test_run(f, xs, [-6, 6], [-6, 6], description)


print("-" * 80)
print("TESTING NM with mod")

f = test_func_1_torch
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modification for Rosenbrock w. autog.", f, START_POINTS["test_function_1"],
         MINIMA_POINTS["test_function_1"], nm, plot=True)

f = test_func_1_approx
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modification for Rosenbrock w. approx.", f, START_POINTS["test_function_1"],
         MINIMA_POINTS["test_function_1"], nm, plot=True)

f = test_func_2_torch
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modification for function 2 w. autog.", f, START_POINTS["test_function_2"],
         MINIMA_POINTS["test_function_2"], nm, plot=True)

f = test_func_2_approx
nm = NewtonMethodWithModifiedHessian(f)
run_test("NM w. Hessian Modification for function 2 w. approx.", f, START_POINTS["test_function_2"],
         MINIMA_POINTS["test_function_2"], nm, plot=True)

print("-" * 80)
print("TESTING CG")

f = test_func_1_torch
cg = ConjugateGradientFR(f, 1)
run_test("CG w. FR for Rosenbrock w. autog.", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"], cg,
         plot=True)

f = test_func_1_approx
cg = ConjugateGradientFR(f, 1)
run_test("CG w. FR for Rosenbrock w. approx..", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"],
         cg, plot=True)

f = test_func_2_torch
cg = ConjugateGradientFR(f, 1)
run_test("CG w. FR for TestFunc2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], cg,
         plot=True)

f = test_func_2_approx
cg = ConjugateGradientFR(f, 1)
run_test("CG w. FR for TestFunc2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], cg,
         plot=True)

f = test_func_1_torch
cg = ConjugateGradientPR(f, 0.01)
run_test("CG w. PR for Rosenbrock w. autog.", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"], cg,
         plot=True)

f = test_func_1_approx
cg = ConjugateGradientPR(f, 0.01)
run_test("CG w. PR for Rosenbrock w. approx..", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"],
         cg, plot=True)

f = test_func_2_torch
cg = ConjugateGradientPR(f, 1)
run_test("CG w. PR for TestFunc2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], cg,
         plot=True)

f = test_func_2_approx
cg = ConjugateGradientPR(f, 0.01)
run_test("CG w. PR for TestFunc2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], cg,
         plot=True)

print("-" * 80)
print("TESTING QN")

f = test_func_1_torch
qn = QuasiNewtonBFGS(f, 2, 1)
run_test("QN w. BFGS for Rosenbrock w. autog.", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"],
         qn, plot=True)

f = test_func_1_approx
qn = QuasiNewtonBFGS(f, 2, 1)
run_test("QN w. BFGS for Rosenbrock w. approx.", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"],
         qn, plot=True)

f = test_func_2_torch
qn = QuasiNewtonBFGS(f, 2, 1)
run_test("QN w. BFGS for testfunc 2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"],
         qn, plot=True)

f = test_func_2_approx
qn = QuasiNewtonBFGS(f, 2, 1)
run_test("QN w. BFGS for testfunc2 w. approx.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"],
         qn, plot=True)

f = test_func_1_torch
qn = QuasiNewtonSR1(f, 2, 1)
run_test("QN w. SR1 for Rosenbrock w. autog.", f, START_POINTS["test_function_1"], MINIMA_POINTS["test_function_1"], qn,
         plot=True)

f = test_func_2_approx
qn = QuasiNewtonSR1(f, 2, 1)
run_test("QN w. SR1 for Rosenbrock w. approx.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"],
         qn, plot=True)

f = test_func_2_torch
qn = QuasiNewtonSR1(f, 2, 1)
run_test("QN w. SR1 for testfunc 2 w. autog.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], qn,
         plot=True)

f = test_func_2_approx
qn = QuasiNewtonSR1(f, 2, 1)
run_test("QN w. SR1 for testfunc2 w. approx.", f, START_POINTS["test_function_2"], MINIMA_POINTS["test_function_2"], qn,
         plot=True)
