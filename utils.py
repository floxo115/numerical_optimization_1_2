from matplotlib import pyplot as plt
import numpy as np


def plot_2d_contour(x, y, func):
    X, Y = np.meshgrid(x, y)
    Z = np.stack((X, Y), axis=2)
    Z = np.apply_along_axis(func, 2, Z)

    cs = plt.contourf(X, Y, Z)
    plt.colorbar(cs)

    return plt
