import numpy as np


# estimate gradient with central difference formula (Nocedal 196)
def cdf_grad(f, x, epsilon=10 ** -6):
    size = x.size
    gradient = np.zeros(size, dtype=np.float64)

    for i in range(size):
        e = np.zeros(size)
        e[i] = 1
        gradient[i] = (f(x + epsilon * e) - f(x - epsilon * e)) / (2 * epsilon)

    return gradient


# estimate hessian empirically (Nocedal 202)
def cdf_hessian(f, x, epsilon=10 ** -7):
    size = x.size

    hessian = np.zeros((size, size), dtype=np.float64)

    for row in range(size):
        for col in range(size):
            e_i = np.zeros(size, dtype=np.float64)
            e_j = np.zeros(size, dtype=np.float64)
            e_i[row] = 1
            e_j[col] = 1

            hessian[row, col] = (f(x + epsilon * e_i + epsilon * e_j) -
                                 f(x + epsilon * e_i) -
                                 f(x + epsilon * e_j) + f(x)) / epsilon ** 2

    # hessian = (hessian + hessian.T) / 2
    return hessian
