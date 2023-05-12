
def backtracking_line_search(f, x, d, alpha, rho=0.5, beta=10e-4):
    f_x = f.function(x)
    nabla_f_x = f.get_gradient(x)
    y, g = f_x, nabla_f_x

    while f.function(x + alpha * d) > y + beta * alpha * (nabla_f_x @ d):
        alpha *= rho

    return alpha
