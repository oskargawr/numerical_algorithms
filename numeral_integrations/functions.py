import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# rectangle rule


def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x)
    return h * np.sum(y), x, y


# trapezoidal rule


def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1])), x, y


# simpsons rule
def simpson_rule_area(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return integral, x, y


# spline csi


def spline_interpolation_area(a_beg, b_fin, n, func):
    x = np.linspace(a_beg, b_fin, n + 1)
    y = func(x)
    h = (b_fin - a_beg) / n

    B = np.zeros(n + 1)
    row = [0, n]
    col = [0, n]
    data = [1, 1]

    for i in range(1, n):
        row.extend([i, i, i])
        col.extend([i, i - 1, i + 1])
        data.extend([4 * h, h, h])
        B[i] = 3 * ((y[i + 1] - y[i]) / h - (y[i] - y[i - 1]) / h)

    A = csr_matrix((data, (row, col)))
    c = spsolve(A, B)

    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h - h * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h)

    total_area = 0
    for i in range(n):
        a_i = y[i]
        b_i = b[i]
        c_i = c[i]
        d_i = d[i]
        h_i = h
        integral = a_i * h_i + b_i * h_i**2 / 2 + c_i * h_i**3 / 3 + d_i * h_i**4 / 4
        total_area += integral

    return total_area
