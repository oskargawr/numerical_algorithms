import numpy as np
import matplotlib.pyplot as plt


def jacobian(A, b, x0, tol=1e-8, max_iter=10000):
    n = len(b)
    x = x0.copy()
    x_prev = x0.copy()
    iter_count = 0

    while iter_count < max_iter:
        for i in range(n):
            sum_val = sum(A[i, j] * x_prev[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_val) / A[i, i]
        if np.linalg.norm(x - x_prev) < tol:
            return x
        x_prev = x.copy()
        iter_count += 1
    return None


def prepare_matrix(n):
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        A[i, i] = 1
        if i > 0:
            if i == 1:
                A[i - 1, i] = 0
                A[i, i - 1] = -0.5
            elif i == n - 1:
                A[i - 1, i] = -0.5
            else:
                A[i, i - 1] = -0.5
        if i < n - 1:
            A[i, i + 1] = -0.5
        if i == 0:
            b[i] = 1
    return A, b


A, b = prepare_matrix(6)
print(A)

x = jacobian(A, b, x0=np.zeros(len(b)))
print("x =", x)
