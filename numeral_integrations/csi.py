import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import time
from scipy.special import ellipe

# ---- Interpolacja spline'owa i obliczanie pola ----


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


def plot_spline_interpolation(func, a_beg, b_fin, n):
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

    def spline(x_val):
        for i in range(n):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                return y[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        return None

    x_dense = np.linspace(a_beg, b_fin, 1000)
    y_dense = np.array([spline(x_val) for x_val in x_dense])
    plt.plot(x_dense, y_dense, label="Spline interpolation")
    plt.scatter(x, y, color="red", label="Data points")
    plt.legend()
    plt.title(f"Spline interpolation for n={n}")
    plt.show()


# ---- Testowanie metody spline'owej dla różnych funkcji i wartości n ----


n_values = [30, 1000, 10000, 100000]

# ---- Obliczanie pola dla półokręgu ----
print("Obliczanie pola dla półokręgu")


def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


for n in n_values:
    start_time = time.time()
    area = spline_interpolation_area(-1, 1, n, half_of_a_circle)
    end_time = time.time()
    pi_approx = 2 * area
    true_pi = np.pi
    error = np.abs(pi_approx - true_pi)
    print(
        f"n: {n}, Przybliżona wartość π: {pi_approx}, Błąd: {error}, Czas: {end_time - start_time} s"
    )
    if n == 30:
        plot_spline_interpolation(half_of_a_circle, -1, 1, n)

# Parabola
print("Obliczanie pola dla paraboli")


def parabola_func(x):
    return x**2


for n in n_values:
    start_time = time.time()
    area = spline_interpolation_area(0, 1, n, parabola_func)
    end_time = time.time()
    true_area = 1 / 3
    error = np.abs(area - true_area)
    print(f"n: {n}, Pole: {area}, Błąd: {error}, Czas: {end_time - start_time} s")
    if n == 30:
        plot_spline_interpolation(parabola_func, 0, 1, n)

# Pole elipsy

elipse_params = [{"a": 3, "b": 1}, {"a": 5, "b": 8}, {"a": 10, "b": 3}]
print("Obliczanie pola dla elipsy")


def ellipse_func(x, a, b):
    return b * np.sqrt(1 - (x / a) ** 2)


def real_ellipse_circumference(a, b):
    e_sq = 1 - b**2 / a**2
    return 4 * a * ellipe(e_sq)


for params in elipse_params:
    a, b = params["a"], params["b"]
    for n in n_values:
        start_time = time.time()
        area = spline_interpolation_area(-a, a, n, lambda x: ellipse_func(x, a, b))
        end_time = time.time()
        true_area = np.pi * a * b / 2
        error = np.abs(area - true_area)
        print(f"n: {n}, Pole: {area}, Błąd: {error}, Czas: {end_time - start_time} s")
        if n == 30:
            plot_spline_interpolation(lambda x: ellipse_func(x, a, b), -a, a, n)

# Pole dla wykresu sinus na przedziale [0, π]
print("Obliczanie pola dla sinusa na przedziale [0, π]")
for n in n_values:
    start_time = time.time()
    area = spline_interpolation_area(0, np.pi, n, np.sin)
    end_time = time.time()
    true_area = 2
    error = np.abs(area - true_area)
    print(f"n: {n}, Pole: {area}, Błąd: {error}, Czas: {end_time - start_time} s")
    if n == 30:
        plot_spline_interpolation(np.sin, 0, np.pi, n)


# ------- KRZYWE -------
# obliczanie długości krzywej za pomocą interpolacji spline'owej


def curve_length(a, b, n, f_prime):
    def integrand(x):
        return np.sqrt(1 + f_prime(x) ** 2)

    length = spline_interpolation_area(a, b, n, integrand)
    return length


# Testowanie metody spline'owej dla różnych wartości n
n_values = [30, 1000, 10000, 100000]

# Obliczanie długości krzywej dla półokręgu
print("Obliczanie długości krzywej dla półokręgu")


def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


def half_of_a_circle_derivative(x):
    return -x / np.sqrt(1 - x**2)


for n in n_values:
    start_time = time.time()
    length = curve_length(
        -np.sqrt(2) / 2, np.sqrt(2) / 2, n, half_of_a_circle_derivative
    )
    end_time = time.time()
    circle_circumference = 4 * length
    true_circumference = np.pi
    pi_approx = 2 * length
    error = np.abs(np.pi - pi_approx)
    print(
        f"n: {n}, Obwód koła: {circle_circumference}, Przyblizenie pi: {pi_approx}, Błąd: {error}, Czas: {end_time - start_time} s"
    )

# Obliczanie długości krzywej dla elipsy

print("Obliczanie długości krzywej dla elipsy")


def ellipse(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def ellipse_derivative(x, a, b):
    return np.divide(-b * x, a**2 * np.sqrt(1 - x**2 / a**2), where=np.abs(x) != a)


for params in elipse_params:
    a, b = params["a"], params["b"]
    for n in n_values:
        start_time = time.time()
        length = curve_length(-a, a, n, lambda x: ellipse_derivative(x, a, b))
        end_time = time.time()
        ellipse_circumference = 2 * length
        true_circumference = real_ellipse_circumference(a, b)
        error = np.abs(ellipse_circumference - true_circumference)
        print(
            f"n: {n}, Obwód elipsy: {ellipse_circumference}, Błąd: {error}, Czas: {end_time - start_time} s"
        )


# Obliczanie długości krzywej dla sinusa na przedziale [0, 2π]
print("Obliczanie długości krzywej dla sinusa na przedziale [0, 2π]")
for n in n_values:
    start_time = time.time()
    length = curve_length(0, 2 * np.pi, n, np.cos)
    end_time = time.time()
    true_sin_length, _ = quad(lambda x: np.sqrt(1 + np.cos(x) ** 2), 0, 2 * np.pi)
    error = np.abs(length - true_sin_length)
    print(f"n: {n}, Długość: {length}, Błąd: {error}, Czas: {end_time - start_time} s")
    if n == 30:
        plot_spline_interpolation(np.cos, 0, 2 * np.pi, n)
