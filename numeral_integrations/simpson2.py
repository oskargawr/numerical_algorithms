import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import ellipe

# ---- Area calculation using Simpson's Rule ----


def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


def simpson_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return integral, x, y


def plot_simpson_parabolas(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    x_dense = np.linspace(a, b, 1000)
    y_dense = f(x_dense)
    plt.plot(x_dense, y_dense, label="f(x)")

    for i in range(0, n, 2):
        xi = x[i : i + 3]
        yi = y[i : i + 3]

        coefs = np.polyfit(xi, yi, 2)
        p = np.poly1d(coefs)

        x_parabola = np.linspace(xi[0], xi[2], 100)
        y_parabola = p(x_parabola)

        plt.plot(
            x_parabola, y_parabola, "r--", label="Simpson Parabola" if i == 0 else ""
        )
        plt.fill_between(x_parabola, 0, y_parabola, color="gray", alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Simpson's Rule Approximation")
    plt.show()


# Parametry całkowania
n_values = [30, 10000, 1000000, 100000000]

# Testowanie metody Simpsona dla różnych n dla koła
a, b = -1, 1
for n in n_values:
    start_time = time.time()
    if n == 30:
        integral, x, y = simpson_rule(half_of_a_circle, a, b, n)
        plot_simpson_parabolas(half_of_a_circle, a, b, n)
    else:
        integral, x, y = simpson_rule(half_of_a_circle, a, b, n)
    end_time = time.time()
    pi_approx = 2 * integral
    true_pi = np.pi
    error = np.abs(pi_approx - true_pi)
    print(
        f"n: {n}, Przybliżona wartość π: {pi_approx}, Błąd: {error}, Czas: {end_time - start_time} s"
    )

# ---- Area calculation for ellipses ----


def ellipse_func(x, a, b):
    return b * np.sqrt(1 - x**2 / a**2)


def simpson_rule_ellipse(f, a, b, n, ax, bx):
    h = (bx - ax) / n
    x = np.linspace(ax, bx, n + 1)
    y = f(x, a, b)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return integral


def plot_ellipse_and_approximation(f, a, b, n):
    x_dense = np.linspace(-a, a, 1000)
    y_dense = f(x_dense, a, b)
    plt.plot(x_dense, y_dense, label=f"Elipsa: a={a}, b={b}")

    h = (2 * a) / n
    x = np.linspace(-a, a, n + 1)
    y = f(x, a, b)

    for i in range(n // 2):
        xi = x[2 * i : 2 * i + 3]
        yi = y[2 * i : 2 * i + 3]

        coefs = np.polyfit(xi, yi, 2)
        p = np.poly1d(coefs)

        x_parabola = np.linspace(xi[0], xi[2], 100)
        y_parabola = p(x_parabola)

        plt.plot(x_parabola, y_parabola, "r--")
        plt.fill_between(x_parabola, 0, y_parabola, color="gray", alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Simpson's Rule Approximation for Ellipse: a={a}, b={b}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


ellipse_params = [{"a": 3, "b": 1}, {"a": 5, "b": 8}, {"a": 10, "b": 3}]
for params in ellipse_params:
    a, b = params["a"], params["b"]
    for n in n_values:
        start_time = time.time()
        if n == 30:
            integral = simpson_rule_ellipse(ellipse_func, a, b, n, -a, a)
            plot_ellipse_and_approximation(ellipse_func, a, b, n)
        else:
            integral = simpson_rule_ellipse(ellipse_func, a, b, n, -a, a)
        end_time = time.time()
        ellipse_area = 2 * integral
        true_area = np.pi * a * b
        error = np.abs(ellipse_area - true_area)
        print(
            f"Elipsa: a={a}, b={b}, n={n}, Pole elipsy: {ellipse_area}, Błąd: {error}, Czas: {end_time - start_time} s"
        )

# ---- Area calculation for sine function ----


def f_sin(x):
    return np.sin(x)


for n in n_values:
    start_time = time.time()
    if n == 30:
        integral, x, y = simpson_rule(f_sin, 0, np.pi, n)
        plot_simpson_parabolas(f_sin, 0, np.pi, n)
    else:
        integral, x, y = simpson_rule(f_sin, 0, np.pi, n)
    end_time = time.time()
    true_integral = 2
    error = np.abs(integral - true_integral)
    print(
        f"Sinus, n={n}, Wynik: {integral}, Błąd: {error}, Czas: {end_time - start_time} s"
    )

import numdifftools as nd

import numdifftools as nd

import numdifftools as nd

# ---- Curve length calculation using Simpson's Rule ----


def circle_func(x):
    return np.sqrt(1 - x**2)


def circle_derivative(x):
    return np.divide(-x, np.sqrt(1 - x**2), where=np.abs(x) != 1)


def simpson_rule_curve_length(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return integral, x, y


def curve_length(f, f_prime, a, b, n):
    def integrand(x):
        return np.sqrt(1 + f_prime(x) ** 2)

    length, x, y = simpson_rule_curve_length(integrand, a, b, n)
    return length, x, y


def plot_simpson_curve(f, f_prime, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    x_dense = np.linspace(a, b, 1000)
    y_dense = f(x_dense)
    plt.plot(x_dense, y_dense, label="f(x)")

    for i in range(0, n, 2):
        xi = x[i : i + 3]
        yi = y[i : i + 3]

        coefs = np.polyfit(xi, yi, 2)
        p = np.poly1d(coefs)

        x_parabola = np.linspace(xi[0], xi[2], 100)
        y_parabola = p(x_parabola)

        plt.plot(
            x_parabola, y_parabola, "r--", label="Simpson Parabola" if i == 0 else ""
        )
        plt.fill_between(x_parabola, 0, y_parabola, color="gray", alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Simpson's Rule Approximation for Curve Length")
    plt.show()


# Parametry całkowania
n_values = [30, 10000, 1000000, 100000000]

# Testowanie metody Simpsona dla różnych n dla okręgu
a, b = 0, 1
for n in n_values:
    start_time = time.time()
    if n == 30:
        plot_simpson_curve(circle_func, circle_derivative, a, b, n)
        quarter_circle_length, _, _ = curve_length(
            circle_func, circle_derivative, a, b, n
        )
    else:
        quarter_circle_length, _, _ = curve_length(
            circle_func, circle_derivative, a, b, n
        )
    end_time = time.time()
    circle_circumference = 4 * quarter_circle_length
    true_circumference = 2 * np.pi
    error = np.abs(circle_circumference - true_circumference)
    print(
        f"n: {n}, Obwód koła: {circle_circumference}, Błąd: {error}, Czas: {end_time - start_time} s"
    )

# ---- Curve length calculation for ellipses ----


def ellipse_func(x, a, b):
    return b * np.sqrt(1 - x**2 / a**2)


def ellipse_derivative(x, a, b):
    return np.divide(-b * x, a**2 * np.sqrt(1 - x**2 / a**2), where=np.abs(x) != a)


def real_ellipse_circumference(a, b):
    e_sq = 1 - b**2 / a**2
    return 4 * a * ellipe(e_sq)


ellipse_params = [{"a": 3, "b": 1}, {"a": 5, "b": 8}, {"a": 10, "b": 3}]
for params in ellipse_params:
    a, b = params["a"], params["b"]
    for n in n_values:
        start_time = time.time()
        if n == 30:
            quarter_ellipse_length = plot_simpson_curve(
                lambda x: ellipse_func(x, a, b),
                lambda x: ellipse_derivative(x, a, b),
                -a,
                a,
                n,
            )

        quarter_ellipse_length, _, _ = curve_length(
            lambda x: ellipse_func(x, a, b),
            lambda x: ellipse_derivative(x, a, b),
            -a,
            a,
            n,
        )
        end_time = time.time()
        ellipse_circumference = 2 * quarter_ellipse_length
        true_circumference = real_ellipse_circumference(a, b)
        error = np.abs(ellipse_circumference - true_circumference)
        print(
            f"Elipsa: a={a}, b={b}, n={n}, Obwód elipsy: {ellipse_circumference}, Błąd: {error}, Czas: {end_time - start_time} s"
        )

# ---- Curve length calculation for sine function ----
from scipy.integrate import quad


def sin_derivative(x):
    return np.cos(x)


def f(x):
    return np.sqrt(1 + np.cos(x) ** 2)


for n in n_values:
    start_time = time.time()
    if n == 30:
        sin_length = plot_simpson_curve(np.sin, sin_derivative, 0, 2 * np.pi, n)

    sin_length, _, _ = curve_length(np.sin, sin_derivative, 0, 2 * np.pi, n)
    end_time = time.time()
    true_sin_length, _ = quad(f, 0, 2 * np.pi)
    error = np.abs(sin_length - true_sin_length)
    print(
        f"Sinus, n={n}, Długość krzywej: {sin_length}, Błąd: {error}, Czas: {end_time - start_time} s"
    )
