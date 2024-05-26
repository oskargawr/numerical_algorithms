import numpy as np
import matplotlib.pyplot as plt
import time


# Definicja funkcji do całkowania
def f(x):
    return x**2


# Metoda trapezów
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1])), x, y


# Generowanie wykresów dla różnych wartości n
def plot_trapezoidal(f, a, b, n):
    integral, x, y = trapezoidal_rule(f, a, b, n)
    x_dense = np.linspace(a, b, 1000)
    y_dense = f(x_dense)
    plt.plot(x_dense, y_dense)
    plt.fill_between(x, y, alpha=0.3)
    plt.title(f"Metoda trapezów, n={n}")
    plt.show()
    return integral


# Parametry całkowania
a = 0
b = 1
n_values = [30, 10000, 1000000, 100000000]

# Testowanie metody trapezów dla różnych n
for n in n_values:
    start_time = time.time()
    if n == 30:
        integral = plot_trapezoidal(f, a, b, n)
    end_time = time.time()
    true_integral = 1 / 3
    error = np.abs(integral - true_integral)
    print(f"n: {n}, Wynik: {integral}, Błąd: {error}, Czas: {end_time - start_time} s")


# ---- trapezoidal rule for an ellipse ----
from scipy.special import ellipe


def f_ellipse(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def elipse_area(a, b):
    return np.pi * a * b


def trapezoidal_rule_ellipse(f, a, b, n, ax, bx):
    h = (bx - ax) / n
    x = np.linspace(ax, bx, n + 1)
    y = f(x, a, b)
    integral = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral


def plot_ellipse_and_approximation(f, a, b, n, ax, bx):
    x_dense = np.linspace(ax, bx, 1000)
    y_dense = f(x_dense, a, b)
    plt.plot(x_dense, y_dense, label=f"Elipsa: a={a}, b={b}")
    h = (bx - ax) / n
    x = np.linspace(ax, bx, n + 1)
    y = f(x, a, b)
    plt.fill_between(x, y, alpha=0.3)
    plt.title(f"Metoda trapezów dla elipsy, n={n}")
    plt.show()


# Testowanie metody trapezów dla różnych parametrów elipsy
ellipse_params = [{"a": 3, "b": 1}, {"a": 5, "b": 8}, {"a": 10, "b": 3}]

n_values_ellipse = [30, 10000, 1000000, 100000000]

for params in ellipse_params:
    a, b = params["a"], params["b"]
    for n in n_values_ellipse:
        ellipse = {"f": f_ellipse, "a": a, "b": b, "n": n, "ax": -a, "bx": a}
        start_time = time.time()
        integral = 2 * trapezoidal_rule_ellipse(**ellipse)
        end_time = time.time()
        true_integral = elipse_area(a, b)
        error = np.abs(integral - true_integral)
        print(
            f"Elipsa: a={a}, b={b}, n={n}, Wynik: {integral}, Błąd: {error}, Czas: {end_time - start_time} s"
        )
        if n == 30:
            plot_ellipse_and_approximation(f_ellipse, a, b, n, -a, a)


# ---- trapezoidal rule for a sine function ----
def f_sin(x):
    return np.sin(x)


n_values_sine = [30, 10000, 1000000, 100000000]

for n in n_values_sine:
    start_time = time.time()
    integral, x, y = trapezoidal_rule(f_sin, 0, np.pi, n)
    end_time = time.time()
    true_integral = 2
    error = np.abs(integral - true_integral)
    print(
        f"Sinus, n={n}, Wynik: {integral}, Błąd: {error}, Czas: {end_time - start_time} s"
    )
    if n == 30:
        x_dense = np.linspace(0, np.pi, 1000)
        y_dense = f_sin(x_dense)
        plt.plot(x_dense, y_dense)
        plt.fill_between(x, y, alpha=0.3)
        plt.title(f"Metoda trapezów dla funkcji sinus, n={n}")
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import ellipe
from scipy.integrate import quad

# ---------------- CURVE LENGTH ----------------


# Funkcja koła y = sqrt(1 - x^2)
def circle_func(x):
    return np.sqrt(1 - x**2)


# Pochodna funkcji koła y' = -x / sqrt(1 - x^2), z obsługą wartości brzegowych
def circle_derivative(x):
    return np.divide(-x, np.sqrt(1 - x**2), where=np.abs(x) != 1)


# Metoda trapezów do całkowania
def trapezoidal_rule_curve_length(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1])), x, y


# Funkcja do obliczania długości krzywej
def curve_length(f, f_prime, a, b, n):
    def integrand(x):
        return np.sqrt(1 + f_prime(x) ** 2)

    length, x, y = trapezoidal_rule_curve_length(integrand, a, b, n)
    return length, x, y


# Generowanie wykresów dla różnych wartości n
def plot_trapezoidal_curve(f, f_prime, a, b, n):
    length, x, y = curve_length(f, f_prime, a, b, n)
    x_dense = np.linspace(a, b, 1000)
    y_dense = np.sqrt(1 + f_prime(x_dense) ** 2)
    plt.plot(x_dense, y_dense)
    plt.fill_between(x, y, alpha=0.3)
    plt.title(f"Metoda trapezów, n={n}")
    plt.show()
    return length


# Parametry całkowania
n_values = [30, 10000, 1000000, 100000000]

# Testowanie metody trapezów dla różnych n dla koła
a, b = 0, 1
for n in n_values:
    start_time = time.time()
    if n == 30:
        quarter_circle_length = plot_trapezoidal_curve(
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

# --- ellipse circumference ---


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
            quarter_ellipse_length = plot_trapezoidal_curve(
                lambda x: ellipse_func(x, a, b),
                lambda x: ellipse_derivative(x, a, b),
                -a,
                a,
                n,
            )
        else:
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

# --- sin curve length at [0, 2pi] ---


def sin_derivative(x):
    return np.cos(x)


def f(x):
    return np.sqrt(1 + np.cos(x) ** 2)


for n in n_values:
    start_time = time.time()
    if n == 30:
        sin_length = plot_trapezoidal_curve(np.sin, sin_derivative, 0, 2 * np.pi, n)
    else:
        sin_length, _, _ = curve_length(np.sin, sin_derivative, 0, 2 * np.pi, n)
    end_time = time.time()
    true_sin_length, _ = quad(f, 0, 2 * np.pi)
    error = np.abs(sin_length - true_sin_length)
    print(
        f"Sinus, n={n}, Długość krzywej: {sin_length}, Błąd: {error}, Czas: {end_time - start_time} s"
    )
