import numpy as np
import matplotlib.pyplot as plt
import time


def f(x):
    return x**2


def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x)
    return h * np.sum(y), x, y


a = 0  # lower bound
b = 1  # upper bound
n = 100
integral, x, y = rectangle_rule(f, a, b, n)
# print(integral)

# x_dense = np.linspace(a, b, 100)
# y_dense = f(x_dense)
# plt.plot(x_dense, y_dense)
# plt.bar(x, y, width=(b - a) / n, alpha=0.3, align="edge")
# plt.show()

# calculate the error
true_integral = 1 / 3
error = np.abs(integral - true_integral)
# print(error)


# ---- rectangle rule for a circle
def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


circle = {
    "f": half_of_a_circle,
    "a": -1,
    "b": 1,
    "n": 30,
}

# time_start = time.time()
# integral, x, y = rectangle_rule(**circle)
# time_end = time.time()
# print(integral)

# x_dense = np.linspace(-1, 1, 30)
# y_dense = half_of_a_circle(x_dense)
# plt.plot(x_dense, y_dense)
# plt.bar(x, y, width=(1 - -1) / 30, alpha=0.3, align="edge")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.show()

# pi_approx = 2 * integral
# print("pi approx:", pi_approx)
# print("n:", circle["n"])
# print("Time:", time_end - time_start)
# print("Error:", np.abs(pi_approx - np.pi))

# ---- rectangle rule for a parabola
parabola = {
    "f": lambda x: x**2,
    "a": 0,
    "b": 1,
    "n": 100000000,
}

# time_start = time.time()
# integral, x, y = rectangle_rule(**parabola)
# time_end = time.time()

# x_dense = np.linspace(0, 1, 30)
# y_dense = parabola["f"](x_dense)
# plt.plot(x_dense, y_dense)
# plt.bar(x, y, width=(1 - 0) / 30, alpha=0.3, align="edge")
# plt.show()

true_integral = 1 / 3

# error = np.abs(integral - true_integral)
# print("n:", parabola["n"])
# print("wynik:", integral)
# print("Time:", time_end - time_start)
# print("error: ", error)


# ---- rectangle rule for an ellipse
def f(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def rectangle_rule_elipse(f, a, b, n, ax, bx):
    h = (bx - ax) / n
    x = np.linspace(ax, bx, n)
    y = f(x, a, b)
    integral = h * np.sum(y)
    return integral


def plot_ellipse_and_approximation(f, a, b, n):
    x_dense = np.linspace(-a, a, 1000)
    y_dense = f(x_dense, a, b)
    plt.plot(x_dense, y_dense, label=f"Elipsa: a={a}, b={b}")

    h = (2 * a) / n
    x = np.linspace(-a, a, n)
    y = f(x, a, b)
    plt.bar(x, y, width=(2 * a) / n, alpha=0.3, align="edge")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def elipse_area(a, b):
    return np.pi * a * b


ellipse = {
    "f": f,
    "a": 10,
    "b": 3,
    "n": 10,
    "ax": -10,
    "bx": 10,
}

# time_start = time.time()
# integral = rectangle_rule_elipse(**ellipse)
# time_end = time.time()
# print("przyblizenie: ", 2 * integral)

# plot_ellipse_and_approximation(f, ellipse["a"], ellipse["b"], ellipse["n"])

# calculate the error
# true_integral = elipse_area(ellipse["a"], ellipse["b"])

# error = np.abs(2 * integral - true_integral)
# print("error: ", error)
# print("Time:", time_end - time_start)
# print("n:", ellipse["n"])


# sin
def sin(x):
    return np.sin(x)


def rectangle_rule_sin(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x)
    return h * np.sum(y), x, y


sin = {
    "f": sin,
    "a": 0,
    "b": np.pi,
    "n": 100,
}

# time_start = time.time()
# integral, x, y = rectangle_rule_sin(**sin)
# time_end = time.time()
# print("przyblizenie: ", integral)

# err = np.abs(integral - 2)
# print("error: ", err)
# print("n:", sin["n"])
# print("Time:", time_end - time_start)

# x_dense = np.linspace(0, np.pi, 30)
# y_dense = sin["f"](x_dense)
# plt.plot(x_dense, y_dense)
# plt.bar(x, y, width=(np.pi - 0) / 30, alpha=0.3, align="edge")
# plt.show()


# ---------------- CURVE LENGTH ----------------

# circle circumference radius 1


# Funkcja koła y = sqrt(1 - x^2)
def circle_func(x):
    return np.sqrt(1 - x**2)


# Pochodna funkcji koła y' = -x / sqrt(1 - x^2), z obsługą wartości brzegowych
def circle_derivative(x):
    return np.divide(-x, np.sqrt(1 - x**2), where=np.abs(x) != 1)


# Metoda prostokątów do całkowania
def rectangle_rule_curve_length(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n, endpoint=False) + h / 2  # Środkowe punkty prostokątów
    y = f(x)
    return h * np.sum(y), x, y


# Funkcja do obliczania długości krzywej
def curve_length(f, f_prime, a, b, n):
    def integrand(x):
        return np.sqrt(1 + f_prime(x) ** 2)

    length, x, y = rectangle_rule_curve_length(integrand, a, b, n)
    return length, x, y


# a, b = 0, 1
# n = 100000000
# time_start = time.time()
# quarter_circle_length, x, y = curve_length(circle_func, circle_derivative, a, b, n)
# time_end = time.time()
# # Obwód koła to czterokrotność długości ćwiartki
# circle_circumference = 4 * quarter_circle_length
# print("Obwód koła:", circle_circumference)
# print("Przybliżenie liczby pi:", circle_circumference / 2)
# print("Time:", time_end - time_start)
# print("n", n)
# print("Error:", np.abs(circle_circumference - 2 * np.pi))

# Wykres funkcji i prostokątów
# integral, x_rect, y_rect = rectangle_rule_curve_length(
#     lambda x: np.sqrt(1 + circle_derivative(x) ** 2), a, b, n
# )
# x_dense = np.linspace(a, b, 1000)
# y_dense = np.sqrt(1 + circle_derivative(x_dense) ** 2)
# plt.plot(x_dense, y_dense, label="Integrand sqrt(1 + (f'(x))^2)")
# plt.bar(x_rect, y_rect, width=(b - a) / n, alpha=0.3, align="edge", label="Rectangles")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Metoda prostokątów dla obwodu koła")
# plt.legend()
# plt.show()


# --- ellipse circumference ---
from scipy.special import ellipe


def ellipse_func(x, a, b):
    return b * np.sqrt(1 - x**2 / a**2)


def ellipse_derivative(x, a, b):
    # return -b * x / (a**2 * np.sqrt(1 - x**2 / a**2))
    # watch out for division by zero
    return np.divide(-b * x, a**2 * np.sqrt(1 - x**2 / a**2), where=np.abs(x) != a)


def real_ellipse_circumference(a, b):
    e_sq = 1 - b**2 / a**2
    return 4 * a * ellipe(e_sq)


a, b = 10, 3
n = 100000000

# time_start = time.time()
# quarter_ellipse_length, x, y = curve_length(
#     lambda x: ellipse_func(x, a, b), lambda x: ellipse_derivative(x, a, b), -a, a, n
# )
# time_end = time.time()
# ellipse_circumference = 2 * quarter_ellipse_length
# print("Obwód elipsy:", ellipse_circumference)
# print("Time:", time_end - time_start)
# print("n", n)
# print("real circumference:", real_ellipse_circumference(a, b))
# print("Error:", np.abs(ellipse_circumference - real_ellipse_circumference(a, b)))


# wykres funkcji i prostokątów
# integral, x_rect, y_rect = rectangle_rule_curve_length(
#     lambda x: np.sqrt(1 + ellipse_derivative(x, a, b) ** 2), -a, a, n
# )
# x_dense = np.linspace(-a, a, n)
# y_dense = np.sqrt(1 + ellipse_derivative(x_dense, a, b) ** 2)
# plt.plot(x_dense, y_dense, label="Integrand sqrt(1 + (f'(x))^2)")
# plt.bar(x_rect, y_rect, width=(2 * a) / n, alpha=0.3, align="edge", label="Rectangles")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Metoda prostokątów dla obwodu elipsy")
# plt.legend()
# plt.show()


# --- sin curve length at [0, 2pi] ---
from scipy.integrate import quad


def f(x):
    return np.sqrt(1 + np.cos(x) ** 2)


def sin_derivative(x):
    return np.cos(x)


n = 1000
time_start = time.time()
sin_length, x, y = curve_length(np.sin, sin_derivative, 0, 2 * np.pi, n)
time_end = time.time()

# estimate true value of sin curve length from 0 to 2pi
m = 1
true_sin_length, _ = quad(f, 0, 2 * np.pi)
# print(true_sin_length)

print("Wynik:", sin_length)
print("n:", n)
print("error:", np.abs(sin_length - true_sin_length))
print("Time:", time_end - time_start)

# wykres funkcji i prostokątów
# integral, x_rect, y_rect = rectangle_rule_curve_length(
#     lambda x: np.sqrt(1 + np.cos(x) ** 2), 0, 2 * np.pi, n
# )
# x_dense = np.linspace(0, 2 * np.pi, 1000)
# y_dense = np.sqrt(1 + np.cos(x_dense) ** 2)
# plt.plot(x_dense, y_dense, label="Integrand sqrt(1 + (f'(x))^2)")
# plt.bar(
#     x_rect, y_rect, width=(2 * np.pi) / n, alpha=0.3, align="edge", label="Rectangles"
# )
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Metoda prostokątów dla długości sinusa")
# plt.legend()
# plt.show()
