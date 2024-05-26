# numerical integration using the trapezoidal rule

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2


def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1])), x, y


a = 0  # lower bound
b = 1  # upper bound
n = 3  # number of trapezoids
integral, x, y = trapezoidal_rule(f, a, b, n)
print(integral)

x_dense = np.linspace(a, b, 100)
y_dense = f(x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

# calculate the error
true_integral = 1 / 3
error = np.abs(integral - true_integral)
print(error)


# ---- trapezoidal rule for a circle
def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


circle = {
    "f": half_of_a_circle,
    "a": -1,
    "b": 1,
    "n": 1000,
}

integral, x, y = trapezoidal_rule(**circle)
print(integral)

x_dense = np.linspace(-1, 1, 1000)
y_dense = half_of_a_circle(x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

pi_approx = 2 * integral
print("pi approx:", pi_approx)

# ---- trapezoidal rule for a parabola
parabola = {
    "f": lambda x: x**2,
    "a": 0,
    "b": 1,
    "n": 1000,
}

integral, x, y = trapezoidal_rule(**parabola)
print(integral)

x_dense = np.linspace(0, 1, 1000)
y_dense = parabola["f"](x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

true_integral = 1 / 3
error = np.abs(integral - true_integral)
print(error)


# ---- trapezoidal rule for an ellipse
from scipy.special import ellipe


def f(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def real_ellipse_circumference(a, b):
    e_sq = 1 - b**2 / a**2
    return 4 * a * ellipe(e_sq)


def trapezoidal_rule_elipse(f, a, b, n, ax, bx):
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
    plt.show()


ellipse = {
    "f": f,
    "a": 1,
    "b": 4,
    "n": 10,
    "ax": -1,
    "bx": 1,
}

integral = trapezoidal_rule_elipse(**ellipse)
print(integral)
plot_ellipse_and_approximation(**ellipse)

# calculate the error
a = ellipse["a"]
b = ellipse["b"]
true_integral = np.pi * a * b
error = np.abs(integral - true_integral)
print(error)


# ---- trapezoidal rule for a sine function
def f(x):
    return np.sin(x)


sin = {
    "f": f,
    "a": 0,
    "b": np.pi,
    "n": 10,
}

integral, x, y = trapezoidal_rule(**sin)
print(integral)

x_dense = np.linspace(0, np.pi, 1000)
y_dense = sin["f"](x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

true_integral = 2
error = np.abs(integral - true_integral)
print(error)


# ---------------- CURVE LENGTH USING TRAPEZOIDAL RULE ----------------
# calculate the curve length of a circle using the trapezoidal rule with a circumference of 1


def circle(x):
    return np.sqrt(1 - x**2)


def circle_derivative(x):
    return np.divide(-x, np.sqrt(1 - x**2), where=np.abs(x) != 1)


def curve_length(f, f_derivative, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f_derivative(x)
    return h * np.sum(np.sqrt(1 + y**2)), x, y


circle = {
    "f": circle,
    "f_derivative": circle_derivative,
    "a": 0,
    "b": 1,
    "n": 100000,
}

integral, x, y = curve_length(**circle)
print(integral)

pi_approx = 2 * integral
print("pi approx:", pi_approx)

x_dense = np.linspace(-1, 1, 1000)
y_dense = circle_derivative(x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

# calculate the error
true_integral = np.pi
error = np.abs(integral - true_integral)
print(error)


# ---- curve length of an ellipse
def ellipse(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def ellipse_derivative(x, a, b):
    return np.divide(-b * x, a**2 * np.sqrt(1 - (x**2 / a**2)), where=np.abs(x) != a)


a, b = 1, 4
n = 1000
elipse_length = curve_length(
    lambda x: ellipse(x, a, b), lambda x: ellipse_derivative(x, a, b), -a, a, n
)

integral, x, y = elipse_length
print(integral)
true_integral = 4 * np.pi * a
error = np.abs(integral - true_integral)
print(error)

x_dense = np.linspace(-a, a, 1000)
y_dense = ellipse_derivative(x_dense, a, b)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()

# ---- curve length of a sine function at [0, 2*pi]
sin_length = curve_length(np.sin, np.cos, 0, 2 * np.pi, 1000)
integral, x, y = sin_length
print(integral)
true_integral = 2 * np.pi
error = np.abs(integral - true_integral)

x_dense = np.linspace(0, 2 * np.pi, 1000)
y_dense = np.cos(x_dense)
plt.plot(x_dense, y_dense)
plt.fill_between(x, y, alpha=0.3)
plt.show()
print("err: ", error)
