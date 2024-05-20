import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2


def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x)
    return h * np.sum(y), x, y


a = 0  # lower bound
b = 1  # upper bound
n = 69
integral, x, y = rectangle_rule(f, a, b, n)
print(integral)

x_dense = np.linspace(a, b, 100)
y_dense = f(x_dense)
plt.plot(x_dense, y_dense)
plt.bar(x, y, width=(b - a) / n, alpha=0.3, align="edge")
plt.show()

# calculate the error
true_integral = 1 / 3
error = np.abs(integral - true_integral)
print(error)


# ---- rectangle rule for a circle
def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


circle = {
    "f": half_of_a_circle,
    "a": -1,
    "b": 1,
    "n": 1000,
}

integral, x, y = rectangle_rule(**circle)
print(integral)

x_dense = np.linspace(-1, 1, 1000)
y_dense = half_of_a_circle(x_dense)
plt.plot(x_dense, y_dense)
plt.bar(x, y, width=(1 - -1) / 1000, alpha=0.3, align="edge")
plt.show()

pi_approx = 2 * integral
print("pi approx:", pi_approx)

# ---- rectangle rule for a parabola
parabola = {
    "f": lambda x: x**2,
    "a": 0,
    "b": 1,
    "n": 1000,
}

integral, x, y = rectangle_rule(**parabola)
print(integral)

x_dense = np.linspace(0, 1, 1000)
y_dense = parabola["f"](x_dense)
plt.plot(x_dense, y_dense)
plt.bar(x, y, width=(1 - 0) / 1000, alpha=0.3, align="edge")
plt.show()

true_integral = 1 / 3

error = np.abs(integral - true_integral)
print(error)


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
    plt.show()


ellipse = {
    "f": f,
    "a": 2,
    "b": 1,
    "n": 1000,
    "ax": -2,
    "bx": 2,
}

integral = rectangle_rule_elipse(**ellipse)
print(integral)

plot_ellipse_and_approximation(f, 2, 1, 1000)

# calculate the error
true_integral = np.pi
error = np.abs(integral - true_integral)
print(error)

# ---- rectangle rule for a sine
sin = {
    "f": lambda x: np.sin(x),
    "a": 0,
    "b": np.pi,
    "n": 1000,
}


def rectangle_rule_sin(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)
    y = f(x)
    return h * np.sum(y), x, y


integral, x, y = rectangle_rule_sin(**sin)
print(integral)

err = np.abs(integral - 2)
print(err)

x_dense = np.linspace(0, np.pi, 1000)
y_dense = sin["f"](x_dense)
plt.plot(x_dense, y_dense)
plt.bar(x, y, width=(np.pi - 0) / 1000, alpha=0.3, align="edge")
plt.show()


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


a, b = 0, 1
n = 10000000
quarter_circle_length, x, y = curve_length(circle_func, circle_derivative, a, b, n)
# Obwód koła to czterokrotność długości ćwiartki
circle_circumference = 4 * quarter_circle_length
print("Obwód koła:", circle_circumference)
print("Przybliżenie liczby pi:", circle_circumference / 2)

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
def ellipse_func(x, a, b):
    return b * np.sqrt(1 - x**2 / a**2)


def ellipse_derivative(x, a, b):
    return -b * x / (a**2 * np.sqrt(1 - x**2 / a**2))


a, b = 2, 1
n = 100000

quarter_ellipse_length, x, y = curve_length(
    lambda x: ellipse_func(x, a, b), lambda x: ellipse_derivative(x, a, b), -a, a, n
)
ellipse_circumference = 2 * quarter_ellipse_length
print("Obwód elipsy:", ellipse_circumference)

# draw the ellipse
x_dense = np.linspace(-a, a, 1000)
y_dense = ellipse_func(x_dense, a, b)
plt.plot(x_dense, y_dense, label="Elipsa")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Elipsa")
plt.legend()
plt.show()


# --- sin curve length at [0, 2pi] ---
def sin_derivative(x):
    return np.cos(x)


n = 1000000
sin_length, x, y = curve_length(np.sin, sin_derivative, 0, 2 * np.pi, n)
print("Długość sinusa na przedziale [0, 2pi]:", sin_length)

# draw the sin curve
x_dense = np.linspace(0, 2 * np.pi, 1000)
y_dense = np.sin(x_dense)
plt.plot(x_dense, y_dense, label="Sinus")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sinus")
plt.legend()
plt.show()
