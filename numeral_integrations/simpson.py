# numerical integration using Simpson's rule
import numpy as np
import matplotlib.pyplot as plt


def half_of_a_circle(x):
    return np.sqrt(1 - x**2)


def simpson_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    # integral = (3 / 8) * h * (y[0] + 3 * y[1] + 3 * y[2] + 2 * y[3] + 3 * y[4] + 3 * y[5] + 2 * y[6] + y[7] + y[8] // two possibilities
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

        # coefficients of the parabola (ax^2 + bx + c)
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


circle = {
    "f": half_of_a_circle,
    "a": -1,
    "b": 1,
    "n": 1000,
}

integral, x, y = simpson_rule(**circle)
print(integral)

plot_simpson_parabolas(**circle)

pi_approx = 2 * integral
print("pi approx:", pi_approx)

# -----

parabola = {
    "f": lambda x: x**2,
    "a": 0,
    "b": 1,
    "n": 1000,
}

integral, x, y = simpson_rule(**parabola)
print(integral)

plot_simpson_parabolas(**parabola)

true_integral = 1 / 3

error = np.abs(integral - true_integral)
print(error)

# ------


def f(x, a, b):
    return b * np.sqrt(1 - (x**2 / a**2))


def simpson_rule_elipse(f, a, b, n, ax, bx):
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


# Define the interval and number of subintervals
a_values = [1, 2, 3]  # semi-major axis lengths
b_values = [0.5, 1, 1.5]  # semi-minor axis lengths
n = 10  # number of subintervals (should be even)

for a, b in zip(a_values, b_values):
    integral = simpson_rule_elipse(f, a, b, n, -a, a)
    area_approximation = 2 * integral
    print(f"Approximate area of ellipse with a={a} and b={b}: {area_approximation}")

    # Plot the function and Simpson's rule approximation
    plot_ellipse_and_approximation(f, a, b, n)

# -----
sin_pi = {
    "f": lambda x: np.sin(x),
    "a": 0,
    "b": np.pi,
    "n": 4,
}

integral, x, y = simpson_rule(**sin_pi)
print(integral)

plot_simpson_parabolas(**sin_pi)

true_integral = 2
error = np.abs(integral - true_integral)
print(error)


# ---------- CURVE LENGTH ----------
import numdifftools as nd


def f(x):
    return np.where(np.abs(x) <= 1, np.sqrt(1 - x**2), 0)


def f_prime(x):
    dfdx = nd.Derivative(f, n=1)
    return dfdx(x)


def integrand(x):
    return np.sqrt(1 + f_prime(x) ** 2)


def simpson_rule_integral(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return integral


def plot_function_and_approximation(f, a, b, n):
    x_dense = np.linspace(a, b, 1000)
    y_dense = f(x_dense)
    plt.plot(x_dense, y_dense, label="f(x) = sqrt(1 - x^2)")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

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
    plt.ylabel("f(x)")
    plt.legend()
    plt.title("Simpson's Rule Approximation for sqrt(1 - x^2)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# Define the interval and number of subintervals
a = -1  # lower bound
b = 1  # upper bound
n = 2  # number of subintervals (should be even)

# Check the derivative function for correctness
x_test = np.linspace(a, b, 5)
print("x values:", x_test)
print("f(x) values:", f(x_test))
print("f'(x) values:", f_prime(x_test))

# Compute the integral using Simpson's rule for the length of the curve
length = simpson_rule_integral(integrand, a, b, n)
pi_approximation = (
    2 * length
)  # the length of the full circle is 2 times the length of the semicircle
print(f"Approximate value of π: {pi_approximation}")

# Plot the function and Simpson's rule approximation
plot_function_and_approximation(f, a, b, n)
