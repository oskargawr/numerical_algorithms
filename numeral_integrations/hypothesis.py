import functions as f
import numpy as np
import matplotlib.pyplot as plt
import time


# Definicje funkcji do testu
def f1():
    return lambda x: 4 * np.sin(2 * x) + 2 * np.exp(3)


f1_true_val = 83.649435


def f2():
    return lambda x: 13 * np.sin(x) * 12 * np.cos(x) * 11 * np.tan(x)


f2_true_val = 2040.6683


def f3():
    return lambda x: -(3 ** (1 + 2 * np.cos(3 * x))) * np.log(9) * np.sin(3 * x)


f3_true_val = -0.7541524


def H1():
    test_functions = [f1(), f2(), f3()]
    true_values = [f1_true_val, f2_true_val, f3_true_val]
    n_values = range(100, 10010, 1000)
    a1_errors = {i: [] for i in range(1, 4)}
    a2_errors = {i: [] for i in range(1, 4)}
    a1_times = {i: [] for i in range(1, 4)}
    a2_times = {i: [] for i in range(1, 4)}

    for i, func in enumerate(test_functions):
        true_val = true_values[i]
        print(f"Testing function f{i+1}...")

        for n in n_values:
            start_time = time.time()
            A1, _, _ = f.rectangle_rule(func, 0, 2, n)
            end_time = time.time()
            a1_errors[i + 1].append(np.abs(A1 - true_val))
            a1_times[i + 1].append(end_time - start_time)

            start_time = time.time()
            A2, _, _ = f.trapezoidal_rule(func, 0, 2, n)
            end_time = time.time()
            a2_errors[i + 1].append(np.abs(A2 - true_val))
            a2_times[i + 1].append(end_time - start_time)

            print(
                f"Function {i+1}, n={n}, A1 Error={np.abs(A1 - true_val)}, A2 Error={np.abs(A2 - true_val)}"
            )

    for i in range(1, 4):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(n_values, a1_errors[i], label="A1 - Rectangle", marker="o")
        plt.plot(n_values, a2_errors[i], label="A2 - Trapezoidal", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Error")
        plt.yscale("log")
        plt.title(f"Error for function f{i}")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(n_values, a1_times[i], label="A1 - Rectangle", marker="o")
        plt.plot(n_values, a2_times[i], label="A2 - Trapezoidal", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Time (s)")
        plt.title(f"Time for function f{i}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# H1()


def H2():
    test_functions = [f1(), f2(), f3()]
    true_values = [f1_true_val, f2_true_val, f3_true_val]
    n_values = range(100, 10010, 1000)
    a2_errors = {i: [] for i in range(1, 4)}
    a3_errors = {i: [] for i in range(1, 4)}
    a2_times = {i: [] for i in range(1, 4)}
    a3_times = {i: [] for i in range(1, 4)}

    for i, func in enumerate(test_functions):
        true_val = true_values[i]
        print(f"Testing function f{i+1}...")

        for n in n_values:
            start_time = time.time()
            A2, *_ = f.trapezoidal_rule(func, 0, 2, n)
            end_time = time.time()
            a2_errors[i + 1].append(np.abs(A2 - true_val))
            a2_times[i + 1].append(end_time - start_time)

            start_time = time.time()
            A3, *_ = f.simpson_rule_area(func, 0, 2, n)
            end_time = time.time()
            a3_errors[i + 1].append(np.abs(A3 - true_val))
            a3_times[i + 1].append(end_time - start_time)

            print(
                f"Function {i+1}, n={n}, A2 Error={np.abs(A2 - true_val)}, A3 Error={np.abs(A3 - true_val)}"
            )

    for i in range(1, 4):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(n_values, a2_errors[i], label="A2 - Trapezoidal", marker="o")
        plt.plot(n_values, a3_errors[i], label="A3 - Simpson", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Error")
        plt.title(f"Error for function f{i}")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(n_values, a2_times[i], label="A2 - Trapezoidal", marker="o")
        plt.plot(n_values, a3_times[i], label="A3 - Simpson", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Time (s)")
        plt.title(f"Time for function f{i}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# H2()


def H3():
    test_functions = [f1(), f2(), f3()]
    true_values = [f1_true_val, f2_true_val, f3_true_val]
    n_values = range(100, 10010, 1000)
    a3_errors = {i: [] for i in range(1, 4)}
    a4_errors = {i: [] for i in range(1, 4)}
    a3_times = {i: [] for i in range(1, 4)}
    a4_times = {i: [] for i in range(1, 4)}

    for i, func in enumerate(test_functions):
        true_val = true_values[i]
        print(f"Testing function f{i+1}...")

        for n in n_values:
            start_time = time.time()
            A3, *_ = f.simpson_rule_area(func, 0, 2, n)
            end_time = time.time()
            a3_errors[i + 1].append(np.abs(A3 - true_val))
            a3_times[i + 1].append(end_time - start_time)

            start_time = time.time()
            A4 = f.spline_interpolation_area(0, 2, n, func)
            end_time = time.time()
            a4_errors[i + 1].append(np.abs(A4 - true_val))
            a4_times[i + 1].append(end_time - start_time)

            print(
                f"Function {i+1}, n={n}, A3 Error={np.abs(A3 - true_val)}, A4 Error={np.abs(A4 - true_val)}"
            )

    for i in range(1, 4):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(n_values, a3_errors[i], label="A3 - Simpson", marker="o")
        plt.plot(n_values, a4_errors[i], label="A4 - Spline Interpolation", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Error")
        plt.title(f"Error for function f{i}")
        plt.yscale("log")  # Ustawienie skali logarytmicznej dla błędów
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(n_values, a3_times[i], label="A3 - Simpson", marker="o")
        plt.plot(n_values, a4_times[i], label="A4 - Spline Interpolation", marker="o")
        plt.xlabel("Number of segments (n)")
        plt.ylabel("Time (s)")
        plt.title(f"Time for function f{i}")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


H3()
