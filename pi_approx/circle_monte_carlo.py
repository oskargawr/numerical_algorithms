import random
import matplotlib.pyplot as plt
import numpy as np
import time


def estimate_pi(interval):
    circle_points_x = []
    circle_points_y = []
    square_points_x = []
    square_points_y = []

    for i in range(interval**2):
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)

        distance = rand_x**2 + rand_y**2

        if distance <= 1:
            circle_points_x.append(rand_x)
            circle_points_y.append(rand_y)
        else:
            square_points_x.append(rand_x)
            square_points_y.append(rand_y)

    pi = 4 * len(circle_points_x) / (len(circle_points_x) + len(square_points_x))
    return pi


def find_iterations_for_accuracy():
    n = 1
    pi_estimate = estimate_pi(n)
    while round(pi_estimate, 4) != round(np.pi, 4):
        n += 1
        pi_estimate = estimate_pi(n)
    return n


# execution_times = []

# start_time = time.time()
# n_required = find_iterations_for_accuracy()
# execution_time = time.time() - start_time
# execution_times.append(execution_time)
# print(f"Liczba iteracji potrzebna dla dokładności 0.00001: {n_required}")
# print(f"Czas wykonania dla dokładności 0.0001: {execution_time:.4f} sekundy")


# print("Estimated value of pi:", estimate_pi(50))

# n_test = np.geomspace(10, 10000, 100)
n_test = [10, 100, 1000, 10000]
pi_estimates = []

for n in n_test:
    n = int(n)
    pi_estimates.append(estimate_pi(n))
    print("Liczba iteracji:", n)
    print("Przyblizona wartosc pi:", estimate_pi(n))
    print("Roznica od rzeczywistej wartosci pi:", np.pi - estimate_pi(n))


##### log roznica

# pi_difference = np.abs(np.pi - np.array(pi_estimates))


# plt.plot(
#     n_test,
#     np.abs([est - np.pi for est in pi_estimates]),
#     marker="o",
#     linestyle="-",
#     color="b",
# )
# # plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Liczba iteracji")
# plt.ylabel("Logarytm różnicy")
# plt.title("Logarytm różnicy między rzeczywistą wartością Pi a przybliżoną")
# plt.grid(True)
# plt.show()

# plt.plot(
#     n_test, pi_estimates, marker="o", linestyle="-", color="b", label="Estimated Pi"
# )
# plt.axhline(y=np.pi, color="r", linestyle="--", label="Actual Pi")
# plt.xscale("log")
# plt.xlabel("Number of iterations")
# plt.ylabel("Value of Pi")
# plt.title("Estimation of Pi using Monte Carlo method")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 8))
# plt.scatter(square_points_x, square_points_y, color="blue", label="Poza kołem")
# plt.scatter(circle_points_x, circle_points_y, color="red", label="Wewnątrz koła")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Monte Carlo Estimation of Pi")
# plt.axis("equal")
# plt.grid(True)


# circle = plt.Circle((0, 0), 1, color="black", fill=False)
# plt.gca().add_patch(circle)

# plt.legend()
# plt.show()
