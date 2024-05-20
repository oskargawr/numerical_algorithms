import GraphBuilder
from gaussian import Gaussian
from gaussianPivotal import GaussianPivotal
from MatrixBuilder import MatrixBuilder
from gaussianSeidal import GaussianSeidal
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from scipy.stats import ttest_ind
import numpy as np
import time
from monteCarlo import MonteCarloTest
from RandomGraph import RandomGraph


def draw_graph(graph_dict):
    G = nx.MultiDiGraph()

    for node, edges in graph_dict.items():
        for edge, weight in edges:
            G.add_edge(node, edge, weight=weight, label=weight)

    pos = nx.spring_layout(G)
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw(G, pos, with_labels=True, connectionstyle="arc3, rad = 0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def draw_transformed_graph_with_colors(graph_dict, start, exit, osk, trash_cans):
    G = nx.DiGraph()

    for node, edges in graph_dict.items():
        for edge in edges:
            G.add_edge(node, edge)

    pos = nx.spring_layout(G)

    color_map = []
    for node in G:
        if node in osk:
            color_map.append("red")
        elif node in exit:
            color_map.append("green")
        elif node in trash_cans:
            color_map.append("orange")
        elif node in start:
            color_map.append("yellow")
        else:
            color_map.append("blue")

    nx.draw(G, pos, with_labels=True, node_color=color_map)
    plt.show()


random_graph = RandomGraph(15, 15, 5, 4, 10)
# data2 = random_graph.get_data()
# print(data2)
data2 = {
    "alleys": 19,
    "intersections": 15,
    "osk": [8, 5, 11, 15, 4],
    "exit": [6, 10, 1, 14],
    "start": 11,
    "routes": [
        [1, 2, 8],
        [1, 15, 3],
        [2, 3, 3],
        [2, 12, 5],
        [3, 12, 9],
        [3, 14, 4],
        [3, 13, 10],
        [4, 12, 7],
        [5, 12, 3],
        [6, 8, 2],
        [7, 15, 2],
        [7, 14, 5],
        [7, 13, 3],
        [7, 8, 6],
        [8, 12, 7],
        [9, 12, 5],
        [10, 15, 1],
        [11, 12, 8],
        [12, 14, 2],
    ],
    "trash_cans": 0,
}
# data2 = {
#     "alleys": 1
# }
# print("data: ", data)
# print("data2: ", data2)

graph_builder = GraphBuilder.GraphBuilder(**data2)
graph = graph_builder.build_graph()
# graph_builder.visualize_graph()
transformed_graph = graph_builder.extend_routes()
print("transformed graph: ", transformed_graph)

matrix, vector = graph_builder.build_matrix()
print("Matrix:")
# matrix.display()
print("Vector:")
print(vector)

n = 100000
start_verteces = []
for i in range(len(graph)):
    if i + 1 not in data2["osk"]:
        start_verteces.append(i + 1)

print("start: ", start_verteces)
print("osk: ", data2["osk"])
print("exit: ", data2["exit"])

results = [0] * len(graph)

for start_vertex in start_verteces:
    monte_carlo = MonteCarloTest(
        dict(sorted(transformed_graph.items())),
        n,
        data2["exit"],
        data2["osk"],
        start_vertex,
    )
    success_rate = monte_carlo.run()
    print(
        f"Start vertex: {start_vertex}, number of trials: {n}, success rate: {success_rate}"
    )
    results[start_vertex - 1] = success_rate

print("Results: ", results)

## histogram for monte carlo results with x-axis as the start vertex and y-axis as the success rate
# plt.bar(range(1, len(results) + 1), results)
# plt.xlabel("Start Vertex")
# plt.ylabel("Success Rate")
# plt.show()


gauss_solution = Gaussian(matrix, vector).solve()
print("Gaussian solution: ", gauss_solution)

gauss_pivotal_solution = GaussianPivotal(matrix, vector).solve()
print("Gaussian pivotal solution: ", gauss_pivotal_solution)

gauss_seidal_solution = GaussianSeidal(matrix, vector).solve()
print("Gaussian Seidal solution: ", gauss_seidal_solution)

gauss_solution = gauss_solution[: len(results)]
gauss_pivotal_solution = gauss_pivotal_solution[: len(results)]
gauss_seidal_solution = gauss_seidal_solution[: len(results)]

difference_gauss = np.subtract(results, gauss_solution)
difference_gauss_pivotal = np.subtract(results, gauss_pivotal_solution)
difference_gauss_seidal = np.subtract(results, gauss_seidal_solution)

# Print the differences
print("Difference between Monte Carlo and Gaussian: ", difference_gauss)
print("Difference between Monte Carlo and Gaussian Pivotal: ", difference_gauss_pivotal)
print("Difference between Monte Carlo and Gaussian Seidal: ", difference_gauss_seidal)

print(difference_gauss)
print(difference_gauss_pivotal)
print(difference_gauss_seidal)


## H1
# Function to calculate the error of a solution
def h1():
    def calculate_error(matrix_builder, vector, solution):
        matrix = np.array(MatrixBuilder.to_matrix(matrix_builder))
        return np.linalg.norm(np.dot(matrix, solution) - vector)

    # Lists to store errors
    errors_standard = []
    errors_pivoting = []

    # List to store matrix sizes
    matrix_sizes = []

    for i in range(1, 6):
        data = RandomGraph(10 * i, 10 * i, 3 * i, 2 * i, 3 * i).get_data()

        # print(data)
        matrix_sizes.append(10 * i)
        print("number of intersections: ", 10 * i)
        print("number of alleys: ", 20 * i)
        print("number of osk: ", 3 * i)
        print("number of exit: ", 2 * i)
        print("max alley length: ", 3 * i)
        graph_builder = GraphBuilder.GraphBuilder(**data)
        graph = graph_builder.build_graph()
        transformed_graph = graph_builder.extend_routes()
        matrix, vector = graph_builder.build_matrix()

        gaussian = Gaussian(matrix, vector)
        solution = gaussian.solve()
        error = calculate_error(matrix, vector, solution)
        errors_standard.append(error)

        gaussian_pivotal = GaussianPivotal(matrix, vector)
        solution_pivotal = gaussian_pivotal.solve()
        error_pivotal = calculate_error(matrix, vector, solution_pivotal)
        errors_pivoting.append(error_pivotal)

        print(f"Error standard: {error}")
        print(f"Error pivotal: {error_pivotal}")

    plt.figure(figsize=(10, 6))
    plt.plot(errors_standard, label="Standard Gaussian")
    plt.plot(errors_pivoting, label="Gaussian with Partial Pivoting")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes)
    plt.xlabel("Matrix Size")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.title("Error Comparison")
    plt.legend()
    plt.show()

    errors_difference = np.array(errors_standard) - np.array(errors_pivoting)

    plt.figure(figsize=(10, 6))
    plt.plot(errors_difference, label="Difference in Errors")
    plt.xticks(range(len(matrix_sizes)), matrix_sizes)
    plt.xlabel("Matrix Size")
    plt.ylabel("Error Difference")
    plt.title(
        "Difference in Errors Between Standard Gaussian and Gaussian with Partial Pivoting"
    )
    plt.legend()
    plt.show()


h1()


# H2 - Gaussian Seidal algorithm works for the given task (if it doesnt, meaning the process does not converge to a solution, then provide with examples when it diverges)
# # gaussian_seidal1 = GaussianSeidal(matrix1, vector1)
# solution_seidal1 = gaussian_seidal1.solve()
# print("Solution 1: ", solution_seidal1)
def h2():
    def test_gauss_seidal_convergence(matrix, vector):
        gauss_seidal = GaussianSeidal(matrix, vector)
        solution = gauss_seidal.solve()
        return solution is not None

    def example_of_non_convergence(matrix, vector):
        gauss_seidal = GaussianSeidal(matrix, vector)
        solution = gauss_seidal.solve()
        return solution

    # Generowanie losowych danych i sprawdzenie zbieżności metody Gaussa-Seidela
    non_convergent_examples = []
    for i in range(1, 100):  # Sprawdź trzy przypadki rozbieżności
        random_graph = RandomGraph(10 * i, 10 * i, 3 * i, 2 * i, 3 * i)
        data = random_graph.get_data()

        graph_builder = GraphBuilder.GraphBuilder(**data)
        transformed_graph = graph_builder.extend_routes()
        matrix, vector = graph_builder.build_matrix()

        is_convergent = test_gauss_seidal_convergence(matrix, vector)
        if not is_convergent:
            example_solution = example_of_non_convergence(matrix, vector)
            non_convergent_examples.append((matrix, vector, example_solution))
            print(
                f"Przykład {len(non_convergent_examples)}: Metoda Gaussa-Seidela nie jest zbieżna."
            )

        if len(non_convergent_examples) >= 3:
            break

    # Wyświetlenie wyników
    if len(non_convergent_examples) > 0:
        print(
            "Przykłady rozwiązań, dla których metoda Gaussa-Seidela nie jest zbieżna:"
        )
        for idx, (matrix, vector, solution) in enumerate(
            non_convergent_examples, start=1
        ):
            print(f"Przykład {idx}:")
            print("Macierz:")
            print(matrix)
            print("Wektor:")
            print(vector)
            print("Rozwiązanie:")
            print(solution)
            print()
    else:
        print(
            "Metoda Gaussa-Seidela jest zbieżna dla wszystkich sprawdzonych przypadków."
        )


# h2()


def has_converged(old_solution, new_solution, tolerance=1e-5):
    return np.linalg.norm(new_solution - old_solution) < tolerance


def test_gaussian_seidal(max_iterations=1000, tolerance=1e-5):
    non_convergent_examples = []
    convergent_examples = []
    for i in range(1, 20):  # Check 100 cases
        random_graph = RandomGraph(10 * i, 10 * i, 3 * i, 2 * i, 3 * i)
        data = random_graph.get_data()

        graph_builder = GraphBuilder.GraphBuilder(**data)
        transformed_graph = graph_builder.extend_routes()
        matrix, vector = graph_builder.build_matrix()

        # Initialize the GaussianSeidal algorithm
        gaussian_seidal = GaussianSeidal(matrix, vector)

        old_solution = None
        for _ in range(max_iterations):
            new_solution = gaussian_seidal.solve()
            if old_solution is not None and has_converged(
                np.array(old_solution), np.array(new_solution), tolerance
            ):
                convergent_examples.append((matrix, vector, new_solution))
                break  # Converged, so break the loop
            old_solution = new_solution

        else:  # No break, so did not converge
            non_convergent_examples.append((matrix, vector, old_solution))
            print(
                f"Example {len(non_convergent_examples)}: The Gaussian Seidal method did not converge."
            )

        if (
            len(non_convergent_examples) >= 3
        ):  # Stop after finding 3 non-convergent examples
            break

    return non_convergent_examples


# print(test_gaussian_seidal())


## compare the speed of the Gaussian, Gaussian with partial pivoting and Gaussian Seidal algorithms for randomly generated data and create a chart
def h3():
    def test_gaussian_speed(matrix, vector):
        gaussian = Gaussian(matrix, vector)
        start_time = time.time()
        solution = gaussian.solve()
        end_time = time.time()
        return end_time - start_time

    def test_gaussian_pivotal_speed(matrix, vector):
        gaussian_pivotal = GaussianPivotal(matrix, vector)
        start_time = time.time()
        solution = gaussian_pivotal.solve()
        end_time = time.time()
        return end_time - start_time

    def test_gaussian_seidal_speed(matrix, vector):
        gaussian_seidal = GaussianSeidal(matrix, vector)
        start_time = time.time()
        solution = gaussian_seidal.solve()
        end_time = time.time()
        return end_time - start_time

    # Lists to store the times taken by each algorithm
    times_gaussian = []
    times_gaussian_pivotal = []
    times_gaussian_seidal = []

    # List to store matrix sizes
    matrix_sizes = []

    for i in range(1, 6):
        random_graph = RandomGraph(10 * i, 10 * i, 3 * i, 2 * i, 3 * i)
        data = random_graph.get_data()

        graph_builder = GraphBuilder.GraphBuilder(**data)
        transformed_graph = graph_builder.extend_routes()
        matrix, vector = graph_builder.build_matrix()

        matrix_sizes.append(10 * i)
        print("number of intersections: ", 10 * i)
        print("number of alleys: ", 20 * i)
        print("number of osk: ", 2 * i)
        print("number of exit: ", 2 * i)
        print("number of start: ", 2 * i)
        print("number of trash cans: ", 0)
        print("max alley length: ", 10 * i)

        # Gaussian
        time_gaussian = test_gaussian_speed(matrix, vector)
        times_gaussian.append(time_gaussian)

        # Gaussian with partial pivoting
        time_gaussian_pivotal = test_gaussian_pivotal_speed(matrix, vector)
        times_gaussian_pivotal.append(time_gaussian_pivotal)

        # Gaussian Seidal
        time_gaussian_seidal = test_gaussian_seidal_speed(matrix, vector)
        times_gaussian_seidal.append(time_gaussian_seidal)

    print(f"Total times Gaussian: {sum(times_gaussian)}")
    print(f"Total times Gaussian with Partial Pivoting: {sum(times_gaussian_pivotal)}")
    print(f"Total times Gaussian Seidal: {sum(times_gaussian_seidal)}")
    # Plot the times taken by each algorithm
    plt.figure(figsize=(10, 6))
    plt.plot(times_gaussian, label="Gaussian")
    plt.plot(times_gaussian_pivotal, label="Gaussian with Partial pivoting")

    plt.plot(times_gaussian_seidal, label="Gaussian Seidal")
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.title("Time Comparison")
    plt.legend()
    plt.show()

    # difference between Gausian with Partial Pivoting and Gaussian Seidal
    times_difference = np.abs(
        np.array(times_gaussian_pivotal) - np.array(times_gaussian_seidal)
    )
    # plot the difference too

    plt.figure(figsize=(10, 6))
    plt.plot(times_difference, label="Difference in Times")
    plt.yscale("log")  # Use a logarithmic scale for the y-axis
    plt.xlabel("Matrix Size")
    plt.ylabel("Time Difference (log scale)")
    plt.title(
        "Difference in Times Between Gaussian with Partial Pivoting and Gaussian Seidal"
    )
    plt.legend()
    plt.show()


# h3()
