import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

### zweryfyikuj czy blad nie pojawia sie przez to ze zamieniamy zwykly graf na transformed graph


class MonteCarloTest:
    def __init__(self, graph, n, exit_vertices, osk_vertices, start_vertex):
        self.graph = graph
        self.n = n
        self.exit_vertices = exit_vertices
        self.osk_vertices = osk_vertices
        self.start_vertex = start_vertex
        self.current_vertex = self.start_vertex
        # self.visited = [self.start_vertex]
        self.visited = []

    def run(self):
        success_count = 0
        for _ in range(self.n):
            while True:
                result = self.move()
                if result is not None:
                    break
            if result:
                success_count += 1
            self.current_vertex = self.start_vertex
            # self.visited = [self.start_vertex]
        return success_count / self.n

    def move(self):
        if self.current_vertex in self.exit_vertices:
            return True
        elif self.current_vertex in self.osk_vertices:
            return False
        else:
            # print(self.graph[self.current_vertex])
            # print("possible choice: ", self.graph[self.current_vertex])
            self.current_vertex = random.choice(self.graph[self.current_vertex])
            # print("chosen vertex: ", self.current_vertex)
            # self.visited.append(self.current_vertex)
            return None


# Przykładowy graf
# graph2 = {
#     1: [6, 7, 8, 9],
#     6: [1, 2],
#     2: [6],
#     7: [1, 3],
#     3: [7],
#     8: [1, 4],
#     4: [8],
#     9: [1, 5],
#     5: [9],
# }

# graph = {
#     2: [11, 18],
#     11: [2, 12],
#     12: [11, 13],
#     13: [12, 14],
#     14: [13, 1],
#     1: [14, 15, 19, 22],
#     15: [1, 16],
#     16: [15, 17],
#     17: [16, 18],
#     18: [17, 2],
#     19: [1, 20],
#     20: [19, 3],
#     3: [20, 21, 23, 24],
#     21: [3, 22],
#     22: [21, 1],
#     23: [3, 4],
#     4: [23, 24, 5, 25, 28, 5, 35, 40],
#     24: [4, 3],
#     5: [4, 4, 31, 37],
#     25: [4, 26],
#     26: [25, 27],
#     27: [26, 6],
#     6: [27, 33],
#     28: [4, 29],
#     29: [28, 30],
#     30: [29, 8],
#     8: [30, 38, 9, 9],
#     31: [5, 32],
#     32: [31, 7],
#     7: [32, 36],
#     33: [6, 34],
#     34: [33, 35],
#     35: [34, 4],
#     36: [7, 37],
#     37: [36, 5],
#     38: [8, 39],
#     39: [38, 40],
#     40: [39, 4],
#     9: [8, 8, 41, 42],
#     41: [9, 10],
#     10: [41, 42],
#     42: [10, 9],
# }

# graph = dict(sorted(graph.items()))

# # Przykładowe dane testowe
# # start_vertex = 5
# exit_vertices = [5, 9]
# osk_vertices = [1, 10, 6]

# # start_verteces = [2, 3, 4, 5, 7, 8, 9]
# start_verteces = []


# for i in range(len(graph)):
#     if i + 1 not in osk_vertices:
#         start_verteces.append(i + 1)
# n = [1000]
# # print(start_verteces)

# results = []
# res = [0] * len(graph)


# for start_vertex in start_verteces:
#     for num in n:
#         monte_carlo = MonteCarloTest(
#             graph, num, exit_vertices, osk_vertices, start_vertex
#         )
#         success_rate = monte_carlo.run()
#         # print(
#         #     f"Start vertex: {start_vertex}, number of trials: {num}, success rate: {success_rate}"
#         # )
#         results.append((start_vertex, num, success_rate))
#         res[start_vertex - 1] = success_rate


# # print(res)


# def draw_graph(graph_dict):
#     G = nx.MultiDiGraph()
#     for key in graph_dict:
#         for value in graph_dict[key]:
#             G.add_edge(key, value)
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True)
#     plt.show()


# draw_graph(graph)


# # Plot results
# start_vertices = [result[0] for result in results]
# num_trials = [result[1] for result in results]
# success_rates = [result[2] for result in results]

# plt.figure(figsize=(10, 6))
# plt.bar(start_vertices, success_rates)
# plt.xlabel("Start Vertex")
# plt.ylabel("Success Rate")
# plt.title("Monte Carlo Test Results")

# # Adjust y-axis ticks
# plt.yticks(np.arange(0, 1.1, 0.1))

# # Add a horizontal line at y=0.5
# plt.axhline(y=0.5, color="red", linestyle="dotted")

# plt.show()
# monte_carlo = MonteCarloTest(graph, 100, exit_vertices, osk_vertices, start_vertex)
# print(monte_carlo.run())
