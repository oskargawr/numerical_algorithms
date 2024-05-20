data = {
    1: [6, 7, 8, 9],
    6: [1, 2],
    2: [6],
    7: [1, 3],
    3: [7],
    8: [1, 4],
    4: [8],
    9: [1, 5],
    5: [9],
}

sorted_data = dict(sorted(data.items()))

print(sorted_data)
# data1 = {
#     "alleys": 4,
#     "intersections": 5,
#     "osk": {1: [1]},
#     "exit": {1: [2]},
#     "start": {1: [3]},
#     "routes": [[1, 2, 2], [1, 3, 2], [1, 4, 2], [1, 5, 2]],
#     "trash_cans": {2: [4, 5]},
# }

# data2 = {
#     "alleys": 5,
#     "intersections": 4,
#     "osk": {1: [1]},
#     "exit": {2: [2, 4]},
#     "start": {1: [3]},
#     "routes": [[1, 2, 4], [2, 3, 4], [3, 4, 4], [4, 1, 4], [1, 3, 6]],
#     "trash_cans": {0: []},
# }


# graph_builder1 = GraphBuilder.GraphBuilder(**data1)
# graph_builder2 = GraphBuilder.GraphBuilder(**data2)
# graph_builder3 = GraphBuilder.GraphBuilder(**data3)

# graph1 = graph_builder1.build_graph()
# graph2 = graph_builder2.build_graph()
# graph3 = graph_builder3.build_graph()

# print("Graph 1: ", graph1)
# print("Graph 2: ", graph2)
# print("Graph 3: ", graph3)

# graph_builder1.visualize_graph()
# graph_builder2.visualize_graph()
# graph_builder3.visualize_graph()

# transformed_graph1 = graph_builder1.extend_routes()
# transformed_graph2 = graph_builder2.extend_routes()
# transformed_graph3 = graph_builder3.extend_routes()

# print("Transformed graph 1: ", transformed_graph1)
# draw_graph(transformed_graph1)
# print("Transformed graph 2: ", transformed_graph2)
# draw_graph(transformed_graph2)
# print("Transformed graph 3: ", transformed_graph3)
# draw_graph(transformed_graph3)
# draw_transformed_graph_with_colors(
#     transformed_graph3, data3["start"][1], data3["exit"][1], data3["osk"][1], []
# )

# matrix1, vector1 = graph_builder1.build_matrix()
# matrix1.display()
# print("Vector 1: ", vector1)

# matrix2, vector2 = graph_builder2.build_matrix()
# matrix2.display()
# print("Vector 2: ", vector2)

# matrix3, vector3 = graph_builder3.build_matrix()
# matrix3.display()
# print("Vector 3: ", vector3)

# gaussian1 = Gaussian(matrix1, vector1)
# solution1 = gaussian1.solve()
# print("Solution 1: ", solution1)

# gaussian2 = Gaussian(matrix2, vector2)
# solution2 = gaussian2.solve()
# print("Solution 2: ", solution2)

# gaussian3 = Gaussian(matrix3, vector3)
# solution3 = gaussian3.solve()
# print("Solution 3: ", solution3)


# gaussian_pivotal1 = GaussianPivotal(matrix1, vector1)
# solution_pivotal1 = gaussian_pivotal1.solve()

# gaussian_pivotal2 = GaussianPivotal(matrix2, vector2)
# solution_pivotal2 = gaussian_pivotal2.solve()

# gaussian_pivotal3 = GaussianPivotal(matrix3, vector3)
# solution_pivotal3 = gaussian_pivotal3.solve()
# print("Solution pivotal 3: ", solution_pivotal3)

# print("Solution pivotal 1: ", solution_pivotal1)
# print("Solution pivotal 2: ", solution_pivotal2)

# gaussian_seidal1 = GaussianSeidal(matrix1, vector1)
# solution_seidal1 = gaussian_seidal1.solve()

# gaussian_seidal2 = GaussianSeidal(matrix2, vector2)
# solution_seidal2 = gaussian_seidal2.solve()

# print("Solution seidal 1: ", solution_seidal1)
# print("Solution seidal 2: ", solution_seidal2)

# gaussian_seidal3 = GaussianSeidal(matrix3, vector3)
# solution_seidal3 = gaussian_seidal3.solve()
# print("Solution seidal 3: ", solution_seidal3)


# def generate_random_data(
#     num_intersections,
#     num_alleys,
#     num_osk,
#     num_exit,
#     num_start,
#     max_alley_length,
#     num_trash_cans,
# ):
#     graph = defaultdict(list)
#     for i in range(2, num_intersections + 1):
#         j = random.randint(1, i - 1)
#         weight = random.randint(1, max_alley_length)
#         graph[i].append((j, weight))
#         graph[j].append((i, weight))

#     # print("1 etap: ", graph)

#     while len([edge for edges in graph.values() for edge in edges]) < num_alleys:
#         i = random.randint(1, num_intersections)
#         j = random.randint(1, num_intersections)
#         if i != j and j not in [v for v, w in graph[i]]:
#             weight = random.randint(5, max_alley_length)
#             graph[i].append((j, weight))
#             graph[j].append((i, weight))

#     # print("2 etap: ", graph)

#     vertices = list(range(1, num_intersections + 1))

#     random.shuffle(vertices)
#     osk_vertices = random.sample(vertices, num_osk)
#     exit_vertices = random.sample(set(vertices) - set(osk_vertices), num_exit)
#     start_vertex = random.choice(
#         list(set(vertices) - set(osk_vertices) - set(exit_vertices))
#     )

#     print(graph)

#     routes = [[i, j, w] for i, edges in graph.items() for j, w in edges]
#     trash_cans = {
#         i: random.sample(vertices, num_trash_cans)
#         for i in range(1, num_intersections + 1)
#     }

#     return {
#         "alleys": num_alleys,
#         "intersections": num_intersections,
#         "osk": {i: osk_vertices for i in range(1, num_intersections + 1)},
#         "exit": {i: exit_vertices for i in range(1, num_intersections + 1)},
#         "start": {1: [start_vertex]},
#         "routes": routes,
#         "trash_cans": trash_cans,
#     }


# data = generate_random_data(10, 10, 3, 2, 2, 10, 0)
