import networkx as nx
import matplotlib.pyplot as plt
from MatrixBuilder import MatrixBuilder as Matrix
from collections import OrderedDict


class GraphBuilder:
    def __init__(self, alleys, intersections, osk, exit, start, routes, trash_cans):
        self.alleys = alleys
        self.intersections = intersections
        self.osk = osk
        self.exit = exit
        self.start = start
        self.routes = routes
        self.trash_cans = trash_cans
        self.graph = {}
        self.transformed_graph = {}
        self.intermediate_node_counter = max([max(edge[:2]) for edge in routes]) + 1
        self.node_mapping = {}

    def build_graph(self):
        for edge in self.routes:
            node1, node2, weight = edge
            if node1 in self.graph:
                self.graph[node1].append((node2, weight))
            else:
                self.graph[node1] = [(node2, weight)]
            if node2 in self.graph:
                self.graph[node2].append((node1, weight))
            else:
                self.graph[node2] = [(node1, weight)]
        return self.graph

    def visualize_graph(self):
        G = nx.Graph()
        for node, edges in self.graph.items():
            for edge in edges:
                G.add_edge(node, edge[0], weight=edge[1])
        pos = nx.spring_layout(G)

        # osk_values = [item for sublist in self.osk.values() for item in sublist]
        # exit_values = [item for sublist in self.exit.values() for item in sublist]
        # trash_cans_values = [
        #     item for sublist in self.trash_cans.values() for item in sublist
        # ]
        # start_values = [item for sublist in self.start.values() for item in sublist]
        osk_values = self.osk
        exit_values = self.exit
        trash_cans_values = self.trash_cans
        start_values = self.start

        color_map = []
        for node in G:
            if node in osk_values:
                color_map.append("red")
            elif node in exit_values:
                color_map.append("green")
            # elif node in start_values:
            #     color_map.append("yellow")
            else:
                color_map.append("blue")

        # print(color_map)

        nx.draw(G, pos, node_color=color_map, with_labels=True)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    def extend_routes(self):
        for edge in self.routes:
            node1, node2, weight = edge
            previous_node = node1
            for i in range(weight - 1):
                intermediate_node = self.intermediate_node_counter
                self.add_edge(previous_node, intermediate_node, 1)
                self.node_mapping[intermediate_node] = node1  # Dodajemy mapowanie
                previous_node = intermediate_node
                self.intermediate_node_counter += 1
                if intermediate_node not in self.transformed_graph:
                    self.transformed_graph[intermediate_node] = []
            self.add_edge(previous_node, node2, 1)
            self.node_mapping[node2] = node1  # Dodajemy mapowanie
        transformed_graph = {}
        for node, edges in self.transformed_graph.items():
            transformed_graph[node] = [edge[0] for edge in edges]
        return transformed_graph

    def add_edge(self, node1, node2, weight):
        if node1 in self.transformed_graph:
            self.transformed_graph[node1].append((node2, weight))
        else:
            self.transformed_graph[node1] = [(node2, weight)]
        if node2 in self.transformed_graph:
            self.transformed_graph[node2].append((node1, weight))
        else:
            self.transformed_graph[node2] = [(node1, weight)]

    def build_matrix(self):
        # osk_values = [item for sublist in self.osk.values() for item in sublist]
        osk_values = self.osk

        # exit_values = [item for sublist in self.exit.values() for item in sublist]
        exit_values = self.exit
        # print("osk values: ", osk_values)
        # print("exit values: ", exit_values)
        matrix = Matrix()
        vectors = []

        self.transformed_graph = OrderedDict(sorted(self.transformed_graph.items()))
        # print("Transformed graph: ", self.transformed_graph)
        for node, edges in self.transformed_graph.items():
            row = node - 1
            col = node - 1
            if node in osk_values:
                vectors.append(0)
                matrix.set_value(row, col, 1)
            elif node in exit_values:
                vectors.append(1)
                matrix.set_value(row, col, 1)
            else:
                vectors.append(0)
                matrix.set_value(row, col, 1)
                probability = -1 / len(edges)
                for edge in edges:
                    col = edge[0] - 1
                    matrix.set_value(row, col, probability)

        return matrix, vectors
