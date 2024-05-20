import random
import networkx as nx


class RandomGraph:
    def __init__(
        self, num_intersections, num_alleys, num_osk, num_exit, max_alley_length
    ):
        self.alleys = []
        self.intersections = num_intersections
        self.osks = []
        self.exits = []
        self.graph = {}
        self.generate_random_graph(
            num_intersections, num_alleys, num_osk, num_exit, max_alley_length
        )

    def generate_random_graph(
        self, num_intersections, num_alleys, num_osk, num_exit, max_alley_length
    ):
        osks = random.sample(range(1, num_intersections + 1), num_osk)
        exits = random.sample(
            set(range(1, num_intersections + 1)) - set(osks), num_exit
        )

        connected_intersections = self.generate_initial_intersection_connections(
            num_intersections, num_alleys
        )

        for connection in connected_intersections.edges():
            alley_length = random.randint(1, max_alley_length)
            node1, node2 = connection
            self.alleys.append([node1, node2, alley_length])

        self.osks = osks
        self.exits = exits
        print("graph before adding edges: ", self.alleys)
        self.create_graph()

    def create_graph(self):
        self.set_main_intersections()
        self.add_edges()

    def set_main_intersections(self):
        self.graph = {i: [] for i in range(1, self.intersections + 1)}

    def add_edges(self):
        for alley in self.alleys:
            u, v, length = alley
            self.add_edge(u, v, length)

    def add_edge(self, u, v, length):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        if length == 1:
            self.graph[u].append(v)
            self.graph[v].append(u)
        else:
            for i in range(1, length):
                new_intersection = len(self.graph) + 1
                self.graph[new_intersection] = []
                if i == 1:
                    self.graph[new_intersection].append(u)
                    self.graph[u].append(new_intersection)
                else:
                    self.graph[new_intersection].append(new_intersection - 1)
                    self.graph[new_intersection - 1].append(new_intersection)
                if i == length - 1:
                    self.graph[new_intersection].append(v)
                    self.graph[v].append(new_intersection)

    def generate_initial_intersection_connections(
        self,
        intersections_num,
        edges_num,
    ):
        G = nx.MultiGraph()
        G.add_nodes_from(range(1, intersections_num + 1))

        for _ in range(edges_num):
            node1 = random.randint(1, intersections_num)
            node2 = random.randint(1, intersections_num)
            while node1 == node2:
                node2 = random.randint(1, intersections_num)
            G.add_edge(node1, node2)

        if nx.is_connected(G):
            return G
        else:
            components = list(nx.connected_components(G))
            while len(components) > 1:
                node1 = random.choice(list(components[0]))
                node2 = random.choice(list(components[1]))
                G.add_edge(node1, node2)
                components = list(nx.connected_components(G))
            return G

    def display(self):
        print(self.graph)

    def get_data(self):
        start_vertex = random.choice(range(1, self.intersections + 1))
        routes = self.alleys
        trash_cans = 0

        return {
            "alleys": len(self.alleys),
            "intersections": self.intersections,
            "osk": self.osks,
            "exit": self.exits,
            "start": start_vertex,
            "routes": routes,
            "trash_cans": 0,
        }


# graph = RandomGraph(10, 10, 3, 2, 3)
# graph.display()

# print(graph.get_data())

# print(graph.graph)
