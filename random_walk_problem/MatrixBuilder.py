import GraphBuilder

# example input data:
# Transformed graph 1:  {1: [6, 7, 8, 9], 6: [1, 2], 2: [6], 7: [1, 3], 3: [7], 8: [1, 4], 4: [8], 9: [1, 5], 5: [9]}
# Transformed graph 2:  {1: [5, 16, 17], 5: [1, 6], 6: [5, 7], 7: [6, 2], 2: [7, 8], 8: [2, 9], 9: [8, 10], 10: [9, 3], 3: [10, 11, 21], 11: [3, 12], 12: [11, 13], 13: [12, 4], 4: [13, 14], 14: [4, 15], 15: [14, 16], 16: [15, 1], 17: [1, 18], 18: [17, 19], 19: [18, 20], 20: [19, 21], 21: [20, 3]}


class MatrixBuilder:
    def __init__(self):
        self.matrix = {}

    def set_value(self, node1, node2, value):
        self.matrix[(node1, node2)] = value

    def get_value(self, node1, node2):
        return self.matrix.get((node1, node2), 0)

    def display(self):
        rows = max(node1 for node1, _ in self.matrix) + 1
        cols = max(node2 for _, node2 in self.matrix) + 1
        for i in range(rows):
            for j in range(cols):
                print(round(self.get_value(i, j), 3), end="\t")
            print()

    def find_max_row_in_col(self, col):
        max_row = max(
            range(col, len(self.matrix)), key=lambda i: abs(self.get_value(i, col))
        )
        return max_row

    @staticmethod
    def to_matrix(matrix_builder):
        rows = max(node1 for node1, _ in matrix_builder.matrix) + 1
        cols = max(node2 for _, node2 in matrix_builder.matrix) + 1
        matrix = [
            [matrix_builder.get_value(i, j) for j in range(cols)] for i in range(rows)
        ]
        return matrix

    def get_size(self):
        return max(node1 for node1, _ in self.matrix) + 1
