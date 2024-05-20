class Gaussian:
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector
        self.solution = []

    def forwardElim(self):
        n = len(self.vector)
        for i in range(n):
            for j in range(i + 1, n):
                factor = self.matrix.get_value(j, i) / self.matrix.get_value(i, i)
                self.update_matrix_and_vector(i, j, n, factor)

    def update_matrix_and_vector(self, i, j, n, factor):
        for k in range(i, n):
            updated_value = self.matrix.get_value(
                j, k
            ) - factor * self.matrix.get_value(i, k)
            self.matrix.set_value(j, k, updated_value)
        self.vector[j] -= factor * self.vector[i]

    def backSub(self):
        n = len(self.vector)
        self.solution = [0] * n
        for i in range(n - 1, -1, -1):
            self.solution[i] = self.calculate_solution(i, n)
        return self.solution

    def calculate_solution(self, i, n):
        solution = self.vector[i]
        for j in range(i + 1, n):
            solution -= self.matrix.get_value(i, j) * self.solution[j]
        return solution / self.matrix.get_value(i, i)

    def solve(self):
        # print(self.vector)
        # self.matrix.display()
        self.forwardElim()
        self.backSub()
        return self.solution
