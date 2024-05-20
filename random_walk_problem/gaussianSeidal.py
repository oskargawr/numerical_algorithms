from MatrixBuilder import MatrixBuilder


class GaussianSeidal:
    def __init__(self, matrix_builder, vector):
        self.matrix_builder = matrix_builder
        self.vector = vector
        self.n = matrix_builder.get_size()

    def solve(self):
        x = [0 for _ in range(self.n)]  # initial guess
        for _ in range(1000):  # perform 1000 iterations
            x_new = x.copy()
            for j in range(self.n):
                summ_val = self.vector[j]
                for i in range(self.n):
                    if j != i:
                        summ_val -= self.matrix_builder.get_value(j, i) * x[i]
                x_new[j] = summ_val / self.matrix_builder.get_value(j, j)
            # if max(abs(a - b) for a, b in zip(x, x_new)) < 1e-10:  # convergence check
            #     return x_new
            x = x_new
        return x
