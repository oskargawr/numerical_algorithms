class GaussianPivotal:
    def __init__(self, matrix_builder, vector):
        self.matrix_builder = matrix_builder
        self.vector = vector
        self.n = matrix_builder.get_size()

    def calculate_lu(self):
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                if self.matrix_builder.get_value(i, i) == 0:
                    print("Error: Zero on diagonal!")
                    print("Need algorithm with pivoting")
                    break
                m = self.matrix_builder.get_value(j, i) / self.matrix_builder.get_value(
                    i, i
                )
                for k in range(self.n):
                    self.matrix_builder.set_value(
                        j,
                        k,
                        self.matrix_builder.get_value(j, k)
                        - m * self.matrix_builder.get_value(i, k),
                    )
                self.vector[j] = self.vector[j] - m * self.vector[i]

    def back_subs(self):
        x = [0 for _ in range(self.n)]
        x[self.n - 1] = self.vector[self.n - 1] / self.matrix_builder.get_value(
            self.n - 1, self.n - 1
        )
        for i in range(self.n - 2, -1, -1):
            sum_ = sum(
                self.matrix_builder.get_value(i, j) * x[j] for j in range(i + 1, self.n)
            )
            x[i] = (self.vector[i] - sum_) / self.matrix_builder.get_value(i, i)
        return x

    def solve(self):
        self.calculate_lu()
        return self.back_subs()
