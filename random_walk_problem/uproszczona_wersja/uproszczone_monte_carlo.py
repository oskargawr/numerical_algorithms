import numpy as np


def simulate_park_walk(n, s, num_trials):
    home_count = 0

    for _ in range(num_trials):
        position = 0

        while True:
            direction = np.random.randint(2)

            if direction == 1:
                position += 1

            else:
                position -= 1

            if position == s:
                home_count += 1
                break

            elif position == -n:
                break

    return home_count / num_trials


n = 3
s = 2

num_trials = [10, 100, 1000]

for trial in num_trials:
    print(simulate_park_walk(n, s, trial))


# print(simulate_park_walk(n, s, num_trials))
