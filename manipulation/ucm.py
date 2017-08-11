import numpy as np


def generate_LP(alpha, sigma, m, k):
    num_x_vars = m ** 2

    A0 = -np.eye(num_x_vars + 1)
    B0 = np.zeros(num_x_vars + 1)

    A1 = np.zeros((m, m, m), dtype=float)
    for i in range(m):
        A1[i, i, :] = 1

    b1 = np.ones(m) * k

    A2 = np.zeros((m, m, m), dtype=float)
    for j in range(m):
        A2[j, :, j] = -1

    b2 = -np.ones(m) * k

    A3 = np.zeros((m, m, m), dtype=float)
    for i in range(m):
        A3[i, i, :] = alpha

    b3 = -sigma

    A = np.vstack((A1.reshape(m, num_x_vars), A2.reshape(m, num_x_vars), A3.reshape(m, num_x_vars)))
    column = np.hstack((np.zeros(m), np.zeros(m), -np.ones(m))).reshape(-1, 1)
    A = np.hstack((A, column))

    b = np.hstack((b1, b2, b3))

    A = np.vstack((A0, A))

    b = np.hstack((B0, b))

    c = np.zeros(num_x_vars + 1)
    c[-1] = 1.0

    return A, b, c
