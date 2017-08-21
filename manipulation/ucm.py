import sys
from operator import itemgetter

import numpy as np
from networkx.algorithms import bipartite as bp
from scipy import sparse

from manipulation.solver import LPSolver
from manipulation.utils import makespan


def generate_LP(alpha, sigma, m, k):
    num_x_vars = m ** 2

    A0 = -np.eye(num_x_vars + 1, dtype=float)
    B0 = np.zeros(num_x_vars + 1, dtype=float)

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
    column = np.hstack((np.zeros(m, dtype=float), np.zeros(m, dtype=float), -np.ones(m, dtype=float))).reshape(-1, 1)
    A = np.hstack((A, column))

    b = np.hstack((b1, b2, b3))

    A = np.vstack((A0, A))

    b = np.hstack((B0, b))

    c = np.zeros(num_x_vars + 1, dtype=float)
    c[-1] = 1.0

    return A, b, c


def birkhoff_von_neumann(Y, tol=0.0001):
    if Y.shape[0] != Y.shape[1]:
        raise ValueError('Y.shape[0] != Y.shape[1]')
    if np.any(Y < -tol):
        raise ValueError('np.any(Y < -tol)')

    Y = np.where(Y < tol, 0, Y)

    m = Y.shape[0]

    lambdas = []
    perms = []

    residuals = Y > tol
    while np.any(residuals):

        adj = residuals.astype(int)
        adj = sparse.csr_matrix(adj)

        G = bp.from_biadjacency_matrix(adj)

        M = bp.maximum_matching(G)
        M_ = [(kk, v - m) for kk, v in M.items() if kk < m]

        if len(M_) < m:  # this can happen due to numerical stability issues TODO add test
            break

        M_ = sorted(M_, key=itemgetter(0))  # if tuples sorted by rows, then the columns are the permutation

        rows, columns = zip(*M_)
        perm = np.array(columns)

        assert perm.shape == (m,)

        lambda_ = np.min(Y[rows, columns])

        P = np.zeros((m, m), dtype=float)
        P[rows, columns] = 1.

        lambdas.append(lambda_)
        perms.append(perm)
        Y -= lambda_ * P

        residuals = Y > tol
    return np.array(lambdas), np.array(perms)


def solve(sigma, alpha, m, k, repeat=1):
    if k > 0:
        A, b, c = generate_LP(alpha, sigma, m, k)

        solver = LPSolver(A, b, c)
        solver.solve()

        obj_value = solver.get_objective()
        x = solver.get_x()
        x = x[:-1].reshape(m, m)

        Y = x / k
        lambdas, perms = birkhoff_von_neumann(Y)

        # re-normalize to deal with numerical errors
        lambdas /= np.sum(lambdas)

        min_makespan = sys.maxsize
        minimizing_ballots = None
        for i in range(repeat):
            cosen_ballot_idxs = np.random.choice(np.arange(len(perms), dtype=int), size=k, p=lambdas)
            ballots = perms[cosen_ballot_idxs]
            cur_makespan = makespan(sigma, ballots, alpha)
            if cur_makespan < min_makespan:
                min_makespan = cur_makespan
                minimizing_ballots = ballots

        return obj_value, minimizing_ballots
    else:
        return np.max(sigma), np.zeros((0,m), dtype=int)
