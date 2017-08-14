import unittest

from manipulation.ucm import generate_LP, birkhoff_von_neumann, solve
from manipulation.utils import borda, final_scores
import numpy as np


class TestUCM(unittest.TestCase):
    # @nottest
    def test_generate_LP(self):
        m = 3
        k = 3
        alpha = borda(m)
        sigma = np.ones(m, dtype=int) * 3
        A, B, c = generate_LP(alpha, sigma, m, k)

        self.assertListEqual(A.tolist(), [[-1., -0., -0., -0., -0., -0., -0., -0., -0., -0.],
                                          [-0., -1., -0., -0., -0., -0., -0., -0., -0., -0.],
                                          [-0., -0., -1., -0., -0., -0., -0., -0., -0., -0.],
                                          [-0., -0., -0., -1., -0., -0., -0., -0., -0., -0.],
                                          [-0., -0., -0., -0., -1., -0., -0., -0., -0., -0.],
                                          [-0., -0., -0., -0., -0., -1., -0., -0., -0., -0.],
                                          [-0., -0., -0., -0., -0., -0., -1., -0., -0., -0.],
                                          [-0., -0., -0., -0., -0., -0., -0., -1., -0., -0.],
                                          [-0., -0., -0., -0., -0., -0., -0., -0., -1., -0.],
                                          [-0., -0., -0., -0., -0., -0., -0., -0., -0., -1.],
                                          [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                          [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
                                          [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                                          [-1., 0., 0., -1., 0., 0., -1., 0., 0., 0.],
                                          [0., -1., 0., 0., -1., 0., 0., -1., 0., 0.],
                                          [0., 0., -1., 0., 0., -1., 0., 0., -1., 0.],
                                          [0., 1., 2., 0., 0., 0., 0., 0., 0., -1.],
                                          [0., 0., 0., 0., 1., 2., 0., 0., 0., -1.],
                                          [0., 0., 0., 0., 0., 0., 0., 1., 2., -1.]])

        self.assertListEqual(B.tolist(), [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 3., 3.,
                                          -3., -3., -3., -3., -3., -3.])

        self.assertListEqual(c.tolist(), [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])

    def test_birkhoff_von_neumann(self):
        m = 3
        perms = []
        lambdas = []

        perms.append([1, 2, 0])
        lambdas.append(0.5)

        perms.append([2, 1, 0])
        lambdas.append(0.25)

        perms.append([1, 0, 2])
        lambdas.append(0.25)

        Y = np.zeros((m, m))
        for perm, lam in zip(perms, lambdas):
            Y[np.arange(m), perm] += lam

        lambdas1, perms1 = birkhoff_von_neumann(Y)

        self.assertListEqual(sorted(perms), sorted(perms1.tolist()))
        self.assertListEqual(sorted(lambdas), sorted(lambdas1.tolist()))

    def test_birkhoff_von_neumann_empty(self):
        m = 3

        Y = np.zeros((m, m))

        lambdas1, perms1 = birkhoff_von_neumann(Y)

        self.assertListEqual([], sorted(perms1.tolist()))
        self.assertListEqual([], sorted(lambdas1.tolist()))

    def test_solve(self):
        m = 3
        k = 3
        alpha = borda(m)
        sigma = np.ones(m, dtype=int) * 3

        frac_ms, ballots = solve(sigma, alpha, m, k)

        self.assertAlmostEqual(frac_ms, 6)
        self.assertListEqual(final_scores(sigma, ballots, alpha).tolist(), [5, 5, 5])
