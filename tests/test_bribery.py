import unittest

import numpy as np
from numpy import array

from manipulation.bribery import generate_LP, solve
from manipulation.solver import LPSolver
from manipulation.utils import borda, calculate_sigma, final_scores


class TestBribery(unittest.TestCase):
    # @nottest
    def test_generate_LP(self):
        m = 4  # icl. preferred at idx m-1
        n = 5
        alpha = borda(m)
        ballots = array([[0, 3, 2, 1],
                         [3, 0, 2, 1],
                         [1, 2, 3, 0],
                         [0, 3, 1, 2],
                         [2, 0, 1, 3]])

        sigma = calculate_sigma(alpha, ballots)

        A, b, c, var_dict = generate_LP(alpha, ballots, sigma, m, n)

        expedcted_A = np.matrix([[-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1.,
                                  0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  -1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., -1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., -1.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
                                  1., 0., -1.],
                                 [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., -1.],
                                 [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., -1.],
                                 [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
                                  0., 0., -1.],
                                 [-1., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 1.],
                                 [0., -1., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0., 0.,
                                  0., 0., 1.],
                                 [0., 0., -1., 0., 0., -1., 0., 0., -1., 0., 0., 0., 0.,
                                  0., 0., 1.],
                                 [0., 1., 2., 0., 0., 0., 0., 0., 0., 0., -3., -1., 0.,
                                  -2., -1., 0.],
                                 [0., 0., 0., 0., 1., 2., 0., 0., 0., -3., 0., -2., -3.,
                                  0., -1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1., 2., -2., -2., -3., -1.,
                                  -1., -1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 2.,
                                  3., 1., -3.]])
        expected_b = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -6., -8., -9., 7.])

        expected_c = array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 1.])
        self.assertListEqual(A.todense().tolist(), expedcted_A.tolist())

        self.assertListEqual(b.tolist(), expected_b.tolist())

        self.assertListEqual(c.tolist(), expected_c.tolist())

    def test_solve(self):
        m = 4  # icl. preferred at idx m-1
        n = 5
        alpha = borda(m)
        ballots = array([[0, 3, 2, 1],
                         [3, 0, 2, 1],
                         [1, 2, 3, 0],
                         [0, 3, 1, 2],
                         [2, 0, 1, 3]])

        # sigma = rankings_to_initial_sigmas(alpha, ballots)

        frac, bribed, manip_ballots = solve(alpha, ballots, m,n)
        print(frac, bribed, manip_ballots)

        not_bribed = list(set(range(n)) - bribed)


        all_ballots = np.vstack( ( ballots[not_bribed,:], manip_ballots ) )

        scores = final_scores(np.zeros(m),  all_ballots, alpha   )

        print(scores)

        self.assertGreaterEqual(scores[-1], np.max(scores[:-1]))