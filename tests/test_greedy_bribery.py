import logging
import unittest

import numpy as np
from numpy import array

from manipulation.greedy_bribery import solve
from manipulation.utils import borda, final_scores

BASIC_FORMAT = "(asctime)-15s %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(format=BASIC_FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestGreedyBribery(unittest.TestCase):
    def test_solve(self):
        m = 3
        n = 10
        alpha = borda(m)
        ballots = array([[2, 0, 1],
                         [2, 1, 0],
                         [0, 2, 1],
                         [0, 2, 1],
                         [2, 1, 0],
                         [1, 2, 0],
                         [2, 1, 0],
                         [2, 0, 1],
                         [2, 0, 1],
                         [1, 2, 0]])

        # sigma = rankings_to_initial_sigmas(alpha, ballots)

        bribed, manip_ballots = solve(alpha, ballots, m, n)
        logger.info('bribed={}'.format(bribed))
        logger.info('manip_ballots={}'.format(manip_ballots))

        self.assertListEqual(bribed, [1, 4, 5])
        self.assertListEqual(manip_ballots.tolist(), [[0, 1, 2],
                                                      [0, 1, 2],
                                                      [0, 1, 2]])

        not_bribed = list(set(range(n)) - set(bribed))

        all_ballots = np.vstack((ballots[not_bribed, :], manip_ballots))

        scores = final_scores(np.zeros(m), all_ballots, alpha)

        print(scores)

        self.assertGreaterEqual(scores[-1], np.max(scores[:-1]))

    def test_solve2(self):
        n = 4
        m = 8
        alpha = borda(m)

        ballots = array([[3, 0, 2, 5, 6, 1, 4, 7],
                         [3, 7, 4, 6, 0, 5, 1, 2],
                         [3, 4, 1, 5, 7, 6, 2, 0],
                         [5, 1, 2, 6, 4, 7, 0, 3]])

        bribed, manip_ballots = solve(alpha, ballots, m, n)
        logger.info('bribed={}'.format(bribed))
        logger.info('manip_ballots={}'.format(manip_ballots))

        self.assertListEqual(bribed, [2])
        self.assertListEqual(manip_ballots.tolist(), [[2, 4, 5, 0, 3, 1, 6, 7]])

        not_bribed = list(set(range(n)) - set(bribed))

        all_ballots = np.vstack((ballots[not_bribed, :], manip_ballots))

        scores = final_scores(np.zeros(m), all_ballots, alpha)

        print(scores)

        self.assertGreaterEqual(scores[-1], np.max(scores[:-1]))
