import unittest

from numpy import array

from manipulation.utils import borda, calculate_sigma


class TestUtils(unittest.TestCase):
    def test_calculate_sigma(self):
        m = 4
        alpha = borda(m)

        ballots = array([[3,1,0,2],[2,1,3,0]])

        sigma = calculate_sigma(alpha, ballots)

        self.assertListEqual(sigma.tolist(), [5,2,3,2])

