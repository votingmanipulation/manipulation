import numpy as np


def borda(m):
    return np.arange(m, dtype=int)


def makespan(sigmas, ballots, alpha):
    return np.max(final_scores(sigmas, ballots, alpha))


def final_scores(sigmas, ballots, alpha):
    return sigmas + np.dot(ballots, alpha)
