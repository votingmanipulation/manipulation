import numpy as np

from manipulation.utils import borda


def reverse_unweighted(initial_sigmas, alpha, k):
    if len(initial_sigmas) != len(alpha):
        raise ValueError(len(initial_sigmas) != len(alpha))

    m = len(initial_sigmas)

    current_awarded = initial_sigmas.copy()

    ballots = []

    for ell in range(k):

        # now sort the candidates according to the current awarded score, high to low:
        candidates_sorted = sorted(np.arange(m, dtype=int), key=lambda i: current_awarded[i], reverse=True)
        # now achieve inverse-sort: every candidate gets his rank (high-to-low) in sort
        ballot = np.zeros(m, dtype=int)
        for rank, cand in enumerate(candidates_sorted):
            ballot[cand] = rank

        ballots.append(ballot)
        current_awarded += alpha[ballot]

    return np.array(ballots)
