import logging

import numpy as np
from manipulation import utils

logger = logging.getLogger(__name__)


def solve(alpha, ballots, m, n):
    p = m - 1
    ballots_copy = np.copy(ballots)

    non_bribed = list(range(n))

    curr_sigma = utils.calculate_sigma(alpha, ballots_copy)

    leader = np.argmax(curr_sigma)
    non_bribed_sorted = sorted(non_bribed, key=lambda i: ballots_copy[i, leader] - ballots_copy[i, p], reverse=True)

    while np.max(curr_sigma) != curr_sigma[m - 1]:  # co-winner assumption
        logger.debug('curr_sigma={}'.format(curr_sigma))
        leader = np.argmax(curr_sigma)
        logger.debug('leader={}'.format(leader))

        non_bribed_sorted = sorted(non_bribed_sorted, key=lambda i: ballots_copy[i, leader] - ballots_copy[i, p],
                                   reverse=True)
        logger.debug('non_bribed_sorted={}'.format(non_bribed_sorted))
        to_bribe = non_bribed_sorted[0]
        logger.debug('to_bribe={}'.format(to_bribe))

        his_ballot = ballots_copy[to_bribe, :]
        logger.debug('his_ballot={}'.format(his_ballot))

        curr_sigma_ = curr_sigma - alpha[his_ballot]

        logger.debug('curr_sigma_={}'.format(curr_sigma_))

        new_ballot = np.empty(m, dtype=int)
        new_ballot[:-1] = utils.inverse_reverse_sort(curr_sigma_[:-1])
        new_ballot[-1] = m - 1

        logger.debug('new_ballot={}'.format(new_ballot))

        ballots_copy[to_bribe] = new_ballot
        curr_sigma = curr_sigma_ + alpha[new_ballot]

        logger.debug('curr_sigma={}'.format(curr_sigma))

        non_bribed_sorted = non_bribed_sorted[1:]

    bribed_voters = set(range(n)) - set(non_bribed_sorted)
    bribed_voters = list(bribed_voters)

    return bribed_voters, ballots_copy[bribed_voters, :]
