import numpy as np
import scipy.sparse as sp
from manipulation import ucm
from manipulation.utils import final_scores, calculate_sigma

from manipulation.solver import LPSolver
import logging

logger = logging.getLogger(__name__)


def generate_LP(alpha, ballots, sigma, m, n):
    from collections import defaultdict

    var_dict = defaultdict()
    var_dict.default_factory = var_dict.__len__

    for i in range(m - 1):
        for j in range(m - 1):
            _ = var_dict[('x', i, j)]

    for v in range(n):
        _ = var_dict[('y', v)]

    _ = var_dict['theta']
    _ = var_dict['k']

    var_dict = dict(var_dict)
    num_vars = len(var_dict)

    # trivial
    A0 = -np.eye(num_vars)
    b0 = np.zeros(num_vars, dtype=float)

    # first
    A1 = sp.lil_matrix((1, num_vars))
    col_idxs = [var_dict[('y', v)] for v in range(n)]
    A1[0, col_idxs] = 1.
    A1[0, var_dict['k']] = -1.

    b1 = np.array([0.])

    # second
    A2 = sp.lil_matrix((m - 1, num_vars))
    for i in range(m - 1):
        col_idxs = [var_dict[('x', i, j)] for j in range(m - 1)]
        A2[i, col_idxs] = 1.
        A2[i, var_dict['k']] = -1.
    b2 = np.zeros(m - 1, dtype=float)

    # third
    A3 = sp.lil_matrix((m - 1, num_vars))
    for j in range(m - 1):
        col_idxs = [var_dict[('x', i, j)] for i in range(m - 1)]
        A3[j, col_idxs] = -1.
        A3[j, var_dict['k']] = 1.
    b3 = np.zeros(m - 1, dtype=float)

    # fourth
    A4 = sp.lil_matrix((m - 1, num_vars))
    for i in range(m - 1):
        # added
        col_idxs = [var_dict[('x', i, j)] for j in range(m - 1)]
        A4[i, col_idxs] = alpha[:-1]  # added scores

        # removed
        col_idxs = [var_dict[('y', v)] for v in range(n)]
        A4[i, col_idxs] = -alpha[ballots[:, i]]

        A4[i, var_dict['theta']] = -1.

    b4 = -sigma[:-1]

    # fifth
    A5 = sp.lil_matrix((1, num_vars))

    # removed
    col_idxs = [var_dict[('y', v)] for v in range(n)]
    A5[0, col_idxs] = alpha[ballots[:, -1]]

    A5[0, var_dict['theta']] = 1.
    A5[0, var_dict['k']] = -alpha[-1]

    b5 = sigma[m - 1:m]

    A = sp.vstack([A0, A1, A2, A3, A4, A5])
    b = np.hstack([b0, b1, b2, b3, b4, b5])

    c = np.zeros(num_vars, dtype=float)
    c[var_dict['k']] = 1.

    return A, b, c, var_dict


def solve(alpha, original_ballots, m, n):
    sigma = calculate_sigma(alpha, original_ballots)

    # Phase I
    A, b, c, var_dict = generate_LP(alpha, original_ballots, sigma, m, n)

    # visualize_lp(A, b, c, var_dict)

    solver = LPSolver(np.array(A.todense()), b, c)
    solver.solve()

    vars = solver.get_x()

    y_star = vars[[var_dict[('y', v)] for v in range(n)]]  # get the 'y' variables
    # X_star = vars[[var_dict[('x', i, j)] for i in range(m - 1) for j in range(m - 1)]].reshape((m - 1, m - 1))
    theta_star = vars[var_dict['theta']]
    k_star = vars[var_dict['k']]

    # Phase II: randomized rounding of the y's
    y_tilde = np.random.rand(n) < y_star


    non_bribed_voters = (~y_tilde).nonzero()[0]
    non_bribed_voters = non_bribed_voters.tolist()
    lp_bribed_voters = y_tilde.nonzero()[0]
    lp_bribed_voters = lp_bribed_voters.tolist()


    k_tilde = y_tilde.sum()  # num. of voters we decided upon

    # Phase III: randomized rounding of the x's
    sigma_hat = calculate_sigma(alpha, original_ballots[non_bribed_voters, :])

    # solve the UCM instance
    frac_ms, manip_ballots = ucm.solve(sigma_hat[:-1], alpha[:-1], m - 1, k_tilde, repeat=10)

    # fix manip_ballots to include p
    lp_manip_ballots = np.zeros((k_tilde, m), dtype=int)
    lp_manip_ballots[:, :m - 1] = manip_ballots
    lp_manip_ballots[:, m - 1].fill(m - 1)


    # Phase IV: other manipulations if necessary
    additional_manip_ballots, additional_bribed_voters = greedy_random_algorithm(alpha, m, original_ballots,
                                                                          lp_manip_ballots, non_bribed_voters)


    bribed_voters = lp_bribed_voters + additional_bribed_voters

    strategy = np.vstack((lp_manip_ballots, additional_manip_ballots))

    return k_star, theta_star, bribed_voters, strategy


def greedy_random_algorithm(alpha, m, original_ballots, lp_manip_ballots, non_bribed_voters):
    current_ballots = np.vstack((original_ballots[non_bribed_voters, :], lp_manip_ballots))

    additional_bribed_voters = []

    additional_manip_ballots = []
    scores = final_scores(np.zeros(m), current_ballots, alpha)

    while scores[m - 1] < np.max(scores[:-1]):
        logger.debug((scores, non_bribed_voters))
        to_bribe = np.random.choice(non_bribed_voters)
        logger.debug('removing ballot {}'.format(original_ballots[to_bribe, :]))

        non_bribed_voters.remove(to_bribe)
        additional_bribed_voters.append(to_bribe)

        new_ballot = np.zeros(m, dtype=int)
        new_ballot[m - 1] = m - 1
        new_ballot[:m - 1] = np.random.permutation(m - 1)
        logger.debug('new ballot {}'.format(new_ballot))
        additional_manip_ballots.append(new_ballot)

        current_ballots = np.vstack((original_ballots[non_bribed_voters, :], lp_manip_ballots, additional_manip_ballots))
        scores = final_scores(np.zeros(m), current_ballots, alpha)
    logger.debug((scores, non_bribed_voters))

    if len(additional_manip_ballots) > 0:
        logger.debug('len(additional_manip_ballots) > 0')
        additional_manip_ballots = np.array(additional_manip_ballots)
    else:
        logger.debug('len(additional_manip_ballots) == 0')
        additional_manip_ballots = np.empty((0, m), dtype=int)

    return additional_manip_ballots, additional_bribed_voters
