import numpy as np
import scipy.sparse as sp
from cvxopt.base import spmatrix


def borda(m):
    return np.arange(m, dtype=int)


def makespan(sigmas, ballots, alpha):
    return np.max(final_scores(sigmas, ballots, alpha))


def final_scores(sigmas, ballots, alpha):
    return sigmas + alpha[ballots].sum(axis=0)

def draw_uniform_sigmas(alpha, m, n, rand=None):
    """

    Args:
        m (int)
        n (int:
        rand (Optional [Union[ numpy.random.mtrand.RandomState, None]]):

    Returns:

    """
    rankings = draw_uniform_ballots(m, n, rand)
    return calculate_sigma(alpha, rankings)


def draw_uniform_ballots(m, n, rand=None):
    """

    Args:
       m (int)
       n (int:
       rand (numpy.random.mtrand.RandomState):

    Returns:

    """
    res = []
    for i in range(n):
        perm = rand.permutation(m) if rand else np.random.permutation(m)
        res.append(perm)
    return np.array(res, dtype=int)


def calculate_sigma(alpha, ballots):
    """

    Args:
        alpha:
        ballots:

    Returns:

    """
    return alpha[ballots].sum(axis=0)


def to_cvxopt_spmatrix(A):

    if sp.issparse(A):
        cx = A.tocoo()
    else:
        cx = sp.coo_matrix(A)

    return spmatrix(cx.data, cx.row, cx.col)


def visualize_lp(A, b,c,var_dict):
    try:
        import pandas as pd
    except ImportError as e:
        raise e

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise e

    try:
        import seaborn as sns
    except ImportError as e:
        raise e

    columns = sorted(var_dict.keys(), key=lambda c: var_dict[c])

    df = pd.DataFrame(A.todense(), columns=columns)
    df['b'] = b

    plt.figure(figsize=(16, 16))
    sns.heatmap(df, annot=True, cbar=False)
    plt.show()

