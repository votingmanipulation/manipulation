from io import BytesIO
import os

from sklearn.externals.joblib.parallel import Parallel, delayed

from manipulation.utils import borda, makespan, draw_uniform_ballots
from manipulation import bribery, greedy_bribery
import pandas as pd

import numpy as np
import scipy.sparse as sp

np.random.seed(424234)

S3 = False
folder = 'c:/git/manipulation/results_local'


def run_single(m, n, tt, alpha):
    ballots = draw_uniform_ballots(m, n)
    k_star, theta_star, bribed_voters, strategy = bribery.solve_with_reverse(alpha, ballots, m, n)

    greedy_bribed_voters, greedy_strategy = greedy_bribery.solve(alpha, ballots, m, n)
    k_greedy = len(greedy_bribed_voters)

    data = [{'alpha': alpha, 'n': n, 'm': m, 'tt': tt, 'ballots': ballots, 'k_star': k_star, 'theta_star': theta_star,
             'k_final': len(bribed_voters), 'k_greedy': k_greedy}]
    filename = 'results-m{}-n{}-tt{}.csv'.format(m, n, tt)
    print(filename)
    df = pd.DataFrame.from_records(data,
                                   columns=['n', 'm', 'tt', 'ballots', 'alpha', 'k_star', 'theta_star', 'k_final', 'k_greedy'])
    if S3:  # s3
        import boto3

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, encoding='utf-8')
        s3_resource = boto3.resource('s3')
        s3_resource.Object('bribery', filename).put(Body=csv_buffer.getvalue())
    else:
        abs_path = os.path.join(folder, filename)
        df.to_csv(abs_path, index=False)


if __name__ == '__main__':
    # for n in [2, 4, 8]:
    #     for m in [2, 4, 8]:
    #         alpha = borda(m)
    #         for tt in range(16):
    #             run_single(m, n, tt, alpha)
    Parallel(n_jobs=-1)(
        delayed(run_single)(m, n, tt, borda(m)) for n in [2, 4, 8, 16, 32] for m in
        [2, 4, 8, 16] for tt in range(16))
