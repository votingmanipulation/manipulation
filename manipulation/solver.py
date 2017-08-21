from cvxopt import solvers
from cvxopt.base import matrix
import numpy as np
import logging
import scipy.sparse as sp

from manipulation.utils import to_cvxopt_spmatrix

logger = logging.getLogger(__name__)


import platform

if platform.system() == 'Windows':
    solver = None  # CVXOPT's conelp
else:
    solver = 'glpk'

class LPSolver(object):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c



    def solve(self):
        c_ = matrix(self.c)

        if sp.issparse(self.A):
            G = to_cvxopt_spmatrix(self.A)
        else:
            G = matrix(self.A)

        h = matrix(self.b)

        self.result = solvers.lp(c_, G, h, solver=solver)

        logger.info(self.result)


        # primal_objective_ = self.result['primal objective']
        # status_ = self.result['status']
        #
        # objective = primal_objective_

        _x = self.result['x']
        # _x = np.array(_x).flatten()[:-1]
        # _x = _x.reshape(m, m)
        # _z = self.result['z']
        # self._z = np.array(_z).flatten()[:-1]  # remove the dummy constraint

    def get_x(self):
        return np.array(self.result['x']).flatten()

    def get_objective(self):
        return self.result['primal objective']


