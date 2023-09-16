'''
   This file contains function to solve quadratic quadratic problems using quadprog
'''
import numpy as np
import quadprog


def solve_QP(Q, c, A, b, E=None, d=None):
    """
        Solves the following quadratic program:
        minimize (1/2)x^T Q x + c^T x
        subject to Ax <= b and Ex=d
        (Adapted from: https://scaron.info/blog/quadratic-programming-in-python.html)
    """

    # Perturb Q so it is positive definite
    qp_G = Q + 10 ** (-9) * np.identity(Q.shape[0])

    qp_a = -c
    if E is not None:
        qp_C = -np.vstack([E, A]).T
        qp_b = -np.hstack([d.T, b.T])
        meq = E.shape[0]
    else:  # no equality constraint
        qp_C = -A.T
        qp_b = -b
        meq = 0

    return quadprog.solve_qp(qp_G.astype(np.float64), qp_a.astype(np.float64), qp_C.astype(np.float64), qp_b.astype(np.float64), meq)[0]