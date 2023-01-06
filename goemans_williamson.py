import cvxpy as cvx
import networkx as nx
import numpy as np

def goemans_williamson(g: nx.Graph) -> np.ndarray:
    # SDP formulation of MAX-CUT
    L = np.array(0.25 * nx.laplacian_matrix(g).todense())
    n = len(g.degree)
    M = cvx.Variable((n, n), PSD=True)
    op = cvx.Problem(cvx.Maximize(cvx.trace(L @ M)), [cvx.diag(M) == 1])
    op.solve(solver=cvx.CVXOPT)

    U = np.linalg.cholesky(M.value)

    # random hyperplane through origin
    r = np.random.randn(n)
    r = r / np.linalg.norm(r)

    return np.sign(U @ r)
