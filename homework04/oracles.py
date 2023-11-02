import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import csr_matrix


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = len(self.b)
        norm2 = np.sum(np.square(x))
        return ((1 / m) * np.dot(np.logaddexp(0, -self.b * self.matvec_Ax(x)), np.ones(m)) +
                (self.regcoef / 2) * norm2)

    def grad(self, x):
        m = len(self.b)
        sigma = expit(self.b * self.matvec_Ax(x))
        return -(1 / m) * self.matvec_ATx((1 - sigma) * self.b) + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        n = len(x)
        sigma = expit(self.b * self.matvec_Ax(x))
        return (1 / m) * self.matmat_ATsA(sigma * (1 - sigma)) + self.regcoef * np.eye(n)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    if scipy.sparse.issparse(A):
        def matvec_Ax(x):
            return A @ x
        def matvec_ATx(x):
            return A.T @ x
        def matmat_ATsA(s):
            As = A.multiply(s.reshape(-1, 1))
            return A.T @ As
    else:
        def matvec_Ax(x):
            return np.dot(A, x)
        def matvec_ATx(x):
            return np.dot(A.T, x)
        def matmat_ATsA(s):
            return A.T.dot(A * s.reshape(-1, 1))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    dim = len(x)
    gradient = np.zeros(dim)
    
    for i in range(dim):

        e_i = np.zeros(dim)
        e_i[i] = 1
        gradient[i] = (func(x + eps * e_i) - func(x)) / eps

    return gradient

def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            hessian[i][j] = (func(x + eps * e_i + eps * e_j) - func(x + eps * e_i) - func(x + eps * e_j) + func(x)) / (eps**2)

    return hessian
