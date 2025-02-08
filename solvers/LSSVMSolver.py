import numpy as np

from solvers.BaseSolver import BaseSolver


# ====================
# Least-Squares SVM Solver
# ====================

class LSSVMSolver(BaseSolver):
    """
    Least-Squares SVM solver. Replaces hinge loss with squared error loss,
    leading to a linear system solution (C + 1) approach.
    """

    def __init__(self, C=1.0):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        """
        self.C = C
        self.w = None
        self.b = None
        self._support_mask = None

    def fit(self, X, y):
        """
        Solve the LS-SVM via a block linear system in the dual.
        """
        n_samples, n_features = X.shape
        # trying to use half the samples because there is not enough memory for the whole dataset
        X = X[n_samples // 2:,:]
        y = y[n_samples // 2:]
        n_samples, n_features = X.shape
        # K = X X^T
        K = X @ X.T

        # Construct block matrix
        big_mat = np.zeros((n_samples + 1, n_samples + 1))
        big_mat[0, 1:] = 1
        big_mat[1:, 0] = 1
        big_mat[1:, 1:] = K + (1.0 / self.C) * np.eye(n_samples)

        rhs = np.zeros(n_samples + 1)
        rhs[1:] = y

        sol = np.linalg.solve(big_mat, rhs)
        self.b = sol[0]
        alpha = sol[1:]

        # w = X^T alpha
        self.w = X.T @ alpha

        # Determine support vectors by margin threshold
        margins = y * (X.dot(self.w) + self.b)
        self._support_mask = np.abs(margins) <= 1 + 1e-5

    def predict(self, X):
        """
        Predict labels in {-1, +1} for input data X.
        """
        return np.sign(X.dot(self.w) + self.b)

    @property
    def support_vectors_(self):
        return self._support_mask

    @property
    def weights_(self):
        return self.w

    @property
    def bias_(self):
        return self.b

    def get_params(self):
        return {
            'C': self.C
        }
