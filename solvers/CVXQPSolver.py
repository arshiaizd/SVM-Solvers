import cvxpy as cp
import numpy as np

from solvers.BaseSolver import BaseSolver


# ===============
# CVXPY QP Solver
# ===============

class CVXQPSolver(BaseSolver):
    """
    Solves the linear SVM via the dual quadratic programming formulation using CVXPY with OSQP.
    Objective: minimize (1/2)α^T Q α - 1^T α
               subject to 0 ≤ α_i ≤ C and sum(α_i y_i) = 0.
    """

    def __init__(self, C=1.0, max_iters=1000):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        max_iters : int
            Maximum number of iterations for the solver.
        """
        self.C = C
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.alpha = None
        self._support_mask = None

    def fit(self, X, y):
        """
        Train the model on (X, y) using the dual QP formulation.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) in {-1, +1}
        """

        # reduce the dataset size

        n_samples, n_features = X.shape
        X = X[:n_samples // 16,:]
        y = y[:n_samples // 16]
        n_samples, n_features = X.shape

        # Define dual variable α (alpha)
        alpha = cp.Variable(n_samples)

        # Construct the quadratic term matrix Q where Q_ij = y_i y_j x_i^T x_j
        y_X = y.reshape(-1, 1) * X  # Element-wise multiplication
        Q = y_X @ y_X.T  # (n_samples, n_samples)
        Q = Q + 1e-8 * np.eye(n_samples)
        # Q = X @ X.T
        # z = y.reshape(-1,1) * alpha
        z = cp.scalar_product(y , alpha)
        # Dual objective: (1/2)α^T Q α - sum(α)
        objective = 0.5 * cp.quad_form(alpha, Q , assume_PSD=True) - cp.sum(alpha)

        # Constraints: 0 ≤ α_i ≤ C, sum(α_i y_i) = 0
        constraints = [
            alpha >= 0,
            alpha <= self.C,
            y @ alpha == 0  # Sum constraint
        ]

        # Solve the QP problem using OSQP
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        # Extract optimized α values
        self.alpha = alpha.value

        # Compute primal variables w and b from α
        self.w = (self.alpha * y) @ X  # w = Σ α_i y_i x_i

        # Calculate bias b using support vectors (0 < α_i < C)
        sv = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        if np.any(sv):
            b_values = y[sv] - X[sv] @ self.w
            self.b = np.mean(b_values)
        else:
            self.b = 0  # Fallback if no support vectors

        # Identify support vectors (α_i > 0)
        self._support_mask = self.alpha > 1e-5

    def predict(self, X):
        """
        Predict labels in {-1, +1} for input X.
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
        """
        Return constructor parameters for replicating the solver.
        """
        return {'C': self.C, 'max_iters': self.max_iters}
