import numpy as np
from tqdm.auto import tqdm

from solvers.BaseSolver import BaseSolver


# ====================
# Newton Method Solver
# ====================

class NewtonMethodSolver(BaseSolver):
    """
    Approximates the SVM solution using a Newton-like approach
    with backtracking line search.
    """

    def __init__(self, C=1.0, max_iter=100, tolerance=1e-5, alpha=0.25, beta=0.5):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        max_iter : int
            Maximum number of outer iterations.
        tolerance : float
            Stopping threshold for w changes.
        alpha : float
            Armijo condition constant for line search.
        beta : float
            Factor by which step is reduced if condition not met.
        """
        self.C = C
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.alpha = alpha
        self.beta = beta
        self.w = None
        self.b = None
        self._support_mask = None

    def _cost(self, X, y, w, b):
        """
        SVM cost = 0.5||w||^2 + C * sum(hinge).
        """
        margins = y * (X.dot(w) + b)
        hinge = np.maximum(0, 1 - margins)
        return 0.5 * np.dot(w, w) + self.C * np.sum(hinge)

    def _gradient(self, X, y, w, b):
        """
        Subgradient of the hinge term plus gradient of regularizer.
        """
        margins = y * (X.dot(w) + b)
        viol_mask = (margins < 1).astype(float)
        grad_w = w - self.C * (X.T @ (y * viol_mask))
        grad_b = -self.C * np.sum(y * viol_mask)
        return grad_w, grad_b

    def fit(self, X, y):
        """
        Train model using iterative Newton-like updates with line search.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        prev_w = np.zeros(n_features)

        pbar = tqdm(total=self.max_iter, desc="Newton Training")

        for _ in range(self.max_iter):
            grad_w, grad_b = self._gradient(X, y, self.w, self.b)
            direction_w = -grad_w
            direction_b = -grad_b

            # Backtracking line search
            t = 1.0
            cost_old = self._cost(X, y, self.w, self.b)
            grad_dot_dir = grad_w.dot(direction_w) + grad_b * direction_b

            while True:
                new_w = self.w + t * direction_w
                new_b = self.b + t * direction_b
                cost_new = self._cost(X, y, new_w, new_b)
                if cost_new <= cost_old + self.alpha * t * grad_dot_dir:
                    break
                t *= self.beta

            # Update w, b
            self.w += t * direction_w
            self.b += t * direction_b

            # Check convergence
            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                break
            prev_w = self.w.copy()

            pbar.set_postfix({'cost': f"{cost_old:.3f}", 'step': f"{t:.1e}"})
            pbar.update(1)

        pbar.close()

        # Mark support vectors
        margins = y * (X.dot(self.w) + self.b)
        self._support_mask = np.abs(margins) <= 1 + 1e-5

    def predict(self, X):
        """
        Predict labels in {-1, +1}.
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
            'C': self.C,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'alpha': self.alpha,
            'beta': self.beta
        }
