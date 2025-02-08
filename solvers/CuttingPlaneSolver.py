import numpy as np
from tqdm.auto import tqdm

from solvers.BaseSolver import BaseSolver


# =====================
# Cutting Plane Solver
# =====================

class CuttingPlaneSolver(BaseSolver):
    """
    Cutting plane method for SVM. Iteratively refines solution by focusing on points
    that violate margin constraints, known as 'cuts'.
    """

    def __init__(self, C=1.0, max_iter=1000, tolerance=1e-5,
                 initial_subset_size=5, max_active_size=500, lr=0.01):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        max_iter : int
            Maximum number of cutting plane iterations.
        tolerance : float
            Stopping threshold based on weight vector change.
        initial_subset_size : int
            Number of random points to start in the active set.
        max_active_size : int
            Maximum size of the active set to manage complexity.
        lr : float
            Learning rate for subproblem updates.
        """
        self.C = C
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.initial_subset_size = initial_subset_size
        self.max_active_size = max_active_size
        self.lr = lr
        self.w = None
        self.b = None
        self._support_mask = None

    def fit(self, X, y):
        """
        Train using cutting plane iterations, focusing on margin-violating samples.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        rng = np.random.default_rng()
        # Initialize active set
        if self.initial_subset_size < n_samples:
            active_set = rng.choice(n_samples, self.initial_subset_size, replace=False)
        else:
            active_set = np.arange(n_samples)

        prev_w = np.zeros(n_features)
        pbar = tqdm(total=self.max_iter, desc="Cutting Plane Training")

        for _ in range(self.max_iter):
            # Solve sub problem on active set
            X_active = X[active_set]
            y_active = y[active_set]

            # Compute hinge violations
            margins = y_active * (X_active.dot(self.w) + self.b)
            hinge_mask = (margins < 1).astype(float)

            # Gradient step
            dw = self.w - self.C * np.dot(X_active.T, y_active * hinge_mask)
            db = -self.C * np.sum(y_active * hinge_mask)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Check convergence
            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                break
            prev_w = self.w.copy()

            # Add new violating points to active set
            full_margins = y * (X.dot(self.w) + self.b)
            violated_indices = np.where(full_margins < 1)[0]
            active_set = np.unique(np.concatenate([active_set, violated_indices]))
            if len(active_set) > self.max_active_size:
                active_set = active_set[:self.max_active_size]

            pbar.set_postfix({'||w||': f"{np.linalg.norm(self.w):.2f}"})
            pbar.update(1)

        pbar.close()

        # Determine final support vectors
        final_margins = y * (X.dot(self.w) + self.b)
        self._support_mask = np.abs(final_margins) <= 1 + 1e-5

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
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'initial_subset_size': self.initial_subset_size,
            'max_active_size': self.max_active_size,
            'lr': self.lr
        }
