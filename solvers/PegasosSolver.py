import numpy as np
from tqdm.auto import tqdm

from solvers.BaseSolver import BaseSolver


# ===================
# Pegasos Solver
# ===================

class PegasosSolver(BaseSolver):
    """
    Pegasos (Primal Estimated sub-Gradient SOlver) for SVM.
    Uses a stochastic subgradient approach with projection.
    """

    def __init__(self, C=1.0, max_iters=1000):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        max_iters : int
            Number of subgradient steps.
        """
        self.C = C
        self.max_iters = max_iters
        self.w_aug = None  # augmented weight vector (w + bias)
        self._support_mask = None

    def fit(self, X, y):
        """
        Train SVM using Pegasos method. We augment X with a column of ones
        to incorporate the bias into w_aug.
        """
        n_samples, n_features = X.shape

        # Augment X for bias
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        self.w_aug = np.zeros(n_features + 1)

        pbar = tqdm(total=self.max_iters, desc="Pegasos Training")
        for t in range(1, self.max_iters + 1):
            i = np.random.randint(n_samples)
            if y[i] * (X_aug[i].dot(self.w_aug)) < 1:
                # subgradient = C * w_aug - y_i * x_i_aug
                grad = self.C * self.w_aug - y[i] * X_aug[i]
            else:
                grad = self.C * self.w_aug

            eta_t = 1.0 / (self.C * t)
            self.w_aug -= eta_t * grad

            # Projection step: ||w|| <= 1 / sqrt(C)
            w_norm = np.linalg.norm(self.w_aug)
            max_norm = 1.0 / np.sqrt(self.C)
            if w_norm > max_norm:
                self.w_aug *= (max_norm / w_norm)

            pbar.update(1)
        pbar.close()

        # Support vectors: points within or on margin
        margins = y * (X_aug.dot(self.w_aug))
        self._support_mask = np.abs(margins) <= 1 + 1e-5

    def predict(self, X):
        """
        Predict labels in {-1, +1} by augmenting X and taking sign of w_aug^T x.
        """
        if self.w_aug is None:
            raise ValueError("Model not trained yet.")
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        return np.sign(X_aug.dot(self.w_aug))

    @property
    def support_vectors_(self):
        return self._support_mask

    @property
    def weights_(self):
        # Return all but the last dimension of w_aug
        if self.w_aug is None:
            return None
        return self.w_aug[:-1]

    @property
    def bias_(self):
        # Return the last dimension of w_aug
        if self.w_aug is None:
            return None
        return self.w_aug[-1]

    def get_params(self):
        return {
            'C': self.C,
            'max_iters': self.max_iters
        }
