import numpy as np
from tqdm.auto import tqdm

from solvers.BaseSolver import BaseSolver


# ======================
# Gradient Descent Solver
# ======================

class GradientDescentSolver(BaseSolver):
    """
    Basic gradient descent solver for a linear SVM with hinge loss.
    Minimizes (1/2)||w||^2 + C * sum(max(0, 1 - y * (w^T x + b))).
    """

    def __init__(self, learning_rate=0.01, C=1.0, max_iters=1000):
        """
        Parameters:
        -----------
        learning_rate : float
            The step size for gradient descent updates.
        C : float
            Regularization parameter (penalty for hinge violations).
        max_iters : int
            Number of gradient descent iterations.
        """
        self.lr = learning_rate
        self.C = C
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.loss_history = []
        self._support_mask = None

    def fit(self, X, y):
        """
        Train the model on (X, y).

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Labels in {-1, +1}.
        """
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = 0

        pbar = tqdm(total=self.max_iters, desc="GD Training")
        for _ in range(self.max_iters):
            # Compute margins and hinge gradient
            margins = y * (X.dot(self.w) + self.b)
            hinge_grad = np.where(margins < 1, -1, 0)

            # Gradient of objective
            dw = self.w + self.C * X.T.dot(hinge_grad * y)
            db = self.C * np.sum(hinge_grad * y)

            # Update w, b
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Track loss for analysis
            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - margins))
            self.loss_history.append(loss)

            pbar.set_postfix({'loss': f"{loss:.2f}", '||w||': f"{np.linalg.norm(self.w):.2f}"})
            pbar.update(1)

        pbar.close()

        # Mark support vectors (points near or violating margin)
        margins = y * (X.dot(self.w) + self.b)
        self._support_mask = np.abs(margins) <= 1 + 1e-5

    def predict(self, X):
        """
        Predict labels in {-1, +1} for input data X.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        --------
        preds : ndarray of shape (n_samples,)
            Predicted labels in {-1, +1}.
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
        return {
            'learning_rate': self.lr,
            'C': self.C,
            'max_iters': self.max_iters
        }
