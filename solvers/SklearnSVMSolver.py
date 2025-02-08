import numpy as np
from sklearn.svm import SVC

from solvers.BaseSolver import BaseSolver


# ===============
# Scikit-learn SVM Solver
# ===============

class SklearnSVMSolver(BaseSolver):
    """
    Wrapper around sklearn's SVC to match our solver interface.
    Supports both linear and kernel-based SVMs.
    """

    def __init__(self, C=1.0, kernel='linear', max_iter=100):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter.
        kernel : str
            Kernel type, e.g., 'linear', 'rbf', etc.
        max_iter : int
            Maximum number of iterations for solver (-1 means no limit).
        """
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.model = SVC(C=self.C, kernel=self.kernel, max_iter=self.max_iter)
        self._support_mask = None

    def fit(self, X, y):
        """
        Fit the sklearn SVC model on (X, y).
        """
        self.model.fit(X, y)
        self._support_mask = np.zeros(X.shape[0], dtype=bool)
        self._support_mask[self.model.support_] = True

    def predict(self, X):
        """
        Predict labels (could be multi-class if underlying SVC is multi-class).
        """
        return self.model.predict(X)

    @property
    def support_vectors_(self):
        return self._support_mask

    @property
    def weights_(self):
        """
        Returns the learned weights if kernel='linear', else None.
        """
        if self.kernel == 'linear':
            return self.model.coef_.flatten()
        return None

    @property
    def bias_(self):
        """
        Returns the bias term if kernel='linear'.
        """
        return self.model.intercept_[0]

    def get_params(self):
        return {
            'C': self.C,
            'kernel': self.kernel,
            'max_iter' : self.max_iter
        }
