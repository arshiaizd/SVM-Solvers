import time

import numpy as np
from sklearn.preprocessing import StandardScaler

# ===================
# Binary SVM Wrapper
# ===================

class SVMClassifier:
    """
    A binary SVM classifier that uses any solver conforming to BaseSolver.
    Applies standard scaling internally before training and predicting.
    """

    def __init__(self, solver):
        """
        Parameters:
        -----------
        solver : BaseSolver
            The solver to be used for training (e.g., NewtonMethodSolver, Pegasos).
        """
        self.solver = solver
        self.scaler = None
        self.training_time = None

    def fit(self, X, y):
        """
        Fit the binary classifier with standard scaling applied internally.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) in {-1, +1}
        """
        start_time = time.time()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.solver.fit(X_scaled, y)
        self.training_time = time.time() - start_time

    @property
    def support_vectors_(self):
        return self.solver.support_vectors_

    def predict(self, X):
        """
        Predict labels in {-1, +1} with the trained binary classifier.

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)

        Returns:
        --------
        preds : ndarray of shape (n_samples,) in {-1, +1}.
        """
        X_scaled = self.scaler.transform(X)
        return self.solver.predict(X_scaled)


    def decision_function(self, X):
        """
        the raw margin: w^T x + b.
        solver.weights_ and solver.bias_ must be defined after fit.
        """
        return X.dot(self.solver.weights_) + self.solver.bias_


    def get_metrics(self, X_test, y_test):
        """
        Evaluate classifier on test data.

        Parameters:
        -----------
        X_test : ndarray
            Test features.
        y_test : ndarray
            True labels in {-1, +1}.

        Returns:
        --------
        metrics : dict
            Includes number of support vectors, training time, accuracy,
            learned weights, and learned bias.
        """
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return {
            'support_vectors': np.sum(self.solver.support_vectors_),
            'training_time': self.training_time,
            'test_accuracy': accuracy,
            'weights': self.solver.weights_,
            'bias': self.solver.bias_
        }
