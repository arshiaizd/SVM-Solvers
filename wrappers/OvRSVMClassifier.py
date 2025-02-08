import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from wrappers.SVMClassifier import SVMClassifier


###############################################################################
#    3) One-vs-Rest Multi-class with PCA and Scaling (OvRSVMClassifier)       #
###############################################################################


class OvRSVMClassifier:
    """
    One-vs-Rest multi-class SVM that applies PCA for dimensionality reduction.
    For each class 'c', trains a binary classifier distinguishing 'c' vs all others.
    The prediction is the class with the highest margin.
    """

    def __init__(self, solver, n_components=50):
        """
        Parameters:
        -----------
        solver : BaseSolver
            Solver instance for binary SVM.
        n_components : int
            Number of PCA components to retain.
        """
        self.solver = solver
        self.n_components = n_components
        self.classes_ = None
        self.classifiers_ = {}
        self.scaler = None
        self.pca = None
        self.training_time = None

    def fit(self, X, y):
        """
        Fit OvR multi-class classifier:
          1) Scale data
          2) Apply PCA
          3) Train one binary SVM per class (class vs. rest)
        """
        start_time = time.time()

        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(y)

        # PCA dimensionality reduction
        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # Train binary SVM for each class
        self.classes_ = np.unique(y)

        for c in self.classes_:
            print(f"checking class {c}")
            y_bin = np.where(y == c, 1, -1)
            solver_copy = type(self.solver)(**self.solver.get_params())
            clf = SVMClassifier(solver=solver_copy)
            clf.fit(X_pca, y_bin)
            self.classifiers_[c] = clf

        self.training_time = time.time() - start_time

    def predict(self, X):
        """
        Predict class labels by picking the class whose classifier gives the
        highest decision margin.
        """
        # Scale + PCA transform
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Collect decision_function from each class
        margins = {}
        for c, clf in self.classifiers_.items():
            # We define a special method below for margin
            margins[c] = clf.decision_function(X_pca)

        # Stack each class's margin into a matrix
        n_samples = X_pca.shape[0]
        n_classes = len(self.classes_)
        all_scores = np.zeros((n_samples, n_classes), dtype=float)

        for i, c in enumerate(self.classes_):
            all_scores[:, i] = margins[c]

        # Pick the class with the max margin for each sample
        best_indices = np.argmax(all_scores, axis=1)
        return self.classes_[best_indices]

    def get_metrics(self, X, y):
        """
        Evaluate the trained OvR classifier on test data (X, y).
        Returns accuracy and total support vector count.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)

        total_sv = 0
        for _, clf in self.classifiers_.items():
            total_sv += np.sum(clf.support_vectors_)

        return {
            'support_vectors': total_sv,
            'training_time': self.training_time,
            'test_accuracy': accuracy
        }
