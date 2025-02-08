import time
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from wrappers.SVMClassifier import SVMClassifier


###############################################################################
#    3) One-vs-One Multi-class with PCA and Scaling (OvOSVMClassifier)       #
###############################################################################

class OvOSVMClassifier:
    """
    One-vs-One multi-class classification with PCA dimensionality reduction.
    Trains one binary SVM for each pair of classes, then uses majority vote.
    """

    def __init__(self, solver, n_components=50):
        """
        Parameters:
        -----------
        solver : BaseSolver
            A solver instance for binary SVM.
        n_components : int
            Number of principal components to retain via PCA.
        """
        self.solver = solver
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.classes_ = None
        self.classifiers_ = {}
        self.training_time = None

    def fit(self, X, y):
        """
        Fit OvO multi-class classifier:
          1) Scale features
          2) Reduce dimension with PCA
          3) Train binary classifiers for each pair of classes
        """
        start_time = time.time()

        # Scale and PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(y)

        self.pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # For each pair of classes, train a separate binary classifier
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes_[i]
                class_j = self.classes_[j]

                idx = np.where((y == class_i) | (y == class_j))[0]
                X_ij = X_pca[idx]
                y_ij = y[idx]

                # Convert to {-1, +1}
                y_ij_binary = np.where(y_ij == class_i, -1, 1)

                # Re-instantiate solver with same constructor params
                solver_copy = type(self.solver)(**self.solver.get_params())

                # Train a binary SVMClassifier on this pair
                clf = SVMClassifier(solver=solver_copy)
                clf.fit(X_ij, y_ij_binary)
                self.classifiers_[(class_i, class_j)] = clf

        self.training_time = time.time() - start_time

    def predict(self, X):
        """
        Predicts multi-class labels via majority vote across all pairwise classifiers.
        """
        # Scale + PCA transform test data
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Collect votes from each pairwise classifier
        votes = []
        for (class_i, class_j), clf in self.classifiers_.items():
            pred_binary = clf.predict(X_pca)  # in {-1, +1}
            mapped = np.where(pred_binary == -1, class_i, class_j)
            votes.append(mapped)

        votes = np.vstack(votes).T  # shape (n_samples, n_classifiers)

        final_pred = []
        for row in votes:
            freq = Counter(row)
            most_common = freq.most_common(1)[0][0]
            final_pred.append(most_common)

        return np.array(final_pred)

    def get_metrics(self, X, y):
        """
        Evaluate OvO classifier on test data, returning accuracy and support-vector stats.
        """
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)

        # Sum support vectors across all pairwise classifiers
        total_sv = 0
        for _, clf in self.classifiers_.items():
            total_sv += np.sum(clf.support_vectors_)

        return {
            'support_vectors': total_sv,
            'training_time': self.training_time,
            'test_accuracy': acc
        }
