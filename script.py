import argparse
import numpy as np
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from tqdm.auto import tqdm
from collections import Counter


# Include all the classes from the original code here (BaseSolver, GradientDescentSolver, CVXQPSolver, SklearnSVMSolver, CuttingPlaneSolver, LSSVMSolver, NewtonMethodSolver, PegasosSolver, SVMClassifier, OvOSVMClassifier, OvRSVMClassifier)
from wrappers.OvOSVMClassifier import OvOSVMClassifier
from wrappers.OvRSVMClassifier import OvRSVMClassifier
from solvers.CVXQPSolver import CVXQPSolver
from solvers.GradientDescentSolver import GradientDescentSolver
from solvers.SklearnSVMSolver import SklearnSVMSolver
from solvers.CuttingPlaneSolver import CuttingPlaneSolver
from solvers.LSSVMSolver import LSSVMSolver
from solvers.NewtonMethodSolver import NewtonMethodSolver
from solvers.PegasosSolver import PegasosSolver


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SVM classifiers on MNIST with various solvers.')
    parser.add_argument('--solver', type=str, required=True,
                        choices=['gd', 'cvx', 'sklearn', 'cuttingplane', 'pegasos', 'newton', 'lssvm'],
                        help='SVM solver to use: gd (Gradient Descent), cvx (CVXPY QP), sklearn, cuttingplane, pegasos, newton, lssvm (Least Squares SVM)')
    parser.add_argument('--strategy', type=str, required=True, choices=['ovo', 'ovr'],
                        help='Multiclass strategy: ovo (One-vs-One) or ovr (One-vs-Rest)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Regularization parameter C (default: 1.0)')
    parser.add_argument('--max_iters', type=int, default=1000,
                        help='Maximum number of iterations for the solver (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for gradient-based solvers (default: 0.01)')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of PCA components (default: 50)')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use a subset of the dataset with specified size for faster testing (default: None)')
    args = parser.parse_args()

    # Load MNIST dataset
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target.astype(int)

    # Subset the data if specified
    if args.subset_size is not None:
        X = X[:args.subset_size]
        y = y[:args.subset_size]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize solver based on arguments
    if args.solver == 'gd':
        solver = GradientDescentSolver(learning_rate=args.learning_rate, C=args.C, max_iters=args.max_iters)
    elif args.solver == 'cvx':
        solver = CVXQPSolver(C=args.C, max_iters=args.max_iters)
    elif args.solver == 'sklearn':
        solver = SklearnSVMSolver(C=args.C, kernel='linear', max_iter=args.max_iters)
    elif args.solver == 'cuttingplane':
        solver = CuttingPlaneSolver(C=args.C, max_iter=args.max_iters, lr=args.learning_rate)
    elif args.solver == 'pegasos':
        solver = PegasosSolver(C=args.C, max_iters=args.max_iters)
    elif args.solver == 'newton':
        solver = NewtonMethodSolver(C=args.C, max_iter=args.max_iters)
    elif args.solver == 'lssvm':
        solver = LSSVMSolver(C=args.C)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

    # Initialize multiclass classifier
    if args.strategy == 'ovr':
        classifier = OvRSVMClassifier(solver=solver, n_components=args.n_components)
    else:
        classifier = OvOSVMClassifier(solver=solver, n_components=args.n_components)

    # Train and evaluate
    print(f"Training {args.strategy.upper()} classifier with {args.solver} solver...")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time

    print("Evaluating...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Collect support vectors count
    total_sv = 0
    if args.strategy == 'ovr':
        for clf in classifier.classifiers_.values():
            total_sv += np.sum(clf.support_vectors_)
    else:
        for clf in classifier.classifiers_.values():
            total_sv += np.sum(clf.support_vectors_)

    # Output results
    print("\n=== Results ===")
    print(f"Solver: {args.solver}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total Support Vectors: {total_sv}")
    print(f"Training Time: {training_time:.2f} seconds")


if __name__ == "__main__":
    main()