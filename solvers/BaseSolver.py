
# =====================
# Base Solver Interface
# =====================

class BaseSolver:
    """
    Abstract base class for SVM solvers. Each concrete solver must implement:
      - fit(X, y): Train on data/features X with labels y.
      - predict(X): Predict labels for new data.
      - support_vectors_, weights_, bias_: Properties to access the learned model.
      - get_params(): Return the constructor parameters for solver replication.
    """

    def fit(self, X, y):
        # Train the model on features X with labels y
        raise NotImplementedError

    def predict(self, X):
        # Predict labels for input features X
        raise NotImplementedError

    @property
    def support_vectors_(self):
        # Returns a boolean mask indicating which samples are support vectors
        raise NotImplementedError

    @property
    def weights_(self):
        # Returns the learned weight vector (if applicable)
        raise NotImplementedError

    @property
    def bias_(self):
        # Returns the learned bias term (if applicable)
        raise NotImplementedError

    def get_params(self):
        """
        Must be overridden to return a dictionary of constructor parameters,
        enabling replication of the solver.
        """
        raise NotImplementedError
