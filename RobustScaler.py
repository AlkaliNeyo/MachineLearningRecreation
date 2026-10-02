

import numpy as np
import random

class RobustScaler:

    def __init__(self):
        self.X_median = None
        self.X_IQR = None
        self.n_features = None

    def _to_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X):
        X = self._to_2d(X)
        self.n_features = X.shape[1]
        self.X_median = np.median(X.T, axis=1) # find the medium for each feature 
        X_q1, X_q3 = np.quantile(X.T, [0.25, 0.75], axis=1)
        self.X_IQR = X_q3 - X_q1

    def transform(self, X):
        X = self._to_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )
        X_scaled = (X - self.X_median) / self.X_IQR
        return X_scaled


