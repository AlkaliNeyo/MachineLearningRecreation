

import numpy as np
import random

class RobustScaler:

    def __init__(self):
        self.X_median = None
        self.X_IQR = None

    def scale(self, X):
        self.X_median = np.median(X.T, axis=1) # find the medium for each feature 
        X_q1, X_q3 = np.quantile(X.T, [0.25, 0.75], axis=1)
        self.X_IQR = X_q3 - X_q1
        X_scaled =  (X - self.X_median) / self.X_IQR
        return X_scaled

