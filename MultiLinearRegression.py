import numpy as np
import random
from RobustScaler import RobustScaler
class MultiLinearRegression:
    # has three methods, fit (train the model with dataset), predict(predict the dept with indept), evaluate(return the MSE) 
    # this model will use Robust Scaling to feature scale the data 
    #lr = learning_rate
    
    def __init__(self, lr = 0.01, n_iter = 1000, b = 0):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = b
        self.scaler = RobustScaler()

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #print(n_features)
        self.w = np.zeros(n_features) #make array for weight, equal to the amount of features. 
        X = np.array(X)
        y = np.array(y)

        #Feature Scaling 
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        for i in range(self.n_iter):
            y_pred = X_scaled @ self.w + self.b # add the two arrays together and than add the b 
            error = y - y_pred #creates array based of the difference between the predicted val and the actual value 
            #Calcuate how does we are with MSE 
            #MSE : J(m,b) = 1/n sum(y = (mx+b))^2
            #so for Gradient Descent we need dj(m) and dj(b) (derivative)
            dm = (-2/n_samples) * ( X_scaled.T @ error)
            db = (-2/n_samples) * np.sum(error)

            self.w -= self.lr * dm
            self.b -= self.lr * db

            if i % 100 == 0:
                self.lr = self.lr * 0.70
                #mse = np.mean(error**2) #mean of error ^2
                print(f"iter {i}: weight={self.w}, b={self.b:.4f}")

        #print(f"\nFinal: m={self.m:.4f}, b={self.b:.4f}")

        #return(self.m , self.b)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return  X_scaled @ self.w + self.b
    
    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        y_pred = X_scaled @ self.w + self.b
        error = y - y_pred
        return np.mean(error**2)

     







