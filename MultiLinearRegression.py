import numpy as np
import random

class MultiLinearRegression:
    # has three methods, fit (train the model with dataset), predict(predict the dept with indept), evaluate(return the MSE) 
    # this model will use Robust Scaling to feature scale the data 
    #lr = learning_rate
    
    def __init__(self, lr = 0.01, n_iter = 1000, b = 0):
        self.lr = lr
        self.n_iter = n_iter
        self.m = None
        self.b = b

        #Used for Robust Scaling
        self.X_median = None
        self.X_IQR = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #print(n_features)
        self.m = np.zeros(n_features) #make array for weight, equal to the amount of features. 
        X = np.array(X)
        y = np.array(y)

        #Feature Scaling 
        self.X_median = np.median(X.T, axis=1) # find the medium for each feature 
        X_q1, X_q3 = np.quantile(X.T, [0.25, 0.75], axis=1)
        self.X_IQR = X_q3 - X_q1
        X_scaled =  (X - self.X_median) / self.X_IQR

        for i in range(self.n_iter):
            y_pred = X_scaled @ self.m + self.b # add the two arrays together and than add the b 
            error = y - y_pred #creates array based of the difference between the predicted val and the actual value 

           
            #Calcuate how does we are with MSE 
            #MSE : J(m,b) = 1/n sum(y = (mx+b))^2
            #so for Gradient Descent we need dj(m) and dj(b) (derivative)

            dm = (-2/n_samples) * ( X_scaled.T @ error)
            db = (-2/n_samples) * np.sum(error)

            self.m -= self.lr * dm
            self.b -= self.lr * db

            #if i % 100 == 0:
                #mse = np.mean(error**2) #mean of error ^2
                #print(f"iter {i}: m={self.m}, b={self.b:.4f}")

        #print(f"\nFinal: m={self.m:.4f}, b={self.b:.4f}")

        #return(self.m , self.b)

    def predict(self, X):
        X = np.array(X)
        X_scaled =  (X - self.X_median) / self.X_IQR
        return  X_scaled @ self.m + self.b
    
    def evaluate(self, X, y):
        X = np.array(X)
        X_scaled =  (X - self.X_median) / self.X_IQR
        y_pred = X_scaled @ self.m + self.b
        error = y - y_pred
        return np.mean(error**2)

     







