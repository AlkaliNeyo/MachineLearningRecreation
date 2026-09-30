import numpy as np
import random


# TO DO: Feature Scaling
# Combine with Multiple Linear Regression Model
class SimpleLinearRegression:
    # has three methods, fit (train the model with dataset), predict(predict the dept with indept), evaluate(return the MSE) 

    #lr = learning_rate
    
    def __init__(self, lr = 0.01, n_iter = 3200, m = random.uniform(-1,1), b = random.uniform(-1,1)):
        self.lr = lr
        self.n_iter = n_iter
        self.m = m
        self.b = b

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(X)

        for i in range(self.n_iter):
            y_pred = (self.m * X) + self.b #creates array for what y should be based on x
            error = y - y_pred #creates array based of the difference between the predicted val and the actual value 

            #Calcuate how does we are with MSE 
            #MSE : J(m,b) = 1/n sum(y = (mx+b))^2
            #so for Gradient Descent we need dj(m) and dj(b) (derivative)

            dm = -2/n * np.sum( X * error)
            db = -2/n * np.sum(error)

            self.m -= self.lr * dm
            self.b -= self.lr * db

            if i % 100 == 0:
                mse = np.mean(error**2) #mean of error ^2
                print(f"iter {i}: m={self.m:.4f}, b={self.b:.4f}, mse={mse:.4f}")

        #print(f"\nFinal: m={self.m:.4f}, b={self.b:.4f}")

        #return(self.m , self.b)

    def predict(self, X):
        X = np.array(X)
        return (self.m * X) + self.b
    
    def evaluate(self, X, y):
        y_pred = (self.m * X) + self.b
        error = y - y_pred
        return np.mean(error**2)

     







