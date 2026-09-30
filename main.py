import random
import numpy as np
from SimpleLinearRegression import SimpleLinearRegression
from MultiLinearRegression import MultiLinearRegression
import csv
import pandas as pd
import os

def main():
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'DataSets', 'multiple_linear_regression_dataset_mlr.csv') #multiple_linear_regression_dataset_mlr.csv
    df = pd.read_csv(csv_path)

    #Split the dataset into target and independent variables 
    tar = "income"

    X = df.drop(columns=[tar]).to_numpy()
    y = df[tar].to_numpy() 

    #print(y)

    model = MultiLinearRegression()

    model.fit(X, y)

    new_x = np.array([37, 5])
    print("New predictions: for {}: ".format(new_x), model.predict(new_x))



if __name__ == "__main__":
    main()