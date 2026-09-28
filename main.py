import random
import numpy as np
from model import SimpleLinearRegression
import csv
import pandas as pd
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'DataSets', 'Salary_dataset.csv')
    df = pd.read_csv(csv_path)
    
    df_binary = df[['YearsExperience', 'Salary']]    

    # Taking only the selected two attributes from the dataset
    df_binary.columns = ['YrsOfExp', 'Salary']
    #display the first 5 rows
    #print(df_binary.head())

    X = np.array(df_binary['YrsOfExp'])
    y = np.array(df_binary['Salary'])

    model = SimpleLinearRegression()

    model.fit(X, y)


    new_x = np.array([6, 12])
    print("New predictions:", model.predict(new_x))



if __name__ == "__main__":
    main()