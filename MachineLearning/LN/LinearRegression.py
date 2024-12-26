import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import os
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=UserWarning)

# Download latest version


class MyLinearRegression:
    # Initialize the model
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None
    # Add bias column (1s) to the feature matrix X
    def add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]
    # Computes output
    def output(self, x):
        return self.add_bias(x) @ self.w
    # Calculate the weights using the normal equation: w = (X^T X)^-1 X^T y
    def fit(self):
        X_bias = self.add_bias(self.X)
        self.w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ self.y       
    def score(self):
        return self.w
    # Visualize the data and the regression line (hyperplane)
    def visualize(self):
        plt.scatter(self.X, self.y, color='red', label='Data Points')
        y_pred = self.output(self.X)
        plt.plot(self.X, y_pred, color='green', label='Regression Line')
        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.legend()
        plt.title('Linear Regression Visualization')
        plt.show()
    def prediction(self, X_test):
        return self.output(X_test)
    
    
# Kaggle: Dataset weight-height.csv
 # take data
path = kagglehub.dataset_download("mustafaali96/weight-height")
print("Path to dataset files:", path)
files = os.listdir(path)
dataset_file = os.path.join(path, "weight-height.csv")
data = pd.read_csv(dataset_file)
    
def main():
    '''
    Method: No Sklearn  & Sklearn
    Trick: { Bias Trick }
    Input: weight
    Output: height
    Model: Linear Regression
    Equation: y = w.x
    '''
    X = data.iloc[:200, 2]
    y = data.iloc[:200, 1]
    X = np.array([X]).T
    y = np.array([y]).T
    X_test = [[170], [190]]
    X_test = np.array(X_test)
    req = LinearRegression().fit(X, y)
    ln = MyLinearRegression(X,y)
    ln.fit()
    y_my_pred = ln.prediction(X_test)
    y_pred = req.predict(X_test)
    
    print('Sklearn Prediction: ' ,y_pred)
    print('MyClass Prediction: ' ,y_my_pred)
    ln.visualize()
# Run the program
main()
