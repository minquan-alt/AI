import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import Perceptron


# download stopwords
nltk.download('stopwords')

def train_test_split(X, y, p):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X và y phải có số lượng mẫu bằng nhau.")
    if not (0 < p < 1):
        raise ValueError("p phải là một giá trị giữa 0 và 1.")
    size = X.shape[0]
    train_size = int(size * p)
    indices = np.random.permutation(size)
    train_indices = indices[: train_size]
    test_indices = indices[train_size:]
    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test
    
class PLA:
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w
        self.N = X.shape[0]
    def visualize(self):
        X_pos = []
        X_neg = []
        for i in range(self.N):
            if self.y[i] == 1:
                X_pos.append(self.X[i])
            else:
                X_neg.append(self.X[i])
        X_pos = np.array(X_pos)
        X_neg = np.array(X_neg)
        plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Class 1')
        plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Class -1')
        x1_hyper = np.linspace(-5, 5, 100)
        x2_hyper = -(self.w[0] + self.w[1]*x1_hyper)/self.w[2]
        plt.plot(x1_hyper, x2_hyper, color='green', label='Hyperplane')
        plt.legend()
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perfectly Separable Data')
        plt.show()
    def predict_visualize(self, X_test):
        X_pos = []
        X_neg = []
        X_test_pos = []
        X_test_neg = []
        y_pred = self.predict(X_test)
        
        for i in range(X_test.shape[0]):
            if y_pred[i] == 1:
                X_test_pos.append(X_test[i])
            else:
                X_test_neg.append(X_test[i])
        for i in range(self.N):
            if self.y[i] == 1:
                X_pos.append(self.X[i])
            else:
                X_neg.append(self.X[i])
        X_pos = np.array(X_pos)
        X_neg = np.array(X_neg)
        X_test_pos = np.array(X_test_pos)
        X_test_neg = np.array(X_test_neg)
        plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red', label='Class 1')
        plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='Class -1')
        plt.scatter(X_test_pos[:, 0], X_test_pos[:, 1], color='pink', label='Class 1 - New point')
        plt.scatter(X_test_neg[:, 0], X_test_neg[:, 1], color='#899AE5', label='Class -1 - New point')
        x1_hyper = np.linspace(-5, 5, 100)
        x2_hyper = -(self.w[0] + self.w[1]*x1_hyper)/self.w[2]
        plt.plot(x1_hyper, x2_hyper, color='green', label='Hyperplane')
        
        # plt.legend()
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perfectly Separable Data')
        plt.show()
    def sgn(self, i): # just use for training
        return 1 if np.dot(self.w, np.append([1], self.X[i, :])) >= 0 else -1

    def misclassified_points(self):
        return [i for i in range(self.N) if self.sgn(i) != self.y[i]]

    def fit(self):
        ite = 0
        while True:
            print('ite: ', ite)
            mis_pts = self.misclassified_points()
            if len(mis_pts) == 0:
                break
            i = mis_pts[0]  # Chọn điểm đầu tiên trong danh sách misclassified
            self.w += self.y[i] * np.append([1], self.X[i, :])  # Cập nhật trọng số
            ite += 1
        print(f"Training completed in {ite} iterations")
        print(self.w)
    def predict(self, X_test):
        y_pred = [ 1 if self.w @ np.append([1], X_test[i, :]) >= 0 else -1  for i in range(X_test.shape[0]) ]
        return y_pred
# Đặt seed cố định để tái hiện kết quả
def main():
    np.random.seed(42)

    # Số lượng điểm dữ liệu mỗi lớp
    num_points = 100
    num_features = 2

    # Tạo dữ liệu lớp -1
    X_neg = np.random.randn(num_points, num_features) - [2, 2] 
    y_neg = -1 * np.ones(num_points)  # Nhãn lớp -1

    # Tạo dữ liệu lớp 1
    X_pos = np.random.randn(num_points, num_features) + [2, 2] 
    y_pos = np.ones(num_points)  # Nhãn lớp 1
    
    X_pos += 1e-6 * np.random.randn(*X_pos.shape)
    X_neg += 1e-6 * np.random.randn(*X_neg.shape)

    # Kết hợp hai lớp dữ liệu
    X = np.vstack((X_neg, X_pos))
    y = np.hstack((y_neg, y_pos))
    print(y)
    w = [0, -10, 7]

    pla = PLA(X, y, w)
    pla.fit()
    np.random.seed(42)  # Để đảm bảo kết quả nhất quán
    X_test = np.random.uniform(low=-2.5, high=2.5, size=(100, 2))  # 100 điểm ngẫu nhiên trong không gian [-2.5, 2.5]
    # sklearn
    pla.predict_visualize(X_test)
    clf = Perceptron()
    clf.fit(X, y)
    sklearn_pred = clf.predict(X_test)
    my_pred = pla.predict(X_test)
    similarity = sum(1 for i in range(len(my_pred)) if sklearn_pred[i] == my_pred[i]) / len(my_pred)
    print(f'Similarity: {similarity}%')
main()






