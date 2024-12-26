import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import timeit

class SVM:
    # --------- Distance Measure ------------ #
    def __init__(self, X, y, option='soft', C=1):
        self.X = X
        self.y = y
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)[0]
        self.N = len(y)
        self.C = C
        self.V = (y * X).T
        self.option = option
        self.support_vectors = None
    def norm(self, x, p=1):
        '''
        Default: p = 1
            Specific Case + p = 1 - Manhattan Norm
                        + p = 2 - Euclidean Norm
        '''
        return (np.sum(np.abs(x) ** p) ** (1 / p))

    def classifier(self, x):
        return 1 if self.w @ x + self.b >= 0 else -1
    def find_lamda(self):
        option = self.option
        C = self.C
        '''
        QP Programming:
            Minimize: 1/2 λ^T.(P).λ + (q^T).λ
            Subject to: Gλ <= h, Aλ = b
        Hard Margin:
            Minimize: 1/2 λ^T.(V^T.V).λ - (1^T).λ
            Subject to: -λ <= 0, y^T.λ = 0
        Soft Margin:
            Minimize: 1/2 λ^T.(V^T.V).λ - (1^T).λ
            Subject to: -λ <= 0 <= C, y^T.λ = 0
            (just add C in comparation to Hard Margin)
        '''
        if option == 'hard':
            V = self.V
            # build object function
            P = matrix((V.T @ V)) # P = V^T.V
            q = matrix(-np.ones(self.N)) # 1^T.λ, q = 1
            # build A, b, G, h
            G = matrix(-np.eye(self.N)) # -λ, G = -I (N x N)
            h = matrix(np.zeros(self.N)) # -λ <= 0, h: 0
            A = matrix(self.y.reshape(1, -1)) # y^T.λ, A = y^T
            b = matrix(np.zeros((1, 1))) # y^T.λ = 0, b = 0
            # Solve QP Programming 
            sol = solvers.qp(P, q, G, h, A, b)
            # Format CVXOPT sol = {'x': matrix([0.5, 0.0, 0.3, 0.2, 0.0])}  
            lamda_opt = np.array(sol['x']).flatten()
            return lamda_opt
        if option == 'soft':
            V = self.V
            # build object function
            P = matrix((V.T @ V))  
            q = matrix(-np.ones(self.N))
            # print('P-size: ', P.size)
            # print('q-size: ', q.size)
            # build constraints
            G1 = -np.eye(self.N)
            G2 = np.eye(self.N) 
            G = np.vstack((G1, G2))
            h1 = np.zeros((self.N, 1))
            h2 = C * np.ones((self.N, 1))
            h = np.concatenate((h1, h2))
            G, h = matrix(G), matrix(h)
            # print('G-size: ', G.size)
            # print('h-size: ', h.size)
            A = matrix(self.y.reshape((-1, self.N)))
            b = matrix(np.zeros((1,1)))
            # print('A-size: ', A.size)
            # print('b-size: ', b.size)
            
            # solve QP
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)
            lamda_opt = np.array(sol['x']).flatten()
            return lamda_opt        
    def fit(self):
        option = self.option
        C = self.C
        lamda = self.find_lamda()
        epsilon = 1e-4
        S = np.where((lamda > epsilon) & ((lamda <= C) if option == 'soft' else True))[0]
        if len(S) == 0:
            print("No support vectors found!")
            return
        # calculate w and b
        self.w = np.sum((lamda[S].reshape(-1, 1) * self.y[S]) * self.X[S], axis=0)
        self.b = np.mean(self.y[S] - self.X[S] @ self.w.T)
        self.support_vectors = self.X[S]
    def visualize(self):
        option = self.option
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='autumn')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create grid
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        Z = self.w[0] * XX + self.w[1] * YY + self.b  # Use XX and YY to calculate Z

        # Plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

        # Plot support vectors
        if self.support_vectors is not None and len(self.support_vectors) > 0:
            plt.scatter(self.support_vectors[:, 0], self.support_vectors[:, 1], s=100,
                        facecolors='none', edgecolors='k', linewidth=1.5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        if option == 'hard':
            plt.title('Hard Margin - SVM - Constraint Optimization') 
        else:
            plt.title('Soft Margin - SVM - Constraint Optimization')
        plt.show()
def loadRandomData(num_points=100, random_seed=42):
    np.random.seed(random_seed)
    num_features = 2
    # Create Data Class -1
    X_neg = np.random.randn(num_points, num_features) - [2, 2] 
    y_neg = -1 * np.ones(num_points)  # Nhãn lớp -1
    # Create Data Class 1
    X_pos = np.random.randn(num_points, num_features) + [2, 2] 
    y_pos = np.ones(num_points)  # Nhãn lớp 1
    # Concatenate data
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.hstack((y_pos, y_neg)).reshape(-1,1)
    return X, y

def hard_margin():
    X, y = loadRandomData()
    svm = SVM(X, y, option='hard')
    svm.fit()
    svm.visualize()
def soft_margin():
    X, y = loadRandomData()
    svm = SVM(X, y, option='soft', C=1)
    svm.fit()
    svm.visualize()
def sklearn_build():
     # Sklearn: Hard Margin
    X, y = loadRandomData()
    # Train model
    clf = SVC(kernel='linear', C = 0.4)
    clf.fit(X, y) 
    w = clf.coef_
    b = clf.intercept_
    print('w = ', w)
    print('b = ', b)
execution_time = timeit.timeit('sklearn_build()', globals=globals(), number=1)
print(f"Hàm sklearn_build() chạy trong: {execution_time:.6f} giây")
