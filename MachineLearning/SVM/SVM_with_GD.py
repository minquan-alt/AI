import numpy as np
import matplotlib.pyplot as plt
import timeit

random_state = 42
class softSVM:
    def __init__(self, C):
        '''
        C: trade-off parameter between margin and loss
        '''
        self.C = C
        self.lamda = 1/C
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.n = 0 # number of data points
        self.d = 0 # number of features
    def decision_function(self, X):
        return X @ self.w + self.b
    def cost(self, X):
        return 1/2 * (self.w.T @ self.w) + self.C * np.sum(np.maximum(0, 1 - self.y * self.decision_function(X)))
    def find_support_vectors(self, X):
        sv_idx = np.where((self.y * self.decision_function(X)) < 1)[0]
        return sv_idx
    def gradient_descent(self, sv_idx):
        if len(sv_idx) == 0:
            return np.zeros_like(self.w), 0
        d_w = self.lamda * self.w - (self.C * (self.y[sv_idx] @ self.X[sv_idx])).T
        d_b = -self.C * np.mean(self.y[sv_idx])
        return d_w, d_b
    def fit(self, X, y, lr=1e-3, epochs=500):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.w = np.random.rand(self.d)
        self.b = 0
        cost_values = []
        for epoch in range(epochs):
            sv_idx = self.find_support_vectors(X)
            d_w, d_b = self.gradient_descent(sv_idx)
            self.w -= lr * d_w
            self.b -= lr * d_b
            cost = self.cost(X)
            cost_values.append(cost)
            print(f"Epoch {epoch+1}/{epochs}, Cost: {cost}")
        plt.plot(cost_values)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.title("Cost Function Convergence")
        plt.show()
        self.support_vectors = self.find_support_vectors(X)
    def predict(self, X):
        return np.sign(self.decision_function(X))
    def score(self, X, y):
        predict_labels = self.predict(X)
        return np.mean(y == predict_labels)
    def visualize(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, marker='o', cmap='autumn')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # create grid
        xx = np.linspace(xlim[0], xlim[1], 30) 
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.decision_function(xy).reshape(XX.shape)
        
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['g', 'k', 'g'], levels=[-1, 0, 1], linestyles=['--', '-', '--'])
        # highlight sv
        ax.scatter(self.X[self.support_vectors, 0], self.X[self.support_vectors, 1], s=80, facecolors='none', edgecolors='k', linewidth=1)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
def main():
    np.random.seed(random_state)
    num_points = 300
    num_features = 2
    # Create Data Class -1
    X_neg = np.random.randn(num_points, num_features) - [1, 1] 
    y_neg = -1 * np.ones(num_points)  # Nhãn lớp -1
    # Create Data Class 1
    X_pos = np.random.randn(num_points, num_features) + [2, 2] 
    y_pos = np.ones(num_points)  # Nhãn lớp 1
    # Concatenate data
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.hstack((y_pos, y_neg))
    sm = softSVM(C=100)
    sm.fit(X, y)
    sm.visualize()
main()
        
        
        
        
        
        
        
