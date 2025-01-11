import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # Step size for the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # Number of data points
    N = int(m / 2)  # Points per class
    D = 2  # Dimensions
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # Maximum radius
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # Angle
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # Radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X.T, Y.T

def load_extra_datasets():
    noisy_circles = datasets.make_circles(n_samples=300, factor=0.5, noise=0.3)
    noisy_moons = datasets.make_moons(n_samples=300, noise=0.2)
    blobs = datasets.make_blobs(n_samples=300, random_state=5, centers=6)
    gaussian_quantiles = datasets.make_gaussian_quantiles(mean=None, cov=0.5, 
                                                          n_samples=300, 
                                                          n_features=2, 
                                                          n_classes=2)
    no_structure = np.random.rand(300, 2), None
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
