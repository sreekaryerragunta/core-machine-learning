"""
SVM utilities (minimal - most logic in sklearn).
"""

import numpy as np


def rbf_kernel(X1, X2, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel between X1 and X2.
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    
    Parameters:
    -----------
    X1 : ndarray of shape (n1, n_features)
    X2 : ndarray of shape (n2, n_features)
    gamma : float
        Kernel coefficient
        
    Returns:
    --------
    K : ndarray of shape (n1, n2)
        Kernel matrix
    """
    # Compute squared Euclidean distances
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
               np.sum(X2**2, axis=1) - \
               2 * np.dot(X1, X2.T)
    
    # Apply RBF kernel
    return np.exp(-gamma * sq_dists)


def polynomial_kernel(X1, X2, degree=3, coef0=1):
    """
    Compute polynomial kernel.
    
    K(x, y) = (x^T y + coef0)^degree
    
    Parameters:
    -----------
    X1, X2 : ndarrays
        Input matrices
    degree : int
        Polynomial degree
    coef0 : float
        Independent term
        
    Returns:
    --------
    K : ndarray
        Kernel matrix
    """
    return (np.dot(X1, X2.T) + coef0) ** degree


def linear_kernel(X1, X2):
    """
    Compute linear kernel (dot product).
    
    K(x, y) = x^T y
    """
    return np.dot(X1, X2.T)
