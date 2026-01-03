"""
Decision Tree utilities - Helper functions for tree construction and evaluation.
"""

import numpy as np
from collections import Counter


def entropy(y):
    """
    Calculate entropy of a dataset.
    
    H(S) = -sum(p_i * log2(p_i))
    
    Parameters:
    -----------
    y : array-like
        Target labels
        
    Returns:
    --------
    float : Entropy value
    """
    if len(y) == 0:
        return 0
    
    # Count occurrences of each class
    counts = np.bincount(y)
    probabilities = counts[counts > 0] / len(y)
    
    # Calculate entropy
    return -np.sum(probabilities * np.log2(probabilities))


def gini_impurity(y):
    """
    Calculate Gini impurity of a dataset.
    
    Gini = 1 - sum(p_i^2)
    
    Parameters:
    -----------
    y : array-like
        Target labels
        
    Returns:
    --------
    float : Gini impurity value
    """
    if len(y) == 0:
        return 0
    
    counts = np.bincount(y)
    probabilities = counts[counts > 0] / len(y)
    
    return 1 - np.sum(probabilities ** 2)


def information_gain(y, y_left, y_right, criterion='gini'):
    """
    Calculate information gain from a split.
    
    IG = impurity(parent) - weighted_avg(impurity(children))
    
    Parameters:
    -----------
    y : array-like
        Parent node labels
    y_left : array-like
        Left child labels
    y_right : array-like
        Right child labels
    criterion : str
        'gini' or 'entropy'
        
    Returns:
    --------
    float : Information gain value
    """
    impurity_func = entropy if criterion == 'entropy' else gini_impurity
    
    # Parent impurity
    parent_impurity = impurity_func(y)
    
    # Weighted children impurity
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    weighted_impurity = (n_left / n) * impurity_func(y_left) + \
                       (n_right / n) * impurity_func(y_right)
    
    return parent_impurity - weighted_impurity


def find_best_split(X, y, feature_idx, criterion='gini'):
    """
    Find best threshold to split a continuous feature.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : array-like
        Target labels
    feature_idx : int
        Index of feature to split on
    criterion : str
        'gini' or 'entropy'
        
    Returns:
    --------
    tuple : (best_threshold, best_gain)
    """
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values)
    
    if len(unique_values) == 1:
        return None, 0
    
    best_gain = 0
    best_threshold = None
    
    # Try each midpoint between consecutive unique values
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        
        # Split data
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        # Calculate information gain
        gain = information_gain(y, y_left, y_right, criterion)
        
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    return best_threshold, best_gain


def most_common_label(y):
    """
    Return the most common label in y.
    
    Parameters:
    -----------
    y : array-like
        Target labels
        
    Returns:
    --------
    int : Most common label
    """
    if len(y) == 0:
        return None
    counter = Counter(y)
    return counter.most_common(1)[0][0]


def calculate_leaf_value(y, task='classification'):
    """
    Calculate prediction value for a leaf node.
    
    Parameters:
    -----------
    y : array-like
        Target values in leaf
    task : str
        'classification' or 'regression'
        
    Returns:
    --------
    Prediction value for the leaf
    """
    if task == 'classification':
        return most_common_label(y)
    else:  # regression
        return np.mean(y)


def calculate_mse(y):
    """
    Calculate Mean Squared Error for regression.
    
    Parameters:
    -----------
    y : array-like
        Target values
        
    Returns:
    --------
    float : MSE value
    """
    if len(y) == 0:
        return 0
    return np.var(y)


def accuracy_score(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Calculate confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    n_classes : int, optional
        Number of classes
        
    Returns:
    --------
    ndarray : Confusion matrix
    """
    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    
    return cm
