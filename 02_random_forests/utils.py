"""
Random Forest utilities - Helper functions for ensemble construction.
"""

import numpy as np


def bootstrap_sample(X, y, random_state=None):
    """
    Create a bootstrap sample (random sample with replacement).
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target values
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    X_sample, y_sample : Bootstrap sampled data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    
    return X[indices], y[indices]


def get_random_features(n_features, max_features, random_state=None):
    """
    Get random subset of feature indices.
    
    Parameters:
    -----------
    n_features : int
        Total number of features
    max_features : int or str
        Number of features to sample
        Can be int, 'sqrt', 'log2', or None (all)
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    array : Indices of selected features
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if max_features is None or max_features == n_features:
        return np.arange(n_features)
    
    if max_features == 'sqrt':
        max_features = int(np.sqrt(n_features))
    elif max_features == 'log2':
        max_features = int(np.log2(n_features))
    
    max_features = min(max_features, n_features)
    return np.random.choice(n_features, size=max_features, replace=False)


def majority_vote(predictions):
    """
    Get majority vote from multiple predictions.
    
    Parameters:
    -----------
    predictions : list of arrays
        Predictions from multiple trees
        
    Returns:
    --------
    array : Majority vote for each sample
    """
    # Stack predictions: shape (n_trees, n_samples)
    predictions = np.array(predictions)
    
    # Get majority vote for each sample
    majority = []
    for i in range(predictions.shape[1]):
        votes = predictions[:, i]
        unique, counts = np.unique(votes, return_counts=True)
        majority.append(unique[np.argmax(counts)])
    
    return np.array(majority)


def calculate_oob_score(trees, X, y, oob_indices_list):
    """
    Calculate out-of-bag score for the forest.
    
    Parameters:
    -----------
    trees : list
        Trained decision trees
    X : ndarray
        Full training data
    y : ndarray
        True labels
    oob_indices_list : list of arrays
        OOB indices for each tree
        
    Returns:
    --------
    float : OOB accuracy score
    """
    n_samples = X.shape[0]
    oob_predictions = np.zeros(n_samples)
    oob_counts = np.zeros(n_samples)
    
    # For each sample, collect predictions from trees that didn't see it
    for tree_idx, tree in enumerate(trees):
        oob_indices = oob_indices_list[tree_idx]
        if len(oob_indices) > 0:
            predictions = tree.predict(X[oob_indices])
            oob_predictions[oob_indices] += predictions
            oob_counts[oob_indices] += 1
    
    # Average predictions (only for samples that have OOB predictions)
    mask = oob_counts > 0
    oob_predictions[mask] /= oob_counts[mask]
    oob_predictions = np.round(oob_predictions).astype(int)
    
    # Calculate accuracy on OOB samples
    if np.sum(mask) > 0:
        accuracy = np.mean(oob_predictions[mask] == y[mask])
        return accuracy
    else:
        return 0.0


def calculate_feature_importance(trees, X, y, n_features):
    """
    Calculate feature importance based on mean decrease in impurity.
    
    Parameters:
    -----------
    trees : list
        Trained decision trees  
    X : ndarray
        Training data
    y : ndarray
        Training labels
    n_features : int
        Number of features
        
    Returns:
    --------
    array : Feature importance scores
    """
    importances = np.zeros(n_features)
    
    for tree in trees:
        # Traverse tree and accumulate importance
        tree_importances = _get_tree_feature_importance(tree.root, n_features)
        importances += tree_importances
    
    # Normalize
    if np.sum(importances) > 0:
        importances /= np.sum(importances)
    
    return importances


def _get_tree_feature_importance(node, n_features, importances=None):
    """
    Recursively calculate feature importance for a tree.
    
    Parameters:
    -----------
    node : Node
        Current tree node
    n_features : int
        Number of features
    importances : array, optional
        Accumulated importances
        
    Returns:
    --------
    array : Feature importance scores
    """
    if importances is None:
        importances = np.zeros(n_features)
    
    if node is None or node.is_leaf():
        return importances
    
    # Add importance for this split
    # (In real implementation, would track impurity decrease)
    importances[node.feature] += 1
    
    # Recurse
    _get_tree_feature_importance(node.left, n_features, importances)
    _get_tree_feature_importance(node.right, n_features, importances)
    
    return importances
