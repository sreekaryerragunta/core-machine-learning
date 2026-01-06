# Random Forests - Theory

## What is a Random Forest?

A **Random Forest** is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model.

**Core idea**: Train many decision trees on different subsets of data and features, then aggregate their predictions. "Wisdom of the crowd" for machine learning.

---

## Key Concepts

### 1. Ensemble Learning

Instead of relying on a single model, combine multiple models to improve performance.

**Benefits**:
- **Reduced overfitting**: Individual trees may overfit, but averaging reduces variance
- **Better generalization**: Ensemble is more robust than any single tree
- **Improved accuracy**: Collective decision is often better than individual ones

### 2. Bagging (Bootstrap Aggregating)

**Algorithm**:
1. Create multiple bootstrap samples (random sampling with replacement)
2. Train a decision tree on each sample
3. Aggregate predictions (voting for classification, averaging for regression)

**Why it works**: Different trees see different data → diverse models → better ensemble

---

## How Random Forests Work

### Training Process

```
For each tree t = 1 to n_trees:
    1. Create bootstrap sample (random sample with replacement)
    2. For each node split:
        - Randomly select subset of features (feature randomness)
        - Find best split using only these features
    3. Grow tree to maximum depth (or stopping criterion)

Prediction:
    - Classification: Majority vote across all trees
    - Regression: Average of all tree predictions
```

### Key Hyperparameters

1. **n_estimators**: Number of trees in forest
   - More trees → better performance, but slower
   - Typical: 100-500 trees

2. **max_features**: Number of features to consider for each split
   - √n for classification
   - n/3 for regression
   - Adds randomness and decorrelates trees

3. **max_depth**: Maximum tree depth
   - Controls individual tree complexity
   - Usually left unlimited for Random Forests

4. **min_samples_split**: Min samples to split
   - Prevents overfitting
   - Typical: 2-10

5. **bootstrap**: Whether to use bootstrap samples
   - Almost always True for Random Forests

---

## Bootstrap Sampling

**What**: Sample m data points WITH replacement from dataset of size m.

**Result**: ~63.2% unique samples, ~36.8% duplicates

**Example**:
```
Original: [1, 2, 3, 4, 5]
Bootstrap sample 1: [2, 2, 4, 5, 1]  
Bootstrap sample 2: [3, 1, 5, 5, 4]
Bootstrap sample 3: [1, 4, 2, 2, 3]
```

**Why**: Creates diversity in training data for each tree.

---

## Feature Randomness

At each node, only consider a random subset of features for splitting.

**Number of features**:
- Classification: $\sqrt{n}$ features
- Regression: $n/3$ features

**Effect**: Decorrelates trees by forcing them to consider different features.

Without feature randomness → all trees might use same strong feature first → correlated trees → less benefit from ensemble.

---

## Out-of-Bag (OOB) Error

**Observation**: Each bootstrap sample uses ~63% of data. Remaining ~37% is "out-of-bag".

**OOB Evaluation**:
1. For each sample, find trees that didn't see it during training
2. Make prediction using only those trees
3. Calculate error across all OOB predictions

**Benefit**: Free validation set! No need for separate test set.

**Formula**:
$$OOB\ Error = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i^{OOB})$$

Where $\hat{y}_i^{OOB}$ is prediction from trees that didn't train on sample $i$.

---

## Feature Importance

Measures how useful each feature is for predictions.

### Method 1: Mean Decrease in Impurity
$$Importance(f) = \sum_{t \in trees} \sum_{nodes\ split\ on\ f} \Delta Impurity$$

**Interpretation**: How much each feature reduces impurity when used for splits.

### Method 2: Permutation Importance
1. Calculate OOB error
2. Randomly shuffle feature values  
3. Recalculate OOB error
4. Importance = increase in error

**Interpretation**: How much performance degrades when feature is randomized.

---

## Advantages

1. **High accuracy**: Often achieves state-of-the-art results
2. **Robust to overfitting**: Averaging reduces variance
3. **Handles high-dimensional data**: Feature randomness helps
4. **Feature importance**: Built-in importance scores
5. **Handles mixed data**: Numerical and categorical
6. **Minimal hyperparameter tuning**: Works well with defaults
7. **OOB evaluation**: Free validation set
8. **Parallel training**: Trees can be trained independently

---

## Disadvantages

1. **Less interpretable**: Can't visualize 100 trees easily
2. **Slower prediction**: Must query all trees
3. **Larger model size**: Stores many trees
4. **Not good for extrapolation**: Can't predict beyond training range
5. **Biased toward categorical features**: With many categories

---

## Random Forest vs Single Decision Tree

| Aspect | Decision Tree | Random Forest |
|--------|--------------|---------------|
| **Variance** | High | Low |
| **Bias** | Low (if deep) | Low |
| **Overfitting** | Prone | Resistant |
| **Interpretability** | High | Low |
| **Training time** | Fast | Slower |
| **Prediction time** | Fast | Slower |
| **Accuracy** | Good | Better |

---

## When to Use Random Forests

### Use When:
- Need high accuracy without much tuning
- Have sufficient training data
- Want feature importance analysis
- Okay with black-box model
- Both numerical and categorical features

### Don't Use When:
- Need interpretability (use single tree or linear model)
- Real-time predictions required (very fast)
- Memory constrained (model size matters)
- Extrapolation needed (use linear models)

---

## Classification vs Regression

### Classification
- **Aggregation**: Majority voting
- **max_features**: √n
- **Output**: Most common class across trees

### Regression
- **Aggregation**: Averaging
- **max_features**: n/3
- **Output**: Mean prediction across trees

---

## Practical Tips

### Hyperparameter Tuning:
1. **Start with defaults**: Often work well
2. **Increase n_estimators**: More is usually better
3. **Tune max_features**: Try √n, n/3, log2(n)
4. **Adjust tree depth**: If overfitting, limit depth
5. **Use OOB score**: For quick validation

### Best Practices:
1. **Always use bootstrap=True**: Core of Random Forest
2. **Don't prune individual trees**: Let them overfit, ensemble fixes it
3. **Use multiple cores**: Set n_jobs=-1 for parallel training
4. **Monitor OOB error**: Converges as more trees added
5. **Feature scaling not needed**: Trees don't care about scale

---

## Key Questions

**Q: How many trees should I use?**
A: Start with 100. Add more until OOB error plateaus. Diminishing returns after ~500 trees, but more trees never hurt accuracy (just slower).

**Q: Why feature randomness in addition to bagging?**
A: Without it, all trees might use the same strong features first, creating correlated trees. Feature randomness decorrelates trees, improving ensemble diversity.

**Q: Can Random Forests overfit?**
A: Individual trees overfit, but the ensemble is robust. As you add more trees, training error may not decrease, but test error won't increase either.

---

**Next**: See implementation notebooks for hands-on Random Forest building!
