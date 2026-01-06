# K-Nearest Neighbors - Theory

## What is K-NN?

A **non-parametric**, **instance-based** learning algorithm that classifies based on similarity to training examples.

**Core idea**: Classify a point based on the K nearest points in feature space.

---

## Algorithm

1. **Store** all training data
2. For each test point:
   - Find K nearest training points
   - Classification: Majority vote
   - Regression: Average of K neighbors

---

## Distance Metrics

**Euclidean** (most common):
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Manhattan**:
$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

---

##Key Hyperparameter: K

- **Small K**: Complex boundary, overfitting
- **Large K**: Smooth boundary, underfitting
- **Rule of thumb**: $K = \sqrt{n}$

---

## Advantages

1. **Simple**: Easy to understand
2. **No training**: Just stores data
3. **Naturally handles multi-class**

---

## Disadvantages

1. **Slow prediction**: O(nd) for each prediction
2. **Memory intensive**: Stores all data
3. **Curse of dimensionality**: Poor in high dimensions
4. **Feature scaling required**: Sensitive to scales

---

**Key Point:** \"K-NN classifies by majority vote of K nearest neighbors. It's simple and effective but slow for large datasets and sensitive to feature scaling and irrelevant features.\"
