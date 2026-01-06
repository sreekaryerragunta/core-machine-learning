# Support Vector Machines (SVM) - Theory

## What is a Support Vector Machine?

A **Support Vector Machine** is a powerful classifier that finds the optimal decision boundary (hyperplane) that maximizes the margin between classes.

**Core idea**: Find the hyperplane that has the maximum distance to the nearest data points from any class (maximum margin).

---

## Key Concepts

### 1. Hyperplane

In n-dimensional space, a hyperplane is a flat subspace of dimension n-1.

**2D**: Line (1D)  
**3D**: Plane (2D)  
**nD**: Hyperplane (n-1D)

**Equation**: $w^T x + b = 0$
- $w$: Normal vector (perpendicular to hyperplane)
- $b$: Bias term (position)
- $x$: Data point

### 2. Margin

The **margin** is the distance between the hyperplane and the nearest data point from either class.

**Goal**: Maximize this margin!

**Hard Margin**: All points correctly classified, no violations  
**Soft Margin**: Allow some misclassifications (more practical)

### 3. Support Vectors

The data points closest to the hyperplane are called **support vectors**.

**Key insight**: Only these points matter! The decision boundary is determined entirely by support vectors, not all training data.

---

## Linear SVM

### Optimization Problem

**Primal Form**:
$$\min_{w,b} \frac{1}{2}||w||^2$$
$$\text{subject to: } y_i(w^T x_i + b) \geq 1, \forall i$$

**Interpretation**:
- Minimize $||w||^2$ → Maximize margin ($\frac{2}{||w||})$
- Constraint: All points correctly classified with margin ≥ 1

### Soft Margin SVM

Real data isn't perfectly separable. Allow violations with slack variables $\xi_i$:

$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{m}\xi_i$$
$$\text{subject to: } y_i(w^T x_i + b) \geq 1 - \xi_i, \xi_i \geq 0$$

**C parameter**: Trade-off between margin size and misclassifications
- **Large C**: Small margin, few violations (high variance)
- **Small C**: Large margin, more violations (high bias)

---

## Kernel Trick

For non-linearly separable data, map to higher-dimensional space where it becomes separable.

**Problem**: Explicit mapping is expensive!

**Solution**: Kernel trick - compute dot products in high-dimensional space without explicit mapping.

### Common Kernels

**1. Linear Kernel**:
$$K(x_i, x_j) = x_i^T x_j$$

**2. Polynomial Kernel**:
$$K(x_i, x_j) = (x_i^T x_j + c)^d$$

**3. RBF (Gaussian) Kernel**:
$$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$

**4. Sigmoid Kernel**:
$$K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$$

### Why Kernels Work

Instead of:
1. Map $x → \phi(x)$ (expensive)
2. Compute $\phi(x_i)^T \phi(x_j)$ (expensive)

Do:
- Compute $K(x_i, x_j)$ directly (cheap!)

**Example**: RBF kernel implicitly maps to infinite-dimensional space!

---

## SVM vs Other Classifiers

| Aspect | SVM | Logistic Regression | Decision Tree |
|--------|-----|---------------------|---------------|
| **Decision boundary** | Maximum margin | Probabilistic | Axis-parallel |
| **Non-linear** | Yes (kernels) | No (need feature eng) | Yes (naturally) |
| **Outlier sensitivity** | Medium (soft margin) | High | Low |
| **Interpretability** | Low | Medium | High |
| **Training time** | O(n²) to O(n³) | O(n) | O(n log n) |
| **Works well for** | High-dim, clear margins | Probabilities needed | Interpretability |

---

## Advantages

1. **Effective in high dimensions**: Works well when features >> samples
2. **Memory efficient**: Only stores support vectors
3. **Versatile**: Different kernels for different data
4. **Works with small datasets**: No need for massive data
5. **Good generalization**: Maximum margin principle

---

## Disadvantages

1. **Slow training**: O(n²) or worse for large datasets
2. **Memory intensive**: Kernel matrix can be huge
3. **Hyperparameter tuning**: C, γ require careful selection
4. **Not probabilistic**: Doesn't naturally output probabilities
5. **Kernel choice**: Selecting right kernel is non-trivial

---

## Hyperparameters

### 1. C (Regularization)
- Controls trade-off between margin and violations
- **Small C**: Wide margin, more violations (regularization)
- **Large C**: Narrow margin, fewer violations (fit data closely)
- Typical: 0.1, 1, 10, 100

### 2. γ (Gamma) for RBF
- Defines kernel width
- **Small γ**: Wide kernel, smooth decision boundary
- **Large γ**: Narrow kernel,  complex boundary (overfit risk)
- Typical: 0.001, 0.01, 0.1, 1

### 3. Kernel Type
- **Linear**: Data linearly separable
- **RBF**: General purpose, try first for non-linear
- **Polynomial**: Specific polynomial relationships
- **Custom**: Domain-specific kernels

---

## When to Use SVM

### Use When:
- Clear margin of separation exists
- High-dimensional data (text, images)
- Small to medium dataset
- Need robust classifier
- Non-linear relationships (with kernels)

### Don't Use When:
- Very large datasets (slow training)
- Need probability estimates
- Many noisy/overlapping classes
- Interpretability is critical

---

## Practical Tips

### Training:
1. **Scale features**: SVM is sensitive to feature scales
2. **Start with RBF**: Good default kernel
3. **Grid search C and γ**: Try exponential range
4. **Use validation set**: Prevent overfitting

### Hyperparameter Tuning:
```
C: [0.1, 1, 10, 100, 1000]
γ: [0.001, 0.01, 0.1, 1]
```

### Best Practices:
1. **Always standardize**: Use StandardScaler
2. **Try linear first**: If it works, it's faster
3. **Cross-validate**: Essential for C and γ selection
4. **Monitor support vectors**: Too many → need different C/γ

---

## Key Questions

**Q: Why maximize margin?**
A: Maximum margin provides best generalization. It's the most confident boundary - farthest from both classes. Intuitively, it's least likely to misclassify new data.

**Q: What are support vectors?**
A: Data points closest to the decision boundary. They define the boundary - if you removed all other points, you'd get the same SVM!

**Q: When to use kernel vs linear SVM?**
A: Try linear first (faster). If accuracy is insufficient and you suspect non-linear relationships, try RBF kernel. Use polynomial if you know specific degree relationship.

**Q: How does C affect the model?**
A: C controls regularization. Low C → prioritize large margin (more regularization). High C → prioritize fitting data (less regularization, risk overfitting).

---

**Next**: See implementation notebooks for hands-on SVM building!
