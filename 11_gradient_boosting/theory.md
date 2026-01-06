# Gradient Boosting - Theory

## What is Gradient Boosting?

**Gradient Boosting Machine (GBM)** - builds an ensemble of trees sequentially, where each tree corrects errors of previous trees using gradient descent.

**Core idea**: Fit new models to the residuals (errors) of previous models, gradually reducing overall error.

---

## Algorithm

1. Initialize model with constant: $F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$
2. For m = 1 to M:
   - Compute residuals: $r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$
   - Fit tree $h_m(x)$ to residuals $r_{im}$
   - Find optimal step size: $\gamma_m = \arg\min_\gamma \sum L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$
   - Update model: $F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)$
3. Output: $F_M(x)$

---

## Key Concepts

### Residual Fitting
Each new tree predicts the **negative gradient** (residuals) of the loss function.

**For regression (MSE)**:
- Residual = $y - F_{m-1}(x)$ (actual - prediction)
- Very intuitive: fit error directly!

**For classification (log-loss)**:
- Residual = gradient of log-loss
- Less intuitive but same principle

### Learning Rate (ν)
Shrinkage parameter that controls each tree's contribution:
$$F_m = F_{m-1} + \nu \cdot \gamma_m h_m(x)$$

- **Small ν** (0.01-0.1): Slower learning, better generalization, need more trees
- **Large ν** (0.5-1.0): Faster learning, risk overfitting, fewer trees

### Tree Depth
**Shallow trees** (depth 3-8):
- Called "weak learners"
- Fast to train
- Less overfitting
- Standard choice

---

## Gradient Boosting vs AdaBoost

| Aspect | Gradient Boosting | AdaBoost |
|--------|-------------------|----------|
| **Minimizes** | Arbitrary loss function | Exponential loss |
| **Trains on** | Residuals/gradients | Reweighted samples |
| **Update** | Learning rate × tree | Alpha × tree |
| **Flexibility** | Any differentiable loss | Limited |
| **Trees** | Usually deeper (3-8) | Stumps (depth 1) |

---

## Hyperparameters

### 1. n_estimators
**Number of boosting rounds**
- More trees → better fit
- Too many → overfitting
- Typical: 100-1000
- Use early stopping!

### 2. learning_rate
**Shrinkage parameter**
- Lower → need more trees, better generalization
- Higher → faster, risk overfit
- Typical: 0.01-0.3
- **Rule**: n_estimators × learning_rate ≈ constant

### 3. max_depth
**Tree complexity**
- Typical: 3-8
- Deeper → more complex interactions
- Too deep → overfitting

### 4. subsample
**Fraction of samples per tree**
- < 1.0: Stochastic gradient boosting
- Adds randomness → reduces overfitting
- Typical: 0.5-1.0

---

## Regularization Techniques

1. **Shrinkage (learning_rate)**: Standard approach
2. **Subsampling**: Train each tree on random subset
3. **Tree constraints**: max_depth, min_samples_split
4. **Early stopping**: Stop when validation error increases

---

## XGBoost Enhancements

XGBoost (Extreme Gradient Boosting) adds:

1. **Regularized objective**: L1/L2 on leaf weights
2. **Second-order derivatives**: Use Hessian for better approximation
3. **Sparsity awareness**: Handles missing values natively
4. **Parallelization**: Column-wise parallel tree building
5. **Cache optimization**: Better hardware utilization
6. **Built-in cross-validation**: Automatic tuning

---

## Advantages

1. **State-of-art performance**: Often wins Kaggle
2. **Handles complex patterns**: Non-linear relationships
3. **Feature interactions**: Automatically captures
4. **Flexible loss functions**: Any differentiable loss
5. **Missing values**: Can handle (XGBoost)
6. **Feature importance**: Built-in

---

## Disadvantages

1. **Slow training**: Sequential, can't parallelize boosting
2. **Sensitive to noise**: Can overfit noisy data
3. **Many hyperparameters**: Needs careful tuning
4. **Less interpretable**: Complex ensemble
5. **Memory intensive**: Stores many trees

---

## Practical Tips

1. **Start with**: 100 trees, lr=0.1, depth=3
2. **Tune in order**: n_estimators → learning_rate → max_depth → subsample
3. **Use early stopping**: Monitor validation set
4. **Grid search**: Final tuning
5. **Feature engineering**: Still important!

---

**Key Point:** "Gradient Boosting fits sequential trees to residuals using gradient descent. Each tree corrects previous errors. Small learning rate + many trees = best generalization."
