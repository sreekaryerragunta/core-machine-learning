# Ridge and Lasso Regression - Theory

## Motivation: Why Regularization?

**Problem with ordinary least squares (OLS)**:
- **Overfitting**: Complex models fit noise in training data
- **Multicollinearity**: Correlated features → unstable coefficients
- **High variance**: Small changes in data → large changes in coefficients

**Solution**: Add penalty term to cost function that discourages large coefficients.

---

## Ridge Regression (L2 Regularization)

### Model

Minimize:
$$J(w) = \frac{1}{2m}\sum_{i=1}^{m}(w^Tx^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

**Components**:
- First term: Mean Squared Error (fit to data)
- Second term: L2 penalty (complexity penalty)
- $\lambda$: Regularization strength (hyperparameter)

### Closed-Form Solution

Ridge has analytical solution (unlike Lasso):

$$w = (X^TX + \lambda I)^{-1}X^Ty$$

**Key insight**: Adding $\lambda I$ ensures $X^TX + \lambda I$ is invertible even when $X^TX$ is singular!

### Properties

1. **Never eliminates features**: All coefficients shrink but stay non-zero
2. **Handles multicollinearity**: Adds bias, reduces variance
3. **Smooth shrinkage**: Coefficients decrease smoothly as $\lambda$ increases
4. **Works with p > n**: Can fit models with more features than samples
5. **Computationally efficient**: Closed-form solution

### Effect of λ

- **λ = 0**: Ordinary least squares (no regularization)
- **λ → ∞**: All coefficients → 0 (only intercept remains)
- **Optimal λ**: Balance between bias and variance (find via cross-validation)

---

## Lasso Regression (L1 Regularization)

### Model

Minimize:
$$J(w) = \frac{1}{2m}\sum_{i=1}^{m}(w^Tx^{(i)} - y^{(i)})^2 + \frac{\lambda}{m}\sum_{j=1}^{n}|w_j|$$

**Key difference**: Absolute value penalty instead of squared

### No Closed-Form Solution

L1 penalty is non-differentiable at zero → need iterative methods:
- **Coordinate descent**: Update one coefficient at a time
- **LARS**: Efficient path algorithm
- **Proximal gradient**: Soft-thresholding

### Soft-Thresholding Operator

Core of Lasso optimization:

$$\text{soft}(x, \lambda) = \begin{cases}
x - \lambda & \text{if } x > \lambda \\
0 & \text{if } |x| \leq \lambda \\
x + \lambda & \text{if } x < -\lambda
\end{cases}$$

**Effect**: Coefficients below threshold set to exactly zero!

### Properties

1. **Feature selection**: Sets some coefficients to exactly zero
2. **Sparse solutions**: Selects subset of features
3. **Interpretable**: Fewer features → easier to understand
4. **Unstable with correlations**: Among correlated features, picks arbitrarily
5. **Computational**: Slower than Ridge (iterative)

---

## Ridge vs Lasso vs OLS

| Aspect | OLS | Ridge (L2) | Lasso (L1) |
|--------|-----|------------|------------|
| **Penalty** | None | $\sum w_j^2$ | $\sum |w_j|$ |
| **Solution** | $(X^TX)^{-1}X^Ty$ | Closed-form | Iterative |
| **Feature selection** | No | No | Yes |
| **Coefficients** | Unstable | Shrunk | Sparse |
| **Multicollinearity** | Fails | Handles | Picks one |
| **Geometric shape** | - | Circle (L2 ball) | Diamond (L1 ball) |
| **Best when** | n >> p, uncorrelated | Multicollinearity | Many irrelevant features |

---

## Geometric Interpretation

**Why L1 produces sparsity but L2 doesn't?**

**Ridge (L2)**: Contours are circles
- Minimum likely hits circle at non-axis point
- All coefficients shrink but rarely reach zero

**Lasso (L1)**: Contours are diamonds
- Sharp corners at axes
- Minimum often hits corner → coefficient = 0
- Natural feature selection!

---

## Elastic Net

Combines L1 and L2:

$$J(w) = \text{MSE} + \lambda[(1-\alpha)\frac{1}{2}\sum w_j^2 + \alpha\sum|w_j|]$$

**Parameters**:
- $\lambda$: Overall regularization strength  
- $\alpha$: Mix ratio (0=Ridge, 1=Lasso)

**Benefits**:
- Feature selection (from L1)
- Stability with correlations (from L2)
- Best of both worlds!

**When to use**: Many correlated features + want feature selection

---

## Bias-Variance Tradeoff

**OLS**: Low bias, high variance → overfits
**Ridge/Lasso**: Higher bias, lower variance → generalizes better

**Optimal λ**: Minimizes **test error** (not training error!)

$$\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Small λ**: Low bias, high variance (overfitting)
- **Large λ**: High bias, low variance (underfitting) 
- **Optimal λ**: Balance point

---

## Standardization: Critical!

**Why?** Regularization penalizes large coefficients, but scale affects coefficient magnitude.

**Example**:
- Feature A: range [0, 1] → small coefficient
- Feature B: range [0, 1000] → tiny coefficient
- Without scaling: Feature B gets over-penalized!

**Solution**: Standardize features before regularization
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

**Note**: Don't regularize intercept (it just shifts predictions)

---

## Hyperparameter Tuning

### Grid Search + Cross-Validation

```python
λ_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
For each λ:
    For each fold:
        Train on training folds
        Validate on validation fold
    Average validation scores
Select λ with best average score
```

### Regularization Path

Plot coefficients vs λ:
- See which features disappear first (Lasso)
- Understand coefficient stability
- Visualize shrinkage

---

## Practical Applications

### Ridge:
- **Multicollinearity**: Highly correlated predictors
- **Stability**: Sensitive to small data changes
- **All features relevant**: Keep all, just shrink

### Lasso:
- **Feature selection**: Many irrelevant features
- **Interpretability**: Want sparse model
- **High-dimensional**: p >> n scenarios

### Elastic Net:
- **Genomics**: Thousands of correlated genes
- **Groups of features**: Related features together
- **Best performance**: When unsure between Ridge/Lasso

---

## Degrees of Freedom

**OLS**: df = p (number of features)
**Ridge**: df decreases with λ (effective parameters < p)
**Lasso**: df ≈ number of non-zero coefficients

**Interpretation**: Regularization reduces model complexity even if all features present (Ridge)

---

## Mathematical Insights

### Ridge = MAP with Gaussian Prior

Ridge is equivalent to maximum a posteriori (MAP) estimation with Gaussian prior:
$$w_j \sim N(0, \sigma^2)$$

**Bayesian view**: Regularization = prior belief that weights are small

### Lasso = MAP with Laplace Prior

Lasso uses Laplace prior:
$$w_j \sim \text{Laplace}(0, b)$$

**Heavy tails**: Allows few large weights, many exactly zero

---

## Advantages

**Ridge**:
1. Handles multicollinearity
2. Stable coefficient estimates
3. Closed-form solution (fast)
4. Works when p > n
5. Reduces variance

**Lasso**:
1. Automatic feature selection
2. Interpretable models
3. Handles high dimensions
4. Sparse solutions
5. Identifies relevant features

---

## Disadvantages

**Ridge**:
1. No feature selection
2. All features in model
3. Less interpretable (all features matter)

**Lasso**:
1. Slow(er) - iterative methods
2. Unstable with correlated features
3. Selects at most n features when p > n
4. Arbitrary selection among correlations

---

## When to Use

### Ridge:
- All features potentially relevant
- Features correlated
- Prediction more important than interpretation
- Multicollinearity issues

### Lasso:
- Many irrelevant features
- Want interpretability
- Feature selection goal
- High-dimensional data

### Elastic Net:
- Unsure between Ridge/Lasso
- Groups of correlated features
- Want both stability and selection

---

## Key Questions

**Q: Why does Ridge never set coefficients to exactly zero?**
A: L2 penalty is differentiable everywhere. Gradient pushes toward zero but never forces exact zero. L1 has non-differentiability at zero, creating "kink" that can trap coefficients at zero.

**Q: How to choose between Ridge and Lasso?**
A: If you believe most features are relevant → Ridge. If many irrelevant → Lasso. Unsure → Try both and compare via CV. Want both → Elastic Net.

**Q: Can regularization hurt performance?**
A: If optimal λ is very small (≈0), yes. But cross-validation finds this. Usually, some regularization helps generalization.

**Q: What if I don't know which features are relevant?**
A: Use Lasso for automatic selection, or Elastic Net for robust selection with correlated features.

---

**Next**: See implementation for coefficient paths, cross-validation, and feature selection!
