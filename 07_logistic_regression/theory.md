# Logistic Regression - Theory

## What is Logistic Regression?

A **probabilistic classifier** that models the probability of a binary outcome using the logistic (sigmoid) function. Despite its name, it's a **classification** algorithm, not regression.

**Core idea**: Map linear combination of features to probability using sigmoid function, then threshold at 0.5 for classification.

---

## The Sigmoid Function

transforms any real-valued number into a probability (0 to 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
- **Range**: $(0, 1)$ - perfect for probabilities
- **S-shaped curve**: Smooth transition from 0 to 1
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ - elegant for gradient descent
- **At z=0**: $\sigma(0) = 0.5$ - decision boundary

---

## Binary Logistic Regression

### Model

$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

- $w$: Weight vector (learned parameters)
- $b$: Bias term
- $x$: Feature vector
- $P(y=1|x)$: Probability of positive class

**Decision rule**:
$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### Cost Function: Binary Cross-Entropy (Log Loss)

Cannot use MSE (non-convex for sigmoid). Use **maximum likelihood** instead:

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h(x^{(i)})) + (1-y^{(i)})\log(1-h(x^{(i)}))]$$

Where $h(x) = \sigma(w^Tx + b)$ is the predicted probability.

**Intuition**:
- If $y=1$: Cost = $-\log(h(x))$ → penalize low probabilities for positive examples
- If $y=0$: Cost = $-\log(1-h(x))$ → penalize high probabilities for negative examples

**Properties**:
- **Convex**: Guaranteed global minimum
- **Probabilistic**: Derived from maximum likelihood
- **Smooth**: Differentiable everywhere

### Gradient Descent

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})x_j^{(i)}$$

**Update rule**:
$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

**Note**: Same form as linear regression, but $h(x) = \sigma(w^Tx)$ instead of $w^Tx$!

---

## Multiclass Logistic Regression (Softmax)

Extends binary to $K$ classes using **softmax function**:

$$P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^{K}e^{w_j^Tx}}$$

**Properties**:
- Outputs sum to 1
- Each class has own weight vector $w_k$
- Generalizes sigmoid (K=2 reduces to binary)

**Cost**: Categorical cross-entropy
$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}1\{y^{(i)}=k\}\log(P(y=k|x^{(i)}))$$

---

## Regularization

Prevent overfitting by adding penalty term to cost function.

### L2 Regularization (Ridge)

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}[\cdots] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

- **Effect**: Shrinks all weights toward zero
- **Keeps all features**: No feature selection
- **λ**: Regularization strength (higher → more regularization)

### L1 Regularization (Lasso)

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}[\cdots] + \frac{\lambda}{m}\sum_{j=1}^{n}|w_j|$$

- **Effect**: Drives some weights to exactly zero
- **Feature selection**: Sparse solutions
- **Non-differentiable** at zero (requires special optimization)

### Elastic Net

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}[\cdots] + \lambda[(1-\alpha)\frac{1}{2}\sum w_j^2 + \alpha\sum|w_j|]$$

- **Combines L1 + L2**: Best of both worlds
- **α**: Mix ratio (0=Ridge, 1=Lasso)

---

## Optimization Methods

### 1. Gradient Descent
- **Batch**: Use all data per iteration
- **Stochastic (SGD)**: Use one sample per iteration
- **Mini-batch**: Use subset per iteration
- **Convergence**: Slow but guaranteed

### 2. Newton's Method
- **Uses Hessian**: Second-order information
- **Faster convergence**: Quadratic near optimum
- **Memory intensive**: O(n²) for Hessian

### 3. L-BFGS
- **Quasi-Newton**: Approximates Hessian
- **Best for small-medium datasets**: Fast convergence
- **sklearn default**: For logistic regression

### 4. Coordinate Descent
- **For L1**: Handles non-differentiability
- **Updates one weight at a time**: Simple updates

---

## Probability Calibration

Raw predicted probabilities may not reflect true probabilities.

**Calibration**: Adjust probabilities to match empirical frequencies

**Methods**:
1. **Platt Scaling**: Fit sigmoid on validation set
2. **Isotonic Regression**: Non-parametric, monotonic

**When to calibrate**:
- Need accurate probability estimates
- Imbalanced classes
- Threshold tuning

---

## Model Evaluation

### Classification Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$ - Of predicted positives, how many correct?
- **Recall**: $\frac{TP}{TP + FN}$ - Of actual positives, how many found?
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ - Harmonic mean

### Probabilistic Metrics
- **Log Loss**: Penalizes confident wrong predictions heavily
- **Brier Score**: $(p - y)^2$ - MSE of probabilities
- **AUC-ROC**: Area under ROC curve - threshold-independent

### Threshold Tuning
Default 0.5 may not be optimal!
- **Imbalanced classes**: Adjust threshold based on costs
- **Precision-recall tradeoff**: Higher threshold → higher precision, lower recall

---

## Logistic vs Linear Regression

| Aspect | Logistic | Linear |
|--------|----------|--------|
| **Output** | Probability (0-1) | Continuous value |
| **Hypothesis** | $\sigma(w^Tx)$ | $w^Tx$ |
| **Cost** | Cross-entropy | MSE |
| **Use case** | Classification | Regression |
| **Decision boundary** | Non-linear (in input space) | Linear |

---

## Assumptions

1. **Binary outcome** (or can be extended to multiclass)
2. **Independence**: Observations are independent
3. **No multicollinearity**: Features not highly correlated
4. **Linear relationship**: Between log-odds and features
5. **Large sample size**: More data → better estimates

**Violations**:
- Multicollinearity: Use regularization
- Non-linearity: Add polynomial features or use tree models
- Small sample: Use regularization, simpler model

---

## Advantages

1. **Probabilistic output**: Not just classes, but confidence
2. **Interpretable**: Coefficients show feature importance
3. **Fast**: O(nd) training, O(d) prediction
4. **No hyperparameters**: (except regularization λ)
5. **Extensions**: Easily extends to multiclass
6. **Regularization**: Built-in overfitting prevention
7. **Online learning**: Can update with new data

---

## Disadvantages

1. **Linear decision boundary**: Struggles with complex patterns
2. **Feature engineering**: May need polynomial/interaction terms
3. **Outlier sensitive**: Can affect coefficients
4. **Assumes independence**: Violated by correlated features
5. **Not great for small datasets**: Needs sufficient samples per class

---

## When to Use

### Use When:
- **Binary or multiclass** classification
- **Interpretability important**: Need to explain predictions
- **Probabilistic predictions**: Need confidence scores
- **Baseline model**: Quick first attempt
- **Linear separability**: Classes roughly separable by hyperplane
- **Real-time predictions**: Fast inference needed

### Don't Use When:
- **Highly non-linear**: Complex decision boundaries
- **Many features, few samples**: Prone to overfitting
- **Image/text raw data**: Use deep learning
- **Need best accuracy**: Try ensemble methods

---

## Practical Tips

### Feature Preprocessing:
1. **Scale features**: Use StandardScaler (critical!)
2. **Handle missing**: Impute or drop
3. **Encode categorical**: One-hot or target encoding
4. **Create interactions**: If suspected non-linearity

### Regularization:
1. **Always use regularization**: Prevents overfitting
2. **Start with L2**: λ = 0.01, 0.1, 1, 10, 100
3. **Try L1 for feature selection**: If many irrelevant features
4. **Cross-validate**: To find optimal λ

### Imbalanced Classes:
1. **Class weights**: Penalize minority class errors more
2. **SMOTE**: Synthetic oversampling
3. **Threshold tuning**: Adjust based on cost matrix
4. **Stratified sampling**: Preserve class distribution

### Debugging:
1. **Check probabilities**: Should be between 0-1
2. **Plot decision boundary**: Visualize 2D case
3. **Feature importance**: Look at coefficient magnitudes
4. **Learning curves**: Detect over/underfitting

---

## Key Questions

**Q: Why use sigmoid instead of just thresholding linear output?**
A: Sigmoid provides probabilistic interpretation and makes the optimization problem convex. Raw linear output doesn't represent uncertainty.

**Q: Why is cross-entropy better than MSE for classification?**
A: Cross-entropy is derived from maximum likelihood and is convex for sigmoid. MSE with sigmoid is non-convex with many local minima.

**Q: How do coefficients relate to odds?**
A: $e^{w_j}$ is the **odds ratio** - how much odds of success multiply when $x_j$ increases by 1 (holding others constant).

**Q: What if classes aren't linearly separable?**
A: Add polynomial features, use kernel trick (not common for logistic regression), or switch to non-linear model (SVM with RBF, trees).

**Q: When to use One-vs-Rest vs Softmax for multiclass?**
A: Softmax if classes mutually exclusive. One-vs-Rest if not (multi-label problem) or if you want probabilistic calibration per class.

---

**Next**: See implementation notebooks for hands-on Logistic Regression mastery!
