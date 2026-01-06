# AdaBoost - Theory

## What is AdaBoost?

**Adaptive Boosting** - a meta-learning algorithm that combines many weak learners into a strong classifier.

**Core idea**: Train weak models sequentially, each focusing on mistakes of previous models, then combine via weighted voting.

---

## Algorithm

1. Initialize sample weights: $w_i = \frac{1}{m}$ for all samples
2. For t = 1 to T:
   - Train weak learner $h_t$ on weighted samples
   - Calculate error: $\epsilon_t = \sum_{i:h_t(x_i) \neq y_i} w_i$
   - Calculate classifier weight: $\alpha_t = \frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})$
   - Update sample weights:
     - Wrong predictions: $w_i \leftarrow w_i \cdot e^{\alpha_t}$
     - Correct predictions: $w_i \leftarrow w_i \cdot e^{-\alpha_t}$
   - Normalize weights
3. Final classifier: $H(x) = \text{sign}(\sum_{t=1}^{T}\alpha_t h_t(x))$

---

## Key Concepts

### Weak Learner
**Definition**: Classifier slightly better than random guessing (error < 0.5)

**Common choices**:
- Decision stump (depth-1 tree)
- Shallow decision trees
- Simple rules

### Sample Weights
- **Initially**: All samples equal weight ($1/m$)
- **After each round**: Misclassified samples get higher weight
- **Effect**: Next learner focuses on hard examples

### Classifier Weight (α)
$$\alpha_t = \frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})$$

- **Low error** (ε → 0): α → ∞ (high confidence)
- **High error** (ε → 0.5): α → 0 (no confidence)
- **Error > 0.5**: Negative α (flip predictions)

---

## Why It Works

**Boosting = Reducing Bias + Variance**

1. **Sequential learning**: Each model corrects previous errors
2. **Weighted focus**: Hard examples get more attention
3. **Ensemble diversity**: Models focus on different aspects
4. **Adaptive**: Algorithm adapts to data difficulty

---

## Advantages

1. **Simple**: Easy to understand and implement
2. **No hyperparameters**: Just number of estimators
3. **Fast**: Weak learners are fast
4. **Versatile**: Works with any weak learner
5. **Feature importance**: Can derive from base learners
6. **Theory-backed**: Proven convergence properties

---

## Disadvantages

1. **Sensitive to noise**: Focuses on outliers (can overfit)
2. **Sensitive to weak learners**: Needs error < 0.5
3. **Sequential**: Can't parallelize like Random Forest
4. **Class imbalance**: May struggle with imbalanced data

---

## AdaBoost vs Random Forest

| Aspect | AdaBoost | Random Forest |
|--------|----------|---------------|
| **Training** | Sequential | Parallel |
| **Sample weighting** | Yes | No (bootstrap) |
| **Base learner** | Weak (stumps) | Full trees |
| **Focus** | Hard examples | Diverse samples |
| **Overfitting** | Can overfit noise | Resistant |
| **Speed** | Slower (sequential) | Faster (parallel) |

---

## Practical Tips

1. **Start with stumps**: Decision stumps often work best
2. **Monitor validation**: Stop if validation error increases
3. **Limit estimators**: 50-200 often sufficient
4. **Clean data**: Remove noise/outliers first
5. **Class weights**: Handle imbalance before boosting

---

**Key Point:** "AdaBoost sequentially trains weak learners, increasing weights of misclassified samples so subsequent models focus on hard examples. Final prediction is weighted vote of all weak learners."
