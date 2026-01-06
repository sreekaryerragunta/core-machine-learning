# Naive Bayes - Theory

## What is Naive Bayes?

A **probabilistic classifier** based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class.

**Core idea**: Calculate probability of each class given the features, pick the most probable class.

---

## Bayes' Theorem

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

- $P(y|X)$: **Posterior** - probability of class $y$ given features $X$
- $P(X|y)$: **Likelihood** - probability of features given class
- $P(y)$: **Prior** - probability of class
- $P(X)$: **Evidence** - probability of features (constant for all classes)

**For classification**: Choose class with highest posterior
$$\hat{y} = \arg\max_y P(y|X) = \arg\max_y P(X|y) \cdot P(y)$$

---

## The "Naive" Assumption

**Assumption**: Features are conditionally independent given the class.

$$P(X|y) = P(x_1, x_2, ..., x_n|y) = P(x_1|y) \cdot P(x_2|y) \cdot ... \cdot P(x_n|y)$$

**Why "naive"?** This assumption is often violated in real data (features are correlated), but Naive Bayes still works surprisingly well!

---

## Types of Naive Bayes

### 1. Gaussian Naive Bayes
For continuous features, assume Gaussian distribution:
$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

### 2. Multinomial Naive Bayes
For discrete count features (e.g., word counts):
$$P(x_i|y) = \frac{count(x_i, y) + \alpha}{count(y) + \alpha n}$$

### 3. Bernoulli Naive Bayes
For binary features:
$$P(x_i|y) = P(i|y)x_i + (1 - P(i|y))(1 - x_i)$$

---

## Training Process

1. **Calculate priors**: $P(y) = \frac{count(y)}{total\ samples}$
2. **Calculate likelihoods**: 
   - Gaussian: Compute $\mu$ and $\sigma$ for each feature per class
   - Multinomial: Count feature occurrences per class
3. **Store parameters**: Keep $\mu, \sigma$ (Gaussian) or counts (Multinomial)

---

## Prediction Process

For each class $y$:
1. Start with prior: $P(y)$
2. Multiply by feature likelihoods: $\prod_i P(x_i|y)$
3. Pick class with highest probability

**Log-space trick** (avoids underflow):
$$\log P(y|X) = \log P(y) + \sum_i \log P(x_i|y)$$

---

## Advantages

1. **Fast**: O(nd) training, O(cd) prediction
2. **Simple**: Easy to implement and understand
3. **Works with small data**: Few parameters to estimate
4. **Handles high dimensions**: Scales well with features
5. **Probabilistic**: Outputs probabilities, not just classes
6. **Online learning**: Can update with new data easily

---

## Disadvantages

1. **Independence assumption**: Rarely true in practice
2. **Zero probability**: If feature never seen in training
3. **Assumes distribution**: Gaussian NB assumes normal distribution
4. **Poor with correlated features**: Violates independence

---

## When to Use

### Use When:
- Need fast baseline
- Text classification (spam, sentiment)
- High-dimensional data
- Small training set
- Need probability estimates
- Features roughly independent

### Don't Use When:
- Features highly correlated
- Need best possible accuracy
- Complex feature interactions important

---

## Practical Tips

1. **Smoothing**: Add Laplace smoothing ($\alpha = 1$) to avoid zero probabilities
2. **Log probabilities**: Always use log-space to prevent underflow
3. **Feature scaling**: Not needed! NB doesn't use distances
4. **Baseline first**: Great starting point before complex models

---

## Key Questions

**Q: Why does it work despite naive assumption?**
A: Even if independence is violated, the ranking of class probabilities often remains correct. We care about relative probabilities, not absolute values.

**Q: What's the zero probability problem?**
A: If a feature value never appears for a class in training, $P(x_i|y) = 0$, making entire posterior zero. Solution: Laplace smoothing.

**Q: When is Gaussian NB appropriate?**
A: When features are continuous and roughly normally distributed. For other distributions, results degrade but often still useful.

---

**Next**: See implementation notebooks for hands-on Naive Bayes!
