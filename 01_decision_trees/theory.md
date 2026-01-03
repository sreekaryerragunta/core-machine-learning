# Decision Trees - Theory

## What is a Decision Tree?

A **Decision Tree** is a supervised learning algorithm that makes predictions by recursively splitting data based on feature values, creating a tree-like structure of decisions.

**Core idea**: Ask a series of questions about features to partition data into increasingly pure subsets.

---

## How Decision Trees Work

### Tree Structure
- **Root Node**: Starting point with all data
- **Internal Nodes**: Decision points (feature tests)
- **Branches**: Outcomes of tests
- **Leaf Nodes**: Final predictions (class labels or values)

### Example
```
[All Data]
    |
Is Age > 30?
   / \
 Yes  No
  |    |
[A]  Is Income > 50K?
        / \
      Yes  No
       |    |
      [B]  [C]
```

---

## Key Concepts

### 1. Purity Measures

**Goal**: Find splits that create **pure** subsets (all same class).

#### Entropy (Information Theory)
Measures uncertainty/disorder in a dataset.

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

Where:
- $p_i$ = proportion of class $i$
- $c$ = number of classes

**Properties**:
- $H = 0$: Pure (all one class)
- $H = 1$: Maximum impurity (50-50 split for binary)
- Lower entropy = more pure

**Example**: 
- [10 A, 0 B]: $H = 0$ (pure)
- [5 A, 5 B]: $H = 1$ (maximum impurity)
- [8 A, 2 B]: $H = -0.8\log_2(0.8) - 0.2\log_2(0.2) = 0.72$

#### Gini Impurity
Probability of misclassification if randomly labeled.

$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

**Properties**:
- Gini = 0: Pure
- Gini = 0.5: Maximum impurity (binary)
- Computationally faster than entropy

**Comparison**:
- **Entropy**: More sensitive to changes, theoretical foundation
- **Gini**: Faster computation, more balanced splits
- Both work well in practice

---

### 2. Information Gain

Measures how much a split reduces uncertainty.

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $S$ = current dataset
- $A$ = feature being tested
- $S_v$ = subset where feature $A$ has value $v$

**Algorithm**: Choose feature with **highest information gain**.

**Example**:
```
Parent: [6 Yes, 4 No] → H = 0.97

Split on "Weather":
  Sunny: [2 Yes, 3 No] → H = 0.97
  Rainy: [4 Yes, 1 No] → H = 0.72

Weighted avg: (5/10)*0.97 + (5/10)*0.72 = 0.845
Information Gain: 0.97 - 0.845 = 0.125
```

---

## Tree Construction Algorithms

### ID3 (Iterative Dichotomiser 3)
- Uses **entropy** and **information gain**
- Only categorical features
- No pruning
- Can overfit

### C4.5 (Successor to ID3)
- Handles continuous features (uses thresholds)
- Uses **gain ratio** to avoid bias toward features with many values
- Includes pruning
- More robust

### CART (Classification and Regression Trees)
- Binary splits only
- Uses **Gini impurity** for classification
- Uses **MSE** for regression
- Supports pruning
- Most common (used by sklearn)

---

## Building a Decision Tree (CART Algorithm)

### Pseudocode
```
function BuildTree(data, features):
    if stopping_criterion_met:
        return LeafNode(majority_class)
    
    best_feature, best_split = find_best_split(data, features)
    
    if no_improvement:
        return LeafNode(majority_class)
    
    left_data, right_data = split_data(data, best_feature, best_split)
    
    left_subtree = BuildTree(left_data, features)
    right_subtree = BuildTree(right_data, features)
    
    return DecisionNode(best_feature, best_split, left_subtree, right_subtree)
```

### Stopping Criteria
1. All samples in node belong to same class (pure)
2. Maximum depth reached
3. Minimum samples per node reached
4. No feature provides information gain
5. All features have same value

---

## Handling Different Feature Types

### Categorical Features
- Test: "Feature == value?"
- Can split into multiple branches (ID3)
- Or binary: "Feature == X vs Feature != X" (CART)

### Continuous Features
- Test: "Feature <= threshold?"
- Try multiple thresholds
- Choose threshold with best information gain
- Binary split only

**Finding best threshold**:
1. Sort data by feature
2. Try midpoint between consecutive unique values
3. Calculate information gain for each
4. Choose best

---

## Overfitting and Pruning

### Problem: Overfitting
- Tree too deep → memorizes training data
- Poor generalization to new data
- High variance

### Solution 1: Pre-Pruning (Early Stopping)
Stop growing tree before it's fully grown:
- Max depth limit
- Min samples per split
- Min samples per leaf
- Max number of leaf nodes
- Min information gain threshold

**Pros**: Fast, prevents overfitting
**Cons**: May stop too early (underfitting)

### Solution 2: Post-Pruning
1. Build full tree
2. Remove branches that don't improve validation performance
3. Use techniques like:
   - Reduced Error Pruning
   - Cost Complexity Pruning (sklearn's default)

**Cost Complexity Pruning**:
$$R_\alpha(T) = R(T) + \alpha |T|$$

Where:
- $R(T)$ = error rate of tree
- $|T|$ = number of leaf nodes
- $\alpha$ = regularization parameter

Choose $\alpha$ that minimizes cross-validation error.

---

## Advantages

1. **Interpretable**: Easy to visualize and explain
2. **No feature scaling**: Splits based on thresholds, not distances
3. **Handles mixed data**: Both categorical and numerical
4. **Non-linear**: Captures complex patterns
5. **Feature importance**: Can rank feature usefulness
6. **Fast predictions**: O(log n) after training

---

## Disadvantages

1. **Overfitting**: Without pruning, memorizes training data
2. **Instability**: Small data changes → very different trees
3. **Greedy**: Locally optimal splits may miss global optimum  
4. **Bias**: Favors features with more values
5. **Not smooth**: Decision boundaries are axis-parallel rectangles

---

## When to Use Decision Trees

### Use When:
- Need interpretability (explain decisions)
- Mixed feature types (categorical + numerical)
- No time for feature scaling
- Non-linear relationships
- Feature interactions important

### Don't Use When:
- Need smooth decision boundaries
- High-dimensional sparse data (like text)
- Want most accurate model (use ensembles instead)
- Small dataset (high variance)

---

## Regression Trees

Same algorithm but:
- **Split criterion**: Minimize MSE instead of Gini/Entropy
- **Leaf prediction**: Mean of target values in node
- **Evaluation**: MSE, R², MAE

$$MSE(S) = \frac{1}{|S|}\sum_{i \in S}(y_i - \bar{y})^2$$

---

## Feature Importance

Measures how useful each feature is for splits.

$$Importance(f) = \sum_{nodes\ split\ on\ f} (N_{node} \times \Delta Impurity)$$

Where:
- $N_{node}$ = number of samples at node
- $\Delta Impurity$ = reduction in impurity from split

**Interpretation**: Higher = more important for predictions.

---

## Practical Tips

### Hyperparameters to Tune:
- `max_depth`: Deeper = more complex (try 3-10)
- `min_samples_split`: Min samples to split (try 2-50)
- `min_samples_leaf`: Min samples in leaf (try 1-20)
- `max_features`: Features to consider per split
- `criterion`: gini vs entropy (usually similar)

### Best Practices:
1. **Start shallow**: Begin with max_depth=3-5
2. **Cross-validate**: Use k-fold to find best hyperparameters
3. **Visualize**: Plot first few levels to understand
4. **Ensemble**: Use Random Forest instead for better performance
5. **Feature engineering**: Trees don't create new features

---

## Key Questions

**Q: What's the difference between ID3, C4.5, and CART?**
A: ID3 uses entropy (categorical only). C4.5 improves ID3 with continuous features and pruning. CART uses Gini impurity, binary splits, supports regression, and is most widely used.

**Q: How do decision trees handle missing values?**
A: Advanced implementations use surrogate splits or send samples down both branches with weights. Simple approach: impute before training.

**Q: Why are decision trees prone to overfitting?**
A: They can grow arbitrarily deep to perfectly fit training data, memorizing noise. High variance – small data changes cause completely different trees.

---

**Next**: See implementation notebooks for hands-on tree building!
