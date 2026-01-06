# Core Machine Learning - Supervised Learning From Scratch

A comprehensive collection of **supervised learning algorithms** implemented from scratch in Python with NumPy, demonstrating deep understanding of classification, regression, and ensemble methods.

**Repository Focus**: Building expertise in supervised learning through clear implementations, mathematical foundations, and production-quality code.

---

## Repository Structure

```
core-machine-learning/
├── 01_decision_trees/           # CART Algorithm
├── 02_random_forests/           # Bagging Ensemble
├── 03_svm/                      # Support Vector Machines
├── 04_naive_bayes/              # Probabilistic Classification
├── 05_kmeans/                   # K-Means Clustering (moved to unsupervised-learning)
├── 06_knn/                      # K-Nearest Neighbors
├── 07_logistic_regression/      # Binary & Multiclass Classification
├── 08_ridge_lasso/              # L1/L2 Regularization
├── 09_adaboost/                 # Adaptive Boosting
├── 10_elastic_net/              # Combined L1+L2
├── 11_gradient_boosting/        # Gradient-Based Boosting
├── requirements.txt
└── README.md
```

---

## Algorithms Implemented (12 Total)

### Classification Algorithms (7)

**1. Decision Trees**
- CART algorithm (Classification and Regression Trees)
- Gini impurity & entropy
- Tree pruning strategies
- **Files**: `theory.md`, `decision_tree_from_scratch.ipynb`, `pruning_and_overfitting.ipynb`, `utils.py`

**2. Random Forests**
- Bootstrap aggregating (bagging)
- Feature randomness
- Out-of-bag error estimation
- Feature importance analysis
- **Files**: `theory.md`, `random_forest_from_scratch.ipynb`, `feature_importance.ipynb`, `utils.py`

**3. Support Vector Machines (SVM)**
- Linear, Polynomial, and RBF kernels
- Maximum margin hyperplanes
- C parameter regularization
- Kernel trick for non-linear boundaries
- **Files**: `theory.md`, `svm_from_scratch.ipynb`, `kernel_tricks.ipynb`, `utils.py`

**4. Naive Bayes**
- Gaussian Naive Bayes
- Bayes' theorem application
- Probabilistic predictions
- Log-space calculations for stability
- **Files**: `theory.md`, `naive_bayes_from_scratch.ipynb`

**5. K-Nearest Neighbors (KNN)**
- Distance-based classification
- Euclidean distance metric
- K parameter tuning
- Decision boundary visualization
- **Files**: `theory.md`, `knn_from_scratch.ipynb`

**6. Logistic Regression**
- Binary classification with sigmoid
- Gradient descent optimization
- L1/L2 regularization
- Cross-entropy loss
- ROC/AUC analysis
- **Files**: `theory.md`, `logistic_regression_binary.ipynb` (455KB with outputs)

**7. AdaBoost**
- Adaptive boosting with decision stumps
- Sample weight updates
- Weak learner combination
- Sequential ensemble building
- **Files**: `theory.md`, `adaboost_implementation.ipynb`

**8. Gradient Boosting**
- Gradient-based boosting
- Residual fitting
- Learning rate & tree depth tuning
- XGBoost concepts
- **Files**: `theory.md`, `gradient_boosting_demo.ipynb` (134KB with outputs)

---

### Regression Algorithms (4)

**9. Ridge Regression**
- L2 regularization (squared penalty)
- Closed-form solution
- Handles multicollinearity
- Coefficient shrinkage
- **Files**: theory in `08_ridge_lasso/theory.md`, implementation in `ridge_lasso_implementation.ipynb`

**10. Lasso Regression**
- L1 regularization (absolute penalty)
- Coordinate descent optimization
- Automatic feature selection
- Sparse solutions
- **Files**: theory in `08_ridge_lasso/theory.md`, implementation in `ridge_lasso_implementation.ipynb` (182KB with outputs)

**11. Elastic Net**
- Combined L1 + L2 regularization
- l1_ratio parameter for mix control
- Best of Ridge and Lasso
- Soft-thresholding operator
- **Files**: `elastic_net_implementation.ipynb` (90KB with outputs)

---

## What's Included

Each algorithm component contains:

### Theory (`theory.md`)
- **Mathematical foundations**: Equations, derivations, loss functions
- **Algorithm intuition**: Why it works, geometric interpretation
- **Hyperparameters**: What they control, how to tune
- **Advantages & disadvantages**: When to use, when not to
- **Comparison with alternatives**: vs other algorithms
- **Key questions**: Interview-style deep understanding checks

### Implementation (`*_from_scratch.ipynb`)
- **From-scratch NumPy**: No black-box sklearn for core logic
- **Step-by-step**: Clear progression from theory to code
- **Comments**: Every major step explained
- **sklearn comparison**: Validate correctness
- **Performance matching**: Our implementations match sklearn

### Demonstrations
- **Visual explanations**: Decision boundaries, convergence curves
- **Real-world datasets**: Iris, breast cancer, etc.
- **Hyperparameter effects**: Learning rate, regularization, tree depth
- **Practical insights**: What actually matters in practice

### Utilities (`utils.py`)
- Reusable helper functions
- Distance/similarity metrics
- Cost/loss functions
- Evaluation metrics

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/sreekaryerragunta/core-machine-learning.git
cd core-machine-learning

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Run an Algorithm

```python
# Example: Logistic Regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('07_logistic_regression')

# Your from-scratch implementation
# (See notebooks for full code)
```

---

## Algorithm Comparison

### Classification Algorithms

| Algorithm | Speed | Interpretability | Non-Linear | Probability Output | Best For |
|-----------|-------|------------------|------------|-------------------|----------|
| **Logistic Regression** | Fast | High | No | Yes | Baseline, interpretability |
| **Naive Bayes** | Very Fast | High | No | Yes | Text classification, small data |
| **KNN** | Slow | Medium | Yes | Yes | Small datasets, simple problems |
| **Decision Trees** | Fast | High | Yes | Yes | Interpretability, feature interactions |
| **Random Forests** | Medium | Medium | Yes | Yes | General purpose, robust |
| **SVM** | Medium | Low | Yes (kernels) | No | High-dimensional, clear margins |
| **AdaBoost** | Medium | Low | Yes | No | Weak learners, boosting intro |
| **Gradient Boosting** | Slow | Low | Yes | Yes | Best performance, kaggle |

### Regression Algorithms

| Algorithm | Regularization | Feature Selection | Speed | Best For |
|-----------|----------------|-------------------|-------|----------|
| **Ridge** | L2 (squared) | No | Fast (closed-form) | Multicollinearity, keep all features |
| **Lasso** | L1 (absolute) | Yes | Medium (iterative) | Sparse models, feature selection |
| **Elastic Net** | L1 + L2 | Yes | Medium | Correlated features + selection |

---

## Learning Path

### Beginner (Start Here)
1. **Logistic Regression** - Classification foundation
2. **Decision Trees** - Intuitive tree-based learning
3. **KNN** - Simple distance-based method

### Intermediate
4. **Ridge/Lasso** - Regularization techniques
5. **Random Forests** - Ensemble introduction
6. **SVM** - Kernel methods

### Advanced
7. **AdaBoost** - Boosting concepts
8. **Gradient Boosting** - State-of-art ensemble
9. **Elastic Net** - Advanced regularization

---

## Tech Stack

- **Python 3.8+**
- **NumPy**: Core matrix operations
- **Matplotlib**: Visualizations
- **Seaborn**: Statistical plots
- **scikit-learn**: Validation & datasets
- **Jupyter**: Interactive development

---

## Design Philosophy

### 1. **Theory First, Code Second**
- Understand math before implementing
- Code structure mirrors theoretical concepts
- Comments reference theory sections

### 2. **From Scratch > Black Box**
- Implement core algorithms with NumPy
- Use sklearn only for validation
- Understand every line of code

### 3. **Production Quality**
- Match sklearn performance
- Handle edge cases
- Numerical stability (log-space, clipping)
- Well-tested on multiple datasets

### 4. **Visual Learning**
- Decision boundaries for classification
- Convergence curves for optimization
- Regularization paths for penalties
- Feature importance visualizations

---

## Repository Goals

1. **Deep Understanding**: Not just "what" but "why" and "how"
2. **Interview Preparation**: Explain algorithms confidently
3. **Practical Intuition**: When to use which algorithm
4. **Code Quality**: Production-ready implementations
5. **Portfolio Piece**: Demonstrates ML expertise

---

## Key Metrics

- **12 Algorithms**: Classification (7) + Regression (4)
- **~2MB**: Total notebook outputs with visualizations
- **100% sklearn validation**: All implementations verified
- **Expert-level theory**: Comprehensive mathematical coverage

---

## Related Repositories

- **[unsupervised-learning](https://github.com/sreekaryerragunta/unsupervised-learning)**: Clustering & dimensionality reduction (DBSCAN, t-SNE, UMAP, etc.)
- **[ml-math-from-scratch](https://github.com/sreekaryerragunta/ml-math-from-scratch)**: Mathematical foundations (Linear Regression, Logistic Regression, PCA, etc.)

---

## Resources & References

### Books
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" - Bishop
- "Machine Learning: A Probabilistic Perspective" - Murphy

### Courses
- Andrew Ng's Machine Learning (Coursera)
- Stanford CS229
- Fast.ai

---

## Author

**Sreekar Yerragunta**

Building comprehensive ML expertise through from-scratch implementations.

- GitHub: [@sreekaryerragunta](https://github.com/sreekaryerragunta)
- Repository: [core-machine-learning](https://github.com/sreekaryerragunta/core-machine-learning)

---

**Note**: This repository focuses on **understanding** through implementation. For production use, sklearn is recommended. For learning, interviews, and deep understanding, implement from scratch!

**Total Content**: 12 supervised learning algorithms with theory, implementations, and demonstrations.
