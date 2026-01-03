# Core Machine Learning

**From-scratch implementations of fundamental ML algorithms with deep theoretical explanations**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Based-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository demonstrates mastery of **core machine learning algorithms** through:
- **Theory**: Mathematical foundations and intuitions
- **Implementation**: From-scratch using NumPy (no sklearn for training)
- **Applications**: Real-world datasets and use cases
- **Comparison**: Validation against scikit-learn

**Target audience**: Data scientists, ML engineers, and anyone building strong ML foundations

---

## Repository Structure

```
core-machine-learning/
├── 01_decision_trees/          # Tree-based classification and regression
├── 02_random_forests/          # Ensemble learning with bagging
├── 03_svm/                     # Support Vector Machines
├── 04_naive_bayes/             # Probabilistic classification
├── 05_k_means/                 # Clustering algorithm
├── 06_knn/                     # K-Nearest Neighbors
└── datasets/                   # Sample datasets
```

---

## Algorithms Covered

### 1. Decision Trees
- Entropy, Information Gain, Gini Impurity
- CART algorithm implementation
- Tree pruning and overfitting prevention
- Visualization of decision boundaries

### 2. Random Forests
- Bootstrap aggregating (bagging)
- Feature randomness
- Out-of-bag error estimation
- Feature importance calculation

### 3. Support Vector Machines (SVM)
- Maximum margin classifier
- Kernel trick (linear, RBF, polynomial)
- Soft margin classification
- Non-linear decision boundaries

### 4. Naive Bayes
- Bayes theorem application
- Gaussian and Multinomial variants
- Text classification
- Spam detection

### 5. K-Means Clustering
- Lloyd's algorithm
- K-Means++ initialization
- Elbow method for choosing K
- Image compression application

### 6. K-Nearest Neighbors (KNN)
- Distance-based classification
- Multiple distance metrics
- Lazy learning approach
- Curse of dimensionality

---

## What Makes This Portfolio Stand Out

### Deep Understanding
- Complete mathematical derivations
- Clear explanations of "why" and "when"
- Algorithmic intuition development

### Clean Implementation
- NumPy-only implementations
- Modular, reusable code
- Comprehensive documentation
- Production-quality standards

### Practical Focus
- Real datasets and applications
- Performance comparisons
- Best practices
- Hyperparameter tuning insights

---

## Getting Started

### Prerequisites
```bash
Python 3.8+
NumPy
Matplotlib
Seaborn
Pandas
Scikit-learn (for comparison only)
Jupyter Notebook
```

### Installation
```bash
git clone https://github.com/sreekaryerragunta/core-machine-learning.git
cd core-machine-learning
pip install -r requirements.txt
```

### Usage
Each algorithm folder contains:
- `theory.md` - Mathematical foundations
- Implementation notebooks - Step-by-step from-scratch builds
- Application notebooks - Real-world use cases
- `utils.py` - Reusable helper functions

Start with any algorithm's theory document, then explore the implementation notebooks.

---

## Key Learnings

After completing this repository, you'll understand:
- How classic ML algorithms work internally
- When to use each algorithm
- Trade-offs between different approaches
- How to implement algorithms from mathematical foundations
- Performance optimization techniques

---

## Portfolio Context

This is part of a comprehensive ML portfolio:
1. **ml-math-from-scratch** - Mathematical foundations (Linear/Logistic Regression, PCA)
2. **core-machine-learning** ← You are here
3. **deep-learning-from-scratch** - Neural networks and backpropagation
4. **applied-ml-projects** - End-to-end ML systems
5. **capstone-ai-systems** - Production-scale AI applications

---

## Contributing

This is a personal learning portfolio. Feel free to:
- Report issues
- Suggest improvements
- Use as a learning resource

---

## Author

**Sreekar Yerragunta**
- GitHub: [@sreekaryerragunta](https://github.com/sreekaryerragunta)
- Focus: Machine Learning, Data Science, AI Systems

---

## License

MIT License - feel free to use this for learning purposes.

---

## Acknowledgments

Built from first principles using:
- Academic papers and textbooks
- numpy documentation
- scikit-learn for validation

**Note**: All implementations are educational. For production use, leverage optimized libraries like scikit-learn.
