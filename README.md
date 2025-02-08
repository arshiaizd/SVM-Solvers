# SVM Solver

This project implements and benchmarks various SVM (Support Vector Machine) solvers for handwritten digit classification on the MNIST dataset. It supports multiple optimization strategies (gradient descent, quadratic programming, cutting-plane methods, etc.) and both One-vs-One (OvO) and One-vs-Rest (OvR) multi-class classification.

![MNIST Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Project Overview

### Key Features
- **7 SVM Solvers**:
  - Gradient Descent
  - CVXPY QP (Quadratic Programming)
  - Scikit-learn (Reference Implementation)
  - Cutting Plane Method
  - Pegasos (Stochastic Subgradient)
  - Newton Method
  - Least-Squares SVM (LS-SVM)
- **Multi-Class Strategies**: OvO and OvR.
- **Dimensionality Reduction**: PCA for feature compression.
- **Scalability**: Subset sampling for faster experimentation.

### Dataset
The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) contains 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. The dataset is split into:
- **60,000 training images**
- **10,000 test images**

The code automatically downloads and preprocesses the data, with optional PCA to reduce the 784 raw features to 50–100 components.

---

## Solvers

### 1. Gradient Descent
Primal solver minimizing hinge loss with L2 regularization.  
**Pros**: Simple, no complex dependencies.  
**Cons**: Slow convergence, hyperparameter-sensitive.

```bash
python script.py --solver gd --strategy ovo --C 1.0 --max_iters 1000 --learning_rate 0.01
