# SVM Solvers
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

## Installation

Ensure you have Python 3 installed. Then, install the required dependencies:

```sh
pip install numpy scipy scikit-learn cvxpy tqdm
```

## Usage
The script provides a command-line interface (CLI) to run different solvers with configurable hyperparameters.

### Running the Script
Use the following command:

```sh
python SVM.py --solver <solver_name> --strategy <ovr/ovo> --C <regularization> --max_iters <iterations> --learning_rate <lr> --n_components <PCA_components> --subset_size <samples>
```

### Arguments

- `--solver`: Choose from `gd`, `cvx`, `sklearn`, `cuttingplane`, `pegasos`, `newton`, `lssvm`.
- `--strategy`: Select multi-class strategy: `ovo` (One-vs-One) or `ovr` (One-vs-Rest).
- `--C`: Regularization parameter (default: 1.0).
- `--max_iters`: Number of training iterations (default: 1000).
- `--learning_rate`: Learning rate for gradient-based solvers (default: 0.01).
- `--n_components`: Number of PCA components (default: 50).
- `--subset_size`: Use a subset of the dataset for faster testing (optional).

## Example Commands

### 1. Run SVM with Gradient Descent (OvR strategy):
```sh
python SVM.py --solver gd --strategy ovr --C 1.0 --max_iters 1000 --learning_rate 0.01 --n_components 50
```

### 2. Run SVM with CVXPY Quadratic Programming (OvO strategy):
```sh
python SVM.py --solver cvx --strategy ovo --C 1.0 --max_iters 100
```

### 3. Run Scikit-Learn SVM with Linear Kernel (OvR strategy):
```sh
python SVM.py --solver sklearn --strategy ovr --C 1.0 --max_iters -1
```

### 4. Run Cutting Plane Method with Learning Rate 0.1:
```sh
python SVM.py --solver cuttingplane --strategy ovr --C 1.0 --max_iters 1000 --learning_rate 0.1
```

### 5. Run Pegasos Method with 5000 training samples for faster results:
```sh
python SVM.py --solver pegasos --strategy ovo --C 1.0 --max_iters 1000 --subset_size 5000
```

### 6. Run Newton Method:
```sh
python SVM.py --solver newton --strategy ovr --C 1.0 --max_iters 100
```

### 7. Run Least Squares SVM:
```sh
python SVM.py --solver lssvm --strategy ovr --C 1.0
```

## Results
The script outputs:

- **Solver used**
- **Multi-class strategy (OvO/OvR)**
- **Test accuracy**
- **Total number of support vectors**
- **Training time**



## License
This project is open-source and distributed under the MIT License.

