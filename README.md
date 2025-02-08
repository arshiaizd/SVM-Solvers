# SVM Solver Benchmark

## Overview
This project implements various Support Vector Machine (SVM) solvers and evaluates their performance on the MNIST dataset. It supports multiple solvers, including:

- **Gradient Descent** (GD)
- **Quadratic Programming with CVXPY** (CVX)
- **Scikit-Learn's SVC** (Sklearn)
- **Cutting Plane Method**
- **Pegasos Method**
- **Newton’s Method**
- **Least Squares SVM** (LS-SVM)

The implementation supports **One-vs-One (OvO)** and **One-vs-Rest (OvR)** strategies for multi-class classification and allows dimensionality reduction using **Principal Component Analysis (PCA)**.

## Dataset
The project uses the **MNIST** dataset, which contains 70,000 images of handwritten digits (0-9), each with 784 features (28×28 pixel grayscale images). The dataset is loaded using `fetch_openml` and split into training and test sets.

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

## Contribution
Feel free to contribute by adding new solvers, improving efficiency, or optimizing the existing implementations. Fork the repository and submit a pull request!

## License
This project is open-source and distributed under the MIT License.

