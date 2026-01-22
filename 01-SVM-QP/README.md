# Support Vector Machine (SVM) Classification Project (QP + Kernels)

This project implements a binary Support Vector Machine (SVM) classifier **from scratch** using a **Quadratic Programming (QP)** formulation solved via `quadprog`. It supports multiple kernels (linear, RBF, polynomial), evaluates train/test accuracy, produces data visualizationn, and includes utilities to load and normalize CSV datasets.



## Author / Contact
- Jiaqi Zhang — jiaqi_zhang7@brown.edu  

## Environment
Developed and tested with:
- Python **3.12.11**
- matplotlib **3.10.5**
- pandas **2.3.2**
- scikit-learn **1.7.1**
- numpy **2.3.2**
- pytorch **2.7.1**
- jupyter
- pytest **8.4.1**
- quadprog

One can use `environment.yaml` in the Repo to set up the above environment.

To set up, run:

`conda env create -f environment.yml`

`conda activate svm`  

## Core Methods and Functions
- `solve_QP(Q, c, A, b, E=None, d=None)`: The function written to solve the quadratic program. (Equality constraints `Ex = d` are supported in the wrapper signature but not used in this project’s SVM formulation.) Inputs: `Q`: (n, n) matrix. `c`: (n,) vector. `A`: (k, n) matrix. `b` : (k,) vector. Outputs: (n,) vector of the optimal decision variables.

- `qp_example()`: A small demonstration function showing how to call `solve_QP` on a toy QP problem. It prints the optimal solution to confirm that the wrapper is working.

- `linear_kernel(xi, xj)`: Implements the standard dot product.

- `rbf_kernel(xi, xj, gamma=0.1)`: Radial Basis Function (Gaussian) kernel.

- `polynomial_kernel(xi, xj, c=2, d=2)`: Polynomial kernel.

- `class SVM(kernel_func=linear_kernel, lambda_param=0.1)`: A QP-based SVM classifier using a specified kernel.

- `__init__(self, kernel_func=linear_kernel, lambda_param=.1)`: It stores:`self.kernel_func`: kernel function used to compute similarity 
`self.lambda_param`: regularization strength used in the QP objective. No training happens here; it only sets hyperparameters.

-`train(self, inputs, labels)`: Fits the model by solving a QP. It does follows:
  - 1.Stores training data in `self.train_inputs`, `self.train_labels`.
  - 2.Builds the Gram matrix G using the chosen kernel.
  - 3.Constructs QP objective (Q, c) and inequality constraints (A, b).
  - 4.Calls solve_QP(...) to obtain the optimal variable vector.
  - 5.Stores the first m components as:`self.alpha`: (m,) coefficients used in prediction.

-`_get_gram_matrix(self)`: Computes the Gram matrix for the training set.

-`_objective_function(self, G)`: Builds the QP objective matrices Q and c for the optimization problem.

-`_inequality_constraint(self, G)`: Builds inequality constraints `Ax <= b` for the QP.

-`predict(self, inputs)`: Predicts labels for new data.Output: (n_samples,) numpy array of predicted labels in {-1, +1}

-`accuracy(self, inputs, labels)`: Computes classification accuracy.

-`test_svm(train_data, test_data, kernel_func=linear_kernel, lambda_param=.1)`: Convenience function to: 
    - Instantiate an SVM with the chosen kernel and lambda_param. 
    - Train on train_data.
    - Compute and print: training accuracy and test accuracy

- `read_data(file_name)`: Reads a CSV dataset and returns normalized features and binary labels.

## Notes
- Results should be reproducible by using the pinned package versions above and running the project in the gaussnb conda environment.