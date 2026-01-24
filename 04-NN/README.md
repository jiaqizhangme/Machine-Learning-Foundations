# Build and Train 1- and 2-Layer Neural Networks (SGD + Backprop) From Scratch

This project implements **shallow neural networks from scratch** and trains them using **mini-batch stochastic gradient descent (SGD)** on the Wine Quality dataset. It includes both a **1-layer linear model** (bias included via an augmented feature column) and a **2-layer network with one hidden layer** trained via **manual backpropagation** (explicit gradients for weights and biases). The workflow covers **data normalization**, reproducible train/test splitting, and evaluation with **learning curves (train vs test MSE)**, **predicted-vs-true plots with R²**, and **residual distribution analysis** to assess convergence and generalization.



**Repository contents**
- `nn.ipynb` — jupyter notebook containing the full write-up, derivations, experiments, and visualization results.
- `environment.yaml` — environment setup file.
- `wine.txt` — dataset that contains attributes of wines as well as wine quality. The data is formatted as follows: The first column
of data in each file is the dependent variable (the observations $y$)
and all other columns are the independent input variables ($x_{1}, x_{2}, \ldots, x_{n}$). For more information about the dataset, one can find it [here](https://archive.ics.uci.edu/ml/datasets/wine).


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

`conda activate nn`  



## Core Methods and Functions

### Utility Functions

#### `l2_loss(predictions, Y)`
Computes the total squared error between model predictions and targets.

- **Inputs**
  - `predictions`: model outputs (reshaped internally as needed)
  - `Y`: target values
- **Output**
  - Total squared error (sum of squared residuals)

---

#### `sigmoid(x)`
Applies the sigmoid function element-wise to an input array.

- **Inputs**
  - `x`: numpy array (any shape)
- **Output**
  - numpy array of the same shape, with values squashed to `(0, 1)`
- **Notes**
  - Implemented with `np.where(...)` to improve stability for large positive/negative inputs.

---

#### `sigmoid_derivative(x)`
Computes the derivative of the sigmoid function.

- **Inputs**
  - `x`: numpy array (any shape)
- **Output**
  - numpy array of the same shape containing sigmoid derivatives
- **Used for**
  - Backpropagation in the 2-layer network.

---

## `class OneLayerNN`
Implements a 1-layer neural network (a linear model) trained with mini-batch SGD. Bias is handled by appending a column of ones to the feature matrix.

### `__init__(batch_size=1)`
Initializes training hyperparameters and model state.

- **What it sets**
  - `batch_size` (mini-batch size)
  - `learning_rate` (default `0.001`)
  - `epochs` (default `25`)
  - `weights` (initialized in `train`)
  - `v` (stores forward-pass outputs)
  - `train_mse_history`, `val_mse_history` (optional learning-curve logs)

---

### `train(X, Y, print_loss=True, X_val=None, Y_val=None, record_history=False)`
Trains the model using mini-batch stochastic gradient descent.

- **Inputs**
  - `X`: 2D numpy array `(n_samples, n_features+1)` (last column is bias term of 1s)
  - `Y`: 1D numpy array `(n_samples,)`
  - `print_loss`: whether to print epoch loss
  - `X_val`, `Y_val`: optional validation data for logging
  - `record_history`: if True, records per-epoch MSE in history buffers
- **What it does**
  - Initializes `weights` randomly in `[0, 1]`
  - Repeats for `epochs`:
    - Shuffles the dataset
    - Iterates over mini-batches
    - Runs `forward_pass()` and `backward_pass()` for each batch
    - Optionally logs training/validation MSE for learning curves
- **Output**
  - Stores learned parameters in `self.weights`

---

### `forward_pass(X)`
Computes the model output for a batch and stores it.

- **Inputs**
  - `X`: 2D array `(batch_size, n_features+1)`
- **What it does**
  - Computes linear predictions and stores them in `self.v`
- **Output**
  - Updates `self.v` (shape `(1, batch_size)`)

---

### `backward_pass(X, Y)`
Performs one SGD step on a batch.

- **Inputs**
  - `X`: batch features
  - `Y`: batch targets
- **What it does**
  - Calls `backprop(X, Y)` to compute gradients
  - Calls `gradient_descent(grad)` to update weights

---

### `backprop(X, Y)`
Computes the gradient of the squared loss with respect to weights for a batch.

- **Inputs**
  - `X`: batch features
  - `Y`: batch targets
- **Output**
  - Gradient array with the same shape as `weights`

---

### `gradient_descent(grad_W)`
Updates weights using the computed gradient.

- **Inputs**
  - `grad_W`: gradient array for `weights`
- **What it does**
  - Applies an SGD update using `learning_rate`

---

### `loss(X, Y)`
Computes total loss (sum of squared errors) over a dataset.

- **Inputs**
  - `X`: feature matrix
  - `Y`: targets
- **Output**
  - Total squared error loss value

---

### `average_loss(X, Y)`
Computes mean squared error over a dataset.

- **Inputs**
  - `X`: feature matrix
  - `Y`: targets
- **Output**
  - Mean squared error (MSE)

---

## `class TwoLayerNN`
Implements a 2-layer neural network (one hidden layer) trained with mini-batch SGD and manual backpropagation. Hidden layer uses sigmoid activation; output layer is linear.

### `__init__(hidden_size, batch_size=1, activation=sigmoid, activation_derivative=sigmoid_derivative)`
Initializes training hyperparameters, activation functions, and parameter placeholders.

- **What it sets**
  - `hidden_size`, `batch_size`
  - `learning_rate` (default `0.01`)
  - `epochs` (default `25`)
  - Parameters initialized in `train`: `wh`, `bh`, `wout`, `bout`
  - Cached forward values: `a1`, `v1`, `a2`, `v2`
  - `train_mse_history`, `val_mse_history` (optional learning-curve logs)

---

### `_get_layer2_bias_gradient(x, y)`
Computes the output-layer bias gradient for a single example.

- **Inputs**
  - `x`: single feature vector
  - `y`: single target value
- **Output**
  - Gradient for `bout`

---

### `_get_layer2_weights_gradient(x, y)`
Computes the output-layer weight gradient for a single example.

- **Inputs**
  - `x`: single feature vector
  - `y`: single target value
- **Output**
  - Gradient for `wout`

---

### `_get_layer1_bias_gradient(x, y)`
Computes the hidden-layer bias gradient for a single example (backpropagated through the hidden activation).

- **Inputs**
  - `x`: single feature vector
  - `y`: single target value
- **Output**
  - Gradient for `bh`

---

### `_get_layer1_weights_gradient(x, y)`
Computes the hidden-layer weight gradient for a single example.

- **Inputs**
  - `x`: single feature vector
  - `y`: single target value
- **Output**
  - Gradient for `wh`

---

### `train(X, Y, print_loss=True, X_val=None, Y_val=None, record_history=False)`
Trains the 2-layer network using mini-batch SGD with manual backpropagation.

- **Inputs**
  - `X`: 2D numpy array `(n_samples, n_features)`
  - `Y`: 1D numpy array `(n_samples,)`
  - `print_loss`: whether to print epoch loss
  - `X_val`, `Y_val`: optional validation set for logging
  - `record_history`: if True, records per-epoch MSE into history buffers
- **What it does**
  - Initializes `wh`, `wout` randomly in `[0, 1]` and biases to zeros
  - Repeats for `epochs`:
    - Shuffles dataset
    - Iterates over mini-batches
    - Runs `forward_pass()` and `backward_pass()` for each batch
    - Optionally logs training/validation MSE for learning curves
- **Output**
  - Stores learned parameters in `wh`, `bh`, `wout`, `bout`

---

### `forward_pass(X)`
Runs the forward computation and stores intermediate activations.

- **Inputs**
  - `X`: 2D numpy array `(batch_size, n_features)`
- **What it does**
  - Computes hidden pre-activations/activations and final outputs
- **Output**
  - Updates cached tensors (`a1`, `v1`, `a2`, `v2`)

---

### `backward_pass(X, Y)`
Performs one SGD step on a mini-batch.

- **Inputs**
  - `X`: batch features
  - `Y`: batch targets
- **What it does**
  - Calls `backprop(X, Y)` to compute gradients
  - Calls `gradient_descent(...)` to update parameters

---

### `backprop(X, Y)`
Computes averaged gradients for `wh`, `bh`, `wout`, and `bout` over a mini-batch.

- **Inputs**
  - `X`: batch features
  - `Y`: batch targets
- **What it does**
  - Computes per-example gradients using helper methods
  - Averages gradients across the batch
- **Output**
  - Returns `(grad_wh, grad_bh, grad_wout, grad_bout)`

---

### `gradient_descent(grad_wh, grad_bh, grad_wout, grad_bout)`
Updates all parameters using SGD.

- **Inputs**
  - Gradients for each parameter matrix/vector
- **What it does**
  - Applies SGD update steps using `learning_rate`

---

### `loss(X, Y)`
Computes total squared error loss over a dataset.

- **Inputs**
  - `X`: feature matrix
  - `Y`: targets
- **Output**
  - Total squared error loss value

---

### `average_loss(X, Y)`
Computes mean squared error (MSE) over a dataset.

- **Inputs**
  - `X`: feature matrix
  - `Y`: targets
- **Output**
  - Mean squared error (MSE)

---

## Evaluation Helpers (used for plots)

#### `predict_one(model, Xb)`
Runs a forward pass using `OneLayerNN` and returns predictions as a 1D array.

- **Inputs**
  - `model`: trained `OneLayerNN`
  - `Xb`: feature matrix with bias column
- **Output**
  - 1D numpy array of predictions

---

#### `predict_two(model, X)`
Runs a forward pass using `TwoLayerNN` and returns predictions as a 1D array.

- **Inputs**
  - `model`: trained `TwoLayerNN`
  - `X`: feature matrix (no bias column)
- **Output**
  - 1D numpy array of predictions

---

#### `r2_score_manual(y_true, y_pred)`
Computes R² score from scratch for evaluation.

- **Inputs**
  - `y_true`: true target values
  - `y_pred`: predicted values
- **Output**
  - R² score (float)




## Notes
- Results should be reproducible by using the pinned package versions above and running the project in the gaussnb conda environment.
- Some Plots and Graphs for Preview:

    ![1](./nn_learning_curves.png)

    ![2](./nn_residuals.png)

    ![3](./nn_pred_vs_true.png)