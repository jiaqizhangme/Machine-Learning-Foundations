# Regularized Logistic Regression With Cross-Validation Evaluation

This project implements regularized binary Logistic Regression from scratch and uses it as a clean baseline to study generalization and model selection. The model is trained with mini-batch stochastic gradient descent on the logistic (log-loss) objective with L2/Tikhonov regularization (controlled by λ), includes a full prediction/accuracy pipeline with a 0.5 sigmoid threshold, and provides automated hyperparameter tuning by sweeping λ and comparing training error vs. validation error vs. k-fold cross-validation error (with a semilog λ plot to visualize the bias–variance trade-off). The project also includes pytest-based unit tests to verify that training and prediction behave correctly on small synthetic datasets, and a simple data loader that reads train/validation CSVs and appends a bias column.

**Repository contents**
- `logistic_code.ipynb` — jupyter notebook containing the full write-up, derivations, experiments, and visualization results.
- `environment.yaml` — environment setup file.
- `X_train.csv`, `X_val.csv` — training/validation feature matrices for the cancer patient dataset (each row is a patient sample; columns are numeric clinical features used for prediction).
- `Y_train.csv`, `Y_val.csv` — corresponding binary labels for the training/validation splits (one label per patient indicating cancer status; encoded as 0/1).


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

`conda activate logi`  


## Core Methods and Functions
### `sigmoid_function(x)`
Computes the logistic sigmoid:
\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]
Used to convert linear scores into probabilities for logistic regression.

---

## `class RegularizedLogisticRegression`
Implements **binary regularized logistic regression** trained with **mini-batch SGD**. The optimization target is logistic loss + L2 (Tikhonov) regularization.

### `__init__(batch_size=15)`
Initializes training hyperparameters and model state:
- `learningRate = 1e-5`
- `num_epochs = 10000`
- `batch_size` (mini-batch size)
- `weights` (initialized in `train`)
- `lmbda` (regularization strength, default `1` but overwritten during sweeps)

---

### `train(X, Y)`
Trains the model using **mini-batch stochastic gradient descent**.

**Inputs**
- `X`: 2D numpy array `(n_samples, n_features+1)` where the last column is a **bias term of 1s**
- `Y`: 1D numpy array `(n_samples,)` with labels in `{0, 1}`

**What it does**
- Initializes `weights` to zeros
- Repeats for `num_epochs`:
  - Shuffles the dataset
  - Iterates over mini-batches
  - Computes predictions: `h = sigmoid(X_batch @ weights)`
  - Computes gradient of logistic loss:
    \[
    \nabla = \frac{1}{m} X^T(h - y)
    \]
  - Adds L2 regularization gradient: `+ 2 * lmbda * weights`
  - Updates weights: `weights -= learningRate * gradient`

**Output**
- Stores learned weights in `self.weights` with shape `(1, n_features+1)`

---

### `predict(X)`
Generates **binary predictions**.

**Inputs**
- `X`: 2D array `(n_samples, n_features+1)` with bias column included

**What it does**
- Computes probability scores: `sigmoid(X @ weights.T)`
- Thresholds at `0.5`:
  - `>= 0.5 → 1`
  - `< 0.5 → 0`

**Output**
- 1D numpy array `(n_samples,)` of predicted labels in `{0,1}`

---

### `accuracy(X, Y)`
Computes classification accuracy.

**Inputs**
- `X`: feature matrix with bias column
- `Y`: true labels

**Output**
- Accuracy in `[0,1]`, computed as `(# correct) / (n_samples)`

---

### `runTrainTestValSplit(lambda_list, X_train, Y_train, X_val, Y_val)`
Sweeps a list of regularization strengths and records **training/validation error**.

**Inputs**
- `lambda_list`: list of λ values
- `X_train, Y_train`: training data
- `X_val, Y_val`: validation data

**What it does**
- For each `λ`:
  - sets `self.lmbda = λ`
  - trains the model on training set
  - computes:
    - training error = `1 - accuracy(train)`
    - validation error = `1 - accuracy(val)`

**Outputs**
- `train_errors`: list aligned to `lambda_list`
- `val_errors`: list aligned to `lambda_list`

---

### `_kFoldSplitIndices(dataset, k)`
Helper for k-fold CV: splits indices into `k` (roughly) equal folds.

**Inputs**
- `dataset`: numpy array whose first dimension is the number of samples
- `k`: number of folds

**What it does**
- Shuffles indices and splits them into `k` groups

**Output**
- `indices_split`: list of `k` numpy arrays (each is a fold of indices)

---

### `runKFold(lambda_list, X, Y, k=3)`
Runs **k-fold cross-validation** for each λ and returns mean CV error.

**Inputs**
- `lambda_list`: list of λ values
- `X, Y`: dataset used for CV (often train+val combined)
- `k`: number of folds (default 3)

**What it does**
- Creates folds using `_kFoldSplitIndices`
- For each `λ`:
  - for each fold `i`:
    - uses fold `i` as validation
    - uses remaining folds as training
    - trains model, measures validation error = `1 - accuracy`
  - averages validation errors across folds

**Output**
- `k_fold_errors`: list of mean CV errors aligned to `lambda_list`

---

### `plotError(lambda_list, train_errors, val_errors, k_fold_errors)`
Plots error curves vs λ on a log-scale x-axis.

**What it shows**
- training error
- validation error
- k-fold (CV) error

Uses `plt.semilogx(...)` to visualize performance across λ spanning multiple orders of magnitude.

---

## Data / Driver Utilities

### `extract()`
Loads train/validation splits from CSV files and appends a bias column.

**Reads**
- `data/X_train.csv`, `data/y_train.csv`
- `data/X_val.csv`, `data/y_val.csv`

**What it does**
- Converts labels into 1D numpy arrays
- Appends a column of ones to `X_train` and `X_val` for the bias term

**Returns**
- `X_train, X_val, Y_train, Y_val`

---

### `main()`
End-to-end experiment runner.

**What it does**
1. Loads data via `extract()`
2. Trains a baseline model and prints train/validation accuracy
3. Sweeps a `lambda_list` to compute:
   - training error
   - validation error
4. Runs k-fold cross validation on the combined train+val set
5. Prints all errors and calls `plotError(...)` for visualization



## Notes
- Results should be reproducible by using the pinned package versions above and running the project in the gaussnb conda environment.
- Some Plots and Graphs for Preview:

Title: Plot of Regularization parameter VS. Three Types of Errors

![plot](./Picture1.png)