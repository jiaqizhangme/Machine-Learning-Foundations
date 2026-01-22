# Gaussian Naive Bayes (GNB) Classification Project

This project implements and evaluates **Gaussian Naive Bayes (GNB)** as a probabilistic, generative classifier on the **Iris** dataset. The workflow includes a comparison against **scikit-learn’s `GaussianNB`** to validate correctness and reproducibility.

**Repository contents**
- `Final_Project.ipynb` — jupyter notebook containing the full write-up, derivations, experiments, and results.
- `iris.csv` and `iris.rst.txt` — the source data file of `load_iris` dataset.
- `environment.yaml` — environment setup file.
-  `Final Presentation.pdf` — pdf file of the presentation of the project.
-  `report.pdf` — pdf file of the `Final_Project.ipynb`.
## Authors / Contact
- Jiaqi Zhang — jiaqi_zhang7@brown.edu  
- Yize Zhao — yize_zhao@brown.edu  
- Ghirish Thaenraj — ghirish_thaenraj@brown.edu  

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

`conda activate gaussnb`

## Core Methods and Functions
- `load_iris()`: Loads feature matrix X and integer labels y for Iris. Also provides feature_names and target_names for readable labels.

- `train_test_split`(X, y, test_size=..., random_state=...): Splits the dataset into training and test sets to evaluate generalization performance.

- `__init__ (var_smoothing=1e-9)`: Initializes hyperparameters and placeholders for learned parameters (class priors, per-class means/variances). var_smoothing controls the magnitude of variance smoothing for numerical stability.

- `_validate_input_X(X)`: Ensures the feature input is numeric, 2D, and shaped consistently (e.g., reshapes 1D inputs to (n_samples, 1)).

- `_validate_input_y(y)`: Ensures labels are in a valid, consistent array format.

- `train(X, y)` / `fit(X, y)`: Learns model parameters from data.

- `_gaussian_log_pdf(X, mean, var)`: Computes the log of the Gaussian probability density for each feature. Using log-space avoids underflow when multiplying many small probabilities.

- `predict_log_proba(X)`: Computes log posterior probabilities for each class: Normalizes in log-space (log-sum-exp style) so outputs are valid log-probabilities.

- `predict_proba(X)`: Converts predict_log_proba(X) into standard probabilities (exp + normalization), returning a (n_samples, n_classes) probability matrix.

- `predict(X)`: Predicts the most likely label for each sample by taking argmax over class posteriors.

- `loss(X, y)`: Computes negative log-likelihood (NLL): how well the model assigns probability to the true class labels (lower is better).

- `GaussianNB()`: Reference implementation used for correctness checks and baseline comparison.

- `accuracy_score(y_true, y_pred)`: Computes accuracy to compare custom predictions vs scikit-learn predictions and evaluate test performance.
  
- `run_unit_tests()`: Runs unit tests (via pytest or internal checks) to verify correctness of shape handling, smoothing behavior, probability outputs, and edge cases.

- `sklearn_comparison_test()`: a compatibility/verification test that checks whether your custom GaussianNaiveBayes implementation behaves the same as scikit-learn’s GaussianNB on the same train/test split.
  
- `verify_exact_match_test()`: Compares custom GNB outputs to scikit-learn’s GaussianNB (e.g., predict, predict_proba, predict_log_proba) within numerical tolerances.

- `visualize_comparison_metrics()`: Produces comparison plots (e.g., custom vs scikit-learn probability/log-probability alignment, performance metrics).

- `visualize_results()`: Generates final result visualizations (e.g., confusion matrix, decision boundary slice using 2 features, and other interpretability plots).

## Notes

- Results should be reproducible by using the pinned package versions above and running the project in the gaussnb conda environment.


- The implementation emphasizes numerical stability (log-domain computation + variance smoothing) and scikit-learn-style interfaces (fit, predict, predict_proba).

