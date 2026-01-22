# Machine Learning Foundations: From Theory to Practice  
**Author: Jiaqi Zhang** · **Master's Student in Data-Enabled Computational Science (DECES) @ Brown University**

This repository is a curated portfolio of **core machine learning algorithms implemented from scratch** (NumPy-based), with an emphasis on mathematical foundations, numerical stability, visualizations, and rigorous evaluation.

---

## 🚀 Repository Structure

| Project | Key Focus | Tech Stack |
| :--- | :--- | :--- |
| **[Support Vector Machine for Classification Solved as a Quadratic Program](./01-SVM-QP/)** | Convex Optimization, Margins, Visualization | Python, NumPy, Matplotlib, quadprog |
| **[Model Complexity & Overfitting (Decision Trees)](./HW_05-Decision-Tree/)** | Bias–Variance Trade-off, Hyperparameter Tuning | Python, NumPy, Matplotlib |
| **[Regularized Logistic Regression + Cross-Validation](./HW_04-Logistic-Regression/)** | Regularization, Cross-Validation, Baselines | Python, NumPy, pytest|
| **[2-Layer Neural Network (SGD)](./HW_08-Neural-Net-SGD/)** | Backpropagation, Optimization Dynamics, Generalization | Python, NumPy, Matplotlib |
| **[Gaussian Naive Bayes from Scratch](./05-Gaussian-Naive-Bayes/)** | Numerical Stability, Generative Modeling, Unit Tests | Python, NumPy, pytest, scikit-learn
(benchmarking)|

---

## 🧠 Notes
- All core algorithms in this repository are implemented **from scratch** (no training via high-level ML libraries).
- **Verification & Testing:** `pytest` is used for unit tests to validate correctness (edge cases, numerical stability).
- **Benchmarking:** `Scikit-learn` is used only as a **reference implementation** for comparison where applicable.

---

## 🛠️ Tech Specs
- **Language:** Python 3.12.11
- **Core Libraries:** NumPy, Matplotlib
- **Utilities:** pytest, quadprog, Jupyter  
- **Benchmarking:** Scikit-learn


## 📧 Contact
For inquiries regarding these implementations or potential collaborations:
- **Email**: jiaqi_zhang7@brown.edu
- **LinkedIn**: www.linkedin.com/in/jiaqi-zhang-95abba355