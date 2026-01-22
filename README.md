# Machine-Learning-Foundations

**Author: Jiaqi Zhang**

This repository serves as a professional portfolio of my work in fundamental machine learning algorithms and statistical data analysis.

##  Projects Included

### 1. [Gaussian Naive Bayes from Scratch](./01-Gaussian-Naive-Bayes/)
- **Core:** Implemented a GNB classifier from first principles using NumPy.
- **Key Features:** Handled numerical stability with log-space probability calculations and implemented variance smoothing to match Scikit-learn benchmarks.

### 2. Statistical Case Studies (Upcoming)
- **Fairness Analysis:** Investigating model bias in demographic datasets.
- **Robust Validation:** Group-based splitting strategies for correlated healthcare data.

##  Tech Stack
- **Language:** Python 3.12.11
- **Libraries:** NumPy 2.3.2, Pandas 2.3.2, Scikit-learn 1.7.1, Matplotlib 3.10.5, pytorch 2.7.1, jupyter, pytest 8.4.1, quadprog

# Machine Learning Foundations: From Theory to Practice
**Author: Jiaqi Zhang** **Master's Student in Data-Enabled Computational Science @ Brown University**

This repository is a curated collection of core machine learning implementations and statistical case studies. It demonstrates a transition from first-principles mathematical derivations to robust, real-world model diagnostics.

## 🚀 Repository Structure

| Project | Key Focus | Tech Stack |
| :--- | :--- | :--- |
| **[01. Gaussian Naive Bayes from Scratch](./01-Gaussian-Naive-Bayes/)** | Numerical Stability, Generative Modeling | Python, NumPy |
| **[02. AI Ethics & Fairness Analysis](./02-AI-Ethics-Fairness/)** | Bias Mitigation, Sensitivity Analysis | Python, Pandas |
| **[03. Robust Medical Data Validation](./03-Robust-Medical-Validation/)** | Correlated Data, i.i.d. Violation | Python, Scikit-learn |
| **[04. Model Complexity & Overfitting](./04-Model-Complexity-Analysis/)** | Bias-Variance Trade-off, Tree Pruning | Python, Matplotlib |

---

## 🔍 Featured Projects

### 1. Gaussian Naive Bayes (GNB) from Scratch
Implemented a GNB classifier using only NumPy to understand the underlying probabilistic mechanics.
* **Log-Space Computation**: Prevented numerical underflow by performing inference in the log-probability domain.
* **Variance Smoothing (Epsilon)**: Implemented additive smoothing to handle zero-variance features, achieving exact parity with Scikit-learn's implementation.



### 2. Algorithmic Fairness in Demographic Data
A deep dive into the ethics of predictive modeling. 
* **The Insight**: Investigated the removal of sensitive attributes (race, sex). Demonstrated that **Accuracy $\neq$ Fairness**; removing features does not eliminate bias if underlying correlations exist in the remaining feature space.
* **Call to Action**: Emphasized the need for causal analysis rather than simple feature exclusion.

### 3. Robust Validation for Correlated Healthcare Data
A critical study on model evaluation in clinical settings where multiple samples exist per subject.
* **The Challenge**: Identified that random splitting breaks the **i.i.d. assumption** (Independent and Identically Distributed), leading to overly optimistic validation scores.
* **The Solution**: Implemented **Group-based Data Splitting**, ensuring that all samples from the same patient remain in either the training or testing set, never both.



### 4. Quantitative Analysis of Model Complexity
Diagnosing the behavior of Decision Trees on the Spam dataset.
* **Hyperparameter Tuning**: Quantified the relationship between `max_depth` and empirical loss.
* **Bias-Variance Trade-off**: Visualized the transition from underfitting to overfitting, identifying the optimal complexity threshold for generalization.



---

## 🛠️ Core Competencies
- **Mathematical Implementation**: Translating probabilistic formulas into stable Python code.
- **Statistical Rigor**: Identifying data leakage and correlation issues in complex datasets.
- **Responsible AI**: Evaluating the socio-technical impacts of algorithmic decisions.

## 📧 Contact
For inquiries regarding these implementations or potential collaborations:
- **Email**: jiaqi_zhang7@brown.edu
- **LinkedIn**: www.linkedin.com/in/jiaqi-zhang-95abba355