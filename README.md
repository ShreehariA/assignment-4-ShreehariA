# **GMM-Based Synthetic Sampling for Imbalanced Data (DA5401 A4)**

**Student Name:** Shreehari Anbazhagan\
**Roll Number:** DA25C020

---

## **Project Overview**

This assignment explores **Gaussian Mixture Model (GMM)-based synthetic sampling** as a strategy for addressing **class imbalance** in fraud detection. The dataset contains highly imbalanced classes where fraudulent transactions form a very small fraction.

The workflow involves:

* Training a **baseline logistic regression model** on the original imbalanced data.
* Applying **GMM-based oversampling** to generate realistic synthetic samples for the minority (fraudulent) class.
* Combining oversampling with **Clustering-Based Undersampling (CBU)** to balance both classes.
* Comparing model performance across these strategies using **precision, recall, and F1-score**, which are more relevant than accuracy in imbalanced contexts.

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Folder Structure & Files

```
project-root/
│
├─ main.ipynb              # Core Jupyter Notebook with all code, visualizations, and explanations
├─ python-version          # Python version used for reproducibility
├─ pyproject.toml          # Project dependencies
├─ uv.lock                 # Locked dependency versions (for uv sync)
├─ .gitignore              # Excludes dataset files
├─ Instructions/           # Assignment PDF (problem statement & instructions)
    └─ DA5401 A3 Clustering Based Sampling.pdf
└─ datasets/               # Dataset folder
   └─ creditcard.csv       # creditcard fraud detection dataset (from Kaggle)
```

### Notes:

* Run `uv sync` to install dependencies exactly as tested.
* The dataset is **not pushed to GitHub**. Please download it via Kaggle:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Plots using **plotly are not visible in github** as they use JavaScript to render, Please run them locally or use nbviewer.
* The notebook is **self-contained**: all preprocessing, PCA, and classification steps are reproducible without manual intervention.

---

## Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
```

---

## **Analysis Workflow**

1. **Data Analysis & Baseline Model**

   * Loaded and analyzed the class imbalance in `creditcard.csv`.
   * Trained Logistic Regression on the imbalanced dataset (75/25 stratified split).
   * Evaluated using precision, recall, and F1-score for the minority class.

2. **GMM-Based Oversampling**

   * Fitted a Gaussian Mixture Model (GMM) on the minority class.
   * Chose the number of mixture components (`k`) based on **AIC/BIC criteria**.
   * Generated synthetic samples to balance the dataset.

3. **Rebalancing with CBU + GMM**

   * Applied **Clustering-Based Undersampling** on the majority class to reduce redundancy.
   * Enriched the minority class with GMM-sampled synthetic data.
   * Produced a balanced dataset for training.

4. **Model Training & Comparative Evaluation**

   * Trained Logistic Regression on:

     * Baseline imbalanced dataset.
     * GMM-balanced dataset.
     * GMM + CBU balanced dataset.
   * Compared performance metrics to assess trade-offs between recall and precision.

---

## **Key Findings**

| Model                     | Precision | Recall | F1-score | Interpretation                                                                            |
| ------------------------- | --------- | ------ | -------- | ----------------------------------------------------------------------------------------- |
| **Baseline LR (M1)**      | 0.85      | 0.63   | 0.72     | Performs cautiously, with good precision but misses many fraud cases.                     |
| **GMM Oversampling (M2)** | 0.56      | 0.85   | 0.68     | Improves fraud detection (recall) but sacrifices precision, leading to more false alarms. |
| **GMM + CBU (M3)**        | 0.01      | 0.91   | 0.01     | Overpredicts fraud, achieving very high recall but unusable precision.                    |

---

## **Conclusion and Recommendations**

**Benefits and Drawbacks of Resampling Methods:**

* GMM generates realistic and diverse samples by learning the minority distribution, making it theoretically stronger than naive oversampling.
* However, oversampling introduces noise and overlap with the majority class, reducing precision.
* Adding CBU further worsens precision by oversimplifying majority class structure.

**Performance Summary:**

* Baseline strikes a better balance between precision and recall.
* GMM oversampling helps recall but inflates false positives.
* GMM + CBU leads to extreme imbalance in metrics and poor practical utility.

**Recommendation:**
GMM-based oversampling is useful when **maximizing recall** is the priority (e.g., in fraud detection, where missing a fraud is costlier than false alarms). However, it should not be used alone — additional regularization, threshold tuning, or hybrid ensemble methods are required to control false positives. For this dataset, the **baseline logistic regression (M1)** remains the most balanced approach, with GMM oversampling (M2) only recommended in contexts where recall is paramount.