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

| Model                     | Precision | Recall | F1-score | Interpretation                                                                       |
| ------------------------- | --------- | ------ | -------- | ------------------------------------------------------------------------------------ |
| **Baseline LR (M1)**      | 0.85      | 0.63   | 0.72     | Balanced model; strong precision but misses some fraud cases.                        |
| **GMM Oversampling (M2)** | 0.08      | 0.86   | 0.15     | Detects more fraud (recall) but produces many false positives due to poor precision. |
| **GMM + CBU (M3)**        | 0.07      | 0.87   | 0.12     | Extreme recall with collapsed precision; predicts almost everything as fraud.        |

---

## **Conclusion and Recommendations**

**Benefits and Drawbacks of Resampling Methods:**

* GMM generates realistic and diverse samples by learning the minority distribution, making it theoretically stronger than naive oversampling.
* However, oversampling introduces noise and overlap with the majority class, reducing precision.
* Adding CBU further worsens precision by oversimplifying majority class structure.

**Performance Summary**

* **Baseline (M1):** Provides the most balanced performance with strong precision and reasonable recall.
* **GMM Oversampling (M2):** Greatly boosts recall (0.86) but at a steep cost — precision drops to 0.08, leading to many false positives.
* **GMM + CBU (M3):** Pushes recall slightly higher (0.87) but precision collapses further (0.07), making it impractical despite its minority coverage.

**Recommendation:**
If **maximizing recall** is critical (e.g., fraud detection), GMM-based oversampling (M2 or M3) can be useful but must be paired with **threshold tuning, regularization, or ensemble methods** to reduce false positives. For general use, **baseline logistic regression (M1)** remains the most balanced and reliable option.