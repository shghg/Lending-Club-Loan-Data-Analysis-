# Lending-Club-Loan-Data-Analysis-
Perfect ✅ — here’s an upgraded **README.md** with **badges**, a **screenshot/graph section**, and a clean professional look. This version is recruiter-ready and visually appealing on GitHub.

---

# 📊 Lending Club Loan Data Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview

This project analyzes the **Lending Club loan dataset (2007–2015)** to predict the likelihood of loan default. Using **exploratory data analysis (EDA)**, **feature engineering**, and a **deep learning model**, the goal is to build a predictive system that identifies risk patterns in borrower profiles.

The workflow demonstrates practical **data preprocessing**, **imbalanced classification handling**, and **model evaluation** with metrics like **AUC** to assess real-world financial risk prediction tasks.

---

## 🗂 Dataset

* Source: Historical Lending Club loan data (2007–2015)
* Rows: \~9,578
* Features:

  * Borrower credit information (e.g., `fico`, `int.rate`, `dti`, `revol.util`)
  * Loan details (e.g., `installment`, `purpose`)
  * Target variable: **`not.fully.paid`** (binary: 1 = default, 0 = fully paid)

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Removed duplicates and handled missing values
* Encoded categorical variables (`purpose`) via one-hot encoding
* Normalized numerical features using `StandardScaler`

### 🔹 Exploratory Data Analysis (EDA)

* **Target Imbalance**: Defaults are much fewer than fully paid loans
* **Key Trends**:

  * Higher **interest rates** correlate with higher default rates
  * Lower **FICO scores** strongly predict default
  * Certain loan purposes (e.g., *small business*, *educational*) are riskier

### 🔹 Feature Engineering

* Correlation matrix used to detect multicollinearity
* Dropped highly correlated features (threshold > 0.85)
* Final dataset contained **19 engineered features**

### 🔹 Modeling

* Built a **deep learning model** with TensorFlow/Keras:

  ```text
  Input → Dense(64, ReLU) → Dropout(0.3)  
        → Dense(32, ReLU) → Dropout(0.2)  
        → Dense(1, Sigmoid)
  ```
* Optimizer: **Adam (lr = 0.001)**
* Loss: **Binary Crossentropy**
* Metrics: Accuracy, **AUC**
* Early stopping applied

---

## 📈 Results

| Metric    | Score                                 |
| --------- | ------------------------------------- |
| Accuracy  | \~80%                                 |
| AUC       | \~0.65                                |
| Precision | Low on default class (needs tuning)   |
| Recall    | Limited for minority class (defaults) |

⚠️ Model predicts majority (non-default) cases well but struggles on minority class → highlights importance of **class imbalance handling**.

---

## 📊 Sample Visualizations

### Interest Rate vs Loan Default

<img src="https://github.com/shghg/Lending-Club-Loan-Data-Analysis/assets/interest_rate_vs_default.png" width="500">  

### FICO Score Distribution by Loan Status

<img src="https://github.com/shghg/Lending-Club-Loan-Data-Analysis/assets/fico_distribution.png" width="500">  

*(Replace with your saved plots from notebook or export them as `.png` and upload to repo → then adjust the image paths.)*

---

## 🔑 Key Insights

* Borrower creditworthiness (FICO, interest rate, DTI) is critical in default prediction.
* Loan purpose impacts risk — *small business* and *educational* loans are more likely to default.
* Class imbalance is a major challenge; methods like **SMOTE**, **threshold tuning**, or **ensemble learning** can improve minority detection.

---

## 🚀 Next Steps

* Hyperparameter tuning & threshold optimization
* Compare with tree-based models (Random Forest, XGBoost)
* Try dimensionality reduction (PCA, autoencoders)
* Deploy model with Flask/FastAPI for real-time scoring

---

## 🛠️ Tech Stack

* **Languages:** Python (3.x)
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, TensorFlow/Keras
* **Environment:** Jupyter Notebook

---

## 📂 Project Structure

```
├── Lending Club Loan Data Analysis.ipynb   # Main notebook
├── Lending Club Loan Data Analysis.pdf     # Report
├── loan_data.csv                           # Dataset
├── output.csv / input.csv                  # Processed data samples
├── loan_default_model.h5                   # Trained model (saved weights)
├── plots/                                  # (Optional) EDA & results graphs
└── README.md                               # Project documentation
```

---
## 🙌 Acknowledgments

* **Lending Club** for providing the dataset.
* Techniques inspired by **imbalanced classification** practices in financial risk modeling.


