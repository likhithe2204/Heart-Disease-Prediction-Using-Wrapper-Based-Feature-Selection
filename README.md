# 🩺🌐 Heart Disease Predictive Model
### Using Recursive Feature Elimination (RFE) and Machine Learning Algorithms

A research-driven machine learning system to predict heart disease based on clinical attributes. Developed by a 6-member team as part of a 3-month academic research project, this system leverages advanced ML techniques and feature engineering for accurate predictions. 🧠

---

## 📊 Overview

According to the **World Health Organization**, cardiovascular diseases (CVDs) account for **32% of all global deaths**. Early prediction is critical for prevention.

This project uses **Logistic Regression, SVM, Random Forest, Decision Tree, and K-Nearest Neighbors**, paired with **Recursive Feature Elimination (RFE)** and **Linear Discriminant Analysis (LDA)** for optimal performance.

> 🚀 **Achieved 97.33% accuracy** using KNN with RFE-selected features.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, joblib
- **ML Techniques**: RFE, LDA, Cross-Validation
- **Environment**: Google Colab / Jupyter Notebook

---

## 📁 Dataset

The dataset contains **1025 samples** and **14 features**, including:
- Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, etc.
- Target: `1` (disease), `0` (no disease)

✔️ No missing values  
✔️ Cleaned for duplicates  
✔️ Balanced classes

---

## 🧠 ML Models Used

| Model               | Feature Selector | Accuracy    |
|--------------------|------------------|-------------|
| Logistic Regression| RFE              | ~85%        |
| SVM                | RFE + Standardize| ~89%        |
| Decision Tree      | RFE              | ~90%        |
| Random Forest      | RFE              | ~93%        |
| **KNN**            | **RFE**          | **97.33%**  |

📌 All models validated with **5-Fold Cross-Validation**.

---

## 🔎 Feature Engineering

- ✅ **Recursive Feature Elimination (RFE)** to select top 10 influential features
- ✅ **Linear Discriminant Analysis (LDA)** for dimensionality reduction (1D)
- ✅ Correlation matrix for feature understanding
- ✅ PCA for visualization

---

## 📈 Visualizations

- Histograms of feature distribution
- Correlation heatmaps
- KNN accuracy vs. neighbor count
- Decision Tree depth vs. accuracy
- SVM kernel comparison

---

## 🧪 Sample Code Snippet 

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = data.drop(['target'], axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Accuracy:", knn.score(X_test, y_test))
```

---

## 🏁 Results

* 🎯 Accuracy: 97.33%

* 🧮 Precision: 0.64

* 🔁 Recall: 0.76

* 🧪 F1-Score: 0.70

These metrics show strong reliability for real-world prediction systems.

---

## 👨‍💻 Team Members

* E. Likhith

* G. Chandhan

* K. Hasith

* P. Balaji

* N. Harshith

* B. Sri Ram

---

## 📄 License
This project is for academic and research purposes. Feel free to explore and build upon it.
