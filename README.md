# 💼 Banking Term Deposit Prediction

This project uses machine learning to predict whether a bank client will subscribe to a term deposit based on personal, financial, and contact-related attributes. The dataset is obtained from a real-world direct marketing campaign conducted by a Portuguese bank.

---

## 📁 Dataset

**Source**: UCI Machine Learning Repository  
**File**: `bank-full.csv`  
**Target Variable**: `y` — indicates if the client subscribed to a term deposit (`yes`/`no`)

---

## 🧠 Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## ⚙️ Machine Learning Workflow
 
. **EDA (Exploratory Data Analysis)**  
   - Class imbalance visualization  
   - Correlation analysis and boxplots

. **Feature Engineering**  
   - Label encoding and one-hot encoding  
   - Date and campaign-related feature transformation

. **Resampling**  
   - Applied **SMOTE** to handle class imbalance

. **Model Training & Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix

---

## 🔍 Key Insights

- **XGBoost** achieved the best F1-score on the minority class.
- Handling class imbalance significantly improved recall for subscribed clients.
- SMOTE proved effective for resampling without losing information.

---

## 📊 Results Summary (Before SMOTE)

| Model              | Accuracy | Recall (Subscribed) | F1-score (Subscribed) |
|-------------------|----------|----------------------|------------------------|
| Logistic Regression | 90%     | 31%                 | 0.42                   |
| Random Forest      | 91%     | 38%                  | 0.48                   |
| XGBoost            | 91%     | 48%                  | 0.54                   |

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Model explainability with SHAP
- Deployment as a web service (e.g., with Flask or Streamlit)

---

## 📦 Installation

```bash
pip install -r requirements.txt
