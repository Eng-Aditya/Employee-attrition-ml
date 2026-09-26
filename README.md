# 📊 Employee Attrition Prediction and Risk Scoring System

A machine learning-based web application for predicting employee attrition risk and providing interpretable employee-level risk insights.

The system combines machine learning, risk scoring, SHAP explainability, and an interactive Streamlit dashboard to help analyze employee attrition patterns.

---

## 🚀 Live Application

🌐 **Streamlit App:**

https://employee-attrition-ml-figlqr98zimvucvhxkkl.streamlit.app/

---

## 📌 Project Overview

Employee attrition is an important business problem because unexpected employee departures can affect productivity, hiring costs, workforce planning, and organizational stability.

This project develops an end-to-end machine learning system that:

- Predicts employee attrition probability
- Converts predictions into Low, Medium, and High risk categories
- Provides employee-level risk profiles
- Explains predictions using SHAP
- Analyzes attrition risk across departments
- Allows users to perform What-If simulations
- Provides an interactive Streamlit dashboard

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze employee attrition patterns.
2. Prepare and preprocess employee data for machine learning.
3. Train classification models for attrition prediction.
4. Handle class imbalance.
5. Evaluate model performance.
6. Convert prediction probabilities into employee risk scores.
7. Explain individual predictions using SHAP.
8. Build an interactive business-oriented dashboard.
9. Deploy the application using Streamlit.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-Learn | Machine learning and preprocessing |
| XGBoost | Model experimentation |
| SMOTE | Class imbalance handling |
| SHAP | Model explainability |
| Streamlit | Web application |
| Plotly | Interactive visualizations |
| Git & GitHub | Version control and project hosting |

---

## 📂 Project Structure

```text
Employee-attrition-ml/
│
├── app/
│   └── app.py
│
├── data/
│   └── Palo Alto Networks.csv
│
├── models/
│   ├── attrition_pipeline.pkl
│   └── risk_config.pkl
│
├── notebook/
│   └── EDA.ipynb
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The project uses an employee dataset containing demographic, job-related, compensation, satisfaction, and work-environment attributes.

The target variable is:

```text
Attrition
```

where:

```text
Yes → Employee attrition
No  → Employee remains
```

The dataset contains approximately **1,470 employee records**.

---

## ⚙️ Data Preprocessing

The preprocessing workflow includes:

- Missing-value checks
- Removal of unnecessary columns
- Categorical feature encoding
- Numerical feature preprocessing
- Feature scaling where required
- Stratified train-test splitting
- Class imbalance handling

Columns such as employee identifiers and constant-value fields were excluded where appropriate.

---

## 🧠 Feature Engineering

Additional features were created to improve analysis and business interpretation, including:

- Income Per Year
- Promotion Delay
- Engagement Score
- Is Stressed
- Is New Manager

These features provide additional information about employee career progression, engagement, and workplace conditions.

---

## 🤖 Machine Learning

Multiple classification approaches were considered, including:

- Logistic Regression
- Random Forest
- XGBoost

The deployed application currently uses:

**Logistic Regression**

The trained preprocessing pipeline and model are stored in the `models/` directory.

---

## 📈 Risk Scoring

The model generates an attrition probability for each employee.

The probability is converted into a percentage-based risk score.

The current application uses a prediction threshold of:

```text
0.40
```

Risk categories are represented as:

```text
Low
Medium
High
```

The risk score is intended as a predictive indicator and should not be interpreted as a certainty that an employee will leave.

---

## 🔍 SHAP Explainability

SHAP is used to explain individual employee predictions.

The application identifies factors that contribute toward:

- Higher predicted attrition risk
- Lower predicted attrition risk

Examples of employee-level factors may include:

- Job Satisfaction
- Overtime
- Business Travel
- Job Involvement
- Performance Rating
- Distance From Home
- Years Since Last Promotion
- Number of Companies Worked
- Relationship Satisfaction

The application presents these factors using user-friendly feature names.

---

## 🖥️ Application Features

### 1. 📊 Dashboard

Provides an overall workforce risk overview including:

- Total employees
- High-risk employees
- Medium-risk employees
- Low-risk employees
- Overall attrition rate
- Employee risk distribution

---

### 2. 👤 Employee Profile

Allows users to select an employee record and view:

- Employee information
- Predicted attrition probability
- Risk score
- Risk category
- SHAP-based explanation
- Factors increasing predicted risk
- Factors reducing predicted risk

---

### 3. 🏢 Department Analysis

Provides department-level risk analysis including:

- Employee count
- Actual attrition rate
- Average predicted risk
- High-risk employee count

This helps identify differences in attrition risk across organizational groups.

---

### 4. 🔮 What-If Analysis

Allows users to modify selected employee attributes and observe how the model prediction changes.

The application provides:

- Original risk score
- What-If risk score
- Risk change
- Updated risk category

This feature demonstrates model sensitivity to selected employee attributes.

> What-If results represent model sensitivity and should not be interpreted as causal effects.

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Eng-Aditya/Employee-attrition-ml.git
```

### 2. Navigate to the project

```bash
cd Employee-attrition-ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app/app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

## 🌐 Deployment

The application is deployed using Streamlit.

Live application:

https://employee-attrition-ml-figlqr98zimvucvhxkkl.streamlit.app/

The deployed application automatically uses the code and model files available in the GitHub repository.

---

## ⚠️ Limitations

This project is intended for educational and analytical purposes.

Important limitations include:

- Model predictions are probabilistic rather than certain outcomes.
- SHAP explanations describe model behavior rather than causal relationships.
- What-If analysis represents model sensitivity rather than guaranteed business outcomes.
- Employee attrition can be influenced by factors that are not present in the dataset.
- Predictions should not be used as the sole basis for employment decisions.

---

## 🔮 Future Improvements

Potential future improvements include:

- Model hyperparameter tuning
- Advanced model comparison
- Model monitoring
- Automated retraining
- Fairness and bias analysis
- Additional workforce analytics
- Time-based attrition forecasting
- More advanced intervention simulations
- Authentication and role-based access
- Cloud database integration

---

## 👨‍💻 Author

**Aditya**

GitHub:

https://github.com/Eng-Aditya

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow:

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Data Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Risk Scoring
  ↓
SHAP Explainability
  ↓
Interactive Streamlit Dashboard
  ↓
Deployment
```

The final system combines machine learning prediction with explainable risk analysis in an interactive business-oriented application.