# Employee Attrition Prediction and Risk Scoring System

A machine learning-based web application for predicting employee attrition risk and analyzing workforce risk using employee data.

## 📌 Project Overview

This project uses machine learning to estimate the probability that an employee may leave an organization.

The system provides:

- Employee-level attrition risk prediction
- Risk categorization into Low, Medium, and High
- Department-level risk analysis
- What-If analysis for selected employee factors
- Interactive dashboard
- Model-based risk scoring
- SHAP-based model explanations

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn
- SHAP
- Plotly
- Streamlit
- Joblib

## 🤖 Machine Learning

The application currently uses:

**Logistic Regression**

The model predicts the probability of employee attrition.

The prediction threshold used by the application is:

**0.40**

Risk categories:

- 🟢 Low Risk
- 🟠 Medium Risk
- 🔴 High Risk

## 📊 Application Features

### 1. Dashboard

Provides an overall view of the workforce, including:

- Total employees
- High-risk employees
- Medium-risk employees
- Low-risk employees
- Overall attrition rate
- Employee risk distribution

### 2. Employee Risk Profile

Allows users to select an employee record and view:

- Predicted attrition risk score
- Risk category
- Employee details
- Risk explanation

### 3. Department Analysis

Provides department-level analysis including:

- Number of employees
- Actual attrition rate
- Average predicted risk score
- High-risk employee count
- Risk distribution by department

### 4. What-If Analysis

Allows selected employee factors to be changed and the predicted risk to be recalculated.

The application compares:

- Original Risk
- What-If Risk
- Risk Change

This helps demonstrate model sensitivity to selected employee attributes.

## 🚀 How to Run the Application

### 1. Clone the repository

```bash
git clone https://github.com/Eng-Aditya/Employee-attrition-ml.git