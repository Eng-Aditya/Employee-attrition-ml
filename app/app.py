import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "attrition_pipeline.pkl"
CONFIG_PATH = BASE_DIR / "models" / "risk_config.pkl"
DATA_PATH = BASE_DIR / "data" / "Palo Alto Networks.csv"


# ============================================================
# LOAD MODEL, CONFIGURATION AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_config():
    return joblib.load(CONFIG_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
risk_config = load_config()
df = load_data()

threshold = risk_config["threshold"]

# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.title("📊 Employee Attrition")
st.sidebar.caption("Risk Scoring System")

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Dashboard",
        "👤 Employee Profile",
        "🏢 Department Analysis",
        "🔮 What-If Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write(
    f"👥 Employees: {len(df):,}"
)

st.sidebar.write(
    f"🎯 Prediction Threshold: {threshold:.2f}"
)

st.sidebar.write(
    "🤖 Model: Logistic Regression"
)

# ============================================================
# PREDICT ATTRITION PROBABILITY
# ============================================================

X = df.drop(columns=["Attrition"])

probabilities = model.predict_proba(X)[:, 1]


# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================

threshold = risk_config["threshold"]


def get_risk_category(probability):

    if probability < 0.20:
        return "Low"

    elif probability < threshold:
        return "Medium"

    else:
        return "High"


# ============================================================
# ADD PREDICTIONS TO DATAFRAME
# ============================================================

df["Risk Score"] = probabilities * 100

df["Risk Category"] = [
    get_risk_category(probability)
    for probability in probabilities
]


# ============================================================
# TITLE
# ============================================================

if page == "📊 Dashboard":
    st.title("📊 Employee Attrition Prediction & Risk Scoring System")

    st.caption(
        "Machine Learning–Based Employee Attrition Risk Analysis Dashboard"
    )

    st.divider()


    # ============================================================
    # KPI CALCULATIONS
    # ============================================================

    total_employees = len(df)

    high_risk = (df["Risk Category"] == "High").sum()

    medium_risk = (df["Risk Category"] == "Medium").sum()

    low_risk = (df["Risk Category"] == "Low").sum()

    actual_attrition = df["Attrition"].sum()

    attrition_rate = (
        actual_attrition / total_employees
    ) * 100


    # ============================================================
    # KPI CARDS
    # ============================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        st.metric(
            "Total Employees",
            total_employees
        )


    with col2:
        st.metric(
            "🔴 High Risk",
            high_risk
        )


    with col3:
        st.metric(
            "🟠 Medium Risk",
            medium_risk
        )


    with col4:
        st.metric(
            "🟢 Low Risk",
            low_risk
        )


    with col5:
        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.2f}%"
        )


    st.divider()


    # ============================================================
    # RISK DISTRIBUTION
    # ============================================================

    st.subheader("🎯 Employee Risk Distribution")


    risk_counts = (
        df["Risk Category"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Category",
        "Employees"
    ]


    fig_risk = px.pie(
        risk_counts,
        names="Risk Category",
        values="Employees",
        hole=0.4,
        title="Distribution of Employees by Risk Category"
    )


    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

    st.divider()

    # ============================================
    # DEPARTMENT RISK SUMMARY
    # ============================================

    st.subheader("🏢 Department Risk Summary")

    department_risk = (
        df.groupby("Department")["Risk Score"]
        .mean()
        .reset_index()
        .sort_values("Risk Score", ascending=False)
    )

    fig_department = px.bar(
        department_risk,
        x="Department",
        y="Risk Score",
        title="Average Attrition Risk Score by Department",
        labels={
            "Risk Score": "Average Risk Score (%)",
            "Department": "Department"
        },
        text="Risk Score"
    )

    fig_department.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_department.update_layout(
        yaxis_title="Average Risk Score (%)",
        xaxis_title="Department",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(
        fig_department,
        use_container_width=True
    )

    # ============================================================
    # DEPARTMENT ANALYSIS
    # ============================================================

    st.subheader("🏢 Department Analysis")


    department_data = (
        df.groupby("Department")
        .agg(
            Employees=("Department", "size"),
            Attrition_Rate=("Attrition", "mean")
        )
        .reset_index()
    )


    department_data["Attrition_Rate"] = (
        department_data["Attrition_Rate"] * 100
    )


    fig_department = px.bar(
        department_data,
        x="Department",
        y="Attrition_Rate",
        text="Attrition_Rate",
        title="Attrition Rate by Department",
        labels={
            "Attrition_Rate": "Attrition Rate (%)"
        }
    )


    fig_department.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )


    st.plotly_chart(
        fig_department,
        use_container_width=True
    )


    # ============================================================
    # HIGH-RISK EMPLOYEES
    # ============================================================

    st.subheader("🔴 High-Risk Employees")


    high_risk_employees = df[
        df["Risk Category"] == "High"
    ].copy()


    high_risk_employees = high_risk_employees.sort_values(
        "Risk Score",
        ascending=False
    )


    display_columns = [
        "Age",
        "Department",
        "JobRole",
        "JobSatisfaction",
        "OverTime",
        "MonthlyIncome",
        "Risk Score",
        "Risk Category"
    ]


    st.dataframe(
        high_risk_employees[display_columns],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# EMPLOYEE RISK PROFILE
# ============================================================

if page == "👤 Employee Profile":
    st.divider()

    st.subheader("👤 Employee Risk Profile")

    st.write(
        "Select an employee record to view their individual attrition risk."
    )


    # Employee selection
    employee_index = st.selectbox(
        "Select Employee Record",
        range(len(df)),
        format_func=lambda x: f"Employee Record {x + 1}"
    )


    # Get selected employee
    selected_employee = df.iloc[employee_index]


    # ------------------------------------------------------------
    # SELECTED EMPLOYEE RISK
    # ------------------------------------------------------------

    risk_score = selected_employee["Risk Score"]

    risk_category = selected_employee["Risk Category"]


    # ------------------------------------------------------------
    # DISPLAY RISK
    # ------------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}%"
        )


    with col2:

        st.metric(
            "Risk Category",
            risk_category
        )


    # ------------------------------------------------------------
    # RISK MESSAGE
    # ------------------------------------------------------------

    if risk_category == "High":

        st.error(
            "🔴 High Risk: This employee has a relatively high predicted probability of attrition."
        )

    elif risk_category == "Medium":

        st.warning(
            "🟠 Medium Risk: This employee has a moderate predicted probability of attrition."
        )

    else:

        st.success(
            "🟢 Low Risk: This employee has a relatively low predicted probability of attrition."
        )


    # ------------------------------------------------------------
    # EMPLOYEE DETAILS
    # ------------------------------------------------------------

    st.subheader("Employee Details")


    detail_columns = [
        "Age",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "OverTime",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]


    employee_details = selected_employee[detail_columns]


    st.dataframe(
        employee_details.to_frame(name="Value"),
        use_container_width=True
    )

    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    st.divider()

    st.subheader("🔍 Why is this employee at risk?")

    st.write(
        "SHAP explains which employee features contributed most "
        "to the model's prediction."
    )


    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    st.divider()

    st.subheader("🔍 Why is this employee at risk?")

    st.write(
        "SHAP explains which employee features contributed most "
        "to the model's prediction."
    )

    try:
        # Get preprocessing and machine learning model
        preprocessor = model.named_steps["preprocessor"]
        ml_model = model.named_steps["model"]

        # Get complete input data for selected employee
        employee_input = X.iloc[[employee_index]]

        # Transform employee data
        transformed_employee = preprocessor.transform(
            employee_input
        )

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Background data for SHAP
        background_data = X.sample(
            n=min(100, len(X)),
            random_state=42
        )

        background_transformed = preprocessor.transform(
            background_data
        )

        # Logistic Regression → LinearExplainer
        explainer = shap.LinearExplainer(
            ml_model,
            background_transformed
        )

        # Generate SHAP explanation
        shap_explanation = explainer(
            transformed_employee
        )

        employee_shap_values = shap_explanation.values[0]

        # Create SHAP dataframe
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": employee_shap_values
        })

        # Calculate importance
        shap_df["Importance"] = (
            shap_df["SHAP Value"].abs()
        )

        # Get top 10 features
        shap_df = (
            shap_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(10)
            .sort_values("SHAP Value")
        )

        # Create chart
        fig_shap = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            title="Top Factors Influencing This Prediction",
            labels={
                "SHAP Value": "Impact on Prediction",
                "Feature": "Feature"
            }
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

        st.info(
            "Positive SHAP values push the prediction toward "
            "higher attrition risk, while negative SHAP values "
            "push it toward lower risk."
        )

    except Exception as e:
        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )

# ============================================================
# DEPARTMENT-LEVEL RISK ANALYSIS
# ============================================================

if page == "🏢 Department Analysis":
    st.divider()

    st.subheader("🏢 Department-Level Risk Analysis")

    st.write(
        "Explore attrition and predicted risk across different departments."
    )


# ------------------------------------------------------------
# DEPARTMENT FILTER
# ------------------------------------------------------------

    departments = sorted(df["Department"].unique())

    selected_department = st.selectbox(
        "Select Department",
        departments
    )


# Filter department
    department_df = df[
        df["Department"] == selected_department
    ].copy()


# ------------------------------------------------------------
# DEPARTMENT KPIs
# ------------------------------------------------------------

    department_employees = len(department_df)

    department_attrition = (
        department_df["Attrition"].mean() * 100
    )

    department_avg_risk = (
        department_df["Risk Score"].mean()
    )


    department_high_risk = (
        department_df["Risk Category"] == "High"
    ).sum()


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "Employees",
            department_employees
        )


    with col2:
        st.metric(
            "Actual Attrition Rate",
            f"{department_attrition:.2f}%"
        )


    with col3:
        st.metric(
            "Average Risk Score",
            f"{department_avg_risk:.2f}%"
        )


    with col4:
        st.metric(
            "High-Risk Employees",
            department_high_risk
        )


# ------------------------------------------------------------
# RISK DISTRIBUTION FOR DEPARTMENT
# ------------------------------------------------------------

    st.subheader(
        f"Risk Distribution — {selected_department}"
    )


    department_risk_counts = (
        department_df["Risk Category"]
        .value_counts()
        .reset_index()
    )

    department_risk_counts.columns = [
        "Risk Category",
        "Employees"
    ]


    fig_department_risk = px.pie(
        department_risk_counts,
        names="Risk Category",
        values="Employees",
        hole=0.4,
        title=f"Risk Categories in {selected_department}"
    )


    st.plotly_chart(
        fig_department_risk,
        use_container_width=True
    )


# ------------------------------------------------------------
# JOB ROLE ANALYSIS
# ------------------------------------------------------------

    st.subheader(
        f"Job Role Analysis — {selected_department}"
    )


    role_data = (
        department_df
        .groupby("JobRole")
        .agg(
            Employees=("JobRole", "size"),
            Attrition_Rate=("Attrition", "mean"),
            Average_Risk=("Risk Score", "mean")
        )
        .reset_index()
    )


    role_data["Attrition_Rate"] = (
        role_data["Attrition_Rate"] * 100
    )


    fig_roles = px.bar(
        role_data,
        x="JobRole",
        y="Attrition_Rate",
        title="Attrition Rate by Job Role",
        labels={
            "Attrition_Rate": "Attrition Rate (%)",
            "JobRole": "Job Role"
        }
    )


    fig_roles.update_layout(
        xaxis_tickangle=-45
    )


    st.plotly_chart(
        fig_roles,
        use_container_width=True
    )

# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

if page == "🔮 What-If Analysis":
    st.divider()

    st.subheader("🔮 What-If Analysis")

    st.write(
        "Adjust selected employee factors to see how the "
        "predicted attrition risk changes."
    )
    what_if_index = st.selectbox(
        "Select Employee Record",
        range(len(X)),
        format_func=lambda x: f"Employee Record {x + 1}"
    )

    # ------------------------------------------------------------
    # CREATE A COPY OF THE SELECTED EMPLOYEE
    # ------------------------------------------------------------

    what_if_employee = X.iloc[[what_if_index]].copy()


    # ------------------------------------------------------------
    # INPUT CONTROLS
    # ------------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        what_if_job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                what_if_employee["JobSatisfaction"].iloc[0]
            )
        )
    

        what_if_environment = st.slider(
            "Environment Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                what_if_employee["EnvironmentSatisfaction"].iloc[0]
            )
        )


    with col2:

        what_if_overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"],
            index=(
                1
                if what_if_employee["OverTime"].iloc[0] == "Yes"
                else 0
            )
        )

        what_if_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=100000,
            value=int(
                what_if_employee["MonthlyIncome"].iloc[0]
            ),
            step=500
        )


    # ------------------------------------------------------------
    # UPDATE EMPLOYEE DATA
    # ------------------------------------------------------------

    what_if_employee.loc[
        what_if_employee.index[0],
        "JobSatisfaction"
    ] = what_if_job_satisfaction


    what_if_employee.loc[
        what_if_employee.index[0],
        "EnvironmentSatisfaction"
    ] = what_if_environment


    what_if_employee.loc[
        what_if_employee.index[0],
        "OverTime"
    ] = what_if_overtime


    what_if_employee.loc[
        what_if_employee.index[0],
        "MonthlyIncome"
    ] = what_if_income


    # ------------------------------------------------------------
    # PREDICT NEW RISK
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # ORIGINAL EMPLOYEE RISK
    # ------------------------------------------------------------

    original_probability = model.predict_proba(
        X.iloc[[what_if_index]]
    )[0][1]

    original_score = original_probability * 100

    if original_probability >= 0.60:
        original_category = "High"

    elif original_probability >= threshold:
        original_category = "Medium"

    else:
        original_category = "Low"

    try:

        what_if_probability = model.predict_proba(
            what_if_employee
        )[0][1]

        what_if_score = what_if_probability * 100

        risk_change = what_if_score - original_score

        # Determine risk category
        if what_if_probability >= 0.60:

            what_if_category = "High"

        elif what_if_probability >= threshold:

            what_if_category = "Medium"

        else:

            what_if_category = "Low"

        # --------------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------------

        st.subheader("📊 What-If Prediction")
        st.write("### 🔄 Before vs What-If")

        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

        with comparison_col1:
            st.metric(
                "Original Risk",
                f"{original_score:.2f}%"
            )

        with comparison_col2:
            st.metric(
                "What-If Risk",
                f"{what_if_score:.2f}%"
            )

        with comparison_col3:
            st.metric(
                "Risk Change",
                f"{risk_change:+.2f} pts"
            )
        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Attrition Risk",
                f"{what_if_score:.2f}%"
            )

        with result_col2:

            st.metric(
                "Risk Category",
                what_if_category
            )

        st.progress(
            min(what_if_probability, 1.0)
        )

        st.caption(
            "This simulation changes selected employee attributes "
            "and recalculates the model prediction. It represents "
            "model sensitivity, not a causal estimate."
        )

    except Exception as e:

        st.error(
            f"What-If prediction could not be generated: {e}"
        )