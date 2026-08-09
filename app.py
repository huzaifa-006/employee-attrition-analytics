import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "employee_attrition_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "employee_attrition_features.pkl"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    features = joblib.load(
        FEATURE_PATH
    )

    return model, features


model, feature_columns = load_model()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_attrition(employee_data):

    employee_df = pd.DataFrame(
        [employee_data]
    )

    employee_df = employee_df[
        feature_columns
    ]

    prediction = model.predict(
        employee_df
    )[0]

    probability = model.predict_proba(
        employee_df
    )[0, 1]

    if probability < 0.30:
        risk = "Low"

    elif probability < 0.60:
        risk = "Medium"

    else:
        risk = "High"

    return prediction, probability, risk


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title(
    "📊 Employee Attrition Analytics"
)

st.write(
    "Predict employee attrition risk using a trained machine learning model."
)

st.divider()


# --------------------------------------------------
# Employee Information
# --------------------------------------------------

st.header(
    "Employee Information"
)


employee_data = {}


# --------------------------------------------------
# Employee Basic Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    employee_data["Age"] = age


with col2:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    employee_data["Gender"] = (
        1 if gender == "Male" else 0
    )


with col3:

    experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=40,
        value=5
    )

    employee_data[
        "Experience_Years"
    ] = experience


# --------------------------------------------------
# Job Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    salary = st.number_input(
        "Salary",
        min_value=0,
        max_value=1000000,
        value=80000,
        step=5000
    )

    employee_data[
        "Salary"
    ] = salary


with col2:

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=30,
        value=5
    )

    employee_data[
        "Projects_Completed"
    ] = projects


with col3:

    overtime = st.selectbox(
        "Overtime",
        ["No", "Yes"]
    )

    employee_data[
        "Overtime"
    ] = (
        1 if overtime == "Yes" else 0
    )


# --------------------------------------------------
# Promotion Information
# --------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    promotion = st.selectbox(
        "Promotion in Last 5 Years",
        ["No", "Yes"]
    )

    employee_data[
        "Promotion_Last_5Yrs"
    ] = (
        1 if promotion == "Yes" else 0
    )


with col2:

    remote_work = st.selectbox(
        "Remote Work",
        ["No", "Yes"]
    )

    employee_data[
        "Remote_Work"
    ] = (
        1 if remote_work == "Yes" else 0
    )


# --------------------------------------------------
# Education
# --------------------------------------------------

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)

education_mapping = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}

employee_data[
    "Education_Level"
] = education_mapping[
    education
]


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "Predict Attrition Risk",
    type="primary"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in employee_data
    ]

    if missing_features:

        st.error(
            "Missing model features:"
        )

        st.write(
            missing_features
        )

    else:

        prediction, probability, risk = (
            predict_attrition(
                employee_data
            )
        )

        st.header(
            "Prediction Result"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            if prediction == 1:

                st.error(
                    "Predicted: Employee may leave"
                )

            else:

                st.success(
                    "Predicted: Employee may stay"
                )


        with col2:

            st.metric(
                "Attrition Probability",
                f"{probability:.2%}"
            )


        with col3:

            st.metric(
                "Risk Level",
                risk
            )


        st.divider()


        # --------------------------------------------------
        # Business Interpretation
        # --------------------------------------------------

        st.subheader(
            "Business Interpretation"
        )

        if risk == "High":

            st.warning(
                "This employee has a relatively high "
                "predicted attrition risk. HR may consider "
                "reviewing factors such as workload, career "
                "development, compensation, and engagement."
            )

        elif risk == "Medium":

            st.info(
                "This employee has a moderate predicted "
                "attrition risk. Additional monitoring and "
                "employee engagement activities may be useful."
            )

        else:

            st.success(
                "This employee has a relatively low "
                "predicted attrition risk."
            )


        st.caption(
            "Model predictions are estimates and should "
            "not be used as the sole basis for employment decisions."
        )