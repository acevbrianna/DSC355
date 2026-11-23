import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

#Set up the dashboard
st.set_page_config(
    page_title="Predicting Donor Churn for Nonprofit Organizations",
    layout="wide"
)

#Display dashboard title
st.title("Predicting Donor Churn for Nonprofit Organizations")
st.write("Logistic Regression Classification")

#Load the saved model and test data
dashboard_data = joblib.load("week7_dashboard_data.joblib")

model = dashboard_data["model"]
X_test = dashboard_data["X_test"]
y_test = dashboard_data["y_test"]

#Generate model predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#Calculate model evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

#Display model performance
st.header("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1-Score", f"{f1:.4f}")
col5.metric("ROC-AUC", f"{auc:.4f}")

#Explain model limitations
st.info(
    "The model performed only slightly better than "
    "the majority-class baseline."
)

#Display confusion matrix
st.header("Confusion Matrix")

fig, ax = plt.subplots()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=[0, 1],
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)
plt.close(fig)

#Create interactive prediction section
st.header("Make a Prediction")

st.write(
    "Select a test record and adjust the feature values."
)

#Select a test record
record = st.selectbox(
    "Choose a Test Record",
    range(len(X_test)),
    format_func=lambda x: f"Record {x + 1}"
)

selected_data = X_test.iloc[record]

#Create predictor input fields
input_values = {}

with st.form("prediction_form"):

    for column in X_test.columns:

        if pd.api.types.is_numeric_dtype(X_test[column]):

            value = selected_data[column]

            if pd.isna(value):
                value = X_test[column].median()

            input_values[column] = st.number_input(
                column,
                value=float(value)
            )

        else:

            options = sorted(
                X_test[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if X_test[column].isna().any():
                options.append("(missing)")

            current_value = selected_data[column]

            if pd.isna(current_value):
                current_value = "(missing)"
            else:
                current_value = str(current_value)

            input_values[column] = st.selectbox(
                column,
                options,
                index=(
                    options.index(current_value)
                    if current_value in options
                    else 0
                )
            )

    #Create prediction button
    predict_button = st.form_submit_button(
        "Predict"
    )

#Generate prediction
if predict_button:

    #Prepare selected input data
    input_df = pd.DataFrame(
        [input_values],
        columns=X_test.columns
    )

    #Restore missing values
    input_df = input_df.replace(
        "(missing)",
        np.nan
    )

    #Calculate prediction and probabilities
    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    #Display prediction results
    st.subheader(
        "Prediction Results"
    )

    st.success(
        f"Predicted Class: {prediction}"
    )

    #Display prediction probabilities
    probability_df = pd.DataFrame(
        {
            "Class": [
                str(x)
                for x in model.classes_
            ],
            "Probability": probabilities
        }
    ).set_index(
        "Class"
    )

    st.bar_chart(
        probability_df
    )

    #Display input data
    st.subheader(
        "Input Data"
    )

    st.dataframe(
        input_df
    )

#Add dashboard footer
st.write("---")

st.caption(
    "Predictive Model Dashboard"
)
