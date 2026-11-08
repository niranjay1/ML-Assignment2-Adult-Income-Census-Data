import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Income Classification")
st.caption("NAME - BHARATI NIRANJAY DEVIDAS | BITS ID 2026AK05339 | BITS PILANI - M.TECH AIML | MACHINE LEARNING ASSIGNMENT 2")
st.write(
    "Predict whether an individual's annual income falls into the <=50K or >50K category using "
    "census data and five machine learning classification algorithms."
)
st.divider()

# Saved model filenames
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Upload and model selection: matching heading sizes
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload test data")
    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )

with col2:
    st.subheader("Select Classification Model")
    selected_model = st.selectbox(
        "Choose Model",
        list(model_files.keys()),
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            "age", "workclass", "fnlwgt", "education",
            "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain",
            "capital_loss", "hours_per_week", "native_country",
            "income"
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            st.stop()

        if df.empty:
            st.error("Uploaded dataset is empty.")
            st.stop()

        X = df.drop(columns=["income"])
        y = (
            df["income"].astype(str).str.strip().str.rstrip(".")
        )

        if not y.isin(["<=50K", ">50K"]).all():
            st.error("Income must contain only <=50K or >50K.")
            st.stop()

        # Load selected saved pipeline
        model_path = os.path.join("model", model_files[selected_model])

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = joblib.load(model_path)

        # Generate predictions
        predictions = model.predict(X)
        y_binary = (y == ">50K").astype(int)
        pred_binary = (
            pd.Series(predictions).astype(str) == ">50K"
        ).astype(int)

        # Continuous scores for ROC-AUC
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.decision_function(X)

        # Six evaluation metrics
        accuracy = accuracy_score(y_binary, pred_binary)
        auc = roc_auc_score(y_binary, scores)
        precision = precision_score(
            y_binary, pred_binary, zero_division=0
        )
        recall = recall_score(
            y_binary, pred_binary, zero_division=0
        )
        f1 = f1_score(y_binary, pred_binary, zero_division=0)
        mcc = matthews_corrcoef(y_binary, pred_binary)

        # Dataset information
        st.subheader("Dataset Information")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Input Features", len(X.columns))
        c3.metric("Total Columns", len(df.columns))

        st.success(
            f"Evaluation completed successfully using {selected_model}!"
        )

        # Model performance
        st.subheader("Model Performance")
        st.caption(f"Selected Model: {selected_model}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        st.divider()

        # Confusion matrix and classification report
        # No additional "Prediction Analysis" heading
        col1, col2 = st.columns([1, 1])

        cm = confusion_matrix(
            y_binary, pred_binary, labels=[0, 1]
        )

        with col1:
            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                annot_kws={"size": 10},
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"],
                cbar=False,
                ax=ax
            )
            ax.set_xlabel("Predicted Income", fontsize=9)
            ax.set_ylabel("Actual Income", fontsize=9)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with col2:
            st.subheader("Classification Report")

            report = classification_report(
                y_binary,
                pred_binary,
                target_names=["<=50K", ">50K"],
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )

        # Sample predictions
        st.divider()
        st.subheader("Sample Predictions")

        output = df.copy()
        output["Predicted Income"] = predictions

        st.dataframe(
            output.head(20),
            use_container_width=True
        )

        # Download predictions
        csv_output = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_output,
            file_name="adult_income_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred during evaluation: {e}")

else:
    st.info("Please upload test_data.csv to begin evaluation.")
    st.markdown(
        """
        **Available classification models:**

        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbours
        4. Gaussian Naive Bayes
        5. Support Vector Machine
        """
    )

