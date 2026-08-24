# ============================================================
# LOAN APPROVAL ML DASHBOARD
# STEP 14: ML MODELS + MODEL COMPARISON
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval ML Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("loan_approval_dataset.csv")

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove ID column
    df = df.drop("loan_id", axis=1)

    # Clean text columns
    df["education"] = df["education"].str.strip()
    df["self_employed"] = df["self_employed"].str.strip()
    df["loan_status"] = df["loan_status"].str.strip()

    # Encode education
    df["education"] = df["education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    # Encode self-employed
    df["self_employed"] = df["self_employed"].map({
        "Yes": 1,
        "No": 0
    })

    # Encode target
    df["loan_status"] = df["loan_status"].map({
        "Approved": 1,
        "Rejected": 0
    })

    return df


df = load_data()

# ============================================================
# PREPARE DATA
# ============================================================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# CREATE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    ),

    "Naive Bayes": GaussianNB()
}

# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    trained_models = {}
    results = {}

    for name, model in models.items():

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

        results[name] = {
            "Accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "Precision": precision_score(
                y_test,
                y_pred
            ),

            "Recall": recall_score(
                y_test,
                y_pred
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred
            )
        }

        trained_models[name] = model

    return trained_models, results


trained_models, results = train_models()

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

# ============================================================
# BEST MODEL
# ============================================================

best_model_name = results_df[
    "F1 Score"
].idxmax()

best_model = trained_models[
    best_model_name
]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Loan ML Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Dataset",
        "🤖 Model Comparison",
        "🔮 Prediction"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🏦 Loan Approval Prediction Dashboard"
    )

    st.write(
        "Machine Learning based Loan Approval System"
    )

    st.divider()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Applications",
            len(df)
        )

    with col2:

        st.metric(
            "Approved Loans",
            int((df["loan_status"] == 1).sum())
        )

    with col3:

        st.metric(
            "Rejected Loans",
            int((df["loan_status"] == 0).sum())
        )

    with col4:

        st.metric(
            "Best Model",
            best_model_name
        )

    st.divider()

    # Best Model
    st.subheader("🏆 Best Performing Model")

    best_col1, best_col2, best_col3, best_col4 = st.columns(4)

    with best_col1:
        st.metric(
            "Model",
            best_model_name
        )

    with best_col2:
        st.metric(
            "Accuracy",
            f"{results_df.loc[best_model_name, 'Accuracy']:.2%}"
        )

    with best_col3:
        st.metric(
            "Precision",
            f"{results_df.loc[best_model_name, 'Precision']:.2%}"
        )

    with best_col4:
        st.metric(
            "F1 Score",
            f"{results_df.loc[best_model_name, 'F1 Score']:.2%}"
        )

# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Features",
            df.shape[1] - 1
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Loan Status Distribution")

    status_data = pd.DataFrame({
        "Status": [
            "Rejected",
            "Approved"
        ],
        "Count": [
            int((df["loan_status"] == 0).sum()),
            int((df["loan_status"] == 1).sum())
        ]
    })

    st.bar_chart(
        status_data.set_index("Status")
    )

# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

elif page == "🤖 Model Comparison":

    st.title(
        "🤖 Machine Learning Model Comparison"
    )

    st.subheader("Performance Table")

    display_df = results_df.copy()

    st.dataframe(
        display_df.style.format(
            "{:.2%}",
            subset=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ),
        use_container_width=True
    )

    st.divider()

    # Accuracy Chart
    st.subheader("📊 Accuracy Comparison")

    st.bar_chart(
        results_df["Accuracy"]
    )

    # F1 Score Chart
    st.subheader("📊 F1 Score Comparison")

    st.bar_chart(
        results_df["F1 Score"]
    )

    st.divider()

    # Best Model
    st.success(
        f"🏆 Best Model: {best_model_name}"
    )

    st.write(
        f"**F1 Score:** "
        f"{results_df.loc[best_model_name, 'F1 Score']:.2%}"
    )

    # Confusion Matrix
    st.subheader(
        "🔥 Confusion Matrix - Best Model"
    )

    best_prediction = best_model.predict(
        X_test_scaled
    )

    cm = confusion_matrix(
        y_test,
        best_prediction
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Rejected",
            "Approved"
        ],
        yticklabels=[
            "Rejected",
            "Approved"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix - {best_model_name}"
    )

    st.pyplot(fig)

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Loan Approval Prediction"
    )

    st.info(
        f"Prediction Model: **{best_model_name}**"
    )

    st.subheader(
        "Enter Customer Details"
    )

    col1, col2 = st.columns(2)

    with col1:

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=2
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=5000000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=15000000
        )

        loan_term = st.number_input(
            "Loan Term",
            min_value=1,
            max_value=50,
            value=15
        )

    with col2:

        cibil = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=750
        )

        residential_assets = st.number_input(
            "Residential Asset Value",
            min_value=0,
            value=5000000
        )

        commercial_assets = st.number_input(
            "Commercial Asset Value",
            min_value=0,
            value=2000000
        )

        luxury_assets = st.number_input(
            "Luxury Asset Value",
            min_value=0,
            value=3000000
        )

        bank_assets = st.number_input(
            "Bank Asset Value",
            min_value=0,
            value=4000000
        )

    st.divider()

    if st.button(
        "🔮 Predict Loan Status",
        use_container_width=True
    ):

        # Convert input to numerical values

        education_value = (
            1 if education == "Graduate"
            else 0
        )

        self_employed_value = (
            1 if self_employed == "Yes"
            else 0
        )

        # Create customer dataframe

        new_customer = pd.DataFrame({

            "no_of_dependents": [dependents],

            "education": [education_value],

            "self_employed": [
                self_employed_value
            ],

            "income_annum": [income],

            "loan_amount": [loan_amount],

            "loan_term": [loan_term],

            "cibil_score": [cibil],

            "residential_assets_value": [
                residential_assets
            ],

            "commercial_assets_value": [
                commercial_assets
            ],

            "luxury_assets_value": [
                luxury_assets
            ],

            "bank_asset_value": [
                bank_assets
            ]
        })

        # Scale customer data

        new_customer_scaled = (
            scaler.transform(new_customer)
        )

        # Prediction

        prediction = best_model.predict(
            new_customer_scaled
        )

        probability = (
            best_model.predict_proba(
                new_customer_scaled
            )
        )

        st.divider()

        if prediction[0] == 1:

            st.success(
                "🎉 LOAN APPROVED"
            )

            st.metric(
                "Approval Probability",
                f"{probability[0][1]:.2%}"
            )

        else:

            st.error(
                "❌ LOAN REJECTED"
            )

            st.metric(
                "Rejection Probability",
                f"{probability[0][0]:.2%}"
            )

        st.write(
            f"### Model Used: {best_model_name}"
        )