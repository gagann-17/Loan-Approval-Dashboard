# 🏦 Loan Approval ML Dashboard

A Streamlit-based machine learning dashboard that predicts loan approval status and compares multiple classification models.

## Features

- Dataset overview and preview
- Loan approval/rejection statistics
- Comparison of six machine learning models
- Accuracy, precision, recall, and F1-score metrics
- Best-model selection based on F1 score
- Confusion matrix visualization
- Interactive loan approval prediction

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- Gaussian Naive Bayes

## Project Structure

```text
LoanApprovalMLDashboard/
├── app.py
├── loan_approval_dataset.csv
└── README.md
```

## Requirements

- Python 3.9 or later
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Open PowerShell in the project directory and run:

```powershell
python -m pip install streamlit pandas matplotlib seaborn scikit-learn
```

## Run the Application

```powershell
python -m streamlit run app.py
```

The dashboard will open in your browser at:

```text
http://localhost:8501
```

## Dataset

The application expects a file named `loan_approval_dataset.csv` in the project root.

The dataset contains applicant information such as:

- Number of dependents
- Education
- Employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Asset values
- Loan approval status

## Usage

Use the sidebar to navigate between:

1. **Dashboard** – View overall loan statistics and the best model.
2. **Dataset** – Explore the dataset and loan-status distribution.
3. **Model Comparison** – Compare model performance and view the confusion matrix.
4. **Prediction** – Enter applicant details and predict loan approval status.

## License

This project is intended for educational and demonstration purposes.
