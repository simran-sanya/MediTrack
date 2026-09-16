# MediTrack: 30-Day Readmission Risk & Clinical Decision Support

👉 **Live Interactive Dashboard:** [Click here to view!](https://your-streamlit-link-here)

MediTrack is a machine learning pipeline and interactive clinical dashboard designed to predict the 30-day readmission risk of diabetic patients. By leveraging advanced tree-based models and neural networks, it provides actionable insights and data-driven discharge guidelines to healthcare providers.

## Project Overview

This project includes a fully validated data preprocessing pipeline, model training across multiple algorithms, and a premium interactive web dashboard built with Streamlit.

**Models Evaluated:**
* Logistic Regression
* Random Forest
* Gradient Boosting (Selected for Deployment)
* Neural Baseline (MLP + Embeddings)

All models were strictly evaluated on a canonical test set ($N=19,869$ encounters) to ensure rigorous and fair comparison, completely eliminating data leakage during categorical encoding.

## Features

- **Patient Encounter Analysis:** Predicts readmission risk for individual patients using the Gradient Boosting model.
- **SHAP Explainability:** Provides feature-level explanations for why a patient received a specific risk score.
- **Cross-Model Comparison:** Evaluates and compares the ROC-AUC, PR-AUC, Precision, Recall, and F1-Scores across all four trained models.
- **AI Discharge Assistant:** A guideline-grounded assistant providing conditional discharge plans based on patient diagnosis.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MediTrack.git
   cd MediTrack
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```
   The interactive dashboard will automatically open in your default web browser!

## Data & Artifacts
The `results/` folder contains the serialized models (`.pkl`), prediction probabilities, frozen decision thresholds, and sample test data necessary for the dashboard to run without needing to retrain the models.
