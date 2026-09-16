import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="MediTrack System", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(to bottom right, #0F172A, #1E293B);
        color: #F8FAFC;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        padding-bottom: 1rem;
    }
    
    /* Metric Cards (Glassmorphism) */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.3);
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("MediTrack: 30-Day Readmission Risk & Clinical Decision Support")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    import os
    def get_path(filename):
        # Fallback to current directory if results/ folder isn't found on GitHub
        return f"results/{filename}" if os.path.exists(f"results/{filename}") else filename

    model = joblib.load(get_path("gb_model.pkl"))
    X_test = pd.read_csv(get_path("X_test_sample.csv"), index_col=0)
    y_test = pd.read_csv(get_path("y_test_sample.csv"), index_col=0).squeeze("columns")
    y_pred_proba = np.load(get_path("y_pred_proba.npy"))
    threshold_df = pd.read_csv(get_path("threshold_matrix.csv"))
    dashboard_df = pd.read_csv(get_path("dashboard_diagnosis_summary.csv"))
    with open(get_path("chosen_threshold.txt")) as f:
        threshold = float(f.read().strip())
    
    # Pre-calculated final metrics from the notebook run for the comparison tab
    metrics_data = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.66, "ROC-AUC": 0.6347, "PR-AUC": 0.2023, "Precision": 0.17, "Recall": 0.50, "F1": 0.25},
        {"Model": "Random Forest", "Accuracy": 0.72, "ROC-AUC": 0.6348, "PR-AUC": 0.2029, "Precision": 0.18, "Recall": 0.42, "F1": 0.26},
        {"Model": "Gradient Boosting", "Accuracy": 0.75, "ROC-AUC": 0.6405, "PR-AUC": 0.2067, "Precision": 0.20, "Recall": 0.39, "F1": 0.26},
        {"Model": "Neural Baseline", "Accuracy": 0.73, "ROC-AUC": 0.6729, "PR-AUC": 0.2289, "Precision": 0.20, "Recall": 0.48, "F1": 0.29}
    ])
    
    return model, X_test, y_test, y_pred_proba, threshold_df, dashboard_df, threshold, metrics_data

model, X_test, y_test, y_pred_proba, threshold_df, dashboard_df, THRESHOLD, metrics_data = load_artifacts()

X_test_float = X_test.astype(float)
explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], X_test_float.sample(100, random_state=42))

DISCHARGE_GUIDELINES = {
    "Circulatory": "7-day tele-health check, blood pressure log review, medication reconciliation.",
    "Respiratory": "5-day check-in, inhaler technique review, smoking-cessation resource offer.",
    "Diabetes": "7-day glucose log review, medication reconciliation, dietitian referral.",
}
FORBIDDEN_INTENTS = ["diagnose", "prescribe", "what medicine", "treatment plan", "which drug"]

tab1, tab4, tab2, tab3 = st.tabs(["Patient Risk Assessment", "Cross-Model Comparison", "AI Discharge Assistant", "Clinical Dashboard"])

# ---------- TAB 1: Patient risk + SHAP explanation ----------
with tab1:
    st.header("Patient Encounter Analysis")
    st.caption(f"Powered by Gradient Boosting | Operating decision threshold: **{THRESHOLD:.2f}**")

    encounter_ids = X_test.index.tolist()
    selected_id = st.selectbox("Select Patient Encounter (Test-Set ID):", encounter_ids[:200])

    patient_row = X_test.loc[[selected_id]]
    risk_score = float(model.predict_proba(patient_row)[:, 1][0])
    is_high_risk = risk_score >= THRESHOLD

    col1, col2 = st.columns([1, 2])
    with col1:
        # Gauge Chart for Risk Score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            title = {'text': "Readmission Risk (%)", 'font': {'size': 20, 'color': 'white'}},
            number = {'font': {'color': 'white'}, 'suffix': '%'},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#EF4444" if is_high_risk else "#10B981"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, THRESHOLD*100], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [THRESHOLD*100, 100], 'color': "rgba(239, 68, 68, 0.2)"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': THRESHOLD*100}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if is_high_risk:
            st.error("🚨 HIGH RISK FLAGGED — Clinical Intervention Recommended")
        else:
            st.success("✅ LOW RISK — Standard Discharge Protocol")

    with col2:
        st.subheader("Patient Clinical Snapshot")
        snapshot = patient_row[["time_in_hospital", "number_inpatient", "number_emergency", "num_medication_changes"]].T
        snapshot.columns = ["Value"]
        st.dataframe(snapshot, use_container_width=True)

    st.markdown("---")
    st.subheader("Why this score? (SHAP Factor-Level Explanation)")
    
    with st.spinner("Calculating SHAP values..."):
        shap_values = explainer(patient_row.astype(float))
        top_idx = np.argsort(np.abs(shap_values[0].values))[::-1][:5]
        
        shap_data = []
        for idx in top_idx:
            val = shap_values[0].values[idx]
            shap_data.append({
                "Feature": patient_row.columns[idx],
                "Patient Value": patient_row.iloc[0, idx],
                "Impact on Risk": val,
                "Direction": "Increased Risk" if val > 0 else "Decreased Risk"
            })
        
        shap_df = pd.DataFrame(shap_data)
        
        # Plotly bar chart for SHAP
        fig_shap = px.bar(shap_df, x="Impact on Risk", y="Feature", orientation='h', 
                          color="Direction", color_discrete_map={"Increased Risk": "#EF4444", "Decreased Risk": "#10B981"},
                          title="Top 5 Features Influencing the Prediction")
        fig_shap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                               yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)

# ---------- TAB 4: Cross-Model Comparison ----------
with tab4:
    st.header("Cross-Model Performance Evaluation")
    st.caption("Validating all models on the exact same canonical Test Set (N=19,869)")
    
    st.dataframe(metrics_data, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_roc = px.bar(metrics_data, x="Model", y="ROC-AUC", color="Model", title="ROC-AUC Comparison",
                         color_discrete_sequence=["#38BDF8", "#818CF8", "#C084FC", "#F472B6"])
        fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        fig_roc.update_yaxes(range=[0.5, 0.75])
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col2:
        fig_f1 = px.bar(metrics_data, x="Model", y="F1", color="Model", title="F1-Score Comparison (Positive Class)",
                        color_discrete_sequence=["#38BDF8", "#818CF8", "#C084FC", "#F472B6"])
        fig_f1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_f1, use_container_width=True)

# ---------- TAB 2: RAG assistant with guardrails ----------
with tab2:
    st.header("Guideline-Grounded Discharge Assistant")
    st.caption("This assistant declines diagnostic and prescriptive requests and defers to the clinician.")

    diagnosis_choice = st.selectbox("Primary diagnosis group for this patient:", list(DISCHARGE_GUIDELINES.keys()))
    user_query = st.text_input("Ask the assistant (e.g. 'Summarize discharge guidance for this patient'):")

    if st.button("Submit Query"):
        if any(w in user_query.lower() for w in FORBIDDEN_INTENTS):
            st.error("🚨 **GUARDRAIL TRIGGERED:** I cannot diagnose conditions or prescribe medications. Please defer to the attending clinician for all medical decisions.")
        else:
            guideline = DISCHARGE_GUIDELINES[diagnosis_choice]
            st.info(f"**Discharge summary for care team** ({diagnosis_choice})\n\n{guideline}\n\n*(Source: MediTrack Discharge Protocols 2026)*")

# ---------- TAB 3: Clinical dashboard ----------
with tab3:
    st.header("Clinical Quality Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Readmission Rate by Diagnosis")
        fig1 = px.bar(dashboard_df, x="Diagnosis Category", y="Readmission Rate (%)", color="Diagnosis Category",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Avg Length of Stay by Diagnosis")
        fig2 = px.bar(dashboard_df, x="Diagnosis Category", y="Avg Length of Stay (Days)", color="Diagnosis Category",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("High-Risk Cohort (Current Threshold)")
    cohort = X_test.copy()
    cohort["risk_score"] = model.predict_proba(X_test)[:, 1]
    cohort = cohort[cohort["risk_score"] >= THRESHOLD].sort_values("risk_score", ascending=False)
    
    st.info(f"**{len(cohort)}** out of **{len(X_test)}** test encounters flagged as high-risk ({len(cohort) / len(X_test) * 100:.1f}%)")
    st.dataframe(cohort[["time_in_hospital", "number_inpatient", "number_emergency", "risk_score"]].head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("Threshold Sensitivity Analysis")
    st.dataframe(threshold_df, use_container_width=True)
