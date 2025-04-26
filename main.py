import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import io
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import hdbscan
from datetime import datetime

# Set Page Config
st.set_page_config(page_title=" Agile Task Predictor", layout="wide")

# Constants
DEFAULT_DATA_PATH = "feature_engineered_dataset.csv"
CLASSIFIER_MODEL_PATH = "xgb_classifier_pipeline.joblib"
REGRESSOR_MODEL_PATH = "xgb_regressor_pipeline.joblib"
UPLOAD_DIR = "uploads"

#  Ensure Upload Directory Exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Check Models
if not (os.path.exists(CLASSIFIER_MODEL_PATH) and os.path.exists(REGRESSOR_MODEL_PATH)):
    st.error(" Model files not found. Please upload models and retry.")
    st.stop()

# Load Models
@st.cache_resource
def load_models():
    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    regressor = joblib.load(REGRESSOR_MODEL_PATH)
    return classifier, regressor

classifier, regressor = load_models()

# Upload Dataset or Use Default
st.subheader(" Load Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_default_data(path):
    return pd.read_csv(path)

if uploaded_file is not None:
    st.success(" Custom dataset uploaded successfully!")

    # Save uploaded file to uploads/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploaded_filename = f"{UPLOAD_DIR}/uploaded_{timestamp}.csv"
    with open(uploaded_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f" Uploaded dataset saved as `{uploaded_filename}`")

    data = pd.read_csv(uploaded_file)
else:
    st.info(f"ℹ No file uploaded. Using default `{DEFAULT_DATA_PATH}` dataset.")
    data = load_default_data(DEFAULT_DATA_PATH)

st.dataframe(data.head())

# App Header
st.title(" AI-Enhanced Agile Project Management")

# Sidebar
with st.sidebar:
    st.header(" Project Info")
    st.markdown("""
    **Goal:**  
    Use AI to:
    - Predict task delays 📈
    - Optimize sprint plans 📅
    - Enhance resource management ⚙️

    **Models:**
    - XGBoost (Classification + Regression)
    - HDBSCAN (Clustering)

    Developed by Chris @ Solent University
    """)

    st.subheader(" Advanced Settings")
    selected_percentile = st.slider(
        "Select Dynamic Cutoff Percentile (Resolution Days)",
        min_value=80, max_value=99, value=99, step=1
    )
    business_cutoff = 60  # Max sprint cutoff
    max_allowed_sprint = 42  # Force max 42 days sprint

# Run AI Analysis
if st.button(" Run AI Analysis"):

    with st.spinner("Running AI Models..."):

        # Features
        classification_features = [
            'Story_Point', 'Was_Activated', 'Was_Completed',
            'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
        ]

        regression_features = [
            'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
            'Created_Weekday', 'Created_Hour', 'Resolved_Weekday', 'Resolved_Hour',
            'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
        ]

        clustering_features = list(set(classification_features + regression_features + ['Resolution_Time_Days']))

        # Validate Required Columns
        required_columns = set(classification_features + regression_features + ['Resolution_Time_Days'])
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            st.error(f" Missing columns in uploaded data: {missing_columns}")
            st.stop()

        # Predictions
        data['Predicted_Delay'] = classifier.predict(data[classification_features])
        data['Delay_Probability'] = classifier.predict_proba(data[classification_features])[:, 1]
        data['Predicted_Resolution_Days'] = regressor.predict(data[regression_features])

        # Clustering
        cluster_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        X_cluster = cluster_pipeline.fit_transform(data[clustering_features])
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        cluster_labels = clusterer.fit_predict(X_cluster)

        data['Cluster'] = np.where(cluster_labels == -1, 'Noise', cluster_labels)

        # Results
        st.success(" Analysis complete!")

        st.subheader(" AI Analysis Results")
        cluster_options = ["All"] + sorted(data['Cluster'].unique(), key=lambda x: (str(x) != "Noise", x))
        selected_cluster = st.selectbox(" Select Cluster", cluster_options)

        filtered_data = data if selected_cluster == "All" else data[data['Cluster'] == selected_cluster]
        st.dataframe(filtered_data)

        # --- KPIs ---
        st.markdown("### 🚦 Key Performance Indicators")

        avg_resolution = filtered_data['Predicted_Resolution_Days'].mean()
        delayed_percent = filtered_data['Predicted_Delay'].mean() * 100
        num_clusters = data['Cluster'].nunique()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(" Delayed Tasks (%)", f"{delayed_percent:.1f}%")
        kpi2.metric(" Avg Resolution (Days)", f"{avg_resolution:.1f}")
        kpi3.metric(" Clusters Found", num_clusters)

        # --- 🏗 Task Splitting Recommendation ---
        st.markdown("### 🏗 Task Splitting Recommendation")

        split_threshold_days = 30
        overweight_tasks = filtered_data[filtered_data['Predicted_Resolution_Days'] > split_threshold_days]
        num_overweight_tasks = len(overweight_tasks)
        total_tasks = len(filtered_data)

        overweight_ratio = (num_overweight_tasks / total_tasks) * 100

        if overweight_ratio > 10:
            st.warning(f"⚠️ {overweight_ratio:.1f}% of tasks are too large (> {split_threshold_days} days).")
            st.info("🔧 Recommendation: Break large tasks into 2–3 smaller subtasks to improve sprint success.")
        else:
            st.success(" Most tasks are manageable size for sprint planning.")

        #  Sprint Recommendation
        st.markdown("### 📅 Sprint Recommendation")

        dynamic_cutoff = np.percentile(filtered_data['Predicted_Resolution_Days'], selected_percentile)
        final_cutoff = min(dynamic_cutoff, business_cutoff)

        reasonable_days = filtered_data['Predicted_Resolution_Days'].clip(upper=final_cutoff)
        suggested_sprint_days = np.percentile(reasonable_days, 80)
        suggested_sprint_days = min(suggested_sprint_days, max_allowed_sprint)

        st.info(f" Suggested Sprint Length: **{suggested_sprint_days:.0f} days** (cutoff: {final_cutoff:.1f} days)")

        # --- Safe vs Aggressive Plans ---
        sprint_options = {
            "1 Week (7 days)": 7,
            "2 Weeks (14 days)": 14,
            "3 Weeks (21 days)": 21,
            "1 Month (30 days)": 30,
            "6 Weeks (42 days)": 42
        }

        sprint_days = np.array(list(sprint_options.values()))
        sprint_names = list(sprint_options.keys())

        aggressive_idx = np.argmin(np.abs(sprint_days - suggested_sprint_days))
        aggressive_choice = sprint_names[aggressive_idx]

        safe_idx = np.where(sprint_days >= suggested_sprint_days)[0]
        safe_choice = sprint_names[safe_idx[0]] if len(safe_idx) else aggressive_choice

        st.success(f"🛡 Safe Sprint Plan: **{safe_choice}** |  Aggressive Sprint Plan: **{aggressive_choice}**")

        # --- Sprint Plan Chart ---
        st.markdown("###  Sprint Plan Comparison")

        sprint_df = pd.DataFrame({
            "Plan": ["Suggested", "Safe", "Aggressive"],
            "Days": [
                suggested_sprint_days,
                sprint_options.get(safe_choice, suggested_sprint_days),
                sprint_options.get(aggressive_choice, suggested_sprint_days)
            ]
        })

        fig = px.bar(sprint_df, x="Plan", y="Days", text="Days", height=400)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)

        # --- Downloadable Results ---
        st.markdown("### Download Results")

        csv_download = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(" Download CSV", data=csv_download, file_name="AI_Agile_Predictions.csv", mime="text/csv")

        excel_buffer = io.BytesIO()
        filtered_data.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(
            label=" Download Excel",
            data=excel_buffer,
            file_name="AI_Agile_Predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- Reset App Button ---
if st.button("🔄 Reset App"):
    st.experimental_rerun()
