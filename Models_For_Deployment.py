#!/usr/bin/env python
# coding: utf-8

# ## leveraging AI/ML to enhance agile project management practices and predict risks

# ## Best models

# In[9]:


import pandas as pd
import numpy as np
import joblib
import hdbscan

from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Load Dataset 
print(" Loading dataset")
df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Clean Dataset 
print("Cleaning dataset")
if 'Resolution_Time_Days' in df.columns:
    df['Resolution_Time_Days'] = df['Resolution_Time_Days'].clip(upper=90)

# Train XGBoost Classifier for Delay Prediction 
print("Training XGBoost Classifier (Delay Prediction)")
features_clf = [
    'Story_Point', 'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]
target_clf = 'Is_Delayed'

X_clf = df[features_clf]
y_clf = df[target_clf]

numeric_transformer_clf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor_clf = ColumnTransformer([
    ('num', numeric_transformer_clf, features_clf)
])

xgb_pipeline_classifier = Pipeline([
    ('preprocessor', preprocessor_clf),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    ))
])

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, stratify=y_clf, test_size=0.2, random_state=42
)

xgb_pipeline_classifier.fit(X_train_clf, y_train_clf)

# Classifier Evaluation
y_pred_clf = xgb_pipeline_classifier.predict(X_test_clf)
y_pred_clf_proba = xgb_pipeline_classifier.predict_proba(X_test_clf)[:,1]
print(f" Classifier Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.4f}")
print(f" Classifier ROC AUC: {roc_auc_score(y_test_clf, y_pred_clf_proba):.4f}")

# Train XGBoost Regressor for Resolution Time Prediction 
print("Training XGBoost Regressor (Resolution Time)")
features_reg = [
    'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour', 'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]
target_reg = 'Resolution_Time_Days'

df_reg = df.dropna(subset=[target_reg])
X_reg = df_reg[features_reg]
y_reg = df_reg[target_reg]

numeric_transformer_reg = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor_reg = ColumnTransformer([
    ('num', numeric_transformer_reg, features_reg)
])

xgb_pipeline_regressor = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('regressor', XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ))
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

xgb_pipeline_regressor.fit(X_train_reg, y_train_reg)

# Regressor Evaluation
y_pred_reg = xgb_pipeline_regressor.predict(X_test_reg)
print(f" Regressor RMSE: {mean_squared_error(y_test_reg, y_pred_reg, squared=False):.2f}")
print(f" Regressor MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f" Regressor R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")

# Train HDBSCAN Clustering 
print(" Training HDBSCAN Clustering")

cluster_features = [
    'Story_Point', 'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee',
    'Assignee_Effort_Avg', 'Sprint_Duration_Days',
    'Created_Weekday', 'Created_Hour', 'Resolution_Time_Days',
    'Resolved_Weekday', 'Resolved_Hour'
]

df_cluster = df.dropna(subset=cluster_features)[cluster_features]
preprocessor_cluster = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_cluster = preprocessor_cluster.fit_transform(df_cluster)

hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
hdbscan_clusterer.fit(X_cluster)

# Save full clustering pipeline (preprocessing + model)
clustering_pipeline = Pipeline([
    ('preprocessor', preprocessor_cluster),
    ('clusterer', hdbscan_clusterer)
])

# Save All Models 
print("Saving models")

joblib.dump(xgb_pipeline_classifier, 'xgb_classifier_pipeline.joblib')
print("Saved Classifier: 'xgb_classifier_pipeline.joblib'")

joblib.dump(xgb_pipeline_regressor, 'xgb_regressor_pipeline.joblib')
print("Saved Regressor: 'xgb_regressor_pipeline.joblib'")

joblib.dump(clustering_pipeline, 'hdbscan_cluster_pipeline.joblib')
print("Saved Clustering Pipeline: 'hdbscan_cluster_pipeline.joblib'")

print("\n Retraining completed successfully!")


# In[ ]:




