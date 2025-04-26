#!/usr/bin/env python
# coding: utf-8

# ## leveraging AI/ML to enhance agile project management practices and predict risks

# In[2]:


get_ipython().system('pip install pandas numpy matplotlib scikit-learn')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv('Tawos_Agile.csv', low_memory=False)


# In[5]:


df.shape


# In[6]:


df.head(5)


# ## Data preperation

# In[8]:


# Define date columns
date_cols = [
    'Creation_Date', 'Resolution_Date', 'sprint_start_date', 'sprint_End_Date',
    'Activated_Date', 'Complete_Date', 'project_Start_Date', 'project_Last_Update_Date'
]

# convert date columns
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Display missing values summary
missing_summary = df.isnull().sum().sort_values(ascending=False)
print("Missing Value Summary:\n", missing_summary[missing_summary > 0])

# Drop duplicates
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows")

# Handle missing values 
if 'Story_Point' in df.columns:
    median_story_point = df['Story_Point'].median()
    df['Story_Point'] = df['Story_Point'].fillna(median_story_point)

if 'Resolution' in df.columns:
    df['Resolution'] = df['Resolution'].fillna('Unresolved')

# Convert categorical columns safely
categorical_cols = ['Type', 'Priority', 'Status', 'Change_Type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# display updated data types summary
print("\nUpdated Data Types:")
print(df.dtypes.value_counts())


# In[9]:


df.isnull().sum() #checking data for missing values


# In[10]:


import numpy as np
from sklearn.impute import SimpleImputer

# Identify Numerical and Categorical Columns 
numerical_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Filter out numerical columns that are completely empty (all NaN)
valid_numerical_columns = [col for col in numerical_columns if df[col].notna().any()]

# Impute Numerical Columns using Median 
numerical_imputer = SimpleImputer(strategy='median')
df[valid_numerical_columns] = pd.DataFrame(
    numerical_imputer.fit_transform(df[valid_numerical_columns]),
    columns=valid_numerical_columns,
    index=df.index
)

# Impute Categorical Columns using Most Frequent 
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = pd.DataFrame(
    categorical_imputer.fit_transform(df[categorical_columns]),
    columns=categorical_columns,
    index=df.index
)

# Replace infinite values (just in case) 
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Final Check for Remaining Missing Values 
final_missing = df.isna().sum()
final_missing = final_missing[final_missing > 0]

# Output Results 
if final_missing.empty:
    print("All missing values have been imputed.")
else:
    print("Remaining missing values:\n", final_missing)


# ## EDA (Exploratory Data Analysis)

# In[12]:


# Set visual style
sns.set(style="whitegrid")

# Distributions of numerical features 
numeric_cols = df.select_dtypes(include='number').columns

# Histograms
df[numeric_cols].hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("Distributions of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Boxplots to identify outliers 
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:6], 1):  # show first 6 numeric features
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Remove ID columns from numeric columns 
numeric_cols_no_id = [col for col in df.select_dtypes(include='number').columns if 'id' not in col.lower()]

# Compute correlation matrix
corr = df[numeric_cols_no_id].corr()

# Plot the heatmap 
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={'shrink': .8}
)
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Pairplot of selected features (FIXED version to avoid MemoryError)
selected_features = ['Story_Point', 'In_Progress_Minutes', 'Total_Effort_Minutes', 'Resolution_Time_Minutes']

# Drop missing values first
df_selected = df[selected_features].dropna()

# Sample smaller data after dropping missing values
df_sample = df_selected.sample(n=500, random_state=42)  # 500 rows for faster plotting

# Create pairplot
sns.pairplot(df_sample, corner=True)  # corner=True for a lighter plot
plt.suptitle("Pairwise Relationships of Selected Features", y=1.02)
plt.show()


# In[13]:


# List of categorical columns to visualize
cat_cols = ['Type', 'Priority', 'Status', 'Change_Type', 'Resolution']

# Set figure size and style
plt.figure(figsize=(18, 12))
sns.set(style="whitegrid")

# Loop through and plot each categorical feature
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 3, i)
    if col in df.columns:
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(col)
        plt.ylabel('Count')

plt.tight_layout()
plt.suptitle("Categorical Feature Distributions", fontsize=18, y=1.02)
plt.show()


# ## Feature Engineering

# In[15]:


import pandas as pd

# Ensure date columns are in datetime format
date_cols = [
    'Creation_Date', 'Resolution_Date', 'sprint_start_date', 'sprint_End_Date',
    'Activated_Date', 'Complete_Date'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Duration Features
df['Resolution_Time_Days'] = (df['Resolution_Date'] - df['Creation_Date']).dt.total_seconds() / (60 * 60 * 24)
df['Sprint_Duration_Days'] = (df['sprint_End_Date'] - df['sprint_start_date']).dt.total_seconds() / (60 * 60 * 24)

# Binary Flags
df['Was_Activated'] = df['Activated_Date'].notnull().astype(int)
df['Was_Completed'] = df['Complete_Date'].notnull().astype(int)

# Date Parts from Creation_Date
df['Created_Year'] = df['Creation_Date'].dt.year
df['Created_Month'] = df['Creation_Date'].dt.month
df['Created_Weekday'] = df['Creation_Date'].dt.weekday
df['Created_Hour'] = df['Creation_Date'].dt.hour

# Date Parts from Resolution_Date
df['Resolved_Weekday'] = df['Resolution_Date'].dt.weekday
df['Resolved_Hour'] = df['Resolution_Date'].dt.hour

# Derived Features for RQ1 & RQ2
df['Effort_Per_Story_Point'] = df['Total_Effort_Minutes'] / (df['Story_Point'] + 1e-5)  # Avoid division by zero
df['Is_Delayed'] = (df['Resolution_Time_Days'] > df['Sprint_Duration_Days']).astype(int)

# Task Load and Resource-Based Features
# Count of tasks per assignee
task_counts = df['Assignee_ID'].value_counts().to_dict()
df['Task_Load_Per_Assignee'] = df['Assignee_ID'].map(task_counts)

# Average effort per assignee
avg_effort_per_user = df.groupby('Assignee_ID')['Total_Effort_Minutes'].mean().to_dict()
df['Assignee_Effort_Avg'] = df['Assignee_ID'].map(avg_effort_per_user)

# Save to CSV
df.to_csv('feature_engineered_dataset.csv', index=False)


# In[16]:


df.shape


# In[17]:


df.head(5)


# In[18]:


df.columns


# In[19]:


if 'Is_Delayed' in df.columns:
    print("Value counts for 'Is_Delayed':\n")
    print(df['Is_Delayed'].value_counts(dropna=False))
else:
    print("'Is_Delayed' column not found in the DataFrame.")


# In[20]:


# df.dtypes


# In[21]:


df.describe()


# ## Modeling: Classification: (RQ2 Predicting Delays, Bottlenecks, Risks :Delay Prediction)

# ## Logistic Regression 

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

# Load Dataset 
print("\n Loading feature-engineered dataset")
df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Select Features and Target 
selected_features = [
    'Story_Point', 'In_Progress_Minutes', 'Total_Effort_Minutes',
    'Resolution_Time_Days', 'Sprint_Duration_Days',
    'Effort_Per_Story_Point', 'Was_Activated', 'Was_Completed',
    'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]

# Check available features
available_features = [col for col in selected_features if col in df.columns]
missing_features = list(set(selected_features) - set(available_features))

if missing_features:
    print(f" Missing features (excluded): {missing_features}")

# Define X and y
X = df[available_features].fillna(0)
y = df['Is_Delayed']

# Train-Test Split 
print("\n Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle Class Imbalance with SMOTE 
print("\n Applying SMOTE for balancing")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Logistic Regression Model 
print("\n Training Logistic Regression model")
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_smote, y_train_smote)

# Predictions 
print("\n Making predictions")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation 
print("\n Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_prob)
print(f"\n ROC AUC Score: {auc_score:.4f}")

# Plot ROC Curve 
print("\n Plotting ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Balanced Logistic Regression)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importance (Coefficients) 
print("\n Logistic Regression Feature Coefficients ")
feature_weights = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print(feature_weights)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
feature_weights.plot(kind='barh')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


top_features = feature_weights.abs().sort_values(ascending=False).head(5)
print("\nTop 5 Most Influential Features:")
print(top_features)


# ## RandomForest Classification

# In[27]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load Dataset 
# print("\n Loading dataset")
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Feature Selection 
requested_features = [
    'Story_Point', 'Priority', 'Type', 'Effort_Per_Story_Point',
    'Was_Activated', 'Was_Completed',
    'Task_Load_Per_Assignee', 'Assignee_Effort_Avg',
    'Title_Changed_After_Estimation', 'Change_Type'
]
target = 'Is_Delayed'

# Validate Features
available_features = [col for col in requested_features if col in df.columns]
missing_features = list(set(requested_features) - set(available_features))

if missing_features:
    print(f" Missing features (skipped): {missing_features}")

X = df[available_features]
y = df[target]

# Column Types 
categorical_cols = [col for col in ['Priority', 'Type', 'Change_Type'] if col in X.columns]
numeric_cols = [col for col in available_features if col not in categorical_cols]

# Preprocessing Pipelines 
numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Build Full Pipeline 
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train-Test Split 
print("\n Splitting dataset")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the Pipeline 
print("\n Training Random Forest model")
clf_pipeline.fit(X_train, y_train)

# Predict and Evaluate
print("\n Evaluating model")
y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

print(f"\n Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\n ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Plot Confusion Matrix 
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Plot ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importance 
print("\n Analyzing feature importance...")

# Get final feature names
fitted_preprocessor = clf_pipeline.named_steps['preprocessor']
ohe = fitted_preprocessor.named_transformers_['cat'].named_steps['encoder']
ohe_features = ohe.get_feature_names_out(categorical_cols)
final_feature_names = np.concatenate([numeric_cols, ohe_features])

# Get feature importances
importances = clf_pipeline.named_steps['classifier'].feature_importances_
feature_df = pd.DataFrame({'Feature': final_feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importances 
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()


# ## XGBoost Classification

# In[29]:


import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

# Load Dataset 
# print(" Loading dataset")
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Define Features and Target
features = [
    'Story_Point', 'Priority', 'Type', 'Effort_Per_Story_Point',
    'Was_Activated', 'Was_Completed',
    'Task_Load_Per_Assignee', 'Assignee_Effort_Avg',
    'Title_Changed_After_Estimation', 'Change_Type'
]
target = 'Is_Delayed'

# Filter Existing Columns
features = [f for f in features if f in df.columns]
X = df[features]
y = df[target]

# Preprocessing 
categorical_cols = [col for col in ['Priority', 'Type', 'Change_Type'] if col in X.columns]
numeric_cols = [col for col in features if col not in categorical_cols]

numeric_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost Classifier 
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)

# SMOTE Pipeline 
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb_model)
])

# Train the Pipeline 
print(" Training model...")
xgb_pipeline.fit(X_train, y_train)

# Predict and Evaluate 
print(" Evaluating model")
y_pred = xgb_pipeline.predict(X_test)
y_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("\n XGBoost + SMOTE Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Confusion Matrix Plot 
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Not Delayed', 'Delayed'],
            yticklabels=['Not Delayed', 'Delayed'])
plt.title('XGBoost + SMOTE Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curve Plot 
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost + SMOTE')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature Importances 
print("\n Calculating Feature Importances")

# Refit the categorical transformer separately
if categorical_cols:
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    cat_transformer.fit(X_train[categorical_cols])
    encoded_features = list(cat_transformer.named_steps['encoder'].get_feature_names_out(categorical_cols))
else:
    encoded_features = []

# Combine numeric + encoded categorical feature names
all_feature_names = numeric_cols + encoded_features

# Extract booster feature importance
booster = xgb_pipeline.named_steps['classifier'].get_booster()
importance_dict = booster.get_score(importance_type='weight')

mapped_importances = {
    all_feature_names[int(k[1:])]: v
    for k, v in importance_dict.items()
    if int(k[1:]) < len(all_feature_names)
}

feat_imp_df = pd.DataFrame(mapped_importances.items(), columns=['Feature', 'Importance'])
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Feature Importance Plot 
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature')
plt.title('XGBoost + SMOTE: Top Feature Importances')
plt.tight_layout()
plt.show()

# Save the Full Pipeline 
print("\n Saving model...")
joblib.dump(xgb_pipeline, 'xgb_smote_classifier_pipeline.joblib')
print(" Saved model as 'xgb_smote_classifier_pipeline.joblib'")

# Save Training Report 
print("\n Saving Training Report...")
report_filename = 'xgb_smote_training_report.txt'

with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("## XGBoost + SMOTE Model Training Report\n\n")
    
    f.write("## Classification Results\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- ROC AUC Score: {roc_auc:.4f}\n\n")
    
    f.write("## Classification Report\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n\n")
    
    f.write("## Top 15 Feature Importances\n")
    for idx, row in feat_imp_df.head(15).iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.0f}\n")
    f.write("\n\n")
    
    f.write("## Confusion Matrix\n")
    f.write(np.array2string(conf_matrix, separator=', '))
    f.write("\n")

print(f" Saved Training Report as '{report_filename}'")

print("\n Retraining, Evaluation and Report Saved Successfully!")

# Save best XGBoost pipeline
joblib.dump(xgb_pipeline, 'xgb_production_pipeline.joblib')



# ## Export Best Models for Deployment

# In[31]:


import joblib

# Save best XGBoost pipeline
joblib.dump(xgb_pipeline, 'xgb_production_pipeline.joblib')



# ## Modeling: Regression (RQ1 Enhancing Task Prioritization and Resource Allocation: resource estimation)

# ## Random Forest Regressor

# In[34]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset 
# print(" Loading dataset")
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Drop rows with missing target 
df = df.dropna(subset=['Resolution_Time_Days'])

# Features and Target 
target = 'Resolution_Time_Days'
features = [
    'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour',
    'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]
# Keep only available features
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]

# Preprocessing 
numeric_transformer = SimpleImputer(strategy='median')

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline with Random Forest Regressor 
print("Training Random Forest Regressor")
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Important: use inside pipeline directly
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# Predict and Evaluate 
print("Evaluating model")
y_pred = rf_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f"MAE  = {mae:.2f} days")
print(f"RMSE = {rmse:.2f} days")
print(f"R²   = {r2:.5f}")

# Feature Importances 
print("\n Calculating Feature Importances")
regressor_model = rf_pipeline.named_steps['regressor']
importances = regressor_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances 
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', color='skyblue')
plt.title('Feature Importances - Random Forest Regressor')
plt.tight_layout()
plt.show()

# Residuals Plot 
print("\n Plotting Residuals")
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color='navy')
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Prediction Error (days)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Actual vs Predicted 
print("\n Plotting Actual vs Predicted")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Resolution Time (days)")
plt.ylabel("Predicted Resolution Time (days)")
plt.title("Actual vs. Predicted - Resolution Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# ## XGBoost Regressor

# In[36]:


from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset 
# print("Loading dataset")
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Clip Outliers 
print(" Cleaning dataset (clipping resolution times)")

# Only keep tasks with realistic resolution times (<= 90 days)
df = df[df['Resolution_Time_Days'] <= 90]

# Drop missing targets
df = df.dropna(subset=['Resolution_Time_Days'])

# Define Features and Target 
target = 'Resolution_Time_Days'
features = [
    'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour',
    'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]

# Filter only existing features
features = [f for f in features if f in df.columns]
X = df[features]
y = df[target]

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: Imputer + XGBoost 
print("Training XGBoost Regressor")
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)

# Predict and Evaluate 
print("Evaluating model")
y_pred = xgb_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n XGBoost Regressor Performance:")
print(f"MAE  = {mae:.2f} days")
print(f"RMSE = {rmse:.2f} days")
print(f"R²   = {r2:.4f}")

# Feature Importances 
print("\n Calculating Feature Importances")

regressor_model = xgb_pipeline.named_steps['regressor']
importances = regressor_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances 
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', orient='h', color='skyblue')
plt.title('Feature Importances - XGBoost Regressor')
plt.tight_layout()
plt.show()

# Residuals Plot 
print("\n Plotting Residuals")
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color='purple')
plt.title("Residuals Distribution - XGBoost")
plt.xlabel("Prediction Error (days)")
plt.tight_layout()
plt.show()

# Actual vs Predicted 
print("\n Plotting Actual vs Predicted")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Resolution Time (days)")
plt.ylabel("Predicted Resolution Time (days)")
plt.title("Actual vs. Predicted - XGBoost")
plt.tight_layout()
plt.show()

# Save Trained Model 
print("\n Saving model")
joblib.dump(xgb_pipeline, 'xgb_regressor_resolution_time.joblib')
print(" Model saved as 'xgb_regressor_resolution_time.joblib'")

# Save Evaluation Report 
print("\n Saving Evaluation Report")

report_filename = 'xgb_regressor_training_report.txt'

with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("# XGBoost Regressor Model Training Report\n\n")
    
    f.write("## Regression Results\n")
    f.write(f"- MAE  = {mae:.2f} days\n")
    f.write(f"- RMSE = {rmse:.2f} days\n")
    f.write(f"- R²   = {r2:.4f}\n\n")
    
    f.write("## Top Feature Importances\n")
    for idx, row in feature_importance_df.iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")

print(f" Saved Evaluation Report as '{report_filename}'")

print("\n Retraining, Evaluation, and Saving Completed Successfully!")


# ## Linear Regressor

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset 
# print("Loading dataset...")
#  df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Drop rows with missing target 
df = df.dropna(subset=['Resolution_Time_Days'])

# Features and Target 
target = 'Resolution_Time_Days'
features = [
    'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour',
    'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]

# Ensure features exist in data
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]

# Clip or Clean Targets 
# Clip unrealistic Resolution Times
print("Cleaning Target Variable")
y = np.clip(y, 0, 90)  # Cap at 90 days maximum

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: Imputer + Linear Regression 
print("Training Linear Regression Model")
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

# Predict and Evaluate 
print("Evaluating model")
y_pred = lr_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Linear Regression Performance:")
print(f"MAE  = {mae:.2f} days")
print(f"RMSE = {rmse:.2f} days")
print(f"R²   = {r2:.5f}")

# Coefficients 
print("\n Analyzing Coefficients...")
regressor_model = lr_pipeline.named_steps['regressor']
coefficients = regressor_model.coef_

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=lambda x: np.abs(x), ascending=False)

# Plot Coefficients 
plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', color='skyblue')
plt.title("Linear Regression Coefficients")
plt.tight_layout()
plt.show()

# Residual Plot 
print("\n Plotting Residuals")
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True, color='steelblue')
plt.title("Residuals Distribution - Linear Regression")
plt.xlabel("Prediction Error (days)")
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot 
print("\n Plotting Actual vs Predicted")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Resolution Time (days)")
plt.ylabel("Predicted Resolution Time (days)")
plt.title("Actual vs. Predicted - Linear Regression")
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Export Best Models for Deployment

# In[40]:


import joblib

# Save the trained XGBoost pipeline
joblib.dump(xgb_pipeline, 'xgb_regressor_resolution_time.joblib')

print(" Model saved as 'xgb_regressor_resolution_time.joblib'")


# ## Clustering: (RQ3 AI-Driven Team Collaboration and Communication: Pattern Discovery)

# ## KMeans Clustering

# In[43]:


import os
import warnings

# Safe Thread Settings 
os.environ["OMP_NUM_THREADS"] = "8"

# Suppress known warnings 
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# Load Dataset 
# print("\n Loading feature-engineered dataset")
# df = pd.read_csv("feature_engineered_dataset.csv", low_memory=False)

# Select Features for Clustering 
features = [
    'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee',
    'Assignee_Effort_Avg', 'Sprint_Duration_Days',
    'Created_Weekday', 'Created_Hour', 'Resolution_Time_Days',
    'Resolved_Weekday', 'Resolved_Hour'
]

features = [f for f in features if f in df.columns]
df_clustering = df[features].dropna()

# Clip Outliers
print("\n Clipping extreme outliers")
clip_settings = {
    'Effort_Per_Story_Point': 500,
    'Task_Load_Per_Assignee': 100,
    'Assignee_Effort_Avg': 1000,
    'Resolution_Time_Days': 90
}

for col, clip_value in clip_settings.items():
    if col in df_clustering.columns:
        df_clustering[col] = np.clip(df_clustering[col], 0, clip_value)

# Preprocessing Pipeline 
print("\n Preprocessing features")
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_processed = preprocessor.fit_transform(df_clustering)

# Elbow Method 
print("\n Running Elbow Method")
inertias = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_processed)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.tight_layout()
plt.show()

# Silhouette Score 
print("\n Running Silhouette Score")
silhouette_scores = []

X_sample = resample(X_processed, n_samples=2000, random_state=42)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_sample)
    score = silhouette_score(X_sample, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Cluster Count (Sampled)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply Final KMeans Model 
optimal_k = 4  # You can adjust based on elbow/silhouette
print(f"\n Training Final KMeans with k={optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
df_clustering['Cluster'] = kmeans.fit_predict(X_processed)

# PCA for 2D Projection 
print("\n Running PCA for 2D Visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
df_clustering['PCA1'] = X_pca[:, 0]
df_clustering['PCA2'] = X_pca[:, 1]

# PCA Scatter Plot 
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clustering, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title(f"KMeans Clustering (k={optimal_k}) - PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Boxplots by Cluster 
print("\n Plotting Boxplots by Cluster")
melted = df_clustering.melt(id_vars='Cluster', value_vars=features, var_name='Feature', value_name='Value')

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted, x='Feature', y='Value', hue='Cluster')
plt.xticks(rotation=45)
plt.title("Feature Distributions by Cluster")
plt.tight_layout()
plt.show()

# Cluster Centroids in Original Feature Space 
print("\n Calculating Cluster Centroids")

scaler = preprocessor.named_steps['scaler']
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=features)
centroids_df['Cluster'] = centroids_df.index

print("\nCluster Centroids (approximate values):")
print(centroids_df.round(2))

# Save Clustered Dataset 
save_clustered = input("\n Save the clustered dataset? (y/n): ").strip().lower()

if save_clustered == 'y':
    output_path = "feature_engineered_dataset_with_clusters.csv"
    df_with_clusters = df.copy()
    df_with_clusters = df_with_clusters.dropna(subset=features)
    df_with_clusters['Cluster'] = df_clustering['Cluster'].values
    df_with_clusters.to_csv(output_path, index=False)
    print(f"\n Clustered dataset saved as: {output_path}")


# ##  DBSCAN Clustering

# In[45]:


pip install hdbscan


# In[59]:


import hdbscan
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

# Load Dataset 
# print("\n📥 Loading feature-engineered dataset...")
# df = pd.read_csv("feature_engineered_dataset.csv", low_memory=False)

# Select Features for Clustering 
features = [
    'Story_Point', 'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee',
    'Assignee_Effort_Avg', 'Sprint_Duration_Days',
    'Created_Weekday', 'Created_Hour', 'Resolution_Time_Days'
]
features = [f for f in features if f in df.columns]

df_clustering = df[features].dropna().copy()
print(f" {len(df_clustering)} tasks selected for clustering.")

# Preprocessing 
print("\n Preprocessing features")
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_processed = preprocessor.fit_transform(df_clustering)

# PCA Dimensionality Reduction 
print("\n Applying PCA...")
pca = PCA(n_components=5, random_state=42)
X_reduced = pca.fit_transform(X_processed)

# Auto-Tuning HDBSCAN 
min_cluster_sizes = [5, 10, 20, 30, 50, 100]
results = []

print("\n Running HDBSCAN tuning")
for min_size in min_cluster_sizes:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
    labels = clusterer.fit_predict(X_reduced)

    num_clusters = len(np.unique(labels[labels != -1]))

    if num_clusters > 1:
        X_valid = X_reduced[labels != -1]
        labels_valid = labels[labels != -1]

        if len(X_valid) > 1000:
            X_sample, labels_sample = resample(X_valid, labels_valid, n_samples=1000, random_state=42)
        else:
            X_sample, labels_sample = X_valid, labels_valid

        silhouette = silhouette_score(X_sample, labels_sample)
    else:
        silhouette = np.nan

    results.append({
        'min_cluster_size': min_size,
        'num_clusters': num_clusters,
        'silhouette_score': silhouette
    })

results_df = pd.DataFrame(results)

print("\n Tuning Results:")
print(results_df)

# Plotting Tuning Results 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

sns.lineplot(data=results_df, x='min_cluster_size', y='num_clusters', marker='o', ax=axes[0])
axes[0].set_title('Number of Clusters vs min_cluster_size')
axes[0].set_xlabel('min_cluster_size')
axes[0].set_ylabel('Number of Clusters')
axes[0].grid(True)

sns.lineplot(data=results_df, x='min_cluster_size', y='silhouette_score', marker='o', ax=axes[1])
axes[1].set_title('Silhouette Score vs min_cluster_size')
axes[1].set_xlabel('min_cluster_size')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True)

plt.suptitle('HDBSCAN Auto-Tuning Results', fontsize=16)
plt.show()

# Select Best Clusterer 
print("\n Selecting best clustering model")
filtered = results_df[(results_df['num_clusters'] >= 5) & (results_df['num_clusters'] <= 30)]

if not filtered.empty:
    best_row = filtered.sort_values('silhouette_score', ascending=False).iloc[0]
else:
    best_row = results_df.sort_values('silhouette_score', ascending=False).iloc[0]

best_min_size = int(best_row['min_cluster_size'])
print(f" Best min_cluster_size = {best_min_size}")

# --- Train Final HDBSCAN Model ---
print("\n Training final HDBSCAN...")
final_clusterer = hdbscan.HDBSCAN(min_cluster_size=best_min_size)
final_labels = final_clusterer.fit_predict(X_reduced)

df_clustering['Final_Cluster'] = final_labels

# PCA 2D Visualization 
print("\n Plotting final clusters...")
pca_2d = PCA(n_components=2, random_state=42)
X_pca2d = pca_2d.fit_transform(X_reduced)

df_clustering['PCA1'] = X_pca2d[:, 0]
df_clustering['PCA2'] = X_pca2d[:, 1]

fig, ax = plt.subplots(figsize=(18, 12))
unique_clusters = sorted(df_clustering['Final_Cluster'].unique())
palette = sns.color_palette("husl", len(unique_clusters))

sns.scatterplot(
    data=df_clustering, x='PCA1', y='PCA2',
    hue='Final_Cluster', palette=palette, legend='full', s=60, alpha=0.7, ax=ax
)

ax.set_title(f"HDBSCAN Final Clustering (min_cluster_size={best_min_size})", fontsize=18)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
ax.grid(True)

plt.tight_layout()
plt.show()



# In[63]:


# Load Data 
# print(" Loading feature-engineered dataset...")
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Select Features for Clustering 
cluster_features = [
    'Story_Point', 'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee',
    'Assignee_Effort_Avg', 'Sprint_Duration_Days',
    'Created_Weekday', 'Created_Hour', 'Resolution_Time_Days',
    'Resolved_Weekday', 'Resolved_Hour'
]

df_cluster = df.dropna(subset=cluster_features)[cluster_features].copy()
print(f" {len(df_cluster)} tasks ready for clustering.")

# Build Preprocessing + Clustering Pipeline 
print("Building clustering pipeline")

clustering_pipeline = Pipeline([
    ('preprocessing', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('clusterer', hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10))
])

# Fit the Full Pipeline 
print(" Training HDBSCAN inside pipeline")
X_cluster = df_cluster.values
clustering_pipeline.fit(X_cluster)

# Extract Cluster Labels 
cluster_labels = clustering_pipeline.named_steps['clusterer'].labels_

df_cluster['Cluster'] = np.where(cluster_labels == -1, 'Noise', cluster_labels)

# Analyze Clusters 
print(" Analyzing clusters")

cluster_summary = df_cluster[df_cluster['Cluster'] != 'Noise'].groupby('Cluster').agg({
    'Resolution_Time_Days': ['count', 'mean', 'median']
}).reset_index()

cluster_summary.columns = ['Cluster', 'Count', 'Avg_Resolution_Days', 'Median_Resolution_Days']

print(cluster_summary)

# Auto Label Clusters 
def label_cluster(row):
    if row['Avg_Resolution_Days'] <= 15:
        return "Fast Tasks"
    elif row['Avg_Resolution_Days'] <= 30:
        return "Medium Tasks"
    else:
        return "Slow Tasks"

cluster_summary['Cluster_Label'] = cluster_summary.apply(label_cluster, axis=1)

# Merge labels back
label_mapping = dict(zip(cluster_summary['Cluster'], cluster_summary['Cluster_Label']))
df_cluster['Cluster_Label'] = df_cluster['Cluster'].map(label_mapping)
df_cluster['Cluster_Label'] = df_cluster['Cluster_Label'].fillna('Noise')

#  Plot Cluster Size Distribution 
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cluster, x='Cluster_Label', order=df_cluster['Cluster_Label'].value_counts().index)
plt.title('Cluster Size Distribution (Auto-Labeled)')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save Outputs
print("Saving outputs")

df_cluster.to_csv('hdbscan_clusters_with_labels.csv', index=False)
joblib.dump(clustering_pipeline, 'hdbscan_full_pipeline.joblib')

print("Saved clustered dataset to 'hdbscan_clusters_with_labels.csv'")
print("Saved HDBSCAN pipeline model to 'hdbscan_full_pipeline.joblib'")
print("Clustering + Labeling + Saving completed successfully!")


# ## Agglomerative Clustering

# In[66]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Dataset 
# print("\n Loading feature-engineered dataset")
# df = pd.read_csv("feature_engineered_dataset.csv", low_memory=False)

# Select Features for Clustering 
cluster_features = [
    'Resolution_Time_Days', 'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour', 'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]

# Drop missing values and sample 500 rows
cluster_df = df[cluster_features].dropna()
cluster_df_sample = cluster_df.sample(n=500, random_state=42).copy()

# Preprocessing: Impute + Scale 
print("\n Preprocessing features...")
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_cluster_sample = preprocessor.fit_transform(cluster_df_sample)

# Silhouette Score Analysis 
print("\n Finding best number of clusters with Silhouette Scores")
range_n_clusters = range(2, 10)
silhouette_scores = []

for n in range_n_clusters:
    model = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = model.fit_predict(X_cluster_sample)
    score = silhouette_score(X_cluster_sample, labels)
    silhouette_scores.append(score)
    print(f"n_clusters = {n}: Silhouette Score = {score:.4f}")

# Plot Silhouette Scores 
plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Score vs. Number of Clusters (Agglomerative)")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Best Cluster Count 
best_n = range_n_clusters[np.argmax(silhouette_scores)]
print(f"\n Best number of clusters by silhouette score: {best_n}")

# Final Agglomerative Clustering 
print("\n Running final Agglomerative Clustering")
agglo = AgglomerativeClustering(n_clusters=best_n, linkage='ward')
cluster_labels = agglo.fit_predict(X_cluster_sample)
cluster_df_sample['Cluster'] = cluster_labels

# PCA for 2D Projection 
print("\n PCA projection for visualization")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_sample)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='tab10', s=60)
plt.title(f"Agglomerative Clustering (n={best_n}) - PCA Projection")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Dendrogram 
print("\n Plotting Dendrogram")
linkage_matrix = linkage(X_cluster_sample, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=4)
plt.title("Dendrogram - Agglomerative Clustering (500 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Cluster Means Summary
print("\n Cluster Feature Averages:")
summary = cluster_df_sample.groupby('Cluster').mean().round(2)
print(summary)


# ### Export Best Models for Deployment

# In[71]:


from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score, roc_auc_score

# Load Dataset 
# df = pd.read_csv('feature_engineered_dataset.csv', low_memory=False)

# Train XGBoost Classifier for Is_Delayed 
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

# Train XGBoost Regressor for Resolution_Time_Days 
features_reg = [
    'Sprint_Duration_Days', 'Was_Activated', 'Was_Completed',
    'Created_Weekday', 'Created_Hour',
    'Resolved_Weekday', 'Resolved_Hour',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee', 'Assignee_Effort_Avg'
]
target_reg = 'Resolution_Time_Days'

# Drop missing labels
df = df.dropna(subset=[target_reg])

X_reg = df[features_reg]
y_reg = df[target_reg]

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

# Train HDBSCAN Clusterer 
cluster_features = [
    'Story_Point', 'Was_Activated', 'Was_Completed',
    'Effort_Per_Story_Point', 'Task_Load_Per_Assignee',
    'Assignee_Effort_Avg', 'Sprint_Duration_Days',
    'Created_Weekday', 'Created_Hour', 'Resolution_Time_Days',
    'Resolved_Weekday', 'Resolved_Hour'
]
cluster_df = df[cluster_features].dropna()
cluster_df_sample = cluster_df.sample(n=500, random_state=42)

preprocessor_cluster = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_cluster_sample = preprocessor_cluster.fit_transform(cluster_df_sample)

hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
hdbscan_clusterer.fit(X_cluster_sample)

# Export Models 
joblib.dump(xgb_pipeline_classifier, 'xgb_classifier_pipeline.joblib')
print("Saved XGBoost Classifier as 'xgb_classifier_pipeline.joblib'")

joblib.dump(xgb_pipeline_regressor, 'xgb_regressor_pipeline.joblib')
print("Saved XGBoost Regressor as 'xgb_regressor_pipeline.joblib'")

joblib.dump(hdbscan_clusterer, 'hdbscan_cluster_model.joblib')
print("Saved HDBSCAN Clusterer as 'hdbscan_cluster_model.joblib'")


# In[ ]:




