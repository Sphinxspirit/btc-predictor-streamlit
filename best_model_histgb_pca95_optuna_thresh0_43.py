#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn==1.3.2 numpy==1.24.4 cloudpickle optuna shap


# In[3]:


import sklearn
print(sklearn.__version__)


# In[4]:


pip install --upgrade scikit-learn==1.3.2


# In[2]:


get_ipython().system('pip install optuna')


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import optuna


# In[4]:


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\angxi\OneDrive\Documents\Institute of Data Course\Capstone\3. Modelling\Final_Merged_Dataset (For Modelling).csv")

# Filter out rows with 0% change
df_filtered = df[df["Next_Day_Change"] != 0].copy()

# Create binary target: 0 = Decrease, 1 = Increase
df_filtered["Target"] = df_filtered["Next_Day_Change"].apply(lambda x: 0 if x < 0 else 1)

# Drop technical indicators and future values
columns_to_drop = [
    'Date', 'Next_Day_Change', 'Next_Day_Change_Range',
    'SMA_20', 'SMA_50', 'EMA_10', 'EMA_50', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'ADX', 'RSI_14', 'CCI_20', 'ROC_10', 'BB_Mid', 'BB_Upper', 'BB_Lower',
    'ATR_14', 'HV_30', 'OBV', 'Lag_1', 'Lag_3', 'Lag_7', 'Lag_30',
    'Weekly_Change', 'Monthly_Change', 'Daily_Change'
]
df_filtered.drop(columns=columns_to_drop, inplace=True)

# Time-based split
split_index = int(len(df_filtered) * 0.8)
train_df = df_filtered.iloc[:split_index]
test_df = df_filtered.iloc[split_index:]

X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"]
X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]


# In[5]:


# 📊 Evaluation Function for Binary Classification
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_binary_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # If model supports predict_proba (for ROC & PR curve)
    if hasattr(model, "predict_proba"):
        y_test_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_test_probs = model.decision_function(X_test)

    # Accuracy & F1
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    f1_gap = abs(train_f1 - test_f1)

    print(f"\n📌 {name}")
    print(f"✅ Train Accuracy: {train_acc:.4f}")
    print(f"✅ Test Accuracy:  {test_acc:.4f}")
    print(f"🎯 Train F1 Score: {train_f1:.4f}")
    print(f"🎯 Test F1 Score:  {test_f1:.4f}")
    print(f"⚠️  F1 Gap:        {f1_gap:.4f}")

    print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred))
    print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

    # Confusion Matrix
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title(f'Train Confusion Matrix - {name}')
    axs[0].set_xlabel('Predicted'); axs[0].set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Greens', ax=axs[1])
    axs[1].set_title(f'Test Confusion Matrix - {name}')
    axs[1].set_xlabel('Predicted'); axs[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.grid(True)
    plt.show()


# In[6]:


# 📊 Evaluation Function for Binary Classification with Threshold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_binary_model_with_threshold(name, model, X_train, y_train, X_test, y_test, threshold=0.5):
    model.fit(X_train, y_train)
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    # Apply custom threshold
    y_train_pred = (y_train_probs >= threshold).astype(int)
    y_test_pred = (y_test_probs >= threshold).astype(int)

    # Accuracy & F1
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    f1_gap = abs(train_f1 - test_f1)

    print(f"\n📌 {name} (Threshold = {threshold})")
    print(f"✅ Train Accuracy: {train_acc:.4f}")
    print(f"✅ Test Accuracy:  {test_acc:.4f}")
    print(f"🎯 Train F1 Score: {train_f1:.4f}")
    print(f"🎯 Test F1 Score:  {test_f1:.4f}")
    print(f"⚠️  F1 Gap:        {f1_gap:.4f}")

    print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred))
    print("Classification Report (Test):\n", classification_report(y_test, y_test_pred))

    # Confusion Matrix
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_train, y_train_pred), annot=True, fmt='d', cmap='Blues', ax=axs[0])
    axs[0].set_title(f'Train Confusion Matrix - {name}')
    axs[0].set_xlabel('Predicted'); axs[0].set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Greens', ax=axs[1])
    axs[1].set_title(f'Test Confusion Matrix - {name}')
    axs[1].set_xlabel('Predicted'); axs[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.grid(True)
    plt.show()


# In[7]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the features first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA (retain 95% variance)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✅ PCA reduced dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}")


# In[8]:


import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_iter = trial.suggest_int("max_iter", 100, 500)
    l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)

    # Create model
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        early_stopping=True,
        random_state=42
    )

    # Use 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_index, val_index in kf.split(X_train_pca, y_train):
        X_tr, X_val = X_train_pca[train_index], X_train_pca[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model.fit(X_tr, y_tr)
        y_val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_probs)
        aucs.append(auc)

    return np.mean(aucs)

# Run Optuna
study = optuna.create_study(direction="maximize", study_name="HistGB-PCA-AUC")
study.optimize(objective, n_trials=50)

# Print best results
print("✅ Best trial:")
print(f"AUC: {study.best_value:.4f}")
print("Params:", study.best_params)


# In[9]:


# Build final model
best_params = {
    'learning_rate': 0.2878371454598202,
    'max_depth': 3,
    'max_iter': 374,
    'l2_regularization': 0.6476798637672115
}

final_model = HistGradientBoostingClassifier(
    **best_params,
    early_stopping=True,
    random_state=42
)

final_model.fit(X_train_pca, y_train)

# Predict
y_test_probs = final_model.predict_proba(X_test_pca)[:, 1]

# Retune threshold (optional, recommended)
# Or apply previously tuned threshold of 0.43
y_test_pred = (y_test_probs >= 0.43).astype(int)


# In[10]:


evaluate_binary_model_with_threshold(
    name="HistGB (PCA + Optuna, Threshold 0.43)",
    model=final_model,
    X_train=X_train_pca,
    y_train=y_train,
    X_test=X_test_pca,
    y_test=y_test,
    threshold=0.43
)


# In[11]:


import numpy as np
import joblib
import cloudpickle
import os

# Clean internal HistGradientBoosting structure for Hugging Face compatibility
for estimator in final_model._predictors:
    for tree in estimator:
        if hasattr(tree, '_raw_prediction_tree'):
            node_struct = tree._raw_prediction_tree.nodes
            node_struct['feature_idx'] = node_struct['feature_idx'].astype(np.uint32)
            node_struct['left'] = node_struct['left'].astype(np.int32)
            node_struct['right'] = node_struct['right'].astype(np.int32)
            node_struct['threshold'] = node_struct['threshold'].astype(np.float32)

# Set output directory
output_path = r"C:\Users\angxi\OneDrive\Documents\Institute of Data Course\Capstone\4. Streamlit App Implementation\Final Files for Deployment"

# Save the model with cloudpickle
with open(os.path.join(output_path, "histgb_pca_model_clean.pkl"), "wb") as f:
    cloudpickle.dump(final_model, f)

# Save scaler and PCA with joblib
joblib.dump(scaler, os.path.join(output_path, "scaler.pkl"))
joblib.dump(pca, os.path.join(output_path, "pca.pkl"))

print("✅ All files successfully saved for Hugging Face deployment.")


# In[ ]:




