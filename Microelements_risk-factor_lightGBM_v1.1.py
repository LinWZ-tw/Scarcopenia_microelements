"""
Author: Lin, Wei-Zhi
Version: 1.2.0
Last updated: 20250430
"""
#%% 00 Import Packages
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

#%% 01 Read and Prepare Data
# Load data
df_data = pd.read_csv('import_df.csv')

# Remove rows with 'suspected normal'
df_data = df_data[df_data['group'] != 'suspected normal']

# Define features and target 
features = ['age', 'gender', 'BMI', ... ]

X = df_data[features]
y = df_data['group']

# Map labels manually: Normal=0, Dyna/Sarcopenia=1
label_mapping = {'Normal': 0, 'Dyna/Sarcopenia': 1}
y_numeric = y.map(label_mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

#%% 02 Train LightGBM Model
model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    is_unbalance=True,     # Add this to help LightGBM know class imbalance
    metric='binary_logloss',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

#%% 03 Feature Importance Visualization
# LightGBM's built-in feature importance
lgb.plot_importance(model, importance_type='gain', figsize=(10, 6), title='Feature Importance (Gain)')
plt.show()

# SHAP feature importance
explainer = shap.Explainer(model)
shap_values_test = explainer(X_test)
shap.summary_plot(shap_values_test, X_test)

#%% 04 SHAP Feature Mean Table
shap_values_train = explainer(X_train)
mean_shap = np.abs(shap_values_train.values).mean(axis=0)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'mean_abs_shap': mean_shap
}).sort_values(by='mean_abs_shap', ascending=False)

# Calculate relative weights
feature_importance['weight'] = feature_importance['mean_abs_shap'] / feature_importance['mean_abs_shap'].sum()

print("\n📋 Feature Importance (SHAP):")
print(feature_importance)

#%% PART 1 — Model Evaluation on Test Set
#%% 05 Predict Risk Scores on Test Set
risk_score_test_raw = model.predict_proba(X_test)[:, 1]

# Auto-flip check based on AUC
risk_score_flipped = 1 - risk_score_test_raw
auc_raw = roc_auc_score(y_test, risk_score_test_raw)
auc_flip = roc_auc_score(y_test, risk_score_flipped)

if auc_flip > auc_raw:
    print(f"🔄 Flipping improves AUC: from {auc_raw:.2f} → {auc_flip:.2f}")
    risk_score_test = risk_score_flipped
    flipped = True
    roc_auc = auc_flip
else:
    print(f"✅ No flip needed: AUC = {auc_raw:.2f}")
    risk_score_test = risk_score_test_raw
    flipped = False
    roc_auc = auc_raw

#%% 06 Find Best Cutoff (Youden's Index)
thresholds = np.arange(0.0, 1.0, 0.01)
sensitivity = []
specificity = []
youden_index = []
accuracies = []

for t in thresholds:
    y_pred = (risk_score_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    sensitivity.append(sens)
    specificity.append(spec)
    accuracies.append(acc)
    youden_index.append(sens + spec - 1)

# Find best threshold
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
print(f"Best Cutoff Threshold (maximize Youden's Index): {best_threshold:.2f}")

#%% 07 Confusion Matrix and Metrics at Best Cutoff
y_pred_best = (risk_score_test >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_best)

print("\nConfusion Matrix at Best Cutoff:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Dyna/Sarcopenia'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix (Best Cutoff={best_threshold:.2f})')
plt.grid(False)
plt.show()

# Calculate performance
tn, fp, fn, tp = cm.ravel()
sensitivity_final = tp / (tp + fn)
specificity_final = tn / (tn + fp)
accuracy_final = (tp + tn) / (tp + tn + fp + fn)

print(f"\nPerformance at Best Cutoff:")
print(f"Sensitivity: {sensitivity_final:.2f}")
print(f"Specificity: {specificity_final:.2f}")
print(f"Accuracy: {accuracy_final:.2f}")

#%% 08 Plot Sensitivity, Specificity, Accuracy vs Threshold
plt.figure(figsize=(8,6))
plt.plot(thresholds, sensitivity, label='Sensitivity')
plt.plot(thresholds, specificity, label='Specificity')
plt.plot(thresholds, accuracies, label='Accuracy')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Cutoff ({best_threshold:.2f})')
plt.xlabel('Cutoff Risk Score')
plt.ylabel('Metric Value')
plt.title('Sensitivity, Specificity, Accuracy vs. Cutoff Threshold')
plt.legend()
plt.grid(False)
plt.show()

#%% 09 Plot ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, risk_score_test)

# Find best point on ROC curve
best_idx_for_roc = np.argmin(np.abs(roc_thresholds - best_threshold))
best_fpr = fpr[best_idx_for_roc]
best_tpr = tpr[best_idx_for_roc]

# Plot
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.scatter(best_fpr, best_tpr, color='red', edgecolor='black', s=100, label=f'Best Cutoff ({best_threshold:.2f})')
plt.annotate(f'{best_threshold:.2f}', (best_fpr, best_tpr), textcoords="offset points", xytext=(10, -10),
             ha='center', fontsize=10, color='red')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(False)
plt.show()

#%% PART 2 — Full Risk Scoring for Entire Dataset
#%% 10 Label Risk Score for All Samples
# Apply flip to full data if test score was flipped
risk_score_all_raw = model.predict_proba(X)[:, 1]
risk_score_all = 1 - risk_score_all_raw if flipped else risk_score_all_raw

plt.figure(figsize=(8,6))
plt.hist(risk_score_all, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Risk Score')
plt.ylabel('Number of Samples')
plt.title('Distribution of Risk Scores (All Samples)')
plt.grid(False)
plt.show()

# Combine ID and risk score into DataFrame
sample_id = df_data['ID']
risk_table_all = pd.DataFrame({
    'ID': sample_id,
    'risk_score': risk_score_all
})

# Add original features
for feature in features:
    risk_table_all[feature] = df_data[feature]

# Add true label
if 'group' in df_data.columns:
    risk_table_all['true_label'] = df_data['group']

# Preview
print("\n📋 Risk Table (All Samples):")
print(risk_table_all.head())

# Save
risk_table_all.to_csv('output_risk_score_all_samples.csv', index=False)
# %%

# Assume:
# - risk_score_all = model.predict_proba(X)[:, 1]
# - df_data contains Se in 'X78..Se....He..'

# 1. Create a DataFrame for convenience
se_risk_df = pd.DataFrame({
    'Se': df_data['X78..Se....He..'],
    'risk_score': risk_score_all
})

# 2. Filter samples near the decision threshold ± small margin
margin = 0.01  # you can adjust (e.g., 0.005 ~ 0.02)
near_cutoff = se_risk_df[
    (se_risk_df['risk_score'] >= best_threshold - margin) &
    (se_risk_df['risk_score'] <= best_threshold + margin)
]

# 3. Display the estimated Se cutoff range
print(f"\nSe concentration range around risk score = {best_threshold:.2f}:")
print(near_cutoff.sort_values(by='Se'))

# 4. Optional — compute median Se value at decision boundary
median_se = near_cutoff['Se'].median()
print(f"\nEstimated Se cutoff (median): {median_se:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(df_data['X78..Se....He..'], risk_score_all, alpha=0.6)
plt.axhline(best_threshold, color='red', linestyle='--', label=f'Risk Score = {best_threshold:.2f}')
plt.axvline(median_se, color='green', linestyle='--', label=f'Est. Se Cutoff = {median_se:.2f}')
plt.xlabel('Se concentration')
plt.ylabel('Predicted Risk Score')
plt.title('Se Concentration vs Predicted Risk Score')
plt.legend()
plt.grid(True)
plt.show()
