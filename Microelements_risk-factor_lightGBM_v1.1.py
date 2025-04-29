"""
Author: Lin, Wei-Zhi
Version: 1.1.0
Last updated: 20250429
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
df_data = pd.read_csv('import_df_TableMaker_2groups0429.csv')

# Remove rows with 'suspected normal'
df_data = df_data[df_data['group'] != 'suspected normal']

# Define features and target
features = ['age_at_include', 'sex', 'BMI', 
            'X78..Se....He..', 'X7..Li....No.gas..', 'X9..Be....No.gas..', 
            'X44..Ca....He..', 'X51..V....He..', 'X52..Cr....He..', 
            'X55..Mn....He..', 'X56..Fe....He..', 'X59..Co....He..', 
            'X60..Ni....He..', 'X63..Cu....He..', 'X66..Zn....He..', 
            'X71..Ga....He..', 'X72..Ge....He..', 'X75..As....He..', 
            'X78..Se....He...1', 'X85..Rb....He..', 'X88..Sr....He..', 
            'X90..Zr....He..', 'X95..Mo....He..', 'X107..Ag....He..', 
            'X111..Cd....He..', 'X115..In....He..', 'X118..Sn....He..', 
            'X121..Sb....He..', 'X125..Te....He..', 'X137..Ba....He..', 
            'X182..W....He..', 'X195..Pt....He..', 'X197..Au....He..', 
            'X201..Hg....He..', 'X205..Tl....He..', 'X208..Pb....He..', 
            'X209..Bi....He..', 'X232..Th....He..', 'X238..U....He..']

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
risk_score_test = model.predict_proba(X_test)[:, 1]  # Directly get probability for class 1 (Dyna/Sarcopenia)

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
# Calculate ROC metrics
fpr, tpr, roc_thresholds = roc_curve(y_test, risk_score_test)
roc_auc = roc_auc_score(y_test, risk_score_test)

# Find the index where ROC threshold is closest to the best_threshold
best_idx_for_roc = np.argmin(np.abs(roc_thresholds - best_threshold))

# Get the corresponding best FPR and TPR
best_fpr = fpr[best_idx_for_roc]
best_tpr = tpr[best_idx_for_roc]

# Start plotting
plt.figure(figsize=(8,6))
# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# Plot random guess line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# Plot best cutoff point
plt.scatter(best_fpr, best_tpr, color='red', edgecolor='black', s=100, label=f'Best Cutoff ({best_threshold:.2f})')
# Annotate the best cutoff on the plot
plt.annotate(f'{best_threshold:.2f}',
             (best_fpr, best_tpr),
             textcoords="offset points",
             xytext=(10, -10),
             ha='center',
             fontsize=10,
             color='red')
# Add plot labels
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(False)
plt.show()

#%% PART 2 — Full Risk Scoring for Entire Dataset
#%% 10 Label Risk Score for All Samples
# 1. Extract ID
sample_id = df_data['ID']

# 2. Combine ID and risk score into a new DataFrame
risk_table_all = pd.DataFrame({
    'ID': sample_id,
    'risk_score': risk_score_all
})

# 3. (Optional) Add original features if you want
for feature in features:
    risk_table_all[feature] = df_data[feature]

# 4. (Optional) Add true label if you have it
if 'group' in df_data.columns:
    risk_table_all['true_label'] = df_data['group']

# 5. Show preview
print("\n Risk Table (All Samples):")
print(risk_table_all.head())

# 6. Save to CSV
risk_table_all.to_csv('output_risk_score_all_samples.csv', index=False)
