"""
Author: Lin, Wei-Zhi
Version: 1.0.0
Last updated: 20250429
"""

#%% 00 Import Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#%% 01 Load Your Data
df_data = pd.read_csv('import_data.csv')

#%% 02 Define Covariates and Metal Features
adjustment_features = ['age_at_include', 'sex', 'BMI']

test_features = [
    'X7..Li....No.gas..', 'X9..Be....No.gas..', 'X44..Ca....He..',
    'X51..V....He..', 'X52..Cr....He..', 'X55..Mn....He..',
    'X56..Fe....He..', 'X59..Co....He..', 'X60..Ni....He..',
    'X63..Cu....He..', 'X66..Zn....He..', 'X71..Ga....He..',
    'X72..Ge....He..', 'X75..As....He..', 'X78..Se....He..',
    'X85..Rb....He..', 'X88..Sr....He..', 'X90..Zr....He..',
    'X95..Mo....He..', 'X107..Ag....He..', 'X111..Cd....He..',
    'X115..In....He..', 'X118..Sn....He..', 'X121..Sb....He..',
    'X125..Te....He..', 'X137..Ba....He..', 'X182..W....He..',
    'X195..Pt....He..', 'X197..Au....He..', 'X201..Hg....He..',
    'X205..Tl....He..', 'X208..Pb....He..', 'X209..Bi....He..',
    'X232..Th....He..', 'X238..U....He..']

#%% 03 Define Outcome
# Convert group to binary (Normal=0, Dyna/Sarcopenia=1)
y = df_data['group'].map({'Normal': 0, 'Dyna/Sarcopenia': 1})

#%% 04 Run Adjusted Logistic Regression for Each Metal
results = []

for metal in test_features:
    try:
        # Select covariates + metal
        X = df_data[adjustment_features + [metal]].copy()
        X = sm.add_constant(X)
        
        # Fit logistic regression
        model = sm.Logit(y, X, missing='drop')
        res = model.fit(disp=False)

        # Extract statistics
        coef = res.params[metal]
        p_value = res.pvalues[metal]
        conf_int = res.conf_int().loc[metal]
        conf_low = conf_int[0]
        conf_high = conf_int[1]
        
        results.append({
            'metal': metal,
            'coef': coef,
            'p_value': p_value,
            'conf_low': conf_low,
            'conf_high': conf_high
        })
    
    except Exception as e:
        print(f"⚠Warning: Regression failed for {metal}: {e}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

#%% 05 Calculate Odds Ratios and Confidence Intervals
results_df['OR'] = np.exp(results_df['coef'])
results_df['OR_low'] = np.exp(results_df['conf_low'])
results_df['OR_high'] = np.exp(results_df['conf_high'])

#%% 06 Apply FDR Correction
reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
results_df['p_value_FDR'] = pvals_corrected
results_df['significant'] = reject

# Sort results
results_df = results_df.sort_values(by='p_value')

#%% 07 Output Final Report
final_report = results_df[['metal', 'OR', 'OR_low', 'OR_high', 'p_value', 'p_value_FDR', 'significant']]
final_report.to_csv('output_all_metal_regression_report.csv', index=False)

print("\noutput_all_metal_regression_report.csv'.")

#%% 08 (Optional) Volcano Plot
plt.figure(figsize=(10,8))
plt.scatter(results_df['coef'], -np.log10(results_df['p_value_FDR']),
            c=results_df['significant'].map({True: 'red', False: 'grey'}),
            alpha=0.7)

plt.axhline(-np.log10(0.05), color='blue', linestyle='--', label='FDR=0.05 Threshold')
plt.xlabel('Coefficient (Effect Size)')
plt.ylabel('-log10(FDR-adjusted p-value)')
plt.title('Volcano Plot: Metal Associations (FDR corrected)')
plt.legend()
plt.grid(False)
plt.show()
