"""
Statistical Analysis Script - Trained Tasks (SF and MATB)
Analyzes learning rates, performance differences, and correlations with executive functions
Uses: df_Best_SF_MATB_Mean_Zscores.csv and df_demographics_EFs.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import os
import sys
from datetime import datetime

def get_significance_stars(p_value):
    """Convert p-value to asterisks: ***<0.001, **<0.01, *<0.05, ns>=0.05"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def format_diag_value(value, decimals=4):
    """Format diagnostic values consistently for logs and tables."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def compute_group_residual_variance_tests(result, group_col='Group'):
    """Test whether residual variance differs across groups."""
    tests = {
        'Levene_Group_p': np.nan,
        'Levene_interpretation': 'Not tested',
        'Brown_Forsythe_Group_p': np.nan,
        'Brown_Forsythe_interpretation': 'Not tested',
        'Residual_Variance_Ratio_MaxMin': np.nan,
        'Group_Residual_Variance_Details': ''
    }

    model_df = getattr(getattr(result.model, 'data', None), 'frame', None)
    if model_df is None or group_col not in model_df.columns:
        return tests

    resid_values = np.asarray(result.resid)
    if len(resid_values) != len(model_df):
        return tests

    diag_df = model_df[[group_col]].reset_index(drop=True).copy()
    diag_df['resid'] = resid_values
    diag_df = diag_df.dropna(subset=[group_col, 'resid'])

    group_samples = []
    group_variances = {}
    for group_name, group_data in diag_df.groupby(group_col):
        values = group_data['resid'].values
        if len(values) >= 2:
            group_samples.append(values)
            group_variances[group_name] = float(np.var(values, ddof=1))

    if len(group_samples) < 2:
        return tests

    try:
        _, levene_p = stats.levene(*group_samples, center='mean')
        tests['Levene_Group_p'] = float(levene_p)
        tests['Levene_interpretation'] = 'OK' if levene_p >= 0.05 else 'Potential issue'
    except Exception:
        pass

    try:
        _, brown_forsythe_p = stats.levene(*group_samples, center='median')
        tests['Brown_Forsythe_Group_p'] = float(brown_forsythe_p)
        tests['Brown_Forsythe_interpretation'] = 'OK' if brown_forsythe_p >= 0.05 else 'Potential issue'
    except Exception:
        pass

    if group_variances:
        variance_values = [var for var in group_variances.values() if pd.notna(var)]
        if variance_values and min(variance_values) > 0:
            tests['Residual_Variance_Ratio_MaxMin'] = max(variance_values) / min(variance_values)
        tests['Group_Residual_Variance_Details'] = '; '.join(
            [f"{grp}:var={var:.4f}" for grp, var in sorted(group_variances.items())]
        )

    return tests


def evaluate_mixedlm_assumptions(model_name, result, storage, check_group_variance=False):
    """Run residual diagnostics for a fitted mixed-effects model."""
    resid = pd.Series(result.resid).dropna()
    n_obs = int(len(resid))

    dw_stat = np.nan
    bp_pvalue = np.nan
    sw_pvalue = np.nan

    if n_obs >= 3:
        dw_stat = float(durbin_watson(resid.values))

        exog = getattr(result.model, 'exog', None)
        if exog is not None and n_obs > exog.shape[1]:
            try:
                _, bp_pvalue, _, _ = het_breuschpagan(resid.values, exog)
            except Exception:
                bp_pvalue = np.nan

        try:
            _, sw_pvalue = stats.shapiro(resid.values)
        except Exception:
            sw_pvalue = np.nan

    row = {
        'Model': model_name,
        'N_obs': n_obs,
        'Durbin_Watson': dw_stat,
        'DW_interpretation': 'OK' if pd.notna(dw_stat) and 1.5 <= dw_stat <= 2.5 else 'Potential issue',
        'Breusch_Pagan_p': bp_pvalue,
        'BP_interpretation': 'OK' if pd.notna(bp_pvalue) and bp_pvalue >= 0.05 else 'Potential issue',
        'Shapiro_Wilk_p': sw_pvalue,
        'SW_interpretation': 'OK' if pd.notna(sw_pvalue) and sw_pvalue >= 0.05 else 'Potential issue',
        'Levene_Group_p': np.nan,
        'Levene_interpretation': 'Not tested',
        'Brown_Forsythe_Group_p': np.nan,
        'Brown_Forsythe_interpretation': 'Not tested',
        'Residual_Variance_Ratio_MaxMin': np.nan,
        'Group_Residual_Variance_Details': ''
    }

    if check_group_variance:
        row.update(compute_group_residual_variance_tests(result, group_col='Group'))

    storage.append(row)

    print(f"\nAssumption checks - {model_name}")
    print(f"  Durbin-Watson: {format_diag_value(dw_stat, 3)} ({row['DW_interpretation']})")
    print(f"  Breusch-Pagan p-value: {format_diag_value(bp_pvalue)} ({row['BP_interpretation']})")
    print(f"  Shapiro-Wilk p-value: {format_diag_value(sw_pvalue)} ({row['SW_interpretation']})")
    if check_group_variance:
        print(f"  Levene (group residual variance) p-value: {format_diag_value(row['Levene_Group_p'])} ({row['Levene_interpretation']})")
        print(f"  Brown-Forsythe (group residual variance) p-value: {format_diag_value(row['Brown_Forsythe_Group_p'])} ({row['Brown_Forsythe_interpretation']})")
        print(f"  Residual variance ratio max/min: {format_diag_value(row['Residual_Variance_Ratio_MaxMin'], 3)}")
        if row['Group_Residual_Variance_Details']:
            print(f"  Group residual variances: {row['Group_Residual_Variance_Details']}")


def extract_group_and_interaction_pvalues(result):
    """Extract key group and session-by-group p-values from a fitted model result."""
    pvalues = result.pvalues
    group_p = pvalues.get('Group[T.tRNS]', np.nan)

    interaction_terms = [term for term in pvalues.index if ('Session' in term and 'Group' in term)]
    interaction_values = [pvalues[term] for term in interaction_terms if pd.notna(pvalues[term])]
    min_interaction_p = float(np.min(interaction_values)) if interaction_values else np.nan
    interaction_details = '; '.join([f"{term}:{pvalues[term]:.4f}" for term in interaction_terms if pd.notna(pvalues[term])])

    return group_p, min_interaction_p, interaction_details


def run_gee_sensitivity(model_name, formula, data, group_col, mixedlm_result, storage):
    """Run sandwich-robust marginal confirmation model and compare with MixedLM."""
    mixed_group_p, mixed_interaction_p, mixed_interaction_details = extract_group_and_interaction_pvalues(mixedlm_result)

    row = {
        'Model': model_name,
        'N_obs': len(data),
        'Estimator': 'GEE (sandwich robust covariance)',
        'MixedLM_Group_p': mixed_group_p,
        'MixedLM_Min_SessionGroup_p': mixed_interaction_p,
        'MixedLM_SessionGroup_Details': mixed_interaction_details,
        'GEE_Group_p': np.nan,
        'GEE_Min_SessionGroup_p': np.nan,
        'GEE_SessionGroup_Details': '',
        'Sandwich_Group_p': np.nan,
        'Sandwich_Min_SessionGroup_p': np.nan,
        'Sandwich_SessionGroup_Details': '',
        'Status': 'OK'
    }

    try:
        gee_model = smf.gee(
            formula=formula,
            groups=data[group_col],
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable()
        )
        gee_result = gee_model.fit(cov_type='robust')
        gee_group_p, gee_interaction_p, gee_interaction_details = extract_group_and_interaction_pvalues(gee_result)

        row['GEE_Group_p'] = gee_group_p
        row['GEE_Min_SessionGroup_p'] = gee_interaction_p
        row['GEE_SessionGroup_Details'] = gee_interaction_details
        row['Sandwich_Group_p'] = gee_group_p
        row['Sandwich_Min_SessionGroup_p'] = gee_interaction_p
        row['Sandwich_SessionGroup_Details'] = gee_interaction_details

        print(f"\nSandwich-robust marginal confirmation (GEE) - {model_name}")
        print(f"  Group p-value: MixedLM={format_diag_value(mixed_group_p)} | Sandwich={format_diag_value(gee_group_p)}")
        print(f"  Min Session×Group p-value: MixedLM={format_diag_value(mixed_interaction_p)} | Sandwich={format_diag_value(gee_interaction_p)}")
    except Exception as exc:
        row['Status'] = f"GEE failed: {str(exc)}"
        print(f"\nSandwich confirmation failed for {model_name}: {exc}")

    storage.append(row)

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_date = datetime.now().strftime("%Y_%m_%d")
log_filename = f"{script_name}_{log_date}.txt"

script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, "..", "logs", log_filename)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

logger = Logger(log_path)
sys.stdout = logger

print("="*80)
print("STATISTICAL ANALYSIS - TRAINED TASKS (SF & MATB)")
print("="*80)
print(f"\nLog file: {log_path}")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

lme_assumptions_results = []
gee_sensitivity_results = []

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
transformed_dir = os.path.join(project_dir, "transformed_data")
plot_dir = os.path.join(project_dir, "plots", "Trained_tasks")
os.makedirs(plot_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Data loading and harmonization
# -----------------------------------------------------------------------------
print("\nLoading data files...")
df_best = pd.read_csv(os.path.join(transformed_dir, "df_Best_SF_MATB_Mean_Zscores.csv"))
df_demographics_EF = pd.read_csv(os.path.join(transformed_dir, "df_demographics_EFs.csv"))

print(f"  df_Best_SF_MATB_Mean_Zscores: {df_best.shape}")
print(f"  df_demographics_EFs: {df_demographics_EF.shape}")

print("\nRecoding groups (A -> tRNS, B -> Sham)...")
group_mapping = {"A": "tRNS", "B": "Sham"}

if 'Group' in df_best.columns:
    df_best['Group'] = df_best['Group'].replace(group_mapping)
elif 'Group_MATB' in df_best.columns:
    df_best['Group'] = df_best['Group_MATB'].replace(group_mapping)
elif 'Group_SF' in df_best.columns:
    df_best['Group'] = df_best['Group_SF'].replace(group_mapping)

if 'Group' in df_demographics_EF.columns:
    df_demographics_EF['Group'] = df_demographics_EF['Group'].replace(group_mapping)

group_colors = {"tRNS": "orange", "Sham": "grey"}

# Session order is used both for modeling subsets and for consistent plotting.
session_order = ['E1', 'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'E2', 'E3']
session_to_num = {sess: idx + 1 for idx, sess in enumerate(session_order)}
df_best['session_factor'] = df_best['Session'].map(session_to_num)

df_best = df_best[df_best['Session'].isin(session_order)].copy()

print(f"\nSessions included: {sorted(df_best['Session'].unique())}")
print(f"Total data points: {len(df_best)}")
print(f"Unique participants: {df_best['Participant'].nunique()}")

print("\n" + "="*80)
print("DEMOGRAPHICS")
print("="*80)

demographics_table = df_demographics_EF.groupby('Group').agg(
    N=('Participant', 'count'),
    Men=('Genre', lambda x: (x == 'H').sum()),
    Women=('Genre', lambda x: (x == 'F').sum()),
    Right_Handed=('Lateralite', lambda x: (x == 'Droitier').sum()),
    Left_Handed=('Lateralite', lambda x: (x == 'Gaucher').sum()),
    Mean_Age=('Age', 'mean'),
    SD_Age=('Age', 'std'),
    Mean_Education_Level=('NiveauEtudes', 'mean'),
    Mean_Flight_Hours=('HeuresVol', 'mean'),
    Mean_VGexp=('VGexp', 'mean')
).reset_index()

print(demographics_table.to_string())

print("\n" + "="*80)
print("HYPOTHESIS 1: LEARNING RATE")
print("="*80)
print("H1a: tRNS group will have better learning rate in Space Fortress")
print("H1b: tRNS group will have better learning rate in MATB")

# Compute group-level means and 95% CIs for retained learning-rate visualization.
summary_data_sf = df_best.groupby(['session_factor', 'Session', 'Group']).agg(
    mean_score=('SF_zscore', 'mean'),
    se_score=('SF_zscore', lambda x: x.std() / np.sqrt(len(x))),
    n=('SF_zscore', 'count')
).reset_index()

summary_data_sf['ci_lower'] = summary_data_sf.apply(
    lambda row: row['mean_score'] - t_dist.ppf(0.975, row['n'] - 1) * row['se_score'] if row['n'] > 1 else row['mean_score'], axis=1
)
summary_data_sf['ci_upper'] = summary_data_sf.apply(
    lambda row: row['mean_score'] + t_dist.ppf(0.975, row['n'] - 1) * row['se_score'] if row['n'] > 1 else row['mean_score'], axis=1
)

summary_data_MATB = df_best.groupby(['session_factor', 'Session', 'Group']).agg(
    mean_score=('MATB_z_mean', 'mean'),
    se_score=('MATB_z_mean', lambda x: x.std() / np.sqrt(len(x))),
    n=('MATB_z_mean', 'count')
).reset_index()

summary_data_MATB['ci_lower'] = summary_data_MATB.apply(
    lambda row: row['mean_score'] - t_dist.ppf(0.975, row['n'] - 1) * row['se_score'] if row['n'] > 1 else row['mean_score'], axis=1
)
summary_data_MATB['ci_upper'] = summary_data_MATB.apply(
    lambda row: row['mean_score'] + t_dist.ppf(0.975, row['n'] - 1) * row['se_score'] if row['n'] > 1 else row['mean_score'], axis=1
)

# Training sessions are modeled with a log(session) slope per participant.
df_training = df_best[df_best['Session'].str.match(r'T0[1-9]|T10')].copy()
df_training['Session_Num'] = df_training['Session'].str.replace('T0', '').str.replace('T', '').astype(int)


def fit_log_model(group_data, score_column):
    """Fit log(x) model and return coefficient (learning rate)"""
    valid_data = group_data[['Session_Num', score_column]].dropna()
    if len(valid_data) < 2:
        return np.nan
    X = np.log(valid_data['Session_Num'].values).reshape(-1, 1)
    y = valid_data[score_column].values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]


def print_independent_ttest_summary(title, trns_values, sham_values):
    """Print a standardized independent t-test summary and return test stats."""
    t_stat, p_val = stats.ttest_ind(trns_values, sham_values)
    print("\n" + "-"*80)
    print(title)
    print("-"*80)
    print(f"tRNS: n={len(trns_values)}, mean={trns_values.mean():.4f}, SD={trns_values.std():.4f}")
    print(f"Sham: n={len(sham_values)}, mean={sham_values.mean():.4f}, SD={sham_values.std():.4f}")
    print("\nIndependent t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")
    return t_stat, p_val


def fit_mixedlm_with_reporting(
    heading,
    formula,
    model_data,
    assumption_label,
    gee_label,
    check_group_variance=False
):
    """Fit a MixedLM, print summary, then run assumptions and sandwich sensitivity checks."""
    print("\n" + "-"*80)
    print(heading)
    print("-"*80)
    print(f"Model: {formula}")
    print("Random effects: Participant")

    model = smf.mixedlm(formula, model_data, groups=model_data["Participant"])
    result = model.fit()
    print(result.summary())

    evaluate_mixedlm_assumptions(
        assumption_label,
        result,
        lme_assumptions_results,
        check_group_variance=check_group_variance
    )
    run_gee_sensitivity(
        gee_label,
        formula,
        model_data,
        "Participant",
        result,
        gee_sensitivity_results
    )
    return result


# Post-hoc per-session group comparisons used for significance markers in the kept plots.
def compute_session_group_ttests(eval_df, outcome_col, section_title):
    """Compute and print tRNS vs Sham comparisons in E1/E2/E3 sessions."""
    print("\n" + "-"*80)
    print(section_title)
    print("-"*80)
    group_pvalues = {}
    for session in ['E1', 'E2', 'E3']:
        trns_values = eval_df[(eval_df['Session'] == session) & (eval_df['Group'] == 'tRNS')][outcome_col]
        sham_values = eval_df[(eval_df['Session'] == session) & (eval_df['Group'] == 'Sham')][outcome_col]
        if len(trns_values) > 0 and len(sham_values) > 0:
            t_stat, p_val = stats.ttest_ind(trns_values, sham_values)
            group_pvalues[session] = p_val
            print(
                f"{session}: tRNS (n={len(trns_values)}, m={trns_values.mean():.3f}) "
                f"vs Sham (n={len(sham_values)}, m={sham_values.mean():.3f}), "
                f"t={t_stat:.3f}, p={p_val:.3f}"
            )
    return group_pvalues


# Participant-level learning-rate extraction per task.
learning_rates_SF = df_training.groupby(['Participant', 'Group'], group_keys=False).apply(
    lambda x: pd.Series({'LearningRate_SF': fit_log_model(x, 'SF_zscore')}), include_groups=False
).reset_index()

learning_rates_MATB = df_training.groupby(['Participant', 'Group'], group_keys=False).apply(
    lambda x: pd.Series({'LearningRate_MATB': fit_log_model(x, 'MATB_z_mean')}), include_groups=False
).reset_index()

learning_rates_SF = learning_rates_SF.dropna(subset=['LearningRate_SF'])
learning_rates_MATB = learning_rates_MATB.dropna(subset=['LearningRate_MATB'])

tRNS_sf = learning_rates_SF[learning_rates_SF['Group'] == 'tRNS']['LearningRate_SF']
Sham_sf = learning_rates_SF[learning_rates_SF['Group'] == 'Sham']['LearningRate_SF']
t_stat_sf, p_val_sf = print_independent_ttest_summary("H1a: SF Learning Rate Comparison", tRNS_sf, Sham_sf)

tRNS_matb = learning_rates_MATB[learning_rates_MATB['Group'] == 'tRNS']['LearningRate_MATB']
Sham_matb = learning_rates_MATB[learning_rates_MATB['Group'] == 'Sham']['LearningRate_MATB']
t_stat_matb, p_val_matb = print_independent_ttest_summary("H1b: MATB Learning Rate Comparison", tRNS_matb, Sham_matb)

# Covariate check: whether gaming experience explains part of the learning-rate effect.
print("\n" + "-"*80)
print("H1a/H1b: Learning Rate with VGexp Covariate")
print("-"*80)

lr_sf_cov = learning_rates_SF.merge(df_demographics_EF[['Participant', 'VGexp']], on='Participant', how='left')
lr_sf_cov = lr_sf_cov.dropna(subset=['LearningRate_SF', 'VGexp', 'Group'])
print(f"SF learning rate observations with VGexp: {len(lr_sf_cov)}")
if len(lr_sf_cov) > 4:
    model_lr_sf = smf.ols("LearningRate_SF ~ C(Group) + VGexp", data=lr_sf_cov).fit()
    print(model_lr_sf.summary())
    print(f"Group effect (p): {model_lr_sf.pvalues.get('C(Group)[T.tRNS]', np.nan):.4f}")
    print(f"VGexp effect (p): {model_lr_sf.pvalues.get('VGexp', np.nan):.4f}")
else:
    print("Insufficient data for SF learning-rate covariate model")

lr_matb_cov = learning_rates_MATB.merge(df_demographics_EF[['Participant', 'VGexp']], on='Participant', how='left')
lr_matb_cov = lr_matb_cov.dropna(subset=['LearningRate_MATB', 'VGexp', 'Group'])
print(f"MATB learning rate observations with VGexp: {len(lr_matb_cov)}")
if len(lr_matb_cov) > 4:
    model_lr_matb = smf.ols("LearningRate_MATB ~ C(Group) + VGexp", data=lr_matb_cov).fit()
    print(model_lr_matb.summary())
    print(f"Group effect (p): {model_lr_matb.pvalues.get('C(Group)[T.tRNS]', np.nan):.4f}")
    print(f"VGexp effect (p): {model_lr_matb.pvalues.get('VGexp', np.nan):.4f}")
else:
    print("Insufficient data for MATB learning-rate covariate model")

print("\nComputing learning rates for training sessions (T01-T10)...")
print("\n" + "="*80)
print("HYPOTHESIS 2: PERFORMANCE COMPARISON (TRAINED TASKS)")
print("="*80)
print("H2a: tRNS better than Sham in Space Fortress after training")
print("H2b: tRNS better than Sham in MATB after training")

# Restrict performance analyses to evaluation sessions (baseline, short-term, long-term).
df_eval = df_best[df_best['Session'].isin(['E1', 'E2', 'E3'])].copy()

session_labels = {'E1': 'Reference', 'E2': 'Short-term', 'E3': 'Long-term'}
df_eval['Session_Label'] = df_eval['Session'].map(session_labels)
df_eval['Session_Label'] = pd.Categorical(df_eval['Session_Label'], 
                                          categories=['Reference', 'Short-term', 'Long-term'], 
                                          ordered=True)

df_eval_sf = df_eval[['SF_zscore', 'Session', 'Group', 'Participant']].dropna()
result_sf = fit_mixedlm_with_reporting(
    heading="H2a: Mixed Model for SF Performance",
    formula="SF_zscore ~ Session + Group + Session:Group",
    model_data=df_eval_sf,
    assumption_label="H2a Base: SF_zscore ~ Session + Group + Session:Group",
    gee_label="H2a Base: SF_zscore ~ Session + Group + Session:Group",
    check_group_variance=False
)
sf_pvalues = result_sf.pvalues
sf_group_pvalues = compute_session_group_ttests(df_eval, 'SF_zscore', "SF Performance: Group Comparisons within Each Session")

df_eval_matb = df_eval[['MATB_z_mean', 'Session', 'Group', 'Participant']].dropna()
result_matb = fit_mixedlm_with_reporting(
    heading="H2b: Mixed Model for MATB Performance",
    formula="MATB_z_mean ~ Session + Group + Session:Group",
    model_data=df_eval_matb,
    assumption_label="H2b Base: MATB_z_mean ~ Session + Group + Session:Group",
    gee_label="H2b Base: MATB_z_mean ~ Session + Group + Session:Group",
    check_group_variance=True
)
matb_pvalues = result_matb.pvalues
matb_group_pvalues = compute_session_group_ttests(df_eval, 'MATB_z_mean', "MATB Performance: Group Comparisons within Each Session")


# -----------------------------------------------------------------------------
# Retained Figure 1: Combined learning rates (stacked)
# -----------------------------------------------------------------------------
print("\nCreating vertically stacked learning rates plot (MATB + SF)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0.28})

for group in ['Sham', 'tRNS']:
    data = summary_data_MATB[summary_data_MATB['Group'] == group]
    x_plot = data['session_factor'].replace({13: 14})
    offset = -0.1 if group == 'Sham' else 0.1
    ax1.errorbar(x_plot + offset, data['mean_score'], 
                yerr=[data['mean_score'] - data['ci_lower'], data['ci_upper'] - data['mean_score']],
                fmt='o', color=group_colors[group], label=group, capsize=5, markersize=6)
    
    training_means = df_training[df_training['Group'] == group].groupby('Session_Num')['MATB_z_mean'].mean()
    if len(training_means) > 1:
        slope = learning_rates_MATB[learning_rates_MATB['Group'] == group]['LearningRate_MATB'].mean()
        first_session = training_means.index.min()
        intercept = training_means.loc[first_session] - slope * np.log(first_session)

        x_train = training_means.index.values
        y_fit = slope * np.log(x_train) + intercept
        ax1.plot(x_train + 1, y_fit, color=group_colors[group], linestyle='-', linewidth=2)  # +1 to align with session_factor axis

        y_pos = 0.14 if group == 'Sham' else 0.07
        sign = '+' if intercept >= 0 else '-'
        equation_text = f'{group}: y = {slope:.3f}·ln(x) {sign} {abs(intercept):.3f}'
        ax1.text(0.98, y_pos, equation_text, transform=ax1.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='none', edgecolor=group_colors[group], linewidth=1.5))

ax1.axvspan(1.5, 11.5, alpha=0.2, color='grey', label='Training Period')
ax1.set_ylabel('MATB Z-Score', fontsize=12, fontweight='bold')
ax1.text(-0.08, 1.05, 'A.', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

ax1.text(1, 1.02, 'Baseline', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax1.text(6.5, 1.02, 'Training', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax1.text(12, 1.02, 'ST', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax1.text(14, 1.02, 'LT', ha='center', va='bottom', transform=ax1.get_xaxis_transform(), fontsize=10, fontweight='bold')

break_x = 13
break_dy = 0.025
trans = ax1.get_xaxis_transform()
ax1.plot([break_x - 0.06, break_x - 0.02], [-break_dy, break_dy], 'k-', linewidth=1, 
         clip_on=False, transform=trans, zorder=11)
ax1.plot([break_x + 0.02, break_x + 0.06], [-break_dy, break_dy], 'k-', linewidth=1, 
         clip_on=False, transform=trans, zorder=11)

ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
ax1.set_xticklabels(['E1', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'E2', 'E3'])
ax1.tick_params(axis='x', labelbottom=True)

y_lim = ax1.get_ylim()
ax1.set_ylim(y_lim[0], 1.4)
y_lim = ax1.get_ylim()
y_range = y_lim[1] - y_lim[0]
y_comparison = y_lim[1] - 0.05 * y_range

stars = get_significance_stars(p_val_matb)
if stars != 'ns':
    ax1.plot([13.9, 14.1], [y_comparison, y_comparison], 'k-', linewidth=2)
    ax1.text(14, y_comparison + 0.01 * y_range, f'{stars}', 
        ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.legend(loc='lower center')
ax1.grid(alpha=0.3)

for group in ['Sham', 'tRNS']:
    data = summary_data_sf[summary_data_sf['Group'] == group]
    x_plot = data['session_factor'].replace({13: 14})
    offset = -0.1 if group == 'Sham' else 0.1
    ax2.errorbar(x_plot + offset, data['mean_score'], 
                yerr=[data['mean_score'] - data['ci_lower'], data['ci_upper'] - data['mean_score']],
                fmt='o', color=group_colors[group], label=group, capsize=5, markersize=6)
    
    training_means = df_training[df_training['Group'] == group].groupby('Session_Num')['SF_zscore'].mean()
    if len(training_means) > 1:
        slope = learning_rates_SF[learning_rates_SF['Group'] == group]['LearningRate_SF'].mean()
        first_session = training_means.index.min()
        intercept = training_means.loc[first_session] - slope * np.log(first_session)

        x_train = training_means.index.values
        y_fit = slope * np.log(x_train) + intercept
        ax2.plot(x_train + 1, y_fit, color=group_colors[group], linestyle='-', linewidth=2)  # +1 to align with session_factor axis

        y_pos = 0.14 if group == 'Sham' else 0.07
        sign = '+' if intercept >= 0 else '-'
        equation_text = f'{group}: y = {slope:.3f}·ln(x) {sign} {abs(intercept):.3f}'
        ax2.text(0.98, y_pos, equation_text, transform=ax2.transAxes,
            fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='none', edgecolor=group_colors[group], linewidth=1.5))

ax2.axvspan(1.5, 11.5, alpha=0.2, color='grey', label='Training Period')
ax2.set_xlabel('Session', fontsize=12, fontweight='bold')
ax2.set_ylabel('SF Z-Score', fontsize=12, fontweight='bold')
ax2.text(-0.08, 1.05, 'B.', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

ax2.text(1, 1.02, 'Baseline', ha='center', va='bottom', transform=ax2.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax2.text(6.5, 1.02, 'Training', ha='center', va='bottom', transform=ax2.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax2.text(12, 1.02, 'ST', ha='center', va='bottom', transform=ax2.get_xaxis_transform(), fontsize=10, fontweight='bold')
ax2.text(14, 1.02, 'LT', ha='center', va='bottom', transform=ax2.get_xaxis_transform(), fontsize=10, fontweight='bold')

ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])
ax2.set_xticklabels(['E1', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'E2', 'E3'])

trans = ax2.get_xaxis_transform()
ax2.plot([break_x - 0.06, break_x - 0.02], [-break_dy, break_dy], 'k-', linewidth=1, 
         clip_on=False, transform=trans, zorder=11)
ax2.plot([break_x + 0.02, break_x + 0.06], [-break_dy, break_dy], 'k-', linewidth=1, 
         clip_on=False, transform=trans, zorder=11)

ax2.legend(loc='lower center')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(hspace=0.32)
plt.savefig(os.path.join(plot_dir, 'Combined_LearningRates_Stacked.png'), dpi=300, bbox_inches='tight')
print("  Combined stacked learning rates plot saved")
plt.close()

# -----------------------------------------------------------------------------
# Retained Figure 2: Combined performances (stacked mean + points + 95% CI)
# -----------------------------------------------------------------------------
print("Creating vertically stacked performances plot (Mean + Points + 95% CI)...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

sns.stripplot(data=df_eval, x='Session_Label', y='MATB_z_mean', hue='Group',
              palette=group_colors, ax=ax1, dodge=True, alpha=0.45, legend=False, size=6,
              edgecolor='black', linewidth=0.5)
sns.pointplot(data=df_eval, x='Session_Label', y='MATB_z_mean', hue='Group',
              palette=group_colors, ax=ax1, dodge=0.2, errorbar=('ci', 95), join=True,
              markers='D', linestyles='-', scale=1.0, estimator=np.mean, capsize=0.15)

ax1.set_ylabel('MATB Z-Score', fontsize=12, fontweight='bold')
ax1.set_xlabel('')
ax1.text(-0.08, 1.05, 'A.', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
ax1.legend(title='Group', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

y_lim = ax1.get_ylim()
y_range = y_lim[1] - y_lim[0]
ax1.set_ylim(y_lim[0], y_lim[1] + 0.18 * y_range)
y_lim = ax1.get_ylim()
y_range = y_lim[1] - y_lim[0]
y_top = y_lim[1] - 0.055 * y_range
line_height = 0.01 * y_range

if 'Session[T.E2]' in matb_pvalues:
    stars = get_significance_stars(matb_pvalues['Session[T.E2]'])
    ax1.plot([0, 1], [y_top, y_top], 'k-', linewidth=1.5)
    ax1.text(0.5, y_top + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

if 'Session[T.E3]' in matb_pvalues:
    stars = get_significance_stars(matb_pvalues['Session[T.E3]'])
    y_top2 = y_top - 0.06 * y_range if 'Session[T.E2]' in matb_pvalues else y_top
    ax1.plot([0, 2], [y_top2, y_top2], 'k-', linewidth=1.5)
    ax1.text(1, y_top2 + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

y_offset_positions = [0, 1, 2]
y_group_line = y_top - 0.115 * y_range
for idx, session_label in enumerate(['Reference', 'Short-term', 'Long-term']):
    session_code = ['E1', 'E2', 'E3'][idx]
    if session_code in matb_group_pvalues:
        p_val = matb_group_pvalues[session_code]
        stars = get_significance_stars(p_val)
        x_pos = y_offset_positions[idx]
        ax1.plot([x_pos - 0.075, x_pos + 0.075], [y_group_line, y_group_line], 'k-', linewidth=1)
        ax1.text(x_pos, y_group_line + 0.012 * y_range, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

sns.stripplot(data=df_eval, x='Session_Label', y='SF_zscore', hue='Group',
              palette=group_colors, ax=ax2, dodge=True, alpha=0.45, legend=False, size=6,
              edgecolor='black', linewidth=0.5)
sns.pointplot(data=df_eval, x='Session_Label', y='SF_zscore', hue='Group',
              palette=group_colors, ax=ax2, dodge=0.2, errorbar=('ci', 95), join=True,
              markers='D', linestyles='-', scale=1.0, estimator=np.mean, capsize=0.15)

ax2.set_ylabel('SF Z-Score', fontsize=12, fontweight='bold')
ax2.set_xlabel('Evaluation Session', fontsize=12, fontweight='bold')
ax2.text(-0.08, 1.05, 'B.', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
ax2.legend(title='Group', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

y_lim = ax2.get_ylim()
y_range = y_lim[1] - y_lim[0]
ax2.set_ylim(y_lim[0], y_lim[1] + 0.18 * y_range)
y_lim = ax2.get_ylim()
y_range = y_lim[1] - y_lim[0]
y_top = y_lim[1] - 0.055 * y_range
line_height = 0.01 * y_range

if 'Session[T.E2]' in sf_pvalues:
    stars = get_significance_stars(sf_pvalues['Session[T.E2]'])
    ax2.plot([0, 1], [y_top, y_top], 'k-', linewidth=1.5)
    ax2.text(0.5, y_top + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

if 'Session[T.E3]' in sf_pvalues:
    stars = get_significance_stars(sf_pvalues['Session[T.E3]'])
    y_top2 = y_top - 0.06 * y_range if 'Session[T.E2]' in sf_pvalues else y_top
    ax2.plot([0, 2], [y_top2, y_top2], 'k-', linewidth=1.5)
    ax2.text(1, y_top2 + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

y_offset_positions = [0, 1, 2]
y_group_line = y_top - 0.115 * y_range
for idx, session_label in enumerate(['Reference', 'Short-term', 'Long-term']):
    session_code = ['E1', 'E2', 'E3'][idx]
    if session_code in sf_group_pvalues:
        p_val = sf_group_pvalues[session_code]
        stars = get_significance_stars(p_val)
        x_pos = y_offset_positions[idx]
        ax2.plot([x_pos - 0.075, x_pos + 0.075], [y_group_line, y_group_line], 'k-', linewidth=1)
        ax2.text(x_pos, y_group_line + 0.012 * y_range, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Combined_Performances_Stacked_MeanCI95.png'), dpi=300, bbox_inches='tight')
print("  Combined stacked performances Mean+Points+95%CI plot saved")
plt.close()


# Exploratory correlations between gaming/flight experience and baseline/retention performance.
def print_corr(df, x_col, y_col, x_label, y_label):
    subset = df[[x_col, y_col]].dropna()
    if len(subset) > 2 and subset[x_col].nunique() > 1 and subset[y_col].nunique() > 1:
        r, p = stats.pearsonr(subset[x_col], subset[y_col])
        print(f"{x_label} vs {y_label}: r={r:.4f}, p={p:.4f}, n={len(subset)}")
    else:
        print(f"{x_label} vs {y_label}: insufficient data (n={len(subset)})")

print("\n" + "-"*80)
print("EXPLORATORY: Gaming & Flight Experience Correlations (Baseline E1)")
print("-"*80)

df_baseline_perf = df_best[df_best['Session'] == 'E1'].copy()
baseline_perf = df_baseline_perf.groupby('Participant').agg({
    'SF_zscore': 'mean',
    'MATB_z_mean': 'mean',
    'Flight': 'mean'
}).reset_index()
baseline_perf.columns = ['Participant', 'Base_SF', 'Base_MATB', 'Base_Flight']

corr_base = df_demographics_EF[['Participant', 'VGexp', 'HeuresVol', 'HeuresSimu']].copy()
baseline_merged = corr_base.merge(baseline_perf, on='Participant', how='inner')
print(f"Participants with baseline performance and experience data: {len(baseline_merged)}")

print("\nGaming experience (VGexp) correlations:")
print_corr(baseline_merged, 'VGexp', 'Base_SF', 'Gaming exp', 'SF performance (baseline)')
print_corr(baseline_merged, 'VGexp', 'Base_MATB', 'Gaming exp', 'MATB performance (baseline)')
print_corr(baseline_merged, 'VGexp', 'Base_Flight', 'Gaming exp', 'SUB_Flight SF (baseline)')
# print_corr(baseline_merged, 'VGexp', 'Base_Flight', 'Gaming exp', 'Flight sim performance (baseline)')

print("\nFlight hours (HeuresVol) correlations:")
print_corr(baseline_merged, 'HeuresVol', 'Base_SF', 'Flight hours', 'SF performance (baseline)')
print_corr(baseline_merged, 'HeuresVol', 'Base_MATB', 'Flight hours', 'MATB performance (baseline)')
print_corr(baseline_merged, 'HeuresVol', 'HeuresSimu', 'Flight hours', 'SUB_Flight SF')
# print_corr(baseline_merged, 'HeuresVol', 'HeuresSimu', 'Flight hours', 'Flight simulator experience')

print("\n" + "-"*80)
print("EXPLORATORY: Gaming & Flight Experience Correlations (Retention E2-E3)")
print("-"*80)

df_retention_perf = df_best[df_best['Session'].isin(['E2', 'E3'])].copy()
retention_perf = df_retention_perf.groupby('Participant').agg({
    'SF_zscore': 'mean',
    'MATB_z_mean': 'mean',
    'Flight': 'mean'
}).reset_index()
retention_perf.columns = ['Participant', 'Ret_SF', 'Ret_MATB', 'Ret_Flight']

corr_base = df_demographics_EF[['Participant', 'VGexp', 'HeuresVol', 'HeuresSimu']].copy()
corr_merged = corr_base.merge(retention_perf, on='Participant', how='inner')
print(f"Participants with retention performance and experience data: {len(corr_merged)}")

print("\nGaming experience (VGexp) correlations:")
print_corr(corr_merged, 'VGexp', 'Ret_SF', 'Gaming exp', 'SF performance')
print_corr(corr_merged, 'VGexp', 'Ret_MATB', 'Gaming exp', 'MATB performance')
print_corr(corr_merged, 'VGexp', 'Ret_Flight', 'Gaming exp', 'Flight sim performance')

print("\nFlight hours (HeuresVol) correlations:")
print_corr(corr_merged, 'HeuresVol', 'Ret_SF', 'Flight hours', 'SF performance')
print_corr(corr_merged, 'HeuresVol', 'Ret_MATB', 'Flight hours', 'MATB performance')
print_corr(corr_merged, 'HeuresVol', 'HeuresSimu', 'Flight hours', 'Flight simulator experience')

print("\n" + "-"*80)
print("H2a COVARIATE ANALYSIS: SF Performance with Gaming Experience")
print("-"*80)
print("Model: SF_zscore ~ Session + Group + Session:Group + VGexp")
print("Random effects: Participant")
df_eval_sf_cov = df_eval.merge(df_demographics_EF[['Participant', 'VGexp']], 
                                on='Participant', how='left')
df_eval_sf_cov = df_eval_sf_cov[['SF_zscore', 'Session', 'Group', 'Participant', 'VGexp']].dropna()
print(f"Valid observations with covariates: {len(df_eval_sf_cov)}")

if len(df_eval_sf_cov) > 10:
    model_sf_cov = smf.mixedlm("SF_zscore ~ Session + Group + Session:Group + VGexp", 
                               df_eval_sf_cov, groups=df_eval_sf_cov["Participant"])
    result_sf_cov = model_sf_cov.fit()
    print(result_sf_cov.summary())
    evaluate_mixedlm_assumptions("H2a Covariate: SF_zscore ~ Session + Group + Session:Group + VGexp", result_sf_cov, lme_assumptions_results)
    run_gee_sensitivity("H2a Covariate: SF_zscore ~ Session + Group + Session:Group + VGexp", "SF_zscore ~ Session + Group + Session:Group + VGexp", df_eval_sf_cov, "Participant", result_sf_cov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")

print("\n" + "-"*80)
print("H2b COVARIATE ANALYSIS: MATB Performance with Gaming Experience")
print("-"*80)
print("Model: MATB_z_mean ~ Session + Group + Session:Group + VGexp")
print("Random effects: Participant")
df_eval_matb_cov = df_eval.merge(df_demographics_EF[['Participant', 'VGexp']], 
                                  on='Participant', how='left')
df_eval_matb_cov = df_eval_matb_cov[['MATB_z_mean', 'Session', 'Group', 'Participant', 'VGexp']].dropna()
print(f"Valid observations with covariates: {len(df_eval_matb_cov)}")

if len(df_eval_matb_cov) > 10:
    model_matb_cov = smf.mixedlm("MATB_z_mean ~ Session + Group + Session:Group + VGexp", 
                                 df_eval_matb_cov, groups=df_eval_matb_cov["Participant"])
    result_matb_cov = model_matb_cov.fit()
    print(result_matb_cov.summary())
    evaluate_mixedlm_assumptions("H2b Covariate: MATB_z_mean ~ Session + Group + Session:Group + VGexp", result_matb_cov, lme_assumptions_results, check_group_variance=True)
    run_gee_sensitivity("H2b Covariate: MATB_z_mean ~ Session + Group + Session:Group + VGexp", "MATB_z_mean ~ Session + Group + Session:Group + VGexp", df_eval_matb_cov, "Participant", result_matb_cov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")


# Diagnostics and sensitivity outputs (exported to logs for reporting).
print("\n" + "="*80)
print("MIXED-EFFECTS MODEL ASSUMPTION CHECKS")
print("="*80)
if lme_assumptions_results:
    assumptions_df = pd.DataFrame(lme_assumptions_results)
    assumptions_display = assumptions_df[[
        'Model', 'N_obs', 'Durbin_Watson', 'DW_interpretation',
        'Breusch_Pagan_p', 'BP_interpretation',
        'Shapiro_Wilk_p', 'SW_interpretation',
        'Levene_Group_p', 'Levene_interpretation',
        'Brown_Forsythe_Group_p', 'Brown_Forsythe_interpretation',
        'Residual_Variance_Ratio_MaxMin'
    ]].copy()
    assumptions_display['Durbin_Watson'] = assumptions_display['Durbin_Watson'].map(lambda x: format_diag_value(x, 3))
    assumptions_display['Breusch_Pagan_p'] = assumptions_display['Breusch_Pagan_p'].map(lambda x: format_diag_value(x, 4))
    assumptions_display['Shapiro_Wilk_p'] = assumptions_display['Shapiro_Wilk_p'].map(lambda x: format_diag_value(x, 4))
    assumptions_display['Levene_Group_p'] = assumptions_display['Levene_Group_p'].map(lambda x: format_diag_value(x, 4))
    assumptions_display['Brown_Forsythe_Group_p'] = assumptions_display['Brown_Forsythe_Group_p'].map(lambda x: format_diag_value(x, 4))
    assumptions_display['Residual_Variance_Ratio_MaxMin'] = assumptions_display['Residual_Variance_Ratio_MaxMin'].map(lambda x: format_diag_value(x, 3))
    print(assumptions_display.to_string(index=False))

    assumptions_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_LME_assumptions_{log_date}.csv")
    assumptions_df.to_csv(assumptions_csv_path, index=False)
    print(f"\nAssumption summary table saved to: {assumptions_csv_path}")

    covariate_models = assumptions_df[assumptions_df['Model'].str.contains('Covariate', case=False, na=False)]
    if not covariate_models.empty:
        covariate_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_covariate_assumptions_{log_date}.csv")
        covariate_models.to_csv(covariate_csv_path, index=False)
        print(f"Covariate model assumption summary saved to: {covariate_csv_path}")
else:
    print("No mixed-effects models were successfully fitted for assumption checks.")

print("\n" + "="*80)
print("SANDWICH-ROBUST MARGINAL CONFIRMATION (vs MixedLM)")
print("="*80)
if gee_sensitivity_results:
    gee_df = pd.DataFrame(gee_sensitivity_results)
    gee_display = gee_df[[
        'Model', 'N_obs', 'Estimator', 'MixedLM_Group_p', 'Sandwich_Group_p',
        'MixedLM_Min_SessionGroup_p', 'Sandwich_Min_SessionGroup_p', 'Status'
    ]].copy()
    gee_display['MixedLM_Group_p'] = gee_display['MixedLM_Group_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['Sandwich_Group_p'] = gee_display['Sandwich_Group_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['MixedLM_Min_SessionGroup_p'] = gee_display['MixedLM_Min_SessionGroup_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['Sandwich_Min_SessionGroup_p'] = gee_display['Sandwich_Min_SessionGroup_p'].map(lambda x: format_diag_value(x, 4))
    print(gee_display.to_string(index=False))

    gee_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_GEE_sensitivity_{log_date}.csv")
    gee_df.to_csv(gee_csv_path, index=False)
    print(f"\nSandwich confirmation summary table saved to: {gee_csv_path}")
else:
    print("No models were available for sandwich confirmation checks.")

# -----------------------------------------------------------------------------
# Multiple comparisons - FDR adjustment log (Benjamini-Hochberg)
# Collect  the p-values from the MixedLM results for MATB and SF
# (the covariate MixedLMs including VGexp). 
# -----------------------------------------------------------------------------
pvalue_records = []  # list of (label, pvalue)

# Collect  covariate MixedLM p-values for MATB and SF 
for varname, label_prefix in [('result_matb_cov', 'MATB_cov_mixedlm'), ('result_sf_cov', 'SF_cov_mixedlm')]:
    mdl = globals().get(varname, None)
    if mdl is None:
        continue
    pv = getattr(mdl, 'pvalues', None)
    if pv is None:
        continue
    for term, p in pv.items():
        try:
            pvalue_records.append((f"{label_prefix}_{term}", float(p) if pd.notna(p) else np.nan))
        except Exception:
            continue

# Prepare adjusted-pvalue log and CSV
adjusted_log_path = os.path.join(script_dir, "..", "logs", f"{script_name}_Adjusted_{log_date}.txt")
adjusted_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_Adjusted_{log_date}.csv")
os.makedirs(os.path.dirname(adjusted_log_path), exist_ok=True)

# Build DataFrame from collected records
records = []
for rec in pvalue_records:
    try:
        label = rec[0]
        pval = rec[1]
    except Exception:
        continue
    records.append({'Label': label, 'P_value': float(pval) if np.isfinite(pval) else np.nan})

df_p = pd.DataFrame(records)

with open(adjusted_log_path, 'w', encoding='utf-8') as af:
    af.write("Multiple comparisons adjustment (Benjamini-Hochberg FDR) (MATB & SF covariate MixedLMs)\n")
    af.write("Generated by script: %s\n" % os.path.basename(__file__))
    af.write("Analysis date: %s\n\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    total_collected = len(df_p)
    finite_mask = df_p['P_value'].notna()
    finite_count = int(finite_mask.sum())

    af.write(f"Total p-values collected for models: {total_collected} (finite: {finite_count})\n\n")

    if finite_count <= 1:
        af.write("No multiple-comparison correction necessary: less than two finite p-values collected.\n\n")
        df_p['P_adj_FDR'] = np.nan
        df_p['Included_in_FDR'] = False
        df_p['Significant_after_FDR'] = False
    else:
        try:
            valid_pvals = df_p.loc[finite_mask, 'P_value'].values.astype(float)
            rej, p_adj, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
            df_p['P_adj_FDR'] = np.nan
            df_p['Included_in_FDR'] = False
            df_p['Significant_after_FDR'] = False
            df_p.loc[finite_mask, 'P_adj_FDR'] = p_adj
            df_p.loc[finite_mask, 'Included_in_FDR'] = True
            df_p.loc[finite_mask, 'Significant_after_FDR'] = rej.astype(bool)
        except Exception as exc:
            af.write(f"FDR adjustment failed: {str(exc)}\n\n")

    af.write(f"{'Label':60s} {'p-value':10s} {'p-adj (FDR)':12s} {'Included':9s} {'Significant'}\n")
    af.write('-'*95 + '\n')
    for _, row in df_p.iterrows():
        pv_str = f"{row['P_value']:.4g}" if pd.notna(row['P_value']) else 'NA'
        padj_str = f"{row['P_adj_FDR']:.4g}" if pd.notna(row['P_adj_FDR']) else 'NA'
        included = 'Yes' if bool(row.get('Included_in_FDR', False)) else 'No'
        signif = 'Yes' if bool(row.get('Significant_after_FDR', False)) else 'No'
        af.write(f"{row['Label']:60s} {pv_str:10s} {padj_str:12s} {included:9s} {signif:>9s}\n")

# Save CSV for easier parsing
df_csv = df_p.copy()
df_csv.to_csv(adjusted_csv_path, index=False)

# No additional console output (preserve original logs)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll plots saved to: {plot_dir}")
print("\nGenerated plots:")
print("  1. Combined_LearningRates_Stacked.png - MATB + SF learning rates (vertically stacked)")
print("  2. Combined_Performances_Stacked_MeanCI95.png - MATB + SF performances (mean + points + 95% CI)")
print(f"\nLog saved to: {log_path}")
print("="*80)

sys.stdout = logger.terminal
logger.close()


