"""
Statistical Analysis Script - Untrained Tasks (Workload Assessments)
Analyzes subjective workload (NASA-TLX) and objective workload (Oddball missed targets)
Uses: df_Best_SF_MATB_Mean_Zscores_SIMU.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests
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


def evaluate_mixedlm_assumptions(model_name, result, storage):
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
        'SW_interpretation': 'OK' if pd.notna(sw_pvalue) and sw_pvalue >= 0.05 else 'Potential issue'
    }
    storage.append(row)

    print(f"\nAssumption checks - {model_name}")
    print(f"  Durbin-Watson: {format_diag_value(dw_stat, 3)} ({row['DW_interpretation']})")
    print(f"  Breusch-Pagan p-value: {format_diag_value(bp_pvalue)} ({row['BP_interpretation']})")
    print(f"  Shapiro-Wilk p-value: {format_diag_value(sw_pvalue)} ({row['SW_interpretation']})")


def plot_residual_diagnostics(model_name, result, plot_path):
    """Residual diagnostic figures removed in streamlined plotting mode."""
    print(f"  Residual diagnostics figure skipped: {os.path.basename(plot_path)}")


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
    """Run GEE sensitivity model and compare key p-values with MixedLM."""
    mixed_group_p, mixed_interaction_p, mixed_interaction_details = extract_group_and_interaction_pvalues(mixedlm_result)

    row = {
        'Model': model_name,
        'N_obs': len(data),
        'MixedLM_Group_p': mixed_group_p,
        'MixedLM_Min_SessionGroup_p': mixed_interaction_p,
        'MixedLM_SessionGroup_Details': mixed_interaction_details,
        'GEE_Group_p': np.nan,
        'GEE_Min_SessionGroup_p': np.nan,
        'GEE_SessionGroup_Details': '',
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
        gee_result = gee_model.fit()
        gee_group_p, gee_interaction_p, gee_interaction_details = extract_group_and_interaction_pvalues(gee_result)

        row['GEE_Group_p'] = gee_group_p
        row['GEE_Min_SessionGroup_p'] = gee_interaction_p
        row['GEE_SessionGroup_Details'] = gee_interaction_details

        print(f"\nGEE sensitivity - {model_name}")
        print(f"  Group p-value: MixedLM={format_diag_value(mixed_group_p)} | GEE={format_diag_value(gee_group_p)}")
        print(f"  Min Session×Group p-value: MixedLM={format_diag_value(mixed_interaction_p)} | GEE={format_diag_value(gee_interaction_p)}")
    except Exception as exc:
        row['Status'] = f"GEE failed: {str(exc)}"
        print(f"\nGEE sensitivity failed for {model_name}: {exc}")

    storage.append(row)


def fit_mixedlm_with_reporting(
    model_name,
    formula,
    data,
    group_col,
    assumptions_storage,
    gee_storage,
    run_diagnostics=False,
    diagnostics_filename=None,
):
    """Fit a MixedLM, print summary, run assumption checks, and optional GEE sensitivity."""
    print(f"Valid observations for model: {len(data)}")
    if len(data) <= 10:
        print("Insufficient data for mixed model analysis")
        return None, {}

    model = smf.mixedlm(formula, data, groups=data[group_col])
    result = model.fit()
    print(result.summary())

    evaluate_mixedlm_assumptions(model_name, result, assumptions_storage)
    if run_diagnostics and diagnostics_filename is not None:
        plot_residual_diagnostics(model_name, result, os.path.join(plot_dir, diagnostics_filename))

    run_gee_sensitivity(model_name, formula, data, group_col, result, gee_storage)
    return result, result.pvalues


def compute_session_group_ttests(data, value_col, metric_label):
    """Compute independent t-tests between groups for each evaluation session."""
    print("\n" + "-" * 80)
    print(f"{metric_label}: Group Comparisons within Each Session")
    print("-" * 80)

    pvalues = {}
    for session_code in ['E1', 'E2', 'E3']:
        session_data = data[data['Session'] == session_code]
        trns_data = session_data[session_data['Group'] == 'tRNS'][value_col]
        sham_data = session_data[session_data['Group'] == 'Sham'][value_col]

        if len(trns_data) > 0 and len(sham_data) > 0:
            t_stat, p_val = stats.ttest_ind(trns_data, sham_data)
            pvalues[session_code] = p_val
            print(
                f"{session_code}: tRNS (n={len(trns_data)}, m={trns_data.mean():.3f}) vs "
                f"Sham (n={len(sham_data)}, m={sham_data.mean():.3f}), t={t_stat:.3f}, p={p_val:.3f}"
            )
        else:
            print(f"{session_code}: Insufficient data for comparison")

    return pvalues


# Create log file with script name and date
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_date = datetime.now().strftime("%Y_%m_%d")
log_filename = f"{script_name}_{log_date}.txt"

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(script_dir, "..", "logs", log_filename)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Redirect output to both console and file
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
print("STATISTICAL ANALYSIS - UNTRAINED TASKS (WORKLOAD ASSESSMENTS)")
print("="*80)
print(f"\nLog file: {log_path}")
print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

lme_assumptions_results = []
gee_sensitivity_results = []

# Configuration
project_dir = os.path.dirname(script_dir)
transformed_dir = os.path.join(project_dir, "transformed_data")
plot_dir = os.path.join(project_dir, "plots", "Untrained_tasks")
os.makedirs(plot_dir, exist_ok=True)

# Load data
print("\nLoading data files...")
df_workload = pd.read_csv(os.path.join(transformed_dir, "df_Best_SF_MATB_Mean_Zscores_SIMU.csv"))
df_demographics_EF = pd.read_csv(os.path.join(transformed_dir, "df_demographics_EFs.csv"))

print(f"  df_Best_SF_MATB_Mean_Zscores_SIMU: {df_workload.shape}")
print(f"  df_demographics_EFs: {df_demographics_EF.shape}")
print(f"  Columns: {list(df_workload.columns)}")

# Recode groups: A -> tRNS, B -> Sham
print("\nRecoding groups (A -> tRNS, B -> Sham)...")
group_mapping = {"A": "tRNS", "B": "Sham"}
if 'Group' in df_workload.columns:
    df_workload['Group'] = df_workload['Group'].replace(group_mapping)

# Color scheme
group_colors = {"tRNS": "orange", "Sham": "grey"}

# Filter for evaluation sessions only (E1, E2, E3)
df_eval = df_workload[df_workload['Session'].isin(['E1', 'E2', 'E3'])].copy()

# Create session labels
session_labels = {'E1': 'Reference', 'E2': 'Short-term', 'E3': 'Long-term'}
df_eval['Session_Label'] = df_eval['Session'].map(session_labels)
df_eval['Session_Label'] = pd.Categorical(df_eval['Session_Label'], 
                                          categories=['Reference', 'Short-term', 'Long-term'], 
                                          ordered=True)

print(f"\nSessions included: {sorted(df_eval['Session'].unique())}")
print(f"Total data points: {len(df_eval)}")
print(f"Unique participants: {df_eval['Participant'].nunique()}")

print("\n" + "="*80)
print("WORKLOAD ASSESSMENTS: SUBJECTIVE & OBJECTIVE")
print("="*80)

# =============================================================================
# SUBJECTIVE WORKLOAD (NASA-TLX)
# =============================================================================
print("\n" + "="*80)
print("SUBJECTIVE WORKLOAD: NASA-TLX")
print("="*80)

# Check for missing data
print(f"\nMissing TLX data: {df_eval['TLX_raw_mean'].isna().sum()} / {len(df_eval)}")

# Descriptive statistics
print("\nDescriptive Statistics by Group and Session:")
tlx_stats = df_eval.groupby(['Group', 'Session_Label'])['TLX_raw_mean'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
]).reset_index()
print(tlx_stats.to_string(index=False))

# Mixed model for TLX - fit first to get p-values for plot
print("\n" + "-"*80)
print("Mixed Model for Subjective Workload (NASA-TLX)")
print("-"*80)
print("Model: TLX_raw_mean ~ Session + Group + Session:Group")
print("Random effects: Participant")
df_tlx_clean = df_eval[['TLX_raw_mean', 'Session', 'Group', 'Participant']].dropna()
result_tlx, tlx_pvalues = fit_mixedlm_with_reporting(
    model_name="H1 Base: TLX_raw_mean ~ Session + Group + Session:Group",
    formula="TLX_raw_mean ~ Session + Group + Session:Group",
    data=df_tlx_clean,
    group_col="Participant",
    assumptions_storage=lme_assumptions_results,
    gee_storage=gee_sensitivity_results,
    run_diagnostics=True,
    diagnostics_filename='residuals_TLX_base.png'
)

# Calculate session-wise group comparisons for TLX
tlx_group_pvalues = compute_session_group_ttests(df_tlx_clean, 'TLX_raw_mean', 'TLX')

# Covariate analysis: Add gaming and flight experience
print("\n" + "-"*80)
print("COVARIATE ANALYSIS: TLX with Gaming & Flight Experience")
print("-"*80)
print("Model: TLX_raw_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol")
print("Random effects: Participant")
df_tlx_cov = df_eval.merge(df_demographics_EF[['Participant', 'JeuxIntensif', 'HeuresVol']], 
                           on='Participant', how='left')
df_tlx_cov = df_tlx_cov[['TLX_raw_mean', 'Session', 'Group', 'Participant', 
                         'JeuxIntensif', 'HeuresVol']].dropna()
print(f"Valid observations with covariates: {len(df_tlx_cov)}")

if len(df_tlx_cov) > 10:
    model_tlx_cov = smf.mixedlm("TLX_raw_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", 
                               df_tlx_cov, groups=df_tlx_cov["Participant"])
    result_tlx_cov = model_tlx_cov.fit()
    print(result_tlx_cov.summary())
    evaluate_mixedlm_assumptions("H1 Covariate: TLX_raw_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", result_tlx_cov, lme_assumptions_results)
    run_gee_sensitivity("H1 Covariate: TLX_raw_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", "TLX_raw_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", df_tlx_cov, "Participant", result_tlx_cov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")

# =============================================================================
# OBJECTIVE WORKLOAD (ODDBALL MISSED TARGETS)
# =============================================================================
print("\n" + "="*80)
print("OBJECTIVE WORKLOAD: ODDBALL MISSED TARGETS")
print("="*80)

# Check for missing data
print(f"\nMissing Oddball missed targets data: {df_eval['oddball_percentage_mean'].isna().sum()} / {len(df_eval)}")

# Descriptive statistics
print("\nDescriptive Statistics by Group and Session:")
oddball_stats = df_eval.groupby(['Group', 'Session_Label'])['oddball_percentage_mean'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
]).reset_index()
print(oddball_stats.to_string(index=False))

# Mixed model for Oddball - fit first to get p-values for plot
print("\n" + "-"*80)
print("Mixed Model for Objective Workload (Oddball Missed Targets)")
print("-"*80)
print("Model: oddball_percentage_mean ~ Session + Group + Session:Group")
print("Random effects: Participant")
df_oddball_clean = df_eval[['oddball_percentage_mean', 'Session', 'Group', 'Participant']].dropna()
result_oddball, oddball_pvalues = fit_mixedlm_with_reporting(
    model_name="H2 Base: oddball_percentage_mean ~ Session + Group + Session:Group",
    formula="oddball_percentage_mean ~ Session + Group + Session:Group",
    data=df_oddball_clean,
    group_col="Participant",
    assumptions_storage=lme_assumptions_results,
    gee_storage=gee_sensitivity_results,
    run_diagnostics=True,
    diagnostics_filename='residuals_Oddball_base.png'
)

# Calculate session-wise group comparisons for Oddball
oddball_group_pvalues = compute_session_group_ttests(df_oddball_clean, 'oddball_percentage_mean', 'Oddball')

# Covariate analysis: Add gaming and flight experience
print("\n" + "-"*80)
print("COVARIATE ANALYSIS: Oddball Missed Targets with Gaming & Flight Experience")
print("-"*80)
print("Model: oddball_percentage_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol")
print("Random effects: Participant")
df_oddball_cov = df_eval.merge(df_demographics_EF[['Participant', 'JeuxIntensif', 'HeuresVol']], 
                               on='Participant', how='left')
df_oddball_cov = df_oddball_cov[['oddball_percentage_mean', 'Session', 'Group', 'Participant', 
                                 'JeuxIntensif', 'HeuresVol']].dropna()
print(f"Valid observations with covariates: {len(df_oddball_cov)}")

if len(df_oddball_cov) > 10:
    model_oddball_cov = smf.mixedlm("oddball_percentage_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", 
                                   df_oddball_cov, groups=df_oddball_cov["Participant"])
    result_oddball_cov = model_oddball_cov.fit()
    print(result_oddball_cov.summary())
    evaluate_mixedlm_assumptions("H2 Covariate: oddball_percentage_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", result_oddball_cov, lme_assumptions_results)
    run_gee_sensitivity("H2 Covariate: oddball_percentage_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", "oddball_percentage_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", df_oddball_cov, "Participant", result_oddball_cov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")

# =============================================================================
# SIMULATOR PERFORMANCE
# =============================================================================
print("\n" + "="*80)
print("SIMULATOR PERFORMANCE")
print("="*80)

# Check for missing data
print(f"\nMissing Simulator data: {df_eval['simu_zscore_mean'].isna().sum()} / {len(df_eval)}")

# Descriptive statistics
print("\nDescriptive Statistics by Group and Session:")
simu_stats = df_eval.groupby(['Group', 'Session_Label'])['simu_zscore_mean'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('Min', 'min'),
    ('Max', 'max')
]).reset_index()
print(simu_stats.to_string(index=False))

# Mixed model for Simulator - fit first to get p-values for plot
print("\n" + "-"*80)
print("Mixed Model for Simulator Performance")
print("-"*80)
print("Model: simu_zscore_mean ~ Session + Group + Session:Group")
print("Random effects: Participant")
df_simu_clean = df_eval[['simu_zscore_mean', 'Session', 'Group', 'Participant']].dropna()
result_simu, simu_pvalues = fit_mixedlm_with_reporting(
    model_name="H3 Base: simu_zscore_mean ~ Session + Group + Session:Group",
    formula="simu_zscore_mean ~ Session + Group + Session:Group",
    data=df_simu_clean,
    group_col="Participant",
    assumptions_storage=lme_assumptions_results,
    gee_storage=gee_sensitivity_results
)

# Calculate session-wise group comparisons for Simulator
simu_group_pvalues = compute_session_group_ttests(df_simu_clean, 'simu_zscore_mean', 'Simulator')

# Covariate analysis: Add gaming and flight experience
print("\n" + "-"*80)
print("COVARIATE ANALYSIS: Simulator with Gaming & Flight Experience")
print("-"*80)
print("Model: simu_zscore_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol")
print("Random effects: Participant")
df_simu_cov = df_eval.merge(df_demographics_EF[['Participant', 'JeuxIntensif', 'HeuresVol']], 
                            on='Participant', how='left')
df_simu_cov = df_simu_cov[['simu_zscore_mean', 'Session', 'Group', 'Participant', 
                           'JeuxIntensif', 'HeuresVol']].dropna()
print(f"Valid observations with covariates: {len(df_simu_cov)}")

if len(df_simu_cov) > 10:
    model_simu_cov = smf.mixedlm("simu_zscore_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", 
                                df_simu_cov, groups=df_simu_cov["Participant"])
    result_simu_cov = model_simu_cov.fit()
    print(result_simu_cov.summary())
    evaluate_mixedlm_assumptions("H3 Covariate: simu_zscore_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", result_simu_cov, lme_assumptions_results)
    run_gee_sensitivity("H3 Covariate: simu_zscore_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", "simu_zscore_mean ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", df_simu_cov, "Participant", result_simu_cov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")

# Covariate analysis: Flight experience only
print("\n" + "-"*80)
print("COVARIATE ANALYSIS: Simulator with Flight Experience")
print("-"*80)
print("Model: simu_zscore_mean ~ Session + Group + Session:Group + HeuresVol")
print("Random effects: Participant")
df_simu_flightcov = df_eval.merge(df_demographics_EF[['Participant', 'HeuresVol']], 
                                   on='Participant', how='left')
df_simu_flightcov = df_simu_flightcov[['simu_zscore_mean', 'Session', 'Group', 'Participant', 
                                        'HeuresVol']].dropna()
print(f"Valid observations with covariates: {len(df_simu_flightcov)}")

if len(df_simu_flightcov) > 10:
    model_simu_flightcov = smf.mixedlm("simu_zscore_mean ~ Session + Group + Session:Group + HeuresVol", 
                                        df_simu_flightcov, groups=df_simu_flightcov["Participant"])
    result_simu_flightcov = model_simu_flightcov.fit()
    print(result_simu_flightcov.summary())
    evaluate_mixedlm_assumptions("H3 Flight covariate: simu_zscore_mean ~ Session + Group + Session:Group + HeuresVol", result_simu_flightcov, lme_assumptions_results)
    run_gee_sensitivity("H3 Flight covariate: simu_zscore_mean ~ Session + Group + Session:Group + HeuresVol", "simu_zscore_mean ~ Session + Group + Session:Group + HeuresVol", df_simu_flightcov, "Participant", result_simu_flightcov, gee_sensitivity_results)
else:
    print("Insufficient data for covariate analysis")

# =============================================================================
# STACKED PLOT (MEAN + INDIVIDUAL POINTS + 95% CI ONLY)
# =============================================================================
print("Creating stacked workload and performance plot (Mean + Points + 95% CI)...")
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

# Top row: TLX (left) and Oddball (right)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
# Bottom row: Simulator spans full width, then shrink to 60%
ax3 = fig.add_subplot(gs[1, :])

# ===== Panel A: TLX (Subjective Workload) =====
sns.stripplot(data=df_eval, x='Session_Label', y='TLX_raw_mean', hue='Group',
              palette=group_colors, ax=ax1, dodge=True, alpha=0.45, legend=False, size=6,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'],
              edgecolor='black', linewidth=0.5)
sns.pointplot(data=df_eval, x='Session_Label', y='TLX_raw_mean', hue='Group',
              palette=group_colors, ax=ax1, dodge=0.2, errorbar=('ci', 95), join=True,
              markers='D', linestyles='-', scale=1.0, estimator=np.mean, capsize=0.15,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'])

ax1.set_ylabel('NASA-TLX Score', fontsize=12, fontweight='bold')
ax1.set_xlabel('Evaluation Session', fontsize=12, fontweight='bold')
ax1.set_title('A. Subjective Workload', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Fix Y scale to keep annotations inside
ax1.set_ylim(0, 110)

if len(tlx_pvalues) > 0:
    y_lim = ax1.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    y_top = y_lim[1] - 0.06 * y_range
    line_height = 0.003 * y_range

    if 'Session[T.E2]' in tlx_pvalues:
        stars = get_significance_stars(tlx_pvalues['Session[T.E2]'])
        ax1.plot([0, 1], [y_top, y_top], 'k-', linewidth=1.5)
        ax1.text(0.5, y_top + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    if 'Session[T.E3]' in tlx_pvalues:
        stars = get_significance_stars(tlx_pvalues['Session[T.E3]'])
        y_top2 = y_top - 0.06 * y_range if 'Session[T.E2]' in tlx_pvalues else y_top
        ax1.plot([0, 2], [y_top2, y_top2], 'k-', linewidth=1.5)
        ax1.text(1, y_top2 + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    y_offset_positions = [0, 1, 2]
    y_group_line = y_top - 0.14 * y_range
    for idx, session_label in enumerate(['Reference', 'Short-term', 'Long-term']):
        session_code = ['E1', 'E2', 'E3'][idx]
        if session_code in tlx_group_pvalues:
            p_val = tlx_group_pvalues[session_code]
            stars = get_significance_stars(p_val)
            x_pos = y_offset_positions[idx]
            ax1.plot([x_pos - 0.075, x_pos + 0.075], [y_group_line, y_group_line], 'k-', linewidth=1)
            ax1.text(x_pos, y_group_line + 0.02 * y_range, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

# ===== Panel B: Oddball (Objective Workload) =====
sns.stripplot(data=df_eval, x='Session_Label', y='oddball_percentage_mean', hue='Group',
              palette=group_colors, ax=ax2, dodge=True, alpha=0.45, legend=False, size=6,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'],
              edgecolor='black', linewidth=0.5)
sns.pointplot(data=df_eval, x='Session_Label', y='oddball_percentage_mean', hue='Group',
              palette=group_colors, ax=ax2, dodge=0.2, errorbar=('ci', 95), join=True,
              markers='D', linestyles='-', scale=1.0, estimator=np.mean, capsize=0.15,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'])

ax2.set_ylabel('Oddball Missed Targets (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Evaluation Session', fontsize=12, fontweight='bold')
ax2.set_title('B. Objective Workload', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Fix Y scale to keep annotations inside
ax2_ymin = ax2.get_ylim()[0]
ax2.set_ylim(ax2_ymin, 90)

if len(oddball_pvalues) > 0:
    y_lim = ax2.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    y_top = y_lim[1] - 0.06 * y_range
    line_height = 0.002 * y_range

    if 'Session[T.E2]' in oddball_pvalues:
        stars = get_significance_stars(oddball_pvalues['Session[T.E2]'])
        ax2.plot([0, 1], [y_top, y_top], 'k-', linewidth=1.5)
        ax2.text(0.5, y_top + line_height, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

    if 'Session[T.E3]' in oddball_pvalues:
        stars = get_significance_stars(oddball_pvalues['Session[T.E3]'])
        y_top2 = y_top - 0.06 * y_range if 'Session[T.E2]' in oddball_pvalues else y_top
        ax2.plot([0, 2], [y_top2, y_top2], 'k-', linewidth=1.5)
        ax2.text(1, y_top2 + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    y_offset_positions = [0, 1, 2]
    y_group_line = y_top - 0.14 * y_range
    for idx, session_label in enumerate(['Reference', 'Short-term', 'Long-term']):
        session_code = ['E1', 'E2', 'E3'][idx]
        if session_code in oddball_group_pvalues:
            p_val = oddball_group_pvalues[session_code]
            stars = get_significance_stars(p_val)
            x_pos = y_offset_positions[idx]
            ax2.plot([x_pos - 0.075, x_pos + 0.075], [y_group_line, y_group_line], 'k-', linewidth=1)
            ax2.text(x_pos, y_group_line + 0.02 * y_range, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

# ===== Panel C: Simulator Performance =====
ax3_pos = ax3.get_position()
new_width = ax3_pos.width * 0.6
x_center = ax3_pos.x0 + ax3_pos.width / 2
new_x0 = x_center - new_width / 2 - 0.05
ax3.set_position([new_x0, ax3_pos.y0, new_width, ax3_pos.height])

sns.stripplot(data=df_eval, x='Session_Label', y='simu_zscore_mean', hue='Group',
              palette=group_colors, ax=ax3, dodge=True, alpha=0.45, legend=False, size=6,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'],
              edgecolor='black', linewidth=0.5)
sns.pointplot(data=df_eval, x='Session_Label', y='simu_zscore_mean', hue='Group',
              palette=group_colors, ax=ax3, dodge=0.2, errorbar=('ci', 95), join=True,
              markers='D', linestyles='-', scale=1.0, estimator=np.mean, capsize=0.15,
              order=['Reference', 'Short-term', 'Long-term'], hue_order=['Sham', 'tRNS'])

ax3.set_ylabel('Simulator Performance (Z-Score)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Evaluation Session', fontsize=12, fontweight='bold')
ax3.set_title('C. Flight Simulator Performance', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Remove legends from each subplot and add a single shared legend near panel C
handles, labels = ax3.get_legend_handles_labels()
if ax1.get_legend() is not None:
    ax1.get_legend().remove()
if ax2.get_legend() is not None:
    ax2.get_legend().remove()
if ax3.get_legend() is not None:
    ax3.get_legend().remove()

ax3_pos_updated = ax3.get_position()
fig.legend(handles, labels, title='Group', fontsize=12, loc='upper left',
           bbox_to_anchor=(ax3_pos_updated.x1 + 0.02, ax3_pos_updated.y1 - 0.02))

# Fix Y scale to keep annotations inside
ax3_ymin = ax3.get_ylim()[0]
ax3.set_ylim(ax3_ymin, 2.0)

if len(simu_pvalues) > 0:
    y_lim = ax3.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    y_top = y_lim[1] - 0.06 * y_range
    line_height = 0.002 * y_range

    if 'Session[T.E2]' in simu_pvalues:
        stars = get_significance_stars(simu_pvalues['Session[T.E2]'])
        ax3.plot([0, 1], [y_top, y_top], 'k-', linewidth=1.5)
        ax3.text(0.5, y_top + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    if 'Session[T.E3]' in simu_pvalues:
        stars = get_significance_stars(simu_pvalues['Session[T.E3]'])
        y_top2 = y_top - 0.06 * y_range if 'Session[T.E2]' in simu_pvalues else y_top
        ax3.plot([0, 2], [y_top2, y_top2], 'k-', linewidth=1.5)
        ax3.text(1, y_top2 + line_height, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    y_offset_positions = [0, 1, 2]
    y_group_line = y_top - 0.14 * y_range
    for idx, session_label in enumerate(['Reference', 'Short-term', 'Long-term']):
        session_code = ['E1', 'E2', 'E3'][idx]
        if session_code in simu_group_pvalues:
            p_val = simu_group_pvalues[session_code]
            stars = get_significance_stars(p_val)
            x_pos = y_offset_positions[idx]
            ax3.plot([x_pos - 0.075, x_pos + 0.075], [y_group_line, y_group_line], 'k-', linewidth=1)
            ax3.text(x_pos, y_group_line + 0.02 * y_range, stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.savefig(os.path.join(plot_dir, 'Workload_Simulator_Stacked_MeanCI95.png'), dpi=300, bbox_inches='tight')
print("  Stacked workload and simulator Mean+Points+95%CI plot saved")
plt.close()

# =============================================================================
# EXPLORATORY: GAMING & FLIGHT EXPERIENCE CORRELATIONS
# =============================================================================
print("\n" + "="*80)
print("EXPLORATORY: Gaming & Flight Experience Correlations (Baseline E1)")
print("="*80)

correlation_results = []


def print_corr(data, x_col, y_col, x_label, y_label):
    """Print Pearson correlation with sample size and guard against invalid inputs."""
    corr_data = data[[x_col, y_col]].dropna()
    if len(corr_data) < 3:
        print(f"{x_label} vs {y_label}: insufficient data (n={len(corr_data)})")
        return

    if corr_data[x_col].nunique() < 2 or corr_data[y_col].nunique() < 2:
        print(f"{x_label} vs {y_label}: insufficient variance (n={len(corr_data)})")
        return

    corr, p_val = stats.pearsonr(corr_data[x_col], corr_data[y_col])
    print(f"{x_label} vs {y_label}: r={corr:.4f}, p={p_val:.4f}, n={len(corr_data)}")
    correlation_results.append({
        'Analysis': 'Experience correlation',
        'Subset': y_label,
        'Predictor': x_label,
        'Outcome': y_label,
        'r': corr,
        'p': p_val,
        'N': len(corr_data),
        'Significant': 'YES' if p_val < 0.05 else 'NO'
    })


gaming_exp_col = next((col for col in ['CGexp', 'VGexp', 'JeuxIntensif'] if col in df_demographics_EF.columns), None)
gaming_exp_label = gaming_exp_col if gaming_exp_col is not None else 'Gaming experience'

experience_columns = ['Participant', 'HeuresVol']
if gaming_exp_col is not None:
    experience_columns.append(gaming_exp_col)

baseline_scores = df_eval[df_eval['Session'] == 'E1'].groupby('Participant').agg(
    Base_TLX=('TLX_raw_mean', 'mean'),
    Base_Oddball=('oddball_percentage_mean', 'mean'),
    Base_Simu=('simu_zscore_mean', 'mean')
).reset_index()

baseline_merged = baseline_scores.merge(
    df_demographics_EF[experience_columns].copy(),
    on='Participant',
    how='left'
)

print(f"\nParticipants with baseline workload/performance and experience data: {baseline_merged['Participant'].nunique()}")

if gaming_exp_col is not None:
    print(f"\nGaming experience ({gaming_exp_label}) correlations:")
    print_corr(baseline_merged, gaming_exp_col, 'Base_Simu', gaming_exp_label, 'simulator performance (baseline)')
    print_corr(baseline_merged, gaming_exp_col, 'Base_TLX', gaming_exp_label, 'NASA-TLX (baseline)')
    print_corr(baseline_merged, gaming_exp_col, 'Base_Oddball', gaming_exp_label, 'Oddball score (baseline)')
else:
    print("\nGaming experience correlations: gaming experience column not found")

print("\nFlight hours (HeuresVol) correlations:")
print_corr(baseline_merged, 'HeuresVol', 'Base_Simu', 'Flight hours', 'simulator performance (baseline)')
print_corr(baseline_merged, 'HeuresVol', 'Base_TLX', 'Flight hours', 'NASA-TLX (baseline)')
print_corr(baseline_merged, 'HeuresVol', 'Base_Oddball', 'Flight hours', 'Oddball score (baseline)')

print("\n" + "="*80)
print("EXPLORATORY: Gaming & Flight Experience Correlations (Retention E2-E3)")
print("="*80)

retention_scores = df_eval[df_eval['Session'].isin(['E2', 'E3'])].groupby('Participant').agg(
    Ret_TLX=('TLX_raw_mean', 'mean'),
    Ret_Oddball=('oddball_percentage_mean', 'mean'),
    Ret_Simu=('simu_zscore_mean', 'mean')
).reset_index()

retention_merged = retention_scores.merge(
    df_demographics_EF[experience_columns].copy(),
    on='Participant',
    how='left'
)

print(f"\nParticipants with retention workload/performance and experience data: {retention_merged['Participant'].nunique()}")

if gaming_exp_col is not None:
    print(f"\nGaming experience ({gaming_exp_label}) correlations:")
    print_corr(retention_merged, gaming_exp_col, 'Ret_Simu', gaming_exp_label, 'simulator performance')
    print_corr(retention_merged, gaming_exp_col, 'Ret_TLX', gaming_exp_label, 'NASA-TLX')
    print_corr(retention_merged, gaming_exp_col, 'Ret_Oddball', gaming_exp_label, 'Oddball score')
else:
    print("\nGaming experience correlations: gaming experience column not found")

print("\nFlight hours (HeuresVol) correlations:")
print_corr(retention_merged, 'HeuresVol', 'Ret_Simu', 'Flight hours', 'simulator performance')
print_corr(retention_merged, 'HeuresVol', 'Ret_TLX', 'Flight hours', 'NASA-TLX')
print_corr(retention_merged, 'HeuresVol', 'Ret_Oddball', 'Flight hours', 'Oddball score')

# =============================================================================
# SUMMARY TABLE OF EXPLORATORY CORRELATIONS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE: ALL EXPLORATORY CORRELATIONS")
print("="*80)

if correlation_results:
    correlation_summary_df = pd.DataFrame(correlation_results)
    correlation_summary_df['r'] = correlation_summary_df['r'].map(lambda x: f"{x:.4f}")
    correlation_summary_df['p'] = correlation_summary_df['p'].map(lambda x: f"{x:.4f}")
    correlation_summary_df = correlation_summary_df[[
        'Analysis', 'Subset', 'Predictor', 'Outcome', 'r', 'p', 'N', 'Significant'
    ]]
    print("\n" + "="*160)
    print(correlation_summary_df.to_string(index=False))
    print("="*160)
else:
    print("No valid exploratory correlations were available to summarize")

# =============================================================================
# SUMMARY TABLE OF STATISTICAL RESULTS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE: ALL STATISTICAL RESULTS BY HYPOTHESIS")
print("="*80)

# Helper function to format p-values with significance markers
def format_pvalue(pval):
    """Format p-value with asterisk if p < 0.05"""
    if np.isnan(pval):
        return "N/A"
    elif pval < 0.05:
        return f"{pval:.4f}*"
    else:
        return f"{pval:.4f}"

# Create summary dataframe
summary_results = []

# Overall Measures
overall_measures = [
    ('H1', 'TLX_raw_mean', 'NASA-TLX (Mean)', 'Subjective Workload'),
    ('H2', 'oddball_percentage_mean', 'Oddball Missed Targets (Mean)', 'Objective Workload'),
    ('H3', 'simu_zscore_mean', 'Simulator Performance (Mean)', 'Flight Simulator'),
]

for hyp, measure, short_name, desc in overall_measures:
    df_test = df_eval[[measure, 'Session', 'Group', 'Participant']].dropna()
    
    # Base model
    if len(df_test) > 10:
        model_base = smf.mixedlm(f"{measure} ~ Session + Group + Session:Group", 
                                df_test, groups=df_test["Participant"])
        result_base = model_base.fit()
        
        # Extract p-values for key effects
        group_pval = result_base.pvalues.get('Group[T.tRNS]', np.nan)
        # Look for interaction terms - they can have different names
        interaction_pval = np.nan
        for col in result_base.pvalues.index:
            if 'Session' in col and 'Group' in col:
                interaction_pval = result_base.pvalues[col]
                break
        
        summary_results.append({
            'Hypothesis': hyp,
            'Measure': short_name,
            'Context': desc,
            'Model': 'Base',
            'N': len(df_test),
            'Main Group (p)': format_pvalue(group_pval),
            'Session×Group (p)': format_pvalue(interaction_pval)
        })
    
    # Covariate model
    df_cov = df_eval.merge(df_demographics_EF[['Participant', 'JeuxIntensif', 'HeuresVol']], 
                           on='Participant', how='left')
    df_cov = df_cov[[measure, 'Session', 'Group', 'Participant', 
                     'JeuxIntensif', 'HeuresVol']].dropna()
    
    if len(df_cov) > 10:
        model_cov = smf.mixedlm(f"{measure} ~ Session + Group + Session:Group + JeuxIntensif + HeuresVol", 
                               df_cov, groups=df_cov["Participant"])
        result_cov = model_cov.fit()
        
        group_pval = result_cov.pvalues.get('Group[T.tRNS]', np.nan)
        # Look for interaction terms
        interaction_pval = np.nan
        for col in result_cov.pvalues.index:
            if 'Session' in col and 'Group' in col:
                interaction_pval = result_cov.pvalues[col]
                break
        
        summary_results.append({
            'Hypothesis': hyp,
            'Measure': short_name,
            'Context': desc,
            'Model': 'Covariate',
            'N': len(df_cov),
            'Main Group (p)': format_pvalue(group_pval),
            'Session×Group (p)': format_pvalue(interaction_pval)
        })

# Convert to DataFrame and print
summary_df = pd.DataFrame(summary_results)

print("\n" + "="*160)
print(summary_df.to_string(index=False))
print("="*160)

print("\nHypothesis Definitions:")
print("  H1: Differential improvement in Subjective Workload (NASA-TLX) between groups across sessions")
print("  H2: Differential improvement in Objective Workload (Oddball Missed Targets) between groups across sessions")
print("  H3: Differential improvement in Flight Simulator Performance between groups across sessions")
print("\nNote: Primary test = Session×Group interaction (shown in 'Session×Group (p)' column)")
print("Significance markers: * = p < 0.05")
print("Models: Base (Session + Group + Session:Group) vs Covariate (+ JeuxIntensif + HeuresVol)")
print("Random effects: Participant")

print("\n" + "="*80)
print("MIXED-EFFECTS MODEL ASSUMPTION CHECKS")
print("="*80)
if lme_assumptions_results:
    assumptions_df = pd.DataFrame(lme_assumptions_results)
    assumptions_display = assumptions_df[[
        'Model', 'N_obs', 'Durbin_Watson', 'DW_interpretation',
        'Breusch_Pagan_p', 'BP_interpretation',
        'Shapiro_Wilk_p', 'SW_interpretation'
    ]].copy()
    assumptions_display['Durbin_Watson'] = assumptions_display['Durbin_Watson'].map(lambda x: format_diag_value(x, 3))
    assumptions_display['Breusch_Pagan_p'] = assumptions_display['Breusch_Pagan_p'].map(lambda x: format_diag_value(x, 4))
    assumptions_display['Shapiro_Wilk_p'] = assumptions_display['Shapiro_Wilk_p'].map(lambda x: format_diag_value(x, 4))
    print(assumptions_display.to_string(index=False))

    assumptions_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_LME_assumptions_{log_date}.csv")
    assumptions_df.to_csv(assumptions_csv_path, index=False)
    print(f"\nAssumption summary table saved to: {assumptions_csv_path}")

    covariate_models = assumptions_df[assumptions_df['Model'].str.contains('covariate|Covariate', na=False)]
    if not covariate_models.empty:
        covariate_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_covariate_assumptions_{log_date}.csv")
        covariate_models.to_csv(covariate_csv_path, index=False)
        print(f"Covariate model assumption summary saved to: {covariate_csv_path}")
else:
    print("No mixed-effects models were successfully fitted for assumption checks.")

print("\n" + "="*80)
print("GEE ROBUST SENSITIVITY CHECKS (vs MixedLM)")
print("="*80)
if gee_sensitivity_results:
    gee_df = pd.DataFrame(gee_sensitivity_results)
    gee_display = gee_df[[
        'Model', 'N_obs', 'MixedLM_Group_p', 'GEE_Group_p',
        'MixedLM_Min_SessionGroup_p', 'GEE_Min_SessionGroup_p', 'Status'
    ]].copy()
    gee_display['MixedLM_Group_p'] = gee_display['MixedLM_Group_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['GEE_Group_p'] = gee_display['GEE_Group_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['MixedLM_Min_SessionGroup_p'] = gee_display['MixedLM_Min_SessionGroup_p'].map(lambda x: format_diag_value(x, 4))
    gee_display['GEE_Min_SessionGroup_p'] = gee_display['GEE_Min_SessionGroup_p'].map(lambda x: format_diag_value(x, 4))
    print(gee_display.to_string(index=False))

    gee_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_GEE_sensitivity_{log_date}.csv")
    gee_df.to_csv(gee_csv_path, index=False)
    print(f"\nGEE sensitivity summary table saved to: {gee_csv_path}")
else:
    print("No models were available for GEE sensitivity checks.")

# -----------------------------------------------------------------------------
# Multiple comparisons - FDR adjustment log (Benjamini-Hochberg)
# Collect p-values for Untrained Tasks 
# (MixedLM base and covariate models for TLX, Oddball, and Simulator)..
# -----------------------------------------------------------------------------
pvalue_records = []

# Collect the specific terms for TLX and Oddball base models.

def _collect_terms(pv_map, prefix):
    if pv_map is None:
        return
    # Candidate keys (cover common naming variations from statsmodels)
    key_variants = [
        ('Intercept', 'Intercept'),
        ('const', 'Intercept'),
        ('Session[T.E2]', 'Session[T.E2]'),
        ('Session[T.E3]', 'Session[T.E3]'),
        ('Group[T.tRNS]', 'Group[T.tRNS]'),
        # interaction variants
        ('Session[T.E2]:Group', 'Session[T.E2]:Group'),
        ('Session[T.E3]:Group', 'Session[T.E3]:Group'),
        ('Session[T.E2]:Group[T.tRNS]', 'Session[T.E2]:Group[T.tRNS]'),
        ('Session[T.E3]:Group[T.tRNS]', 'Session[T.E3]:Group[T.tRNS]'),
        ('Group:Session[T.E2]', 'Session[T.E2]:Group'),
        ('Group:Session[T.E3]', 'Session[T.E3]:Group')
    ]

    for candidate, label in key_variants:
        if candidate in pv_map.index:
            try:
                p = pv_map[candidate]
                pvalue_records.append((f"{prefix}_{label}", float(p) if pd.notna(p) else np.nan))
            except Exception:
                continue

# Collect from base mixed models only 
_collect_terms(globals().get('tlx_pvalues', None), 'TLX_base')
_collect_terms(globals().get('oddball_pvalues', None), 'Oddball_base')

# Build DataFrame from collected records
adjusted_log_path = os.path.join(script_dir, "..", "logs", f"{script_name}_Adjusted_{log_date}.txt")
adjusted_csv_path = os.path.join(script_dir, "..", "logs", f"{script_name}_Adjusted_{log_date}.csv")
os.makedirs(os.path.dirname(adjusted_log_path), exist_ok=True)

records = []
for rec in pvalue_records:
    try:
        label = rec[0]
        pval = rec[1]
    except Exception:
        continue
    records.append({'Label': label, 'P_value': float(pval) if np.isfinite(pval) else np.nan})

df_p = pd.DataFrame(records)
if 'P_value' not in df_p.columns:
    df_p = pd.DataFrame(columns=['Label', 'P_value'])

with open(adjusted_log_path, 'w', encoding='utf-8') as af:
    af.write("Multiple comparisons adjustment (Benjamini-Hochberg FDR) - Untrained Tasks\n")
    af.write("Generated by script: %s\n" % os.path.basename(__file__))
    af.write("Analysis date: %s\n\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    total_collected = len(df_p)
    finite_mask = df_p['P_value'].notna()
    finite_count = int(finite_mask.sum())

    af.write(f"Total p-values collected (Untrained Tasks): {total_collected} (finite: {finite_count})\n\n")

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

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll plots saved to: {plot_dir}")
print("\nGenerated plots:")
print("  1. Workload_Simulator_Stacked_MeanCI95.png - stacked workload/simulator (mean + points + 95% CI)")
print(f"\nLog saved to: {log_path}")
print("="*80)

# Close logger
sys.stdout = logger.terminal
logger.close()


