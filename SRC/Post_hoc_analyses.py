"""
Post-hoc Analyses for tRNS Study
1) Comparing learning rates between SF and MATB
2) Comparing retention deltas (E3 - E2) between tRNS and Sham groups for both SF and MATB
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# ============================================================================
# LOAD DATA
# ============================================================================

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'transformed_data')

# Load main dataset with proper group labels
df_best = pd.read_csv(os.path.join(data_dir, 'df_Best_SF_MATB_Mean_Zscores.csv'))

# Keep only necessary columns and create consistent naming
df_SF_MATB = df_best[['Participant', 'Session', 'Group_SF', 'SF_zscore', 'MATB_z_mean']].copy()
df_SF_MATB.rename(columns={'MATB_z_mean': 'MATB_zscore', 'Group_SF': 'Group'}, inplace=True)

# Map group codes to labels
group_mapping = {'A': 'tRNS', 'B': 'Sham'}
df_SF_MATB['Group'] = df_SF_MATB['Group'].map(group_mapping)

# ============================================================================
# COMPUTE LEARNING RATES
# ============================================================================

# Filter training sessions (T01 to T10)
df_training = df_SF_MATB[df_SF_MATB['Session'].str.match(r'T0[1-9]|T10')].copy()
df_training['Session_Num'] = df_training['Session'].str.replace('T0', '').str.replace('T', '').astype(int)

def fit_log_model(group_data, score_column):
    """Fit log(x) model and return coefficient (learning rate)"""
    if len(group_data) < 3:
        return np.nan
    try:
        x = np.log(group_data['Session_Num'].values)
        y = group_data[score_column].values
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return np.nan
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    except:
        return np.nan

# Calculate learning rates per participant
learning_rates_SF = df_training.groupby(['Participant', 'Group'], group_keys=False).apply(
    lambda x: pd.Series({'LearningRate_SF': fit_log_model(x, 'SF_zscore')}), include_groups=False
).reset_index()

learning_rates_MATB = df_training.groupby(['Participant', 'Group'], group_keys=False).apply(
    lambda x: pd.Series({'LearningRate_MATB': fit_log_model(x, 'MATB_zscore')}), include_groups=False
).reset_index()

# Check data availability
print(f"SF learning rates computed: {learning_rates_SF['LearningRate_SF'].notna().sum()} participants")
print(f"MATB learning rates computed: {learning_rates_MATB['LearningRate_MATB'].notna().sum()} participants")

# Merge SF and MATB learning rates
learning_rates_combined = learning_rates_SF.merge(
    learning_rates_MATB[['Participant', 'LearningRate_MATB']], 
    on='Participant', 
    how='inner'
)

# Remove participants with missing data
learning_rates_combined = learning_rates_combined.dropna(subset=['LearningRate_SF', 'LearningRate_MATB'])

print("="*80)
print("POST-HOC ANALYSIS: SF vs MATB Learning Rate Comparison")
print("="*80)

# ============================================================================
# PAIRED T-TEST: SF vs MATB LEARNING RATES (ALL GROUPS COMBINED)
# ============================================================================

print("\n" + "-"*80)
print("Paired t-test: SF vs MATB Learning Rates (All Participants)")
print("-"*80)

print(f"\nSample size: n={len(learning_rates_combined)} participants")
print(f"\nDescriptive statistics:")
print(f"  SF Learning Rate:   mean={learning_rates_combined['LearningRate_SF'].mean():.4f}, "
      f"SD={learning_rates_combined['LearningRate_SF'].std():.4f}")
print(f"  MATB Learning Rate: mean={learning_rates_combined['LearningRate_MATB'].mean():.4f}, "
      f"SD={learning_rates_combined['LearningRate_MATB'].std():.4f}")

# Compute difference
learning_rates_combined['Difference'] = learning_rates_combined['LearningRate_SF'] - learning_rates_combined['LearningRate_MATB']
print(f"\n  Difference (SF - MATB): mean={learning_rates_combined['Difference'].mean():.4f}, "
      f"SD={learning_rates_combined['Difference'].std():.4f}")

# Paired t-test
t_stat, p_val = stats.ttest_rel(learning_rates_combined['LearningRate_SF'], 
                                 learning_rates_combined['LearningRate_MATB'])

print(f"\nPaired t-test results:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_val:.4f}")
print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")

# Effect size (Cohen's d for paired samples)
mean_diff = learning_rates_combined['Difference'].mean()
sd_diff = learning_rates_combined['Difference'].std()
cohens_d = mean_diff / sd_diff

print(f"\nEffect size:")
print(f"  Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Interpretation: {effect_interpretation}")

# ============================================================================
# OVERALL DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "-"*80)
print("Overall Descriptive Statistics (All Groups Combined)")
print("-"*80)

print(f"\nSF Learning Rate:")
print(f"  Mean = {learning_rates_combined['LearningRate_SF'].mean():.4f}")
print(f"  SD   = {learning_rates_combined['LearningRate_SF'].std():.4f}")
print(f"  Min  = {learning_rates_combined['LearningRate_SF'].min():.4f}")
print(f"  Max  = {learning_rates_combined['LearningRate_SF'].max():.4f}")

print(f"\nMATB Learning Rate:")
print(f"  Mean = {learning_rates_combined['LearningRate_MATB'].mean():.4f}")
print(f"  SD   = {learning_rates_combined['LearningRate_MATB'].std():.4f}")
print(f"  Min  = {learning_rates_combined['LearningRate_MATB'].min():.4f}")
print(f"  Max  = {learning_rates_combined['LearningRate_MATB'].max():.4f}")

# ============================================================================
# DESCRIPTIVE BY GROUP
# ============================================================================

print("\n" + "-"*80)
print("Descriptive Statistics by Group")
print("-"*80)

for group in ['tRNS', 'Sham']:
    group_data = learning_rates_combined[learning_rates_combined['Group'] == group]
    print(f"\n{group} Group (n={len(group_data)}):")
    print(f"  SF Learning Rate:   mean={group_data['LearningRate_SF'].mean():.4f}, "
          f"SD={group_data['LearningRate_SF'].std():.4f}")
    print(f"  MATB Learning Rate: mean={group_data['LearningRate_MATB'].mean():.4f}, "
          f"SD={group_data['LearningRate_MATB'].std():.4f}")
    print(f"  Difference (SF - MATB): mean={group_data['Difference'].mean():.4f}, "
          f"SD={group_data['Difference'].std():.4f}")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)

# ============================================================================
# Compute Retention effect stats for both MATB and SF to compare with previous study Chenot et al., 2022
# ============================================================================
print("\n" + "-"*80)
print("Retention (E3 - E2) delta analyses for SF and MATB by Group")
print("-"*80)

# Select E2 and E3 sessions
df_E2 = df_SF_MATB[df_SF_MATB['Session'] == 'E2'][['Participant', 'SF_zscore', 'MATB_zscore', 'Group']].copy()
df_E2.rename(columns={'SF_zscore': 'SF_E2', 'MATB_zscore': 'MATB_E2'}, inplace=True)

df_E3 = df_SF_MATB[df_SF_MATB['Session'] == 'E3'][['Participant', 'SF_zscore', 'MATB_zscore', 'Group']].copy()
df_E3.rename(columns={'SF_zscore': 'SF_E3', 'MATB_zscore': 'MATB_E3'}, inplace=True)

# Merge E2 and E3
df_retention = pd.merge(df_E2, df_E3, on='Participant', suffixes=('_E2', '_E3'))

# Pick Group from E2 
if 'Group_E2' in df_retention.columns:
    df_retention['Group'] = df_retention['Group_E2']

# Compute perf deltas (E3 - E2)
df_retention['Delta_SF'] = df_retention['SF_E3'] - df_retention['SF_E2']
df_retention['Delta_MATB'] = df_retention['MATB_E3'] - df_retention['MATB_E2']

# Drop participants with missing deltas
df_retention = df_retention.dropna(subset=['Delta_SF', 'Delta_MATB', 'Group'])

print(f"Retention delta rows available: {len(df_retention)} participants")

from scipy import stats

def compare_delta_by_group(df, delta_col, group_col='Group'):
    g1 = df[df[group_col] == 'tRNS'][delta_col].dropna()
    g2 = df[df[group_col] == 'Sham'][delta_col].dropna()
    n1, n2 = len(g1), len(g2)
    mean1, mean2 = g1.mean(), g2.mean()
    sd1, sd2 = g1.std(), g2.std()

    # Welch's t-test (unequal variance)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

    # Cohen's d (pooled sd)
    if n1 + n2 - 2 > 0:
        pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else np.nan
    else:
        cohens_d = np.nan

    print(f"\nDelta: {delta_col}")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f} (Welch's t-test)")
    print(f"  tRNS (n={n1}): mean={mean1:.4f}, SD={sd1:.4f}")
    print(f"  Sham (n={n2}): mean={mean2:.4f}, SD={sd2:.4f}")
    print(f"  Cohen's d (tRNS - Sham): {cohens_d:.4f}")

    return {'t_stat': t_stat, 'p_val': p_val, 'mean1': mean1, 'mean2': mean2, 'sd1': sd1, 'sd2': sd2, 'cohens_d': cohens_d}

# Compare Delta_SF
res_sf = compare_delta_by_group(df_retention, 'Delta_SF')

# Compare Delta_MATB
res_matb = compare_delta_by_group(df_retention, 'Delta_MATB')

print("\nAnalysis finished: retention deltas compared between groups.")





# ============================================================================
# STABILISATION ANALYSIS: detect when MATB performance stabilizes per participant
# ============================================================================
out_dir= os.path.join(script_dir, '..', 'logs')
def detect_stabilization(df, score_col='MATB_zscore', session_pattern=r'T0[1-9]|T10',
                         min_sessions=4, slope_eps=1e-4, threshold_pct=0.05):
    """Detect stabilization session per participant using two methods:
    1) Slope method: find earliest training session s.t. the slope of
       score ~ ln(session_num) computed over sessions [s..end] <= slope_eps.
    2) Threshold method: find earliest session where the participant's
       score >= (participant_max - threshold_pct * (participant_max - participant_min)).

    Returns a DataFrame with one row per participant and columns for both detections.
    """
    import re
    dfm = df[['Participant', 'Session', score_col, 'Group']].dropna()
    if dfm.empty:
        return pd.DataFrame()

    # Keep only training sessions and extract numeric session index
    mask = dfm['Session'].str.match(session_pattern)
    df_train = dfm[mask].copy()
    if df_train.empty:
        return pd.DataFrame()

    # derive Session_Num
    df_train['Session_Num'] = df_train['Session'].str.replace('T0', '').str.replace('T', '').astype(int)

    rows = []
    for pid, g in df_train.groupby('Participant'):
        row = {'Participant': pid}
        g_sorted = g.sort_values('Session_Num')
        n = len(g_sorted)
        row['n_training_sessions'] = n
        row['Group'] = g_sorted['Group'].iloc[0] if 'Group' in g_sorted.columns else np.nan

        # threshold method
        pmax = float(g_sorted[score_col].max())
        pmin = float(g_sorted[score_col].min())
        if pmax == pmin:
            thresh = pmax
        else:
            thresh = pmax - threshold_pct * (pmax - pmin)

        thresh_session = np.nan
        for _, rr in g_sorted.iterrows():
            if rr[score_col] >= thresh:
                thresh_session = rr['Session']
                thresh_session_num = int(rr['Session_Num'])
                break
        row['Stabilize_By_Threshold_Session'] = thresh_session
        row['Stabilize_By_Threshold_Session_Num'] = thresh_session_num if not pd.isna(thresh_session) else np.nan
        row['Stabilize_By_Threshold_Value'] = float(thresh)

        # slope method: earliest session where slope over remainder <= slope_eps
        slope_session = np.nan
        slope_value = np.nan
        if n >= min_sessions:
            sess_nums = g_sorted['Session_Num'].values
            scores = g_sorted[score_col].values
            # try each possible start index
            for start_idx in range(0, n):
                x = np.log(sess_nums[start_idx:])
                y = scores[start_idx:]
                if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                    continue
                try:
                    s, intercept = np.polyfit(x, y, 1)
                except Exception:
                    continue
                # small positive slope considered continued improvement; we look for <= eps
                if s <= slope_eps:
                    slope_session = int(sess_nums[start_idx])
                    slope_value = float(s)
                    break

        row['Stabilize_By_Slope_Session_Num'] = slope_session
        row['Stabilize_By_Slope_Value'] = slope_value

        # also provide human-readable session for slope if present
        if not pd.isna(slope_session):
            row['Stabilize_By_Slope_Session'] = f'T{int(slope_session):02d}'
        else:
            row['Stabilize_By_Slope_Session'] = np.nan

        rows.append(row)

    out_df = pd.DataFrame(rows)
    return out_df


# Run stabilization detection (does not execute any external scripts)
print('\n' + '-'*80)
print('STABILIZATION ANALYSIS: detect earliest session where MATB stabilizes per participant')
print('-'*80)

stability_df = detect_stabilization(df_SF_MATB, score_col='MATB_zscore', min_sessions=4, slope_eps=1e-4, threshold_pct=0.05)
if stability_df.empty:
    print('No training data available for stabilization analysis.')
else:
    out_stab = os.path.join(out_dir, 'MATB_stabilization_by_participant.csv')
    try:
        stability_df.to_csv(out_stab, index=False)
        print(f'Stabilization summary saved to: {out_stab}')
    except Exception as e:
        print(f'Failed to save stabilization summary: {e}')

    # Compute group-level mean stabilization session numbers for each detection metric
    try:
        cols = ['Stabilize_By_Threshold_Session_Num', 'Stabilize_By_Slope_Session_Num']
        grp_summary_rows = []
        for col in cols:
            if col in stability_df.columns:
                tmp = stability_df[['Group', col]].copy()
                tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
                grp = tmp.groupby('Group')[col].agg(['mean', 'std', 'count']).reset_index()
                for _, r in grp.iterrows():
                    grp_summary_rows.append({
                        'Group': r['Group'],
                        'Metric': col,
                        'Mean_Stabilization_Session_Num': float(r['mean']) if not pd.isna(r['mean']) else np.nan,
                        'SD_Stabilization_Session_Num': float(r['std']) if not pd.isna(r['std']) else np.nan,
                        'N': int(r['count'])
                    })

        if grp_summary_rows:
            grp_summary_df = pd.DataFrame(grp_summary_rows)
            # Compute overall averages across the group×metric rows
            try:
                mean_of_means = grp_summary_df['Mean_Stabilization_Session_Num'].dropna().mean()
                mean_of_sds = grp_summary_df['SD_Stabilization_Session_Num'].dropna().mean()
                overall = {
                    'Overall_Mean_of_Means': float(mean_of_means) if not pd.isna(mean_of_means) else np.nan,
                    'Overall_Mean_of_SDs': float(mean_of_sds) if not pd.isna(mean_of_sds) else np.nan,
                    'N_group_metric_rows': int(len(grp_summary_df))
                }
                # append overall summary as an "ALL" group row and save combined CSV
                overall_row = {
                    'Group': 'ALL',
                    'Metric': 'Overall',
                    'Mean_Stabilization_Session_Num': overall['Overall_Mean_of_Means'],
                    'SD_Stabilization_Session_Num': overall['Overall_Mean_of_SDs'],
                    'N': overall['N_group_metric_rows']
                }
                combined_df = pd.concat([grp_summary_df, pd.DataFrame([overall_row])], ignore_index=True)
                out_combined = os.path.join(out_dir, 'MATB_stabilization_summary_combined.csv')
                combined_df.to_csv(out_combined, index=False)
                print(f'Combined stabilization summary (group + overall) saved to: {out_combined}')
                print('\nCombined stabilization summary:')
                print(combined_df.to_string(index=False))
            except Exception as e:
                print(f'Failed to compute/save overall stabilization summary: {e}')
    except Exception as e:
        print(f'Failed to compute/save group stabilization summary: {e}')

