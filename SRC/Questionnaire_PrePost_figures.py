import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Set up paths relative to script location
script_dir = Path(__file__).parent
root_dir = script_dir.parent
data_path = root_dir / "Raw_csv" / "df_QuestionnairePrePost.csv"
output_dir = root_dir / "plots"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_csv(data_path)

# Define color palette for groups
palette = {'A': 'orange', 'B': 'grey'}

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ===== Figure 1: Pre condition by Session =====
df_pre = df[df['PrePost'] == 'pre']

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=df_pre, x='Session', y='total', hue='Group', 
            palette=palette, ax=ax, width=0.6, dodge=True, boxprops=dict(alpha=0.5))
sns.stripplot(data=df_pre, x='Session', y='total', hue='Group',
              palette=palette, alpha=0.6, size=8, jitter=True, 
              ax=ax, dodge=True, legend=False)

ax.set_title('Pre - Group A (Stimulation) vs Group B (Sham)', fontsize=14, fontweight='bold')
ax.set_xlabel('Session', fontsize=12)
ax.set_ylabel('Total Symptoms', fontsize=12)
ax.get_legend().set_title('Group')
for text in ax.get_legend().get_texts():
    if text.get_text() == 'A':
        text.set_text('Group A (Stimulation)')
    elif text.get_text() == 'B':
        text.set_text('Group B (Sham)')

plt.tight_layout()
plt.savefig(output_dir / 'Total_Symptoms_Pre_by_Session.pdf', dpi=300, bbox_inches='tight')
print("Saved: Total_Symptoms_Pre_by_Session.pdf")
plt.close()

# ===== Figure 2: Post condition by Session =====
df_post = df[df['PrePost'] == 'post']

fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=df_post, x='Session', y='total', hue='Group',
            palette=palette, ax=ax, width=0.6, dodge=True, boxprops=dict(alpha=0))
sns.stripplot(data=df_post, x='Session', y='total', hue='Group',
              palette=palette, alpha=0.6, size=8, jitter=True,
              ax=ax, dodge=True, legend=False)

ax.set_title('Post - Group A (Stimulation) vs Group B (Sham)', fontsize=14, fontweight='bold')
ax.set_xlabel('Session', fontsize=12)
ax.set_ylabel('Total Symptoms', fontsize=12)
ax.get_legend().set_title('Group')
for text in ax.get_legend().get_texts():
    if text.get_text() == 'A':
        text.set_text('Group A (Stimulation)')
    elif text.get_text() == 'B':
        text.set_text('Group B (Sham)')

plt.tight_layout()
plt.savefig(output_dir / 'Total_Symptoms_Post_by_Session.pdf', dpi=300, bbox_inches='tight')
print("Saved: Total_Symptoms_Post_by_Session.pdf")
plt.close()

# ===== Figure 3: Pre/Post comparison with participant means =====
# Compute mean of all sessions for each participant by condition and group
df_means = df.groupby(['Identifiant', 'PrePost', 'Group'])['total'].mean().reset_index()

# Convert PrePost to categorical with ordered levels
df_means['PrePost'] = pd.Categorical(df_means['PrePost'], categories=['pre', 'post'], ordered=True)

fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(data=df_means, x='PrePost', y='total', hue='Group',
            palette=palette, ax=ax, width=0.6, dodge=True, showcaps=False, 
            boxprops=dict(alpha=0.5))
sns.stripplot(data=df_means, x='PrePost', y='total', hue='Group',
              palette=palette, alpha=0.7, size=8, jitter=True,
              ax=ax, dodge=True, legend=False)

ax.set_xlabel('Pre/Post', fontsize=12)
ax.set_ylabel('Total Score', fontsize=12)
ax.get_legend().set_title('Group')
for text in ax.get_legend().get_texts():
    if text.get_text() == 'A':
        text.set_text('Stimulation')
    elif text.get_text() == 'B':
        text.set_text('Sham')

plt.tight_layout()
plt.savefig(output_dir / 'Total_Symptoms_PrePost_Comparison.pdf', dpi=300, bbox_inches='tight')
print("Saved: Total_Symptoms_PrePost_Comparison.pdf")
plt.close()

# ===== Descriptive Statistics for Pre/Post by Group =====
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS - Pre/Post Comparison by Group")
print("="*60)

desc_stats = df_means.groupby(['PrePost', 'Group'])['total'].agg(['mean', 'std', 'count']).reset_index()
desc_stats.columns = ['PrePost', 'Group', 'Mean', 'SD', 'N']

print("\n", desc_stats.to_string(index=False))

# ===== Repeated Measures ANOVA =====
print("\n" + "="*60)
print("REPEATED MEASURES ANOVA")
print("="*60)
print("Design: PrePost (within) x Group (between)")
print("Dependent variable: Total Symptoms Score\n")

# Fit the model using OLS with participant ID as a random factor
# We'll model this as a mixed model using OLS with participant ID as a categorical factor
model = ols('total ~ C(PrePost) * C(Group) + C(Identifiant)', data=df_means).fit()

# Extract ANOVA table (Type II sum of squares)
anova_table = anova_lm(model, typ=2)

# Filter to show only the main effects and interactions (not participant)
# Extract the rows for PrePost, Group, and their interaction
anova_results = anova_table.loc[['C(PrePost)', 'C(Group)', 'C(PrePost):C(Group)']]
print(anova_results.to_string())

# Calculate effect sizes (partial eta-squared)
print("\n" + "-"*60)
print("Effect Sizes (Partial Eta-Squared)")
print("-"*60)

# Extract sum of squares and calculate partial eta-squared
for idx in anova_results.index:
    ss_effect = anova_table.loc[idx, 'sum_sq']
    ss_residual = anova_table.iloc[-1]['sum_sq']  # Last row is residual
    partial_eta_sq = ss_effect / (ss_effect + ss_residual)
    f_val = anova_table.loc[idx, 'F']
    p_val = anova_table.loc[idx, 'PR(>F)']
    print(f"{idx:25} | F={f_val:8.4f}, p={p_val:8.4f}, η²={partial_eta_sq:8.4f}")

print("\n" + "="*60)

# Print summary
print("\nVisualization complete!")
print("Generated 3 PDF files:")
print("  1. Total_Symptoms_Pre_by_Session.pdf")
print("  2. Total_Symptoms_Post_by_Session.pdf")
print("  3. Total_Symptoms_PrePost_Comparison.pdf")

# Print data summary
print(f"\nData summary:")
print(f"Total participants: {df['Identifiant'].nunique()}")
print(f"Sessions: {', '.join(map(str, sorted(df['Session'].unique())))}")
print(f"Groups: {', '.join(sorted(df['Group'].unique()))}")
print(f"Conditions: {', '.join(sorted(df['PrePost'].unique()))}")
