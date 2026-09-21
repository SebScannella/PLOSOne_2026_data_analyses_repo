# Brain Stimulation Effects on Trained and Untrained Tasks

This repository contains the cleaned analysis pipeline used to assess the effect of brain stimulation (sham vs stimulated/tRNS group) on:

- **Trained tasks** (Space Fortress and MATB)
- **Untrained tasks** (NASA-TLX workload, oddball missed targets, simulator performance)

The analyses focus on repeated measurements across sessions and estimate group effects, session effects, and group-by-session interactions.

## Study Question

Does active brain stimulation (tRNS) produce different performance trajectories than sham across training and evaluation phases?

We address this with mixed-effects models and robustness checks.

## Repository Structure

- `SRC/SF_MATB_Trained_Tasks_Analysis.py`: main analysis for trained tasks
- `SRC/Untrained_Tasks_Analysis.py`: main analysis for untrained tasks
- `SRC/Post_hoc_analyses.py`: post-hoc comparisons for trained-task learning rates
- `SRC/MATB_LMM_Sandwich.r`: robust mixed-model inference example in R (sandwich/CR2)
- `SRC/Questionnaire_PrePost_figures.py`: pre/post questionnaire figure generation
- `transformed_data/`: cleaned input datasets used by the scripts
- `logs/`: dated text outputs + assumptions/sensitivity CSV exports
- `plots/Trained_tasks/` and `plots/Untrained_tasks/`: generated figures

## Data Inputs

### Trained-task script

- `transformed_data/df_Best_SF_MATB_Mean_Zscores.csv`
- `transformed_data/df_demographics_EFs.csv`

### Untrained-task script

- `transformed_data/df_Best_SF_MATB_Mean_Zscores_SIMU.csv`
- `transformed_data/df_demographics_EFs.csv`

### Group coding

The scripts recode groups as:

- `A` -> `tRNS` (stimulated)
- `B` -> `Sham`

## Analysis Strategy

### 1) Trained tasks (SF + MATB)

Script: `SRC/SF_MATB_Trained_Tasks_Analysis.py`

Main steps:

1. Load cleaned trained-task and demographics/EF data.
2. Recode group labels to `Sham`/`tRNS`.
3. Compute **learning rates** on training sessions (`T01`-`T10`) using participant-level log-session slopes.
4. Compare learning rates between groups.
5. Fit mixed-effects models on evaluation sessions (`E1`, `E2`, `E3`) for key outcomes:
   - `SF_zscore ~ Session * Group`
   - `MATB_z_mean ~ Session * Group`
6. Run selected covariate-adjusted models.
7. Test correlations with executive-function indices.
8. Export diagnostics and sensitivity checks.

### 2) Untrained tasks (TLX + Oddball + Simulator)

Script: `SRC/Untrained_Task_Analysis.py`

Main steps:

1. Load cleaned untrained-task and demographics data.
2. Recode group labels to `Sham`/`tRNS`.
3. Keep evaluation sessions (`E1`, `E2`, `E3`) for longitudinal comparisons.
4. Fit mixed-effects models for:
   - `TLX_raw_mean ~ Session * Group`
   - `oddball_percentage_mean ~ Session * Group`
   - `simu_zscore_mean ~ Session * Group`
5. Fit covariate-adjusted versions (notably gaming intensity and flight-hour variables).
6. Run scenario-specific analyses where relevant.
7. Run session-wise group t-tests used in figure annotations.
8. Export diagnostics and sensitivity checks.

## Model Robustness and Assumptions

Both main Python scripts include:

- Mixed-effects modeling (`statsmodels` MixedLM)
- Residual checks:
  - Durbin-Watson
  - Breusch-Pagan
  - Shapiro-Wilk
- Sensitivity comparison against marginal models (GEE-based check in Python outputs)
- CSV export of assumption and sensitivity summaries

Additionally, `SRC/MATB_LMM_Sandwich.r` provides an R-based robust-inference path (sandwich/CR2 style) for MATB-focused checks.

## Outputs

Running the scripts generates dated outputs:

- Text logs in `logs/`:
  - `SF_MATB_Trained_Tasks_Analysis_YYYY_MM_DD.txt`
  - `Untrained_Tasks_Analysis_YYYY_MM_DD.txt`
- Diagnostic CSVs in `logs/`:
  - `*_LME_assumptions_YYYY_MM_DD.csv`
  - `*_covariate_assumptions_YYYY_MM_DD.csv`
  - `*_GEE_sensitivity_YYYY_MM_DD.csv`
- Figures in:
  - `plots/Trained_tasks/`
  - `plots/Untrained_tasks/`

## How to Run

From repository root:

```bash
python SRC/SF_MATB_Trained_Tasks_Analysis.py
python SRC/Untrained_Tasks_Analysis.py
```

Optional post-hoc trained-task comparisons:

```bash
python SRC/Post_hoc_analyses.py
```

Complementary robust R analysis:

```r
source("SRC/MATB_LMM_Sandwich.r")
```

## Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

## Interpretation Notes

- The critical inferential term for stimulation effects over time is typically the **Session x Group interaction**.
- Main group effects indicate average between-group differences across sessions.
- Session effects indicate longitudinal change independent of group.

## Reproducibility Notes

- Scripts assume the existing folder layout and relative paths in this repository.
- Outputs are date-stamped; re-running scripts on a new date creates new log/CSV files.
- Use the cleaned files in `transformed_data/` as the canonical analysis inputs.
