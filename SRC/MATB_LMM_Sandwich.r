# ============================================================
# MATB LMM + VGexp + CR2 robust inference (participant level)
# ============================================================

required_pkgs <- c("lme4", "clubSandwich", "dplyr")
to_install <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(to_install) > 0) install.packages(to_install)

library(lme4)
library(clubSandwich)
library(dplyr)

# 1) Load data
df_best <- read.csv("transformed_data/df_Best_SF_MATB_Mean_Zscores.csv", stringsAsFactors = FALSE)
df_demo <- read.csv("transformed_data/df_demographics_EFs.csv", stringsAsFactors = FALSE)

# 2) Recode groups in both files
recode_group <- function(x) dplyr::recode(x, "A" = "tRNS", "B" = "Sham")

if ("Group" %in% names(df_best)) {
  df_best$Group <- recode_group(df_best$Group)
} else if ("Group_MATB" %in% names(df_best)) {
  df_best$Group <- recode_group(df_best$Group_MATB)
} else if ("Group_SF" %in% names(df_best)) {
  df_best$Group <- recode_group(df_best$Group_SF)
} else {
  stop("No group column found in df_best.")
}

if ("Group" %in% names(df_demo)) {
  df_demo$Group <- recode_group(df_demo$Group)
}

# 3) Keep E1/E2/E3 and add VGexp
df_eval <- df_best |>
  filter(Session %in% c("E1", "E2", "E3")) |>
  left_join(df_demo |> select(Participant, VGexp), by = "Participant") |>
  select(MATB_z_mean, Session, Group, Participant, VGexp) |>
  filter(
    !is.na(MATB_z_mean),
    !is.na(Session),
    !is.na(Group),
    !is.na(Participant),
    !is.na(VGexp)
  )

df_eval$Session <- factor(df_eval$Session, levels = c("E1", "E2", "E3"))
df_eval$Group <- factor(df_eval$Group, levels = c("Sham", "tRNS"))
df_eval$Participant <- factor(df_eval$Participant)

# 4) Fit LMM with VGexp covariate
lmm_matb_vg <- lmer(
  MATB_z_mean ~ Session * Group + VGexp + (1 | Participant),
  data = df_eval,
  REML = TRUE
)

cat("\n===== Conventional LMM summary (with VGexp) =====\n")
print(summary(lmm_matb_vg))

# 5) CR2 robust sandwich covariance (clustered by participant)
V_CR2 <- vcovCR(
  lmm_matb_vg,
  cluster = df_eval$Participant,
  type = "CR2"
)

robust_ct <- coef_test(
  lmm_matb_vg,
  vcov = V_CR2,
  test = "Satterthwaite"
)

cat("\n===== CR2 robust coefficient tests =====\n")
print(robust_ct)

# Robust coefficients + p-values table
robust_df <- as.data.frame(robust_ct)
keep_cols <- c("Coef", "beta", "SE", "tstat", "df_Satt", "p_Satt")
keep_cols <- keep_cols[keep_cols %in% names(robust_df)]

cat("\n===== Robust coefficients and p-values (compact) =====\n")
print(robust_df[, keep_cols, drop = FALSE], row.names = FALSE)

# 6) Residual diagnostics
# Note: CR2 changes inference (SE/p), not residual values themselves.
diag_df <- data.frame(
  fitted = fitted(lmm_matb_vg),
  resid_raw = residuals(lmm_matb_vg, type = "response"),
  resid_pearson = residuals(lmm_matb_vg, type = "pearson"),
  participant = df_eval$Participant
)
diag_df$std_resid <- as.numeric(scale(diag_df$resid_pearson))
diag_df$sqrt_abs_std <- sqrt(abs(diag_df$std_resid))

out_dir <- "plots/Trained_tasks"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

png(
  filename = file.path(out_dir, "MATB_LMM_VGexp_CR2_diagnostics.png"),
  width = 1800, height = 1400, res = 200
)

old_par <- par(no.readonly = TRUE)
par(mfrow = c(2, 2), mar = c(5, 5, 3, 1))

plot(diag_df$fitted, diag_df$resid_pearson,
     xlab = "Fitted values", ylab = "Pearson residuals",
     main = "Residuals vs Fitted", pch = 19, cex = 0.7)
abline(h = 0, lty = 2, col = "red")

qqnorm(diag_df$resid_pearson, main = "Normal Q-Q (Pearson residuals)", pch = 19, cex = 0.7)
qqline(diag_df$resid_pearson, col = "red", lwd = 2)

plot(diag_df$fitted, diag_df$sqrt_abs_std,
     xlab = "Fitted values", ylab = "Sqrt(|standardized residuals|)",
     main = "Scale-Location", pch = 19, cex = 0.7)

boxplot(resid_pearson ~ participant, data = diag_df,
        outline = FALSE, las = 2, cex.axis = 0.6,
        xlab = "Participant", ylab = "Pearson residuals",
        main = "Residuals by Participant")
abline(h = 0, lty = 2, col = "red")

par(old_par)
dev.off()

cat("\nSaved diagnostics to: ", file.path(out_dir, "MATB_LMM_VGexp_CR2_diagnostics.png"), "\n", sep = "")