# ==============================================================================
# 03_gof_metrics.R — Goodness-of-Fit Metrics for All 9 Methods × 3 Models
# ==============================================================================
# Input  : data/drives_clean.csv, R/results/parameters.csv
# Output : R/results/gof_metrics.csv
# ==============================================================================
# Metrics: KS, RMSE, MAE, Anderson-Darling, AIC, AICc, BIC, MRL Error %,
#          Composite Rank (Borda count of rank_RMSE + rank_AD + rank_AIC)
# ==============================================================================

# ── Reproducibility ──────────────────────────────────────────────────────────
set.seed(42)
options(scipen = 999)
options(digits = 6)
Sys.setenv(LANG = "en_US")

# ── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)
library(here)

cat("==================================================================\n")
cat(" Goodness-of-Fit Evaluation — 9 Methods × 3 Models\n")
cat("==================================================================\n\n")

# ── Load data ────────────────────────────────────────────────────────────────
drives <- read_csv(here::here("data", "drives_clean.csv"), show_col_types = FALSE)
params <- read_csv(here::here("R", "results", "parameters.csv"), show_col_types = FALSE)

models <- unique(drives$model)

# ==============================================================================
# CENSORED LOG-LIKELIHOOD FUNCTION
# ==============================================================================
#' Compute censored Weibull log-likelihood at given (k, λ) for ALL data
#'
#' loglik(k,λ) =
#'   Σ_{failed=1} [log(k) - log(λ) + (k-1)*log(t/λ) - (t/λ)^k]
#' + Σ_{failed=0} [-(t/λ)^k]
#'
#' Used for AIC/BIC computation for ALL 9 methods.
#' IMPORTANT: AIC is theoretically derived for MLE only.
#' For non-MLE methods, we evaluate the log-likelihood at their (k,λ) estimates
#' as a comparative convenience, not a theoretically valid AIC.
#'
#' @param t Numeric vector of observation times
#' @param event Binary vector (1 = failure, 0 = censored)
#' @param k Shape parameter
#' @param lam Scale parameter
#' @return Log-likelihood value
censored_loglik <- function(t, event, k, lam) {
  if (is.na(k) || is.na(lam) || k <= 0 || lam <= 0) return(NA_real_)
  
  t_ratio <- t / lam
  
  # Failures: full log-PDF
  ll_fail <- sum(event * (log(k) - log(lam) + (k - 1) * log(t_ratio) - t_ratio^k))
  
  # Censored: log-survival = -(t/λ)^k
  ll_cens <- sum((1 - event) * (-t_ratio^k))
  
  ll_fail + ll_cens
}

# ==============================================================================
# MAIN GoF COMPUTATION
# ==============================================================================

all_gof <- tibble()

for (mdl in models) {
  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  cat("  Model:", mdl, "\n")
  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  
  drive_data <- drives %>% filter(model == mdl)
  t_all      <- drive_data$lifetime_days
  event_all  <- drive_data$failed
  n_total    <- nrow(drive_data)
  n_failures <- sum(event_all)
  cens_pct   <- round((1 - mean(event_all)) * 100, 2)
  
  # Failure times only (for empirical CDF metrics)
  t_failures <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n_f <- length(t_failures)
  
  # Empirical CDF using Benard's formula
  # F_emp_i = (i - 0.3) / (n + 0.4)
  i_seq <- seq_len(n_f)
  F_emp <- (i_seq - 0.3) / (n_f + 0.4)
  
  # Get parameters for this model
  model_params <- params %>% filter(model == mdl)
  
  gof_rows <- list()
  
  for (j in seq_len(nrow(model_params))) {
    mth <- model_params$method[j]
    k   <- model_params$k[j]
    lam <- model_params$lambda[j]
    
    # Skip methods with NA parameters
    if (is.na(k) || is.na(lam)) {
      gof_rows[[j]] <- tibble(
        model = mdl, method = mth,
        KS = NA, RMSE = NA, MAE = NA, AD = NA,
        AIC = NA, AICc = NA, BIC = NA, MRL_err = NA,
        rank_RMSE = NA, rank_AD = NA, rank_AIC = NA,
        Composite_Rank = NA
      )
      cat(sprintf("  %-12s → SKIPPED (NA parameters)\n", mth))
      next
    }
    
    # ── Fitted CDF at failure times ────────────────────────────────────
    F_fit <- 1 - exp(-(t_failures / lam)^k)
    
    # PIT values — clip to avoid log(0) in Anderson-Darling
    z <- pmax(1e-10, pmin(1 - 1e-10, F_fit))
    
    # ── KS (Kolmogorov-Smirnov) ─────────────────────────────────────
    # L-infinity norm: worst-case single-point deviation.
    # CAVEAT 1: When parameters are estimated from the same data,
    # the KS test becomes conservative — do not compare against
    # standard KS critical values. Use as descriptive metric only.
    # CAVEAT 2: Standard KS does not handle right-censored data.
    # Computed over uncensored failure times only — approximate.
    KS <- max(abs(F_emp - F_fit))
    
    # ── RMSE ─────────────────────────────────────────────────────────
    # L2 norm: sensitive to large deviations.
    # One point with deviation 0.10 contributes 0.01;
    # ten points each with deviation 0.032 contribute only 0.0102.
    RMSE <- sqrt(mean((F_emp - F_fit)^2))
    
    # ── MAE ──────────────────────────────────────────────────────────
    # L1 norm: equal weight to every deviation.
    # Most interpretable: "CDF off by X percentage points on average."
    # Always MAE ≤ RMSE ≤ sqrt(n)*MAE.
    MAE <- mean(abs(F_emp - F_fit))
    
    # ── Anderson-Darling ─────────────────────────────────────────────
    # Tail-amplified statistic.
    # Weight = 1/[u*(1-u)]:
    #   u=0.50: weight=4.0   (1.0x baseline)
    #   u=0.10: weight=11.1  (2.8x)
    #   u=0.05: weight=21.1  (5.3x)
    #   u=0.02: weight=51.0  (12.8x)
    # Variance argument: asymptotic Var(empirical CDF at u) = u*(1-u).
    # AD standardizes each squared deviation by its natural variance.
    # Most powerful for tail misfit detection.
    z_rev <- rev(z)
    AD <- -n_f - (1 / n_f) * sum((2 * i_seq - 1) * (log(z) + log(1 - z_rev)))
    
    # ── AIC ──────────────────────────────────────────────────────────
    # Recompute full censored loglik at (k, λ) for ALL observations
    # AIC = -2*loglik + 2*p, where p = 2 (k and λ)
    # IMPORTANT: AIC is theoretically derived for MLE only.
    # For non-MLE methods, this is a comparative convenience.
    ll <- censored_loglik(t_all, event_all, k, lam)
    AIC_val <- if (!is.na(ll)) -2 * ll + 2 * 2 else NA_real_
    
    # ── AICc (small sample correction) ───────────────────────────────
    # For models where n_total < 40 or n_total/2 < 20:
    # AICc = AIC + 2*p*(p+1)/(n-p-1), p = 2
    AICc_val <- NA_real_
    if (!is.na(AIC_val)) {
      if (n_total < 40 || (n_total / 2) < 20) {
        AICc_val <- AIC_val + 2 * 2 * (2 + 1) / (n_total - 2 - 1)
      } else {
        AICc_val <- AIC_val  # AICc ≈ AIC for large n
      }
    }
    
    # ── BIC ──────────────────────────────────────────────────────────
    # BIC = -2*loglik + log(n_total)*p
    # For n > 7: ln(n) > 2, BIC penalizes more heavily than AIC.
    # Same caveat as AIC for non-MLE methods.
    BIC_val <- if (!is.na(ll)) -2 * ll + log(n_total) * 2 else NA_real_
    
    # ── MRL Error % ──────────────────────────────────────────────────
    # MTTF_predicted = λ * Γ(1 + 1/k)
    # MTTF_observed  = mean(t_failures)
    # CAVEAT: t_bar_f uses only observed failures. With heavy censoring,
    # t_bar_f is downward-biased (long-surviving censored units excluded).
    MTTF_pred <- lam * gamma(1 + 1/k)
    MTTF_obs  <- mean(t_failures)
    MRL_err   <- abs(MTTF_pred - MTTF_obs) / MTTF_obs * 100
    
    gof_rows[[j]] <- tibble(
      model = mdl, method = mth,
      KS = KS, RMSE = RMSE, MAE = MAE, AD = AD,
      AIC = AIC_val, AICc = AICc_val, BIC = BIC_val,
      MRL_err = MRL_err,
      rank_RMSE = NA_real_, rank_AD = NA_real_, rank_AIC = NA_real_,
      Composite_Rank = NA_real_
    )
    
    cat(sprintf("  %-12s → KS=%.5f  RMSE=%.6f  AD=%.3f  AIC=%.1f  MRL=%.1f%%\n",
                mth, KS, RMSE, AD,
                ifelse(is.na(AIC_val), NA, AIC_val),
                MRL_err))
  }
  
  # Combine rows for this model
  model_gof <- bind_rows(gof_rows)
  
  # ── Composite Rank (Borda count) ─────────────────────────────────────
  # rank_RMSE = rank(RMSE)    # 1 = best, 9 = worst
  # rank_AD   = rank(AD)
  # rank_AIC  = rank(AIC)
  # Composite = (rank_RMSE + rank_AD + rank_AIC) / 3
  #
  # WHY these three:
  #   RMSE: global body fit (L2)
  #   AD: tail-sensitive (catches what RMSE misses)
  #   AIC: likelihood-based, handles censoring natively
  # WHY NOT others:
  #   BIC excluded: ranks same as AIC for fixed parameter count (p=2)
  #   KS/MAE reported but redundant given RMSE
  #   MRL reported separately: operational metric, not shape metric
  
  valid_rows <- !is.na(model_gof$RMSE)
  
  if (sum(valid_rows) > 0) {
    model_gof$rank_RMSE[valid_rows] <- rank(model_gof$RMSE[valid_rows])
    model_gof$rank_AD[valid_rows]   <- rank(model_gof$AD[valid_rows])
    model_gof$rank_AIC[valid_rows]  <- rank(model_gof$AIC[valid_rows])
    model_gof$Composite_Rank[valid_rows] <- (
      model_gof$rank_RMSE[valid_rows] +
      model_gof$rank_AD[valid_rows] +
      model_gof$rank_AIC[valid_rows]
    ) / 3
  }
  
  # Print ranking
  cat("\n  Composite Ranking:\n")
  ranked <- model_gof %>%
    filter(!is.na(Composite_Rank)) %>%
    arrange(Composite_Rank)
  for (r in seq_len(nrow(ranked))) {
    cat(sprintf("    #%d  %-12s  (Composite = %.2f)\n",
                r, ranked$method[r], ranked$Composite_Rank[r]))
  }
  cat("\n")
  
  all_gof <- bind_rows(all_gof, model_gof)
}

# ==============================================================================
# Save Results
# ==============================================================================

cat("==================================================================\n")
cat(" Saving GoF metrics...\n")
cat("==================================================================\n\n")

gof_path <- here::here("R", "results", "gof_metrics.csv")
write_csv(all_gof, gof_path)
cat("  Saved:", gof_path, "\n")
cat("  Rows:", nrow(all_gof), "\n")

# Print Anderson-Darling critical values reference
cat("\n  Anderson-Darling approximate critical values (Weibull, MLE-estimated):\n")
cat("    α = 0.25:  0.474\n")
cat("    α = 0.10:  0.637\n")
cat("    α = 0.05:  0.757\n")
cat("    α = 0.025: 0.877\n")
cat("    α = 0.01:  1.038\n")
cat("  (When used as ranking metric, critical values not needed.)\n")

# Print delta_AIC interpretation
cat("\n  delta_AIC interpretation:\n")
cat("    0-2:   Both models substantially supported\n")
cat("    2-6:   Moderate evidence for lower-AIC model\n")
cat("    6-10:  Strong evidence\n")
cat("    >10:   Very strong evidence; higher-AIC negligible support\n")

cat("\n[DONE] 03_gof_metrics.R complete.\n")
