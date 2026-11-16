# ==============================================================================
# 02_estimation_methods.R — Nine Weibull Parameter Estimation Methods
# ==============================================================================
# Input  : data/drives_clean.csv
# Output : R/results/parameters.csv
#          R/results/brms_fit_[model].rds (MCMC fits)
# ==============================================================================
# Methods implemented:
#   1. OLS        — Ordinary Least Squares rank regression (= MRR)
#   2. WLS        — Weighted Least Squares (Bergman weights)
#   3. MOM        — Method of Moments
#   4. L-Moments  — L-Moment estimation
#   5. EPF        — Energy Pattern Factor
#   6. MLE        — Maximum Likelihood (Nelder-Mead)
#   7A. MAP       — Bayesian MAP with exponential priors
#   7B. MCMC      — Full Bayesian via brms/Stan (EXTENSION method)
#   8. Percentile — Two-quantile analytical method
# ==============================================================================

# ── Reproducibility ──────────────────────────────────────────────────────────
set.seed(42)
options(scipen = 999)
options(digits = 6)
Sys.setenv(LANG = "en_US")

# ── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)
library(here)

# brms is optional — only needed for Method 7B (MCMC extension)
HAS_BRMS <- requireNamespace("brms", quietly = TRUE)
if (HAS_BRMS) {
  library(brms)
  cat("  brms loaded — MCMC (7B) available.\n")
} else {
  cat("  brms not installed — MCMC (7B) will be skipped.\n")
  cat("  Install with: install.packages('brms')\n")
}

# ── Load Data ────────────────────────────────────────────────────────────────
cat("==================================================================\n")
cat(" Weibull Parameter Estimation — 9 Methods × 3 Models\n")
cat("==================================================================\n\n")

drives <- read_csv(here::here("data", "drives_clean.csv"), show_col_types = FALSE)
models <- unique(drives$model)
cat("  Models:", paste(models, collapse = ", "), "\n")
cat("  Total drives:", format(nrow(drives), big.mark = ","), "\n\n")

# Create results directories
dir.create(here::here("R", "results"), showWarnings = FALSE, recursive = TRUE)
dir.create(here::here("R", "results", "plots"), showWarnings = FALSE)

# ==============================================================================
# SANITY CHECK FUNCTION
# ==============================================================================
#' Validate estimated parameters before saving
#' @param k Shape parameter estimate
#' @param lam Scale parameter estimate
#' @param method_name Name of the estimation method
#' @return TRUE if estimates are valid, FALSE if NA or non-positive
check_estimates <- function(k, lam, method_name) {
  if (is.na(k) || is.na(lam)) {
    warning(paste(method_name, ": NA result — optimization failed"))
    return(FALSE)
  }
  if (k <= 0 || lam <= 0) {
    warning(paste(method_name, ": non-positive parameter"))
    return(FALSE)
  }
  if (k > 20) {
    warning(paste(method_name, ": k =", round(k, 2),
                  "— suspiciously large, possible noise-chasing"))
  }
  if (k < 0.3) {
    warning(paste(method_name, ": k =", round(k, 2),
                  "— suspiciously small, check data quality"))
  }
  return(TRUE)
}

# ==============================================================================
# METHOD 1: OLS (Ordinary Least Squares) — also known as MRR
# ==============================================================================
#' OLS / Median Rank Regression on Weibull probability paper
#'
#' What reliability literature calls "MRR" (Median Rank Regression) is
#' mathematically identical to OLS on the Weibull linearization:
#' same Benard formula, same lm() with no weights, same k and lambda.
#'
#' DERIVATION:
#'   Weibull CDF: F(t) = 1 - exp(-(t/λ)^k)
#'   Take logs:   log(1-F) = -(t/λ)^k
#'   Again:       log(-log(1-F)) = k*log(t) - k*log(λ)
#'   This is linear in log(t):
#'     Y = k*X + b,  where Y = log(-log(1-F)), X = log(t), b = -k*log(λ)
#'   Slope = k,  λ = exp(-intercept/k)
#'
#' MEDIAN RANKS (Benard's formula):
#'   F_i = (i - 0.3) / (n + 0.4)
#'   Why not i/n: last point gives F_n = 1, then ln(-ln(0)) = -Inf.
#'   Benard's formula ≈ median of Beta(i, n-i+1) — best single point
#'   estimate for where the i-th order statistic sits in percentile space.
#'
#' WHY OLS IS BIASED:
#'   i-th order statistic from N uniform draws ~ Beta(i, N-i+1)
#'   Var(F_i) = i*(N-i+1) / [(N+1)^2*(N+2)]
#'   Large at tails (i=1 or i=N), small in middle (i ≈ N/2).
#'   Y_i inherits unequal variance via delta method:
#'   Var(Y_i) ≈ [g'(F_i)]^2 * Var(F_i)
#'   where g'(u) = 1/[(1-u)*(-log(1-u))]
#'   OLS minimizes Σ(r_j^2) equally for all points.
#'   Tail points produce large residuals due to high noise.
#'   OLS tilts the line toward them → k_hat biased.
#'   Chebyshev: P(|r_j - μ_j| ≥ a) ≤ Var(r_j)/a^2.
#'   Large Var at tails → more likely far from true line.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method)
estimate_ols <- function(drive_data) {
  # Use only observed failures, sorted ascending
  t_fail <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n <- length(t_fail)
  
  if (n < 3) return(list(k = NA_real_, lambda = NA_real_, method = "OLS"))
  
  # Median ranks — Benard's formula: F_i = (i - 0.3) / (n + 0.4)
  i <- seq_len(n)
  F_i <- (i - 0.3) / (n + 0.4)
  
  # Weibull linearization
  Y <- log(-log(1 - F_i))   # Y_i = ln(-ln(1 - F_i))
  X <- log(t_fail)           # X_i = ln(t_i)
  
  # Fit OLS: Y = slope*X + intercept, NO weights
  fit <- lm(Y ~ X)
  k_hat   <- unname(coef(fit)[2])           # slope = k
  lam_hat <- exp(-unname(coef(fit)[1]) / k_hat)  # λ = exp(-intercept/k)
  
  list(k = k_hat, lambda = lam_hat, method = "OLS")
}

# ==============================================================================
# METHOD 2: WLS (Weighted Least Squares — Bergman weights)
# ==============================================================================
#' Weighted Least Squares with Bergman (1986) weights
#'
#' Same Y, X, F_i as OLS but with analytically derived weights.
#'
#' BERGMAN WEIGHT DERIVATION (delta method, 4 steps):
#'   Step 1: Var(F_i) = i*(N-i+1) / [(N+1)^2*(N+2)]   [Beta variance]
#'   Step 2: g'(F_i) = 1/[(1-F_i)*(-log(1-F_i))]      [chain rule on Y=g(F)]
#'   Step 3: Var(Y_i) ≈ [g'(F_i)]^2 * Var(F_i)        [delta method]
#'   Step 4: w_i = 1/Var(Y_i)                          [WLS weight]
#'   Combining:
#'     w_i = [(1-F_i)*(-log(1-F_i))]^2 * (N+1)^2*(N+2) / [i*(N-i+1)]
#'
#' WEIGHT PEAK at F ≈ 0.632:
#'   Maximize f(s) = -s*log(s), s = 1-F:
#'   f'(s) = -log(s) - 1 = 0 → s = e^{-1} → F ≈ 0.632 = t = λ
#'   Regression anchors at the natural Weibull characteristic life.
#'
#' TAIL BEHAVIOR:
#'   F_i → 1: (1-F_i) → 0, w_i → 0. Late failures: near-zero influence.
#'   F_i → 0: -log(1-F_i) ≈ F_i → 0, w_i → 0. Early failures: near-zero.
#'
#' IMPORTANT: WLS is NOT robust to genuinely anomalous observations.
#' It down-weights tail points due to predictable Beta variance.
#' A wrong data point at a middle rank receives HIGH WLS weight and
#' can drag the estimate. L-Moments handles true outliers;
#' WLS handles rank-position noise only.
#'
#' WLS CAN OUTRANK MLE: MLE is theoretically optimal under exact model
#' specification. But if the true distribution is only approximately Weibull
#' (nearly always true), MLE's aggressive commitment to the functional form
#' becomes a liability. WLS fits the empirical CDF more loosely and can
#' describe actual data better when the model is slightly misspecified.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method)
estimate_wls <- function(drive_data) {
  t_fail <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n <- length(t_fail)
  
  if (n < 3) return(list(k = NA_real_, lambda = NA_real_, method = "WLS"))
  
  i <- seq_len(n)
  F_i <- (i - 0.3) / (n + 0.4)
  
  Y <- log(-log(1 - F_i))
  X <- log(t_fail)
  
  # Bergman weights: w_i = [(1-F_i)*(-log(1-F_i))]^2 * (N+1)^2*(N+2) / [i*(N-i+1)]
  numerator   <- ((1 - F_i) * (-log(1 - F_i)))^2 * (n + 1)^2 * (n + 2)
  denominator <- i * (n - i + 1)
  wi <- numerator / denominator
  
  # Fit WLS: lm(Y ~ X, weights = wi)
  fit <- lm(Y ~ X, weights = wi)
  k_hat   <- unname(coef(fit)[2])
  lam_hat <- exp(-unname(coef(fit)[1]) / k_hat)
  
  list(k = k_hat, lambda = lam_hat, method = "WLS")
}

# ==============================================================================
# METHOD 3: MOM (Method of Moments)
# ==============================================================================
#' Method of Moments estimator
#'
#' DERIVATION:
#'   Weibull moments via substitution u = (t/λ)^k:
#'     E[T]   = λ * Γ(1 + 1/k)
#'     Var(T) = λ^2 * [Γ(1 + 2/k) - Γ(1 + 1/k)^2]
#'     CV^2   = Γ(1 + 2/k) / Γ(1 + 1/k)^2 - 1
#'
#'   UNIVERSAL PATTERN: λ is a pure scale parameter.
#'   Any dimensionless ratio of moments has λ cancel exactly.
#'   CV^2, τ2 (L-Moments), EPF all depend only on k.
#'   This is referenced by Methods 4 and 5.
#'
#'   Solve CV^2_theoretical(k) = CV^2_sample for k:
#'     h(k) = Γ(1+2/k)/Γ(1+1/k)^2 - 1 - CV^2_sample = 0
#'     Use uniroot (Brent's method) on interval [0.1, 50].
#'     One root guaranteed: CV^2(k) is strictly monotone decreasing in k.
#'   Then: λ_hat = μ / Γ(1 + 1/k_hat)
#'
#' CENSORING BIAS: Complete data only. Even 1% censoring biases the
#' sample mean downward — excluded drives are systematically longer-lived.
#' λ underestimated. Bias exists at ANY nonzero censoring fraction.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method)
estimate_mom <- function(drive_data) {
  t_fail <- drive_data$lifetime_days[drive_data$failed == 1]
  n <- length(t_fail)
  
  if (n < 3) return(list(k = NA_real_, lambda = NA_real_, method = "MOM"))
  
  # Sample moments
  mu    <- mean(t_fail)
  sigma2 <- var(t_fail)
  CV2   <- sigma2 / mu^2    # squared coefficient of variation
  
  # Solve: Γ(1+2/k)/Γ(1+1/k)^2 - 1 = CV^2
  # h(k) = 0 has exactly one root since CV^2(k) is monotone decreasing
  h <- function(k) gamma(1 + 2/k) / gamma(1 + 1/k)^2 - 1 - CV2
  
  k_hat <- tryCatch({
    uniroot(h, lower = 0.1, upper = 50, tol = 1e-10)$root
  }, error = function(e) NA_real_)
  
  if (is.na(k_hat)) return(list(k = NA_real_, lambda = NA_real_, method = "MOM"))
  
  # λ = μ / Γ(1 + 1/k)
  lam_hat <- mu / gamma(1 + 1/k_hat)
  
  list(k = k_hat, lambda = lam_hat, method = "MOM")
}

# ==============================================================================
# METHOD 4: L-MOMENTS
# ==============================================================================
#' L-Moment estimation for Weibull parameters
#'
#' L-statistics are linear combinations of order statistics:
#'   L1 = mean(t)
#'   L2 = (1/2) * E[T_(j) - T_(i)]  for j > i  (mean pairwise difference)
#'
#' L2 SINGLE-SUM FORMULA (derived from pairwise definition):
#'   Pairwise: λ2 = 1/(2*C(n,2)) * Σ_{j>i}(t_(j) - t_(i))
#'   Step 1: Separate double sum into two single sums
#'   Step 2: t_(k) appears (k-1) times as larger, (N-k) times as smaller
#'   Step 3: Rewrite as single sum with weights (2k - N - 1)
#'   Result: λ2_hat = Σ[(2i - n - 1) * t_sorted] / [n*(n-1)]
#'   Weights (2k-N-1): negative early ranks, positive late ranks,
#'   zero at middle. A linear contrast — no squaring anywhere.
#'
#' L-CV: τ2 = λ2 / L1
#'
#' CLOSED-FORM SOLUTION for k (no uniroot needed):
#'   τ2 = 1 - 2^(-1/k)
#'   2^(-1/k) = 1 - τ2
#'   -1/k * ln(2) = ln(1 - τ2)
#'   k = -ln(2) / ln(1 - τ2)
#'
#' Derivation of τ2 = 1 - 2^(-1/k) via integral:
#'   λ2 = λ * Γ(1+1/k) * (1 - 2^(-1/k))
#'   τ2 = λ2/L1  [λ and Γ cancel]
#'
#' ROBUSTNESS: More robust than MOM — linear combinations, no squaring.
#' One outlier at most doubles L2; same outlier quadruples variance.
#' Biased under censoring > ~5%.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method)
estimate_lmoments <- function(drive_data) {
  t_fail <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n <- length(t_fail)
  
  if (n < 3) return(list(k = NA_real_, lambda = NA_real_, method = "L-Moments"))
  
  # L1 = sample mean
  L1 <- mean(t_fail)
  
  # L2 via efficient single-sum formula
  i <- seq_len(n)
  lam2_single <- sum((2 * i - n - 1) * t_fail) / (n * (n - 1))
  
  # === DOUBLE VERIFICATION: pairwise formula ===
  # Pairwise (definitional — slow but exact reference):
  if (n <= 5000) {  # Only for reasonable n to avoid O(n^2)
    pairs <- combn(t_fail, 2)
    lam2_pairwise <- mean(pairs[2, ] - pairs[1, ]) / 2
    
    if (abs(lam2_pairwise - lam2_single) > 1e-6) {
      warning("L-Moments: pairwise and single-sum formulas disagree!")
    }
  }
  
  # L-CV
  tau2 <- lam2_single / L1
  
  if (tau2 <= 0 || tau2 >= 1) {
    return(list(k = NA_real_, lambda = NA_real_, method = "L-Moments"))
  }
  
  # Closed-form k: k = -ln(2) / ln(1 - τ2)
  k_closed <- -log(2) / log(1 - tau2)
  
  # === VERIFICATION: closed-form vs uniroot ===
  k_uniroot <- tryCatch({
    uniroot(function(k) 1 - 2^(-1/k) - tau2,
            lower = 0.1, upper = 50, tol = 1e-12)$root
  }, error = function(e) NA_real_)
  
  if (!is.na(k_uniroot) && abs(k_closed - k_uniroot) > 1e-6) {
    warning(paste("L-Moments: closed-form k =", round(k_closed, 6),
                  "vs uniroot k =", round(k_uniroot, 6)))
  }
  
  k_hat <- k_closed
  
  # λ = L1 / Γ(1 + 1/k)
  lam_hat <- L1 / gamma(1 + 1/k_hat)
  
  list(k = k_hat, lambda = lam_hat, method = "L-Moments")
}

# ==============================================================================
# METHOD 5: EPF (Energy Pattern Factor)
# ==============================================================================
#' Energy Pattern Factor method
#'
#' PHYSICAL ORIGIN: P = (1/2)*ρ*A*v^3. Mean power requires E[v^3],
#' not (E[v])^3. EPF = E[v^3]/(E[v])^3 >= 1 always (Jensen's inequality).
#' EPF = 1: zero variability. Larger EPF: heavier right tail.
#'
#' THEORETICAL EPF:
#'   E[T^n] = λ^n * Γ(1 + n/k)
#'   EPF(k) = Γ(1 + 3/k) / Γ(1 + 1/k)^3   [λ^3 cancels]
#'
#' Solve EPF(k) = EPF_sample:
#'   h(k) = Γ(1+3/k)/Γ(1+1/k)^3 - EPF_hat = 0
#'   Bracket [0.5, 20] as specified in EPF source document.
#'   Monotone: EPF decreases as k increases → one root guaranteed.
#'
#' EMPIRICAL APPROXIMATION (quick check only):
#'   k_approx = 1 + 3.69 / EPF_hat^2
#'   Purely empirical, not derived. Valid only for k ∈ [1, 3].
#'   Always verify with exact root-finding.
#'
#' OUTLIER SENSITIVITY:
#'   Outlier at v=10, mean=5:
#'     Squared deviation: (10-5)^2 = 25
#'     Cube: 10^3 = 1000. Mean cube: 5^3 = 125. Ratio: 1000/125 = 8.
#'   One observation at 2x the mean contributes 8x the mean cube.
#'   Even one anomalous observation can destroy the EPF estimate.
#'
#' No censoring support at any censoring fraction > 0%.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method, k_EPF_exact, k_EPF_approx, k_EPF_approx_pct_error)
estimate_epf <- function(drive_data) {
  t_fail <- drive_data$lifetime_days[drive_data$failed == 1]
  n <- length(t_fail)
  
  if (n < 3) return(list(k = NA_real_, lambda = NA_real_, method = "EPF",
                          k_EPF_exact = NA, k_EPF_approx = NA,
                          k_EPF_approx_pct_error = NA))
  
  # Sample EPF = mean(t^3) / mean(t)^3
  EPF_hat <- mean(t_fail^3) / mean(t_fail)^3
  
  # Exact solution via root-finding
  # h(k) = Γ(1+3/k)/Γ(1+1/k)^3 - EPF_hat
  h <- function(k) gamma(1 + 3/k) / gamma(1 + 1/k)^3 - EPF_hat
  
  k_exact <- tryCatch({
    uniroot(h, lower = 0.5, upper = 20, tol = 1e-10)$root
  }, error = function(e) NA_real_)
  
  if (is.na(k_exact)) {
    return(list(k = NA_real_, lambda = NA_real_, method = "EPF",
                k_EPF_exact = NA, k_EPF_approx = NA,
                k_EPF_approx_pct_error = NA))
  }
  
  # Empirical approximation (quick check only, valid for k ∈ [1,3])
  k_approx <- 1 + 3.69 / EPF_hat^2
  k_approx_pct_error <- abs(k_approx - k_exact) / k_exact * 100
  
  k_hat <- k_exact
  lam_hat <- mean(t_fail) / gamma(1 + 1/k_hat)
  
  list(k = k_hat, lambda = lam_hat, method = "EPF",
       k_EPF_exact = k_exact, k_EPF_approx = k_approx,
       k_EPF_approx_pct_error = k_approx_pct_error)
}

# ==============================================================================
# METHOD 6: MLE (Maximum Likelihood Estimation)
# ==============================================================================
#' Maximum Likelihood Estimation with censored data
#'
#' Uses ALL observations (failures + censored).
#'
#' CENSORED LOG-LIKELIHOOD:
#'   loglik(k,λ) =
#'     Σ_{failed=1} [log(k) - log(λ) + (k-1)*log(t/λ) - (t/λ)^k]
#'   + Σ_{failed=0} [-(t/λ)^k]
#'
#'   Censored term = log(S(t)) = -(t/λ)^k
#'   Small λ: (tc/λ)^k large → model says unit should have failed
#'   → contradicts data → optimizer forces λ upward.
#'   OLS and WLS never see censored units → λ systematically
#'   underestimated under heavy censoring.
#'
#' OPTIMIZATION: Nelder-Mead (optim() default in R).
#'   Works without derivatives. Simplex in 2D = triangle of 3 points.
#'   Each step: identify best B, worst W, middle M.
#'   Centroid of non-worst: x_bar = (B+M)/2.
#'   Try reflection: R = x_bar + (x_bar - W).
#'   If R better than B: expand E = x_bar + 2*(x_bar - W).
#'   If R worse than M: contract C = x_bar + 0.5*(x_bar - W).
#'   If contraction fails: shrink all toward B.
#'   Weibull loglik is unimodal → converges to global max reliably.
#'
#' OPTIMIZE ON LOG SCALE: u = log(k), v = log(λ).
#'   Ensures k > 0, λ > 0 automatically. Better numerical behavior.
#'
#' Gold standard for censored data. Asymptotically efficient.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method, converged)
estimate_mle <- function(drive_data) {
  t <- drive_data$lifetime_days
  event <- drive_data$failed
  n <- length(t)
  
  if (sum(event) < 3) {
    return(list(k = NA_real_, lambda = NA_real_, method = "MLE", converged = NA))
  }
  
  # Negative log-likelihood on log scale: params = c(log(k), log(λ))
  neg_loglik <- function(log_params) {
    k   <- exp(log_params[1])
    lam <- exp(log_params[2])
    
    # Guard against numerical issues
    if (k < 1e-10 || lam < 1e-10) return(1e12)
    
    t_ratio <- t / lam
    
    # Log-PDF for failures: log(k) - log(λ) + (k-1)*log(t/λ) - (t/λ)^k
    ll_fail <- sum(event * (log(k) - log(lam) + (k - 1) * log(t_ratio) - t_ratio^k))
    
    # Log-survival for censored: -(t/λ)^k
    ll_cens <- sum((1 - event) * (-t_ratio^k))
    
    return(-(ll_fail + ll_cens))
  }
  
  # Initial values: use method of moments on failures as starting point
  t_fail <- t[event == 1]
  mean_t <- mean(t_fail)
  
  result <- optim(
    par     = c(log(1.5), log(mean_t)),
    fn      = neg_loglik,
    method  = "Nelder-Mead",
    control = list(maxit = 10000, abstol = 1e-10, reltol = 1e-10)
  )
  
  k_hat   <- exp(result$par[1])
  lam_hat <- exp(result$par[2])
  converged <- (result$convergence == 0)
  
  if (!converged) {
    warning(paste("MLE did not converge — code:", result$convergence))
  }
  
  list(k = k_hat, lambda = lam_hat, method = "MLE", converged = converged)
}

# ==============================================================================
# METHOD 7A: BAYESIAN MAP (Maximum A Posteriori)
# ==============================================================================
#' Bayesian MAP with exponential priors
#'
#' Same censored log-likelihood as MLE, plus log-prior terms.
#'
#' PRIORS:
#'   π(k) ∝ exp(-k/μ_k)      [μ_k = 2, prior mean for shape]
#'   π(λ) ∝ exp(-λ/mean(t_f)) [prior mean = mean of failure times]
#'   log_prior_k   = -k/μ_k
#'   log_prior_lam = -λ/mean(t_failures)
#'
#' MAP OBJECTIVE: maximize log posterior:
#'   log_posterior = loglik + log_prior_k + log_prior_lam
#'   Minimize negative with optim() Nelder-Mead, log scale.
#'
#' CRITICAL — FLAT PRIOR = MLE, shown algebraically:
#'   If π(k) ∝ 1 and π(λ) ∝ 1:
#'   log posterior = loglik + log(1) + log(1) = loglik
#'   MAP with flat prior = MLE exactly. Zero point doing both.
#'   Exponential prior adds penalty -k/μ_k for large k.
#'   This is regularization — prevents optimizer landing on
#'   implausible k values (e.g., k = 4.7 from 8 noisy failures).
#'
#' When MAP differs from MLE: n_failures < ~20.
#' When MAP → MLE: large N. Loglik grows with N, prior fixed.
#' Returns single point (k_hat, λ_hat). No uncertainty.
#' Fundamental limitation: false precision with small samples.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method, converged)
estimate_map <- function(drive_data) {
  t <- drive_data$lifetime_days
  event <- drive_data$failed
  
  if (sum(event) < 3) {
    return(list(k = NA_real_, lambda = NA_real_, method = "MAP", converged = NA))
  }
  
  t_fail <- t[event == 1]
  mean_t_fail <- mean(t_fail)
  
  # Prior hyperparameters
  MU_K   <- 2              # Exponential prior mean for k
  MU_LAM <- mean_t_fail    # Exponential prior mean for λ
  
  # Negative log-posterior on log scale
  neg_log_posterior <- function(log_params) {
    k   <- exp(log_params[1])
    lam <- exp(log_params[2])
    
    if (k < 1e-10 || lam < 1e-10) return(1e12)
    
    t_ratio <- t / lam
    
    # Log-likelihood (same as MLE)
    ll_fail <- sum(event * (log(k) - log(lam) + (k - 1) * log(t_ratio) - t_ratio^k))
    ll_cens <- sum((1 - event) * (-t_ratio^k))
    loglik  <- ll_fail + ll_cens
    
    # Log-priors: exponential
    log_prior_k   <- -k / MU_K
    log_prior_lam <- -lam / MU_LAM
    
    # Negative log-posterior (minimize)
    return(-(loglik + log_prior_k + log_prior_lam))
  }
  
  result <- optim(
    par     = c(log(1.5), log(mean_t_fail)),
    fn      = neg_log_posterior,
    method  = "Nelder-Mead",
    control = list(maxit = 10000, abstol = 1e-10, reltol = 1e-10)
  )
  
  k_hat   <- exp(result$par[1])
  lam_hat <- exp(result$par[2])
  converged <- (result$convergence == 0)
  
  if (!converged) {
    warning(paste("MAP did not converge — code:", result$convergence))
  }
  
  list(k = k_hat, lambda = lam_hat, method = "MAP", converged = converged)
}

# ==============================================================================
# METHOD 7B: FULL BAYESIAN MCMC (brms + Stan)
# ==============================================================================
#' Full Bayesian MCMC via brms/Stan
#'
#' NOTE: This method is an EXTENSION beyond the core course documents.
#' It provides full posterior distributions and credible intervals
#' for honest uncertainty quantification.
#'
#' Same priors as MAP (7A): exponential on k and λ.
#'
#' brms: shape = k. Intercept = log(λ).
#' cens(1-failed) converts our convention (failed=1 observed)
#' to brms convention (cens=1 censored).
#'
#' MAP vs MCMC distinction:
#'   MAP:  one point. Hides uncertainty. Fast.
#'   MCMC: full posterior distribution. Credible intervals. Honest.
#'   With 8 failures: MAP gives k=2.1 (false precision).
#'   MCMC gives k=2.1 [95% CI: 0.9, 4.1] (honest truth).
#'   Wide CI is not a weakness — it is what the data tells you.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @param model_name Name of the drive model (for saving .rds)
#' @return list(k, lambda, method, k_lower_95, k_upper_95,
#'              lam_lower_95, lam_upper_95, fit)
estimate_mcmc <- function(drive_data, model_name) {
  if (sum(drive_data$failed) < 3) {
    return(list(k = NA_real_, lambda = NA_real_, method = "MCMC",
                k_lower_95 = NA, k_upper_95 = NA,
                lam_lower_95 = NA, lam_upper_95 = NA, fit = NULL))
  }
  
  # Clean model name for file paths
  model_clean <- gsub("[^A-Za-z0-9]", "_", model_name)
  rds_path <- here::here("R", "results", paste0("brms_fit_", model_clean, ".rds"))
  
  # Load existing fit if available (MCMC is expensive)
  if (file.exists(rds_path)) {
    cat("    Loading existing MCMC fit:", basename(rds_path), "\n")
    fit <- readRDS(rds_path)
  } else {
    cat("    Running MCMC (this may take 5-30 minutes)...\n")
    
    mean_t <- mean(drive_data$lifetime_days[drive_data$failed == 1])
    
    # brms uses cens() where cens=1 means censored, cens=0 means observed.
    # Our data: failed=1 means observed, failed=0 means censored.
    # So we pass censored = 1 - failed.
    drive_data$censored <- 1 - drive_data$failed
    
    fit <- brm(
      lifetime_days | cens(censored) ~ 1,
      data    = drive_data,
      family  = weibull(),
      prior   = c(
        prior(exponential(0.5), class = shape),            # mean = 2
        prior(normal(0, 10), class = Intercept)            # weakly informative on log(λ)
      ),
      chains  = 4,
      iter    = 2000,
      warmup  = 1000,
      cores   = 4,
      seed    = 42,
      silent  = 2
    )
    
    # Save fit
    saveRDS(fit, rds_path)
    cat("    Saved:", basename(rds_path), "\n")
  }
  
  # Check convergence
  rhat_vals <- rhat(fit)
  if (any(rhat_vals > 1.01, na.rm = TRUE)) {
    warning(paste("MCMC NOT CONVERGED for", model_name,
                  "— max Rhat:", round(max(rhat_vals, na.rm = TRUE), 3)))
  }
  
  # Extract posterior summaries
  ps <- posterior_summary(fit)
  
  # Shape parameter k
  k_hat     <- ps["shape", "Estimate"]
  k_lower   <- ps["shape", "Q2.5"]
  k_upper   <- ps["shape", "Q97.5"]
  
  # Scale parameter λ = exp(Intercept)
  # In brms weibull family, Intercept is log(mu) where mu relates to λ
  # We need to extract λ from the posterior draws properly
  draws <- as_draws_df(fit)
  
  # brms Weibull: mu = exp(Intercept), shape = k
  # The Weibull in brms parameterizes as: mu = λ * Γ(1 + 1/k)
  # So λ = mu / Γ(1 + 1/k) = exp(Intercept) / Γ(1 + 1/shape)
  mu_draws  <- exp(draws$b_Intercept)
  k_draws   <- draws$shape
  lam_draws <- mu_draws / gamma(1 + 1/k_draws)
  
  lam_hat   <- median(lam_draws)
  lam_lower <- quantile(lam_draws, 0.025)
  lam_upper <- quantile(lam_draws, 0.975)
  
  list(k = k_hat, lambda = lam_hat, method = "MCMC",
       k_lower_95 = k_lower, k_upper_95 = k_upper,
       lam_lower_95 = unname(lam_lower), lam_upper_95 = unname(lam_upper),
       fit = fit)
}

# ==============================================================================
# METHOD 8: PERCENTILE / QUANTILE
# ==============================================================================
#' Two-quantile percentile method
#'
#' p1 = 0.16, p2 = 0.84 (pnorm(-1) and pnorm(1) — borrowed from
#' normal distribution, no Weibull justification, arbitrary).
#'
#' ANALYTICAL SOLUTION (subtract linearized Weibull equations):
#'   From F(t) = 1 - exp(-(t/λ)^k):
#'     log(-log(1-p)) = k*log(t) - k*log(λ)
#'   At two points (p1,t1) and (p2,t2):
#'     k = [log(-log(1-p2)) - log(-log(1-p1))] / [log(t2) - log(t1)]
#'     λ = t1 / (-log(1-p1))^(1/k)
#'
#' Uses only 2 of n points. Remaining n-2 contribute nothing.
#' Pedagogical baseline only. Expected to perform worst.
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return list(k, lambda, method)
estimate_percentile <- function(drive_data) {
  t_fail <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n <- length(t_fail)
  
  if (n < 5) return(list(k = NA_real_, lambda = NA_real_, method = "Percentile"))
  
  # Two quantile points
  P1 <- 0.16    # pnorm(-1) ≈ 0.1587
  P2 <- 0.84    # pnorm(1)  ≈ 0.8413
  
  t1 <- quantile(t_fail, P1, names = FALSE)
  t2 <- quantile(t_fail, P2, names = FALSE)
  
  if (t1 <= 0 || t2 <= t1) {
    return(list(k = NA_real_, lambda = NA_real_, method = "Percentile"))
  }
  
  # Analytical solution
  k_hat   <- (log(-log(1 - P2)) - log(-log(1 - P1))) / (log(t2) - log(t1))
  lam_hat <- t1 / (-log(1 - P1))^(1 / k_hat)
  
  list(k = k_hat, lambda = lam_hat, method = "Percentile")
}

# ==============================================================================
# SENSITIVITY ANALYSIS: EPF outlier contamination
# ==============================================================================
#' Run sensitivity analysis comparing EPF, MOM, L-Moments, MLE
#' on clean vs contaminated data (largest failure time × 10)
#'
#' @param drive_data Data frame with columns lifetime_days, failed
#' @return Data frame with sensitivity results
sensitivity_epf_outlier <- function(drive_data) {
  t_fail <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n <- length(t_fail)
  if (n < 5) return(NULL)
  
  # Create contaminated version: replace largest failure with 10x its value
  t_contaminated <- t_fail
  t_contaminated[n] <- t_fail[n] * 10
  
  # Helper: make a mini data frame for methods
  make_df <- function(t_vals) {
    tibble(lifetime_days = t_vals, failed = rep(1, length(t_vals)))
  }
  
  clean_df <- make_df(t_fail)
  contam_df <- make_df(t_contaminated)
  
  methods_to_test <- list(
    EPF = estimate_epf,
    MOM = estimate_mom,
    `L-Moments` = estimate_lmoments,
    MLE = estimate_mle
  )
  
  results <- map_dfr(names(methods_to_test), function(mname) {
    fn <- methods_to_test[[mname]]
    res_clean  <- fn(clean_df)
    res_contam <- fn(contam_df)
    
    k_clean <- res_clean$k
    k_contam <- res_contam$k
    pct_change <- if (!is.na(k_clean) && !is.na(k_contam) && k_clean != 0) {
      abs(k_contam - k_clean) / k_clean * 100
    } else NA_real_
    
    tibble(
      method = mname,
      k_clean = k_clean,
      k_contaminated = k_contam,
      pct_change = pct_change
    )
  })
  
  results
}

# ==============================================================================
# MAIN ESTIMATION LOOP — All 9 methods × 3 models
# ==============================================================================

all_results <- tibble()
sensitivity_results <- tibble()
mcmc_fits <- list()

for (mdl in models) {
  cat("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  cat("  Model:", mdl, "\n")
  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  
  drive_data <- drives %>% filter(model == mdl)
  n_total    <- nrow(drive_data)
  n_failures <- sum(drive_data$failed)
  cens_pct   <- round((1 - mean(drive_data$failed)) * 100, 2)
  
  cat("  N total:", n_total, "| Failures:", n_failures,
      "| Censoring:", cens_pct, "%\n\n")
  
  # --- Run all 9 methods ---
  methods_list <- list(
    estimate_ols(drive_data),
    estimate_wls(drive_data),
    estimate_mom(drive_data),
    estimate_lmoments(drive_data),
    estimate_epf(drive_data),
    estimate_mle(drive_data),
    estimate_map(drive_data),
    tryCatch(estimate_mcmc(drive_data, mdl),
             error = function(e) {
               warning(paste("MCMC failed for", mdl, ":", e$message))
               list(k = NA_real_, lambda = NA_real_, method = "MCMC",
                    k_lower_95 = NA, k_upper_95 = NA,
                    lam_lower_95 = NA, lam_upper_95 = NA, fit = NULL)
             }),
    estimate_percentile(drive_data)
  )
  
  for (res in methods_list) {
    k   <- res$k
    lam <- res$lambda
    mth <- res$method
    
    # Sanity check
    valid <- check_estimates(k, lam, paste(mdl, "-", mth))
    
    if (!valid) {
      k <- NA_real_
      lam <- NA_real_
    }
    
    cat(sprintf("  %-12s → k = %-8s  λ = %-10s  %s\n",
                mth,
                ifelse(is.na(k), "NA", sprintf("%.4f", k)),
                ifelse(is.na(lam), "NA", sprintf("%.2f", lam)),
                ifelse(valid, "✓", "✗ (failed)")))
    
    # Build result row
    row <- tibble(
      model          = mdl,
      method         = mth,
      k              = k,
      lambda         = lam,
      k_lower_95     = ifelse(mth == "MCMC", res$k_lower_95 %||% NA, NA_real_),
      k_upper_95     = ifelse(mth == "MCMC", res$k_upper_95 %||% NA, NA_real_),
      lam_lower_95   = ifelse(mth == "MCMC", res$lam_lower_95 %||% NA, NA_real_),
      lam_upper_95   = ifelse(mth == "MCMC", res$lam_upper_95 %||% NA, NA_real_),
      converged      = if (mth %in% c("MLE", "MAP")) res$converged else NA,
      k_EPF_exact    = if (mth == "EPF") res$k_EPF_exact %||% NA else NA_real_,
      k_EPF_approx   = if (mth == "EPF") res$k_EPF_approx %||% NA else NA_real_,
      k_EPF_approx_pct_error = if (mth == "EPF") res$k_EPF_approx_pct_error %||% NA else NA_real_
    )
    
    all_results <- bind_rows(all_results, row)
    
    # Store MCMC fit for diagnostics
    if (mth == "MCMC" && !is.null(res$fit)) {
      mcmc_fits[[mdl]] <- res$fit
    }
  }
  
  # --- Sensitivity analysis ---
  cat("\n  Running sensitivity analysis (EPF outlier contamination)...\n")
  sens <- sensitivity_epf_outlier(drive_data)
  if (!is.null(sens)) {
    sens$model <- mdl
    sensitivity_results <- bind_rows(sensitivity_results, sens)
    cat("  Sensitivity results:\n")
    for (j in seq_len(nrow(sens))) {
      cat(sprintf("    %-12s | k_clean=%.4f | k_contam=%.4f | change=%.1f%%\n",
                  sens$method[j], sens$k_clean[j], sens$k_contaminated[j],
                  sens$pct_change[j]))
    }
  }
}

# ==============================================================================
# Save Results
# ==============================================================================

cat("\n==================================================================\n")
cat(" Saving results...\n")
cat("==================================================================\n\n")

# Parameters
params_path <- here::here("R", "results", "parameters.csv")
write_csv(all_results, params_path)
cat("  Saved:", params_path, "\n")
cat("  Rows:", nrow(all_results), "(9 methods × ", length(models), "models)\n")

# Sensitivity analysis
sens_path <- here::here("R", "results", "sensitivity_analysis.csv")
write_csv(sensitivity_results, sens_path)
cat("  Saved:", sens_path, "\n")

cat("\n[DONE] 02_estimation_methods.R complete.\n")
