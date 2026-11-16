# ==============================================================================
# 04_plots.R — All 8 Plot Types × 3 Models (24 PNG files)
# ==============================================================================
# Input  : data/drives_clean.csv, R/results/parameters.csv,
#          R/results/gof_metrics.csv
# Output : R/results/plots/*.png (8 plots per model × 3 models = 24 files)
# ==============================================================================

# ── Reproducibility ──────────────────────────────────────────────────────────
set.seed(42)
options(scipen = 999)
options(digits = 6)
Sys.setenv(LANG = "en_US")

# ── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)
library(here)
library(scales)

cat("==================================================================\n")
cat(" Generating Plots — 8 Types × 3 Models\n")
cat("==================================================================\n\n")

# ── Load data ────────────────────────────────────────────────────────────────
drives  <- read_csv(here::here("data", "drives_clean.csv"), show_col_types = FALSE)
params  <- read_csv(here::here("R", "results", "parameters.csv"), show_col_types = FALSE)
gof     <- read_csv(here::here("R", "results", "gof_metrics.csv"), show_col_types = FALSE)

models  <- unique(drives$model)
plot_dir <- here::here("R", "results", "plots")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# CONSISTENT COLOR PALETTE — 9 methods, colorblind-friendly
# ==============================================================================
# These colors are IDENTICAL across all 8 plots and all 3 models.
# 7A (MAP) and 7B (MCMC) always have distinct colors.
METHOD_COLORS <- c(
  "OLS"        = "#E41A1C",
  "WLS"        = "#377EB8",
  "MOM"        = "#4DAF4A",
  "L-Moments"  = "#984EA3",
  "EPF"        = "#FF7F00",
  "MLE"        = "#A65628",
  "MAP"        = "#F781BF",
  "MCMC"       = "#999999",
  "Percentile" = "#DEDE00"
)

# Method order for consistent legend ordering
METHOD_ORDER <- c("OLS", "WLS", "MOM", "L-Moments", "EPF",
                  "MLE", "MAP", "MCMC", "Percentile")

# ==============================================================================
# WEIBULL DISTRIBUTION FUNCTIONS
# ==============================================================================
weibull_cdf    <- function(t, k, lam) 1 - exp(-(t / lam)^k)
weibull_sf     <- function(t, k, lam) exp(-(t / lam)^k)
weibull_pdf    <- function(t, k, lam) (k / lam) * (t / lam)^(k - 1) * exp(-(t / lam)^k)
weibull_hazard <- function(t, k, lam) (k / lam) * (t / lam)^(k - 1)

# ==============================================================================
# HELPER: Clean model name for file paths
# ==============================================================================
clean_model_name <- function(name) {
  gsub("[^A-Za-z0-9]", "_", name)
}

# ==============================================================================
# COMMON THEME
# ==============================================================================
theme_weibull <- function() {
  theme_bw() +
    theme(
      plot.title    = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "grey40"),
      axis.title    = element_text(size = 12),
      axis.text     = element_text(size = 10),
      legend.text   = element_text(size = 9),
      legend.position = "right"
    )
}

# ==============================================================================
# PLOT GENERATION LOOP — 8 plots per model
# ==============================================================================

for (mdl in models) {
  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  cat("  Model:", mdl, "\n")
  cat("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
  
  drive_data <- drives %>% filter(model == mdl)
  model_params <- params %>% filter(model == mdl, !is.na(k), !is.na(lambda))
  model_gof    <- gof %>% filter(model == mdl)
  
  n_total    <- nrow(drive_data)
  n_failures <- sum(drive_data$failed)
  cens_pct   <- round((1 - mean(drive_data$failed)) * 100, 1)
  model_clean <- clean_model_name(mdl)
  
  # Failure times for empirical distributions
  t_failures <- sort(drive_data$lifetime_days[drive_data$failed == 1])
  n_f <- length(t_failures)
  
  # Median ranks (Benard's formula)
  i_seq <- seq_len(n_f)
  F_emp <- (i_seq - 0.3) / (n_f + 0.4)
  
  # Time grid for fitted curves
  t_max <- max(t_failures) * 1.15
  t_grid <- seq(1, t_max, length.out = 500)
  
  subtitle_text <- paste0(mdl, " — N=", format(n_total, big.mark = ","),
                          ", Failures=", n_failures,
                          ", Censoring=", cens_pct, "%")
  
  # Ensure method factor ordering
  model_params <- model_params %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 1 — Weibull Probability Plot
  # ═══════════════════════════════════════════════════════════════════════════
  # X: log(t), Y: log(-log(1-F))
  # Straight line = Weibull fits. Slope = k.
  # Diverging slopes = methods disagree on shape.
  # Curvature = Weibull may not be appropriate.
  
  emp_df <- tibble(x = log(t_failures), y = log(-log(1 - F_emp)))
  
  line_data <- model_params %>%
    rowwise() %>%
    do({
      t_line <- seq(min(t_failures), max(t_failures), length.out = 300)
      F_line <- weibull_cdf(t_line, .$k, .$lambda)
      F_line <- pmin(pmax(F_line, 1e-10), 1 - 1e-10)
      tibble(
        method = .$method,
        x = log(t_line),
        y = log(-log(1 - F_line)),
        k_val = .$k,
        lam_val = .$lambda
      )
    }) %>%
    ungroup() %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  # Create legend labels with k and λ values
  legend_labels <- model_params %>%
    mutate(label = paste0(method, " (k=", round(k, 3), ", λ=", round(lambda, 1), ")")) %>%
    select(method, label) %>%
    deframe()
  
  p1 <- ggplot() +
    geom_point(data = emp_df, aes(x = x, y = y),
               color = "black", shape = 16, size = 1.5, alpha = 0.6) +
    geom_line(data = line_data, aes(x = x, y = y, color = method),
              linewidth = 0.8) +
    scale_color_manual(values = METHOD_COLORS, labels = legend_labels,
                       drop = FALSE) +
    labs(title = "Weibull Probability Plot",
         subtitle = subtitle_text,
         x = "ln(t) [t in days]",
         y = "ln(-ln(1-F(t)))",
         color = "Method") +
    theme_weibull()
  
  ggsave(file.path(plot_dir, paste0("plot1_weibull_probability_", model_clean, ".png")),
         p1, width = 10, height = 7, dpi = 300)
  cat("  [OK] plot1_weibull_probability\n")
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 2 — CDF Comparison
  # ═══════════════════════════════════════════════════════════════════════════
  
  emp_cdf_df <- tibble(t = t_failures, F_emp = F_emp)
  
  fitted_cdf <- model_params %>%
    rowwise() %>%
    do({
      tibble(
        method = .$method,
        t = t_grid,
        F_fit = weibull_cdf(t_grid, .$k, .$lambda)
      )
    }) %>%
    ungroup() %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  p2 <- ggplot() +
    geom_step(data = emp_cdf_df, aes(x = t, y = F_emp),
              color = "black", linewidth = 1.2, linetype = "solid") +
    geom_line(data = fitted_cdf, aes(x = t, y = F_fit, color = method),
              linewidth = 0.8) +
    scale_color_manual(values = METHOD_COLORS, drop = FALSE) +
    labs(title = "CDF Comparison — Empirical vs Fitted",
         subtitle = subtitle_text,
         x = "Time (days)", y = "F(t) — Cumulative Failure Probability",
         color = "Method") +
    ylim(0, 1) +
    theme_weibull()
  
  ggsave(file.path(plot_dir, paste0("plot2_cdf_", model_clean, ".png")),
         p2, width = 10, height = 7, dpi = 300)
  cat("  [OK] plot2_cdf\n")
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 3 — Survival Function
  # ═══════════════════════════════════════════════════════════════════════════
  
  # Compute Kaplan-Meier estimate for robust non-parametric survival evaluation
  km_fit <- survival::survfit(survival::Surv(lifetime_days, failed) ~ 1, data = drive_data)
  emp_surv_df <- tibble(t = km_fit$time, S_emp = km_fit$surv)
  # Ensure the curve starts at t=0, S=1
  emp_surv_df <- bind_rows(tibble(t = 0, S_emp = 1), emp_surv_df)
  
  fitted_surv <- model_params %>%
    rowwise() %>%
    do({
      tibble(
        method = .$method,
        t = c(0, t_grid), # Start prediction at t=0
        S_fit = weibull_sf(c(0, t_grid), .$k, .$lambda)
      )
    }) %>%
    ungroup() %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  p3 <- ggplot() +
    geom_step(data = emp_surv_df, aes(x = t, y = S_emp, linetype = "Kaplan-Meier"),
              color = "black", linewidth = 1.2) +
    geom_line(data = fitted_surv, aes(x = t, y = S_fit, color = method),
              linewidth = 0.8) +
    scale_color_manual(values = METHOD_COLORS, drop = FALSE) +
    scale_linetype_manual(values = c("Kaplan-Meier" = "solid"), name = "Benchmark") +
    labs(title = "Survival (Reliability) Function (with Kaplan-Meier)",
         subtitle = subtitle_text,
         x = "Time (days)", y = "S(t) = Probability of Survival",
         color = "Fitted Method") +
    ylim(0, 1) +
    theme_weibull()
  
  ggsave(file.path(plot_dir, paste0("plot3_survival_", model_clean, ".png")),
         p3, width = 10, height = 7, dpi = 300)
  cat("  [OK] plot3_survival\n")
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 4 — PDF Comparison
  # ═══════════════════════════════════════════════════════════════════════════
  
  fitted_pdf <- model_params %>%
    rowwise() %>%
    do({
      tibble(
        method = .$method,
        t = t_grid,
        f_fit = weibull_pdf(t_grid, .$k, .$lambda)
      )
    }) %>%
    ungroup() %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  p4 <- ggplot() +
    geom_histogram(data = tibble(t = t_failures),
                   aes(x = t, y = after_stat(density)),
                   bins = 40, fill = "grey70", alpha = 0.3,
                   color = "grey50") +
    geom_line(data = fitted_pdf, aes(x = t, y = f_fit, color = method),
              linewidth = 0.8) +
    scale_color_manual(values = METHOD_COLORS, drop = FALSE) +
    labs(title = "PDF Comparison — Histogram vs Fitted Densities",
         subtitle = subtitle_text,
         x = "Time (days)", y = "f(t) — Probability Density",
         color = "Method") +
    theme_weibull()
  
  ggsave(file.path(plot_dir, paste0("plot4_pdf_", model_clean, ".png")),
         p4, width = 10, height = 7, dpi = 300)
  cat("  [OK] plot4_pdf\n")
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 5 — Hazard Function
  # ═══════════════════════════════════════════════════════════════════════════
  # h(t) = (k/λ)*(t/λ)^(k-1)
  # k < 1: decreasing hazard — infant mortality
  # k = 1: constant hazard — random failures (memoryless)
  # k > 1: increasing hazard — wear-out
  # k ≈ 3.5: approximates normal distribution shape
  
  fitted_haz <- model_params %>%
    rowwise() %>%
    do({
      tibble(
        method = .$method,
        t = t_grid,
        h_fit = weibull_hazard(t_grid, .$k, .$lambda)
      )
    }) %>%
    ungroup() %>%
    mutate(method = factor(method, levels = METHOD_ORDER))
  
  # Determine k range for annotation
  k_range <- range(model_params$k)
  
  p5 <- ggplot(fitted_haz, aes(x = t, y = h_fit, color = method)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = METHOD_COLORS, drop = FALSE) +
    labs(title = "Hazard (Failure Rate) Function",
         subtitle = subtitle_text,
         x = "Time (days)", y = "h(t) — Instantaneous Failure Rate",
         color = "Method") +
    annotate("text", x = t_max * 0.02, y = max(fitted_haz$h_fit, na.rm = TRUE) * 0.95,
             label = "k < 1: infant mortality\nk = 1: random (memoryless)\nk > 1: wear-out\nk ≈ 3.5: ≈ normal",
             hjust = 0, vjust = 1, size = 3, color = "grey40") +
    theme_weibull()
  
  ggsave(file.path(plot_dir, paste0("plot5_hazard_", model_clean, ".png")),
         p5, width = 10, height = 7, dpi = 300)
  cat("  [OK] plot5_hazard\n")
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 6 — GoF Heatmap
  # ═══════════════════════════════════════════════════════════════════════════
  # Rows: 9 methods (7A and 7B separate)
  # Columns: KS, RMSE, MAE, AD, AIC, BIC, MRL_err
  # Fill: column-wise min-max normalization [0,1]
  # Green = best (low), Red = worst (high)
  
  heatmap_data <- model_gof %>%
    filter(!is.na(KS)) %>%
    select(method, KS, RMSE, MAE, AD, AIC, BIC, MRL_err)
  
  if (nrow(heatmap_data) > 0) {
    # Column-wise min-max normalization
    metrics_cols <- c("KS", "RMSE", "MAE", "AD", "AIC", "BIC", "MRL_err")
    
    heatmap_norm <- heatmap_data %>%
      mutate(across(all_of(metrics_cols), ~ {
        rng <- range(., na.rm = TRUE)
        if (rng[2] - rng[1] < 1e-12) 0.5 else (. - rng[1]) / (rng[2] - rng[1])
      }, .names = "{.col}_norm"))
    
    # Pivot to long format for ggplot
    heatmap_long <- heatmap_data %>%
      pivot_longer(cols = all_of(metrics_cols), names_to = "metric", values_to = "value")
    
    heatmap_norm_long <- heatmap_norm %>%
      select(method, ends_with("_norm")) %>%
      pivot_longer(cols = -method, names_to = "metric_norm", values_to = "norm_value") %>%
      mutate(metric = str_remove(metric_norm, "_norm"))
    
    heatmap_combined <- heatmap_long %>%
      left_join(heatmap_norm_long, by = c("method", "metric")) %>%
      mutate(
        method = factor(method, levels = rev(METHOD_ORDER)),
        metric = factor(metric, levels = metrics_cols)
      )
    
    p6 <- ggplot(heatmap_combined, aes(x = metric, y = method, fill = norm_value)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = signif(value, 3)), size = 3, color = "black") +
      scale_fill_gradient2(low = "#2ecc71", mid = "#f1c40f", high = "#e74c3c",
                           midpoint = 0.5, limits = c(0, 1),
                           name = "Normalized\n(0=best)") +
      labs(title = "Goodness-of-Fit Heatmap",
           subtitle = subtitle_text,
           x = "Metric", y = "Method") +
      theme_weibull() +
      theme(legend.position = "right",
            axis.text.x = element_text(angle = 30, hjust = 1))
    
    ggsave(file.path(plot_dir, paste0("plot6_gof_heatmap_", model_clean, ".png")),
           p6, width = 10, height = 6, dpi = 300)
    cat("  [OK] plot6_gof_heatmap\n")
  }
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 7 — Composite Rank Bar Chart
  # ═══════════════════════════════════════════════════════════════════════════
  
  rank_data <- model_gof %>%
    filter(!is.na(Composite_Rank)) %>%
    arrange(Composite_Rank) %>%
    mutate(
      method = factor(method, levels = rev(method)),
      rank_color = Composite_Rank
    )
  
  if (nrow(rank_data) > 0) {
    p7 <- ggplot(rank_data, aes(x = Composite_Rank, y = method, fill = rank_color)) +
      geom_col(width = 0.7) +
      geom_text(aes(label = round(Composite_Rank, 2)),
                hjust = -0.1, size = 3.5) +
      scale_fill_gradient(low = "#2ecc71", high = "#e74c3c",
                          name = "Rank") +
      labs(title = "Composite Rank — Method Comparison",
           subtitle = subtitle_text,
           x = "Composite Rank (lower = better)", y = "Method") +
      theme_weibull() +
      theme(legend.position = "none")
    
    ggsave(file.path(plot_dir, paste0("plot7_composite_rank_", model_clean, ".png")),
           p7, width = 9, height = 6, dpi = 300)
    cat("  [OK] plot7_composite_rank\n")
  }
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PLOT 8 — Parameter Estimates vs MLE Reference
  # ═══════════════════════════════════════════════════════════════════════════
  
  mle_row <- model_params %>% filter(method == "MLE")
  
  if (nrow(mle_row) > 0) {
    k_mle   <- mle_row$k[1]
    lam_mle <- mle_row$lambda[1]
    
    param_plot_data <- model_params %>%
      mutate(method = factor(method, levels = rev(METHOD_ORDER)))
    
    # Get MCMC CIs if available
    mcmc_params <- params %>% filter(model == mdl, method == "MCMC")
    
    # K panel
    p8_k <- ggplot(param_plot_data, aes(x = k, y = method, fill = method)) +
      geom_col(width = 0.6) +
      geom_vline(xintercept = k_mle, linetype = "dashed", color = "red",
                 linewidth = 0.8) +
      scale_fill_manual(values = METHOD_COLORS, guide = "none") +
      labs(title = "Shape Parameter (k)",
           x = "k", y = NULL) +
      theme_weibull()
    
    # Add MCMC error bar if available
    if (nrow(mcmc_params) > 0 && !is.na(mcmc_params$k_lower_95[1])) {
      p8_k <- p8_k +
        geom_errorbar(
          data = param_plot_data %>% filter(method == "MCMC") %>%
            mutate(k_lower_95 = mcmc_params$k_lower_95[1], k_upper_95 = mcmc_params$k_upper_95[1]),
          aes(xmin = k_lower_95, xmax = k_upper_95),
          width = 0.3, linewidth = 0.8, color = "black"
        )
    }
    
    # Lambda panel
    p8_lam <- ggplot(param_plot_data, aes(x = lambda, y = method, fill = method)) +
      geom_col(width = 0.6) +
      geom_vline(xintercept = lam_mle, linetype = "dashed", color = "red",
                 linewidth = 0.8) +
      scale_fill_manual(values = METHOD_COLORS, guide = "none") +
      labs(title = "Scale Parameter (λ)",
           x = "λ (days)", y = NULL) +
      theme_weibull()
    
    # Add MCMC error bar if available
    if (nrow(mcmc_params) > 0 && !is.na(mcmc_params$lam_lower_95[1])) {
      p8_lam <- p8_lam +
        geom_errorbar(
          data = param_plot_data %>% filter(method == "MCMC") %>%
            mutate(lam_lower_95 = mcmc_params$lam_lower_95[1], lam_upper_95 = mcmc_params$lam_upper_95[1]),
          aes(xmin = lam_lower_95, xmax = lam_upper_95),
          width = 0.3, linewidth = 0.8, color = "black"
        )
    }
    
    # Combine with patchwork
    library(patchwork)
    p8 <- (p8_k | p8_lam) +
      plot_annotation(
        title = "Parameter Estimates vs MLE Reference",
        subtitle = subtitle_text,
        caption = "Dashed red line = MLE estimate. Error bars for MCMC (7B) only.\nNo uncertainty output for other methods including MAP (7A).",
        theme = theme(
          plot.title = element_text(size = 13, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "grey40"),
          plot.caption = element_text(size = 9, color = "grey50", hjust = 0)
        )
      )
    
    ggsave(file.path(plot_dir, paste0("plot8_parameters_", model_clean, ".png")),
           p8, width = 14, height = 7, dpi = 300)
    cat("  [OK] plot8_parameters\n")
  }
  
  cat("\n")
}

cat("==================================================================\n")
cat(" All plots saved to:", plot_dir, "\n")
cat("==================================================================\n")

# List generated files
plot_files <- list.files(plot_dir, pattern = "\\.png$")
cat("  Total plot files:", length(plot_files), "\n")
for (f in plot_files) cat("    ", f, "\n")

cat("\n[DONE] 04_plots.R complete.\n")
