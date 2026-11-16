# ==============================================================================
# 01_data_prep.R — Data Loading, Cleaning, and Survival Data Construction
# ==============================================================================
# Source : Backblaze Hard Drive Stats Q3 2025
# Input  : Daily CSV files (one per day, ~127 MB each)
# Output : data/drives_clean.csv — one row per drive with lifetime and status
# ==============================================================================

# ── Reproducibility ──────────────────────────────────────────────────────────
set.seed(42)
options(scipen = 999)          # avoid scientific notation in output
options(digits = 6)            # consistent decimal display
Sys.setenv(LANG = "en_US")     # English error messages for reproducibility

# ── Libraries ────────────────────────────────────────────────────────────────
library(tidyverse)
library(here)

# ── Configuration ────────────────────────────────────────────────────────────
# Point DATA_RAW_DIR to the folder containing daily Backblaze CSV files.
# By default, looks in the project's data/ folder. Change if CSVs are elsewhere.
DATA_RAW_DIR <- here::here("..", "data_Q3_2025")
OUTPUT_DIR   <- here::here("data")

cat("==================================================================\n")
cat(" STEP 1 — Data Loading & Preprocessing\n")
cat("==================================================================\n\n")

# ==============================================================================
# Step 1 — Load all daily CSVs and bind them
# ==============================================================================
# Each CSV contains one row per drive per day with columns:
# date, serial_number, model, capacity_bytes, failure, smart_* attributes.
# map_dfr applies read_csv to every file and row-binds the results.
# We select only the columns we need to reduce memory usage (~11 GB raw).
# ==============================================================================

csv_files <- list.files(DATA_RAW_DIR, pattern = "^2025-.*\\.csv$", full.names = TRUE)
cat("  CSV files found:", length(csv_files), "\n")

if (length(csv_files) == 0) {
  stop("No CSV files found in ", DATA_RAW_DIR,
       "\n  Download Backblaze Q3 2025 data and place CSVs there.\n",
       "  See data/README_data.md for instructions.")
}

# Columns we need (minimal set to reduce memory)
COLS_NEEDED <- c("date", "serial_number", "model", "capacity_bytes", "failure")

cat("  Loading and binding all daily CSVs (this may take several minutes)...\n")

df <- csv_files %>%
  map_dfr(function(fp) {
    tryCatch({
      tmp <- read_csv(fp, col_types = cols(
        date          = col_date(format = ""),
        serial_number = col_character(),
        model         = col_character(),
        capacity_bytes = col_double(),
        failure       = col_integer(),
        .default      = col_skip()
      ), show_col_types = FALSE)
      tmp
    }, error = function(e) {
      warning(paste("  [WARN] Skipping", basename(fp), ":", e$message))
      tibble()
    })
  })

cat("  Raw rows loaded:", format(nrow(df), big.mark = ","), "\n")
cat("  Date range:", as.character(min(df$date, na.rm = TRUE)), "to",
    as.character(max(df$date, na.rm = TRUE)), "\n\n")

# ==============================================================================
# Step 2 — Build one row per drive
# ==============================================================================
# For each unique serial_number, compute:
#   model       = drive model (first occurrence)
#   first_seen  = earliest date the drive appeared
#   last_seen   = latest date the drive appeared
#   failed      = 1 if the drive ever reported failure=1, else 0
#   lifetime_days = observation window length (last_seen - first_seen + 1)
#
# The +1 ensures a drive seen only on one day has lifetime_days = 1,
# not 0, which would cause log(0) = -Inf in Weibull linearizations.
# ==============================================================================

cat("  Building one row per drive...\n")

drives <- df %>%
  group_by(serial_number) %>%
  summarise(
    model         = first(model),
    first_seen    = min(date),
    last_seen     = max(date),
    failed        = max(failure),
    lifetime_days = as.numeric(last_seen - first_seen) + 1,
    .groups       = "drop"
  )

cat("  Total unique drives:", format(nrow(drives), big.mark = ","), "\n\n")

# ==============================================================================
# ROBUSTNESS: Defensive cleaning for real-world data issues
# ==============================================================================

n_before <- nrow(drives)

# --- Remove drives with implausibly long lifetimes (>10 years = 3650 days) ---
# In a single-quarter dataset, lifetime > 92 days is the max possible.
# We use 3650 as a generous upper bound for any dataset.
n_long <- sum(drives$lifetime_days > 3650, na.rm = TRUE)
drives <- drives %>% filter(lifetime_days <= 3650)

# --- Remove duplicate serial numbers across models (data entry errors) ---
# Keep first occurrence by model
n_dup_before <- nrow(drives)
drives <- drives %>%
  group_by(serial_number) %>%
  filter(model == first(model)) %>%
  ungroup()
n_removed_dup <- n_dup_before - nrow(drives)

# ==============================================================================
# Step 3 — Select top 3 models by drive count
# ==============================================================================
# We focus on the 3 most common models to ensure adequate sample sizes
# for all estimation methods. More models can be analyzed by changing this.
# ==============================================================================

top3 <- drives %>%
  count(model, sort = TRUE) %>%
  slice_head(n = 3) %>%
  pull(model)

cat("  Top 3 models by drive count:\n")
for (m in top3) {
  n <- sum(drives$model == m)
  cat("    ", m, ":", format(n, big.mark = ","), "drives\n")
}

drives <- drives %>% filter(model %in% top3)

# ==============================================================================
# Step 4 — Remove drives with lifetime_days <= 0
# ==============================================================================
# Drives with lifetime <= 0 are data artifacts (impossible physically).
# log(0) = -Inf would break all Weibull linearization methods.
# ==============================================================================

n_zero <- sum(drives$lifetime_days <= 0, na.rm = TRUE)
drives <- drives %>% filter(lifetime_days > 0)

# ==============================================================================
# Additional robustness: ensure minimum failures per model
# ==============================================================================
# Minimum 5 failures per model is required for any estimation method
# to produce meaningful results. If a top-3 model has fewer, replace
# it with the next most common model that meets the threshold.
# ==============================================================================

failure_check <- drives %>%
  group_by(model) %>%
  summarise(n_failures = sum(failed), .groups = "drop")

models_insufficient <- failure_check %>%
  filter(n_failures < 5) %>%
  pull(model)

if (length(models_insufficient) > 0) {
  cat("\n  WARNING: Models with <5 failures (replacing):",
      paste(models_insufficient, collapse = ", "), "\n")
  
  # Get replacement models from the full dataset
  all_models <- df %>%
    group_by(serial_number) %>%
    summarise(model = first(model), failed = max(failure), .groups = "drop")
  
  replacement_candidates <- all_models %>%
    filter(!(model %in% top3)) %>%
    group_by(model) %>%
    summarise(n = n(), n_failures = sum(failed), .groups = "drop") %>%
    filter(n_failures >= 5, n >= 30) %>%
    arrange(desc(n))
  
  for (bad_model in models_insufficient) {
    if (nrow(replacement_candidates) > 0) {
      new_model <- replacement_candidates$model[1]
      replacement_candidates <- replacement_candidates[-1, ]
      cat("    Replacing", bad_model, "with", new_model, "\n")
      
      new_drives <- df %>%
        filter(model == new_model) %>%
        group_by(serial_number) %>%
        summarise(
          model         = first(model),
          first_seen    = min(date),
          last_seen     = max(date),
          failed        = max(failure),
          lifetime_days = as.numeric(last_seen - first_seen) + 1,
          .groups       = "drop"
        ) %>%
        filter(lifetime_days > 0, lifetime_days <= 3650)
      
      drives <- drives %>%
        filter(model != bad_model) %>%
        bind_rows(new_drives)
    }
  }
}

# ==============================================================================
# Step 5 — Print per-model summary: n_total, n_failures, censoring_%
# ==============================================================================

cat("\n  ────────────────────────────────────────────────────────────────\n")
cat("  Per-Model Summary\n")
cat("  ────────────────────────────────────────────────────────────────\n")

model_summary <- drives %>%
  group_by(model) %>%
  summarise(
    n_total      = n(),
    n_failures   = sum(failed),
    n_censored   = sum(failed == 0),
    censoring_pct = round((1 - mean(failed)) * 100, 2),
    min_lifetime  = min(lifetime_days),
    median_lifetime = median(lifetime_days),
    max_lifetime  = max(lifetime_days),
    .groups       = "drop"
  )

for (i in seq_len(nrow(model_summary))) {
  r <- model_summary[i, ]
  cat(sprintf("  %-35s | N=%s | Failures=%d | Censoring=%.2f%%\n",
              r$model,
              format(r$n_total, big.mark = ","),
              r$n_failures,
              r$censoring_pct))
  cat(sprintf("    Lifetime (days): min=%d, median=%d, max=%d\n",
              r$min_lifetime, r$median_lifetime, r$max_lifetime))
}

# ==============================================================================
# Data Quality Report
# ==============================================================================

cat("\n  === Data Quality Report ===\n")
cat("  Drives removed: lifetime > 3650 days:", n_long, "\n")
cat("  Drives removed: duplicate serial/model:", n_removed_dup, "\n")
cat("  Drives removed: lifetime <= 0:", n_zero, "\n")
cat("  Final dataset: ",
    format(nrow(drives), big.mark = ","), " drives across ",
    length(unique(drives$model)), " models\n\n")

# ==============================================================================
# Step 6 — Save as data/drives_clean.csv
# ==============================================================================

output_path <- file.path(OUTPUT_DIR, "drives_clean.csv")
write_csv(drives, output_path)
cat("  Saved:", output_path, "\n")
cat("  Rows:", format(nrow(drives), big.mark = ","), "\n")
cat("  Columns:", paste(names(drives), collapse = ", "), "\n")

# Also save model summary for use by downstream scripts
summary_path <- file.path(OUTPUT_DIR, "model_summary.csv")
write_csv(model_summary, summary_path)
cat("  Saved:", summary_path, "\n")

cat("\n[DONE] 01_data_prep.R complete.\n")
