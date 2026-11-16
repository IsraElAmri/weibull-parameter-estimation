# ==============================================================================
# packages.R — Install all required R packages for the Weibull analysis project
# ==============================================================================
# Usage: Rscript packages.R
# Run this ONCE before the analysis pipeline.
# ==============================================================================

cat("=== Installing Required R Packages ===\n\n")

# List of required packages
required_packages <- c(
  # Data wrangling & core
  "tidyverse",      # dplyr, ggplot2, readr, purrr, tidyr, stringr, forcats
  "here",           # Robust project-relative paths
  

  # Survival analysis
  "survival",       # Surv objects, survfit, coxph
  "flexsurv",       # Flexible parametric survival models
  "fitdistrplus",   # Distribution fitting utilities
  
  # Plotting
  "ggplot2",        # Grammar of graphics (included in tidyverse)
  "scales",         # Axis formatting helpers
  "patchwork",      # Combine multiple ggplots
  
  # Tables & reporting
  "knitr",          # R Markdown engine
  "rmarkdown",      # Document rendering
  "kableExtra",     # Professional LaTeX/HTML tables
  
  # Goodness-of-fit
  "goftest",        # Anderson-Darling, Cramér-von Mises tests
  
  # Bayesian (Method 7B — MCMC extension)
  "brms",           # Bayesian regression models via Stan
  "rstan",          # Stan interface for R
  "bayesplot",      # MCMC diagnostic plots
  
  # LaTeX for PDF rendering
  "tinytex",        # Lightweight LaTeX distribution
  
  # Flowchart
  "DiagrammeR"      # Decision flowchart in reports
)

# Install missing packages
installed <- installed.packages()[, "Package"]
to_install <- setdiff(required_packages, installed)

if (length(to_install) > 0) {
  cat("Installing:", paste(to_install, collapse = ", "), "\n\n")
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  cat("All packages already installed.\n")
}

# Install TinyTeX for PDF rendering (if not present)
if (!tinytex::is_tinytex()) {
  cat("\nInstalling TinyTeX (lightweight LaTeX distribution)...\n")
  tinytex::install_tinytex()
} else {
  cat("TinyTeX already installed.\n")
}

cat("\n=== Package Installation Complete ===\n")

# Verify critical packages load
cat("\nVerifying critical package loading...\n")
suppressMessages({
  library(tidyverse)
  library(here)
  library(survival)
  library(brms)
  library(ggplot2)
  library(kableExtra)
})
cat("All critical packages loaded successfully.\n")
