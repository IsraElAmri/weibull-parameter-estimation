# Performance Evaluation of Methods for Weibull Distribution Parameter Estimation
## Application to Backblaze Hard Drive Failure Data (Q3 2025)

A complete two-language project (R + Python) implementing **9 Weibull parameter estimation methods**, **8 goodness-of-fit metrics**, and an **interactive Streamlit dashboard** for reliability analysis of hard drive failure data.

---

## 📁 Project Structure

```
weibull_project/
├── data/
│   ├── README_data.md          # Dataset documentation
│   ├── drives_clean.csv        # Generated: one row per drive
│   └── model_summary.csv       # Generated: per-model statistics
├── R/
│   ├── 01_data_prep.R          # Data loading, cleaning, survival data
│   ├── 02_estimation_methods.R # 9 estimation methods + sensitivity analysis
│   ├── 03_gof_metrics.R        # 8 GoF metrics + composite ranking
│   ├── 04_plots.R              # 8 plot types × 3 models = 24 PNGs
│   ├── 05_report.Rmd           # Technical report (code visible)
│   ├── CODE_EXPLAINED.Rmd      # Line-by-line code explanation
│   ├── FINAL_REPORT.Rmd        # Academic paper format
│   └── results/
│       ├── parameters.csv      # 9 methods × 3 models estimates
│       ├── gof_metrics.csv     # All GoF metrics
│       ├── sensitivity_analysis.csv
│       ├── brms_fit_*.rds      # MCMC posterior fits (cached)
│       └── plots/              # 24 PNG files at 300 DPI
├── python/
│   └── dashboard.py            # Streamlit interactive dashboard
├── packages.R                  # R package installer
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Methods Implemented

| # | Method | Principle | Handles Censoring | 
|---|--------|-----------|-------------------|
| 1 | **OLS** (= MRR) | Rank regression on Weibull paper | ✗ |
| 2 | **WLS** | Bergman (1986) delta-method weights | ✗ |
| 3 | **MOM** | Moment matching (CV²) | ✗ |
| 4 | **L-Moments** | Hosking (1990) linear moments | ✗ (>5%) |
| 5 | **EPF** | Energy pattern factor (E[T³]/(E[T])³) | ✗ |
| 6 | **MLE** | Censored log-likelihood, Nelder-Mead | ✓ |
| 7A | **MAP** | Bayesian posterior mode, exponential priors | ✓ |
| 7B | **MCMC** ★ | Full Bayesian via brms/Stan (HMC-NUTS) | ✓ |
| 8 | **Percentile** | Two-quantile analytical method | ✗ |

★ MCMC is an **extension** beyond core course materials — provides full posterior distributions with credible intervals.

## 🏗️ Setup & Installation

### 1. Download Backblaze Data

1. Go to https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
2. Download the **Q3 2025** dataset (or your target quarter)
3. Extract all daily CSV files
4. Either:
   - Place them in `weibull_project/data/`, **or**
   - Update `DATA_RAW_DIR` at the top of `R/01_data_prep.R` to point to your CSV directory

### 2. Install R Packages

```bash
Rscript packages.R
```

This installs: tidyverse, survival, brms, rstan, bayesplot, kableExtra, DiagrammeR, tinytex, and others.

### 3. Install Python Packages

```bash
pip install -r requirements.txt
```

Optional — for exact environment replication:
```bash
pip freeze > requirements_exact.txt
```

## 🚀 Running the Analysis Pipeline

Execute the R scripts **in order** (each depends on the previous):

```bash
# Step 1: Load & clean data → data/drives_clean.csv
Rscript R/01_data_prep.R

# Step 2: Estimate parameters → R/results/parameters.csv
Rscript R/02_estimation_methods.R

# Step 3: Compute GoF metrics → R/results/gof_metrics.csv
Rscript R/03_gof_metrics.R

# Step 4: Generate all plots → R/results/plots/*.png
Rscript R/04_plots.R
```

### Render Reports (Optional)

```bash
# Technical report (echo=TRUE, all code visible)
Rscript -e "rmarkdown::render('R/05_report.Rmd')"

# Line-by-line code explanation (teaching document)
Rscript -e "rmarkdown::render('R/CODE_EXPLAINED.Rmd')"

# Academic paper (echo=FALSE, publication format)
Rscript -e "rmarkdown::render('R/FINAL_REPORT.Rmd')"
```

### Launch Dashboard

```bash
streamlit run python/dashboard.py
```

The dashboard reads **only** from `R/results/` — no statistical computation in Python.

## ⏱️ Runtime Notes

| Step | Approximate Time |
|------|-----------------|
| `01_data_prep.R` | 5–15 min (reads ~11 GB of CSVs) |
| `02_estimation_methods.R` | 5–30 min per model for MCMC |
| `03_gof_metrics.R` | < 1 min |
| `04_plots.R` | 1–3 min |
| Reports (Rmd) | 2–5 min each |
| Dashboard | Instant (reads CSVs) |

**MCMC caching**: Method 7B results are saved as `.rds` files. On subsequent runs, existing fits are loaded instead of refitted. Delete the `.rds` file to force a refit.

## 🎯 Key Features

### Mathematical Rigor
- Every method includes **full derivations** in code comments
- **Sanity checks** on all parameter estimates (k > 0, λ > 0, range warnings)
- **Convergence reporting** for MLE and MAP (optim convergence codes)
- **Double verification** for L-Moments (pairwise vs single-sum formula)
- **EPF approximation check** (exact vs empirical k ≈ 1 + 3.69/EPF²)

### Statistical Completeness
- **8 GoF metrics**: KS, RMSE, MAE, Anderson-Darling, AIC, AICc, BIC, MRL error
- **Composite ranking** via Borda count (rank_RMSE + rank_AD + rank_AIC) / 3
- **Sensitivity analysis**: EPF outlier contamination vs MOM, L-Moments, MLE
- **Decision logic**: method elimination based on censoring thresholds

### Reproducibility
- `set.seed(42)` in all scripts
- `sessionInfo()` in reports
- Exact package versions via `packages.R`
- All constants named and explained — no magic numbers

## 📋 Decision Logic Summary

| Censoring Level | Eliminate | Retain |
|----------------|-----------|--------|
| > 0% | Percentile, MOM, EPF | MLE, MAP, MCMC |
| > ~5% | + L-Moments | MLE, MAP, MCMC |
| > ~20% | + OLS, WLS | MLE, MAP, MCMC |

**Secondary criteria**:
- Prior knowledge + small n → MCMC (honest uncertainty)
- Prior knowledge, quick estimate → MAP
- No prior, large dataset → MLE
- Complete data, outliers → L-Moments
- Complete data, clean → WLS

## 📐 Key Insights

- **OLS = MRR**: Identical computation. No need to implement both.
- **Flat prior + MAP = MLE**: Algebraically proven. Exponential prior adds regularization.
- **WLS can outrank MLE**: When the true distribution is only approximately Weibull.
- **MCMC wide CI ≠ weakness**: It's what the data tells you.
- **EPF is fragile**: One observation at 2× the mean → 8× effect on the cube.

## 📚 References

1. Weibull W. (1951). *J. Applied Mechanics*, 18, 293–297.
2. Benard A. & Bos-Levenbach E.C. (1953). *Statistica*, 7, 163–173.
3. Bergman B. (1986). *J. Materials Science Letters*, 5, 611–614.
4. Akaike H. (1974). *IEEE Trans. Automatic Control*, 19, 716–723.
5. Hosking J.R.M. (1990). *J. Royal Statistical Society B*, 52, 105–124.
6. Schwarz G. (1978). *Annals of Statistics*, 6, 461–464.
7. Bürkner P.C. (2017). *J. Statistical Software*, 80(1), 1–28.
8. Backblaze Inc. (2025). Hard Drive Test Data Q3 2025.
