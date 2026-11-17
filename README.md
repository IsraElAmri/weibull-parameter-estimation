# Weibull Parameter Estimation on Hard Drive Failure Data

[![R](https://img.shields.io/badge/R-4.3+-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comparative statistical reliability analysis and interactive Streamlit dashboard benchmarking 9 Weibull parameter estimation methods on 117,918 hard drives from the Backblaze Q3 2025 dataset (92-day observational window).

## Key Reliability Findings

Across 117,918 hard drives monitored over 92 days, only 251 drives failed (99.69% to 99.88% right-censoring rate across the 3 dominant drive models: TOSHIBA MG07ACA14TA, TOSHIBA MG08ACA16TA, and WDC WUH722222ALE6L4).

1. **Failure-Only Methods Discarding Censored Drives**:
   - Classical methods that ignore right-censored drives (OLS, WLS, MOM, L-Moments, EPF, Percentile) estimate a scale parameter $\lambda \approx 50\text{--}59\text{ days}$.
   - This result is mathematically misleading: predicting a characteristic life of 50-60 days for a fleet where over 99.7% of drives survived the full 92-day window occurs because the algorithms fit only the ~250 failed units.
2. **Censoring-Aware Methods (MLE and Bayesian MCMC)**:
   - Maximum Likelihood Estimation accounting for right-censoring and Bayesian MCMC (via `brms`/Stan) incorporate the full survival history of non-failed drives.
   - MLE and MCMC consistently agree: shape parameter $k \approx 1.0\text{--}1.3$ (indicating a roughly constant hazard rate during early-to-mid service) and realistic fleet characteristic lifetimes $\lambda \approx 11,800\text{--}43,000\text{ days}$ (32 to 118 years).
   - MCMC 95% posterior credible intervals for $k$ include or closely bound 1.0.
3. **Goodness-of-Fit Paradox**:
   - Standard residual metrics (RMSE, Kolmogorov-Smirnov) evaluated only on observed failure timestamps paradoxically rank the failure-only methods best.
   - Information criteria (AIC, BIC) correctly penalize models that cannot account for fleet-wide survival probabilities.

## Estimation Methods Benchmarked

| Method | Implementation | Censoring Handling |
|---|---|---|
| **OLS** | Ordinary Least Squares on Weibull probability paper | Discards censored drives |
| **WLS** | Weighted Least Squares (variance-weighted) | Discards censored drives |
| **MOM** | Method of Moments ($E[X], \text{Var}(X)$) | Discards censored drives |
| **L-Moments** | Linear combinations of order statistics | Discards censored drives |
| **EPF** | Empirical Probability Function | Discards censored drives |
| **Percentile** | Two-point percentile matching ($t_{0.10}, t_{0.90}$) | Discards censored drives |
| **MLE** | Maximum Likelihood with log-likelihood survival term | Right-censoring aware |
| **MAP** | Maximum A Posteriori (Lognormal informative prior) | Right-censoring aware (prior-sensitive) |
| **MCMC** | Full Bayesian Hamiltonian Monte Carlo (brms / Stan) | Right-censoring aware |

## Repository Structure

```
weibull-parameter-estimation/
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â”œâ”€â”€ README.md
â”œâ”€â”€ packages.R
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ R/
â”‚   â”œâ”€â”€ 01_data_prep.R
â”‚   â”œâ”€â”€ 02_estimation_methods.R
â”‚   â”œâ”€â”€ 03_gof_metrics.R
â”‚   â”œâ”€â”€ 04_plots.R
â”‚   â”œâ”€â”€ BLACK_BOX_VS_WHITE_BOX.Rmd
â”‚   â””â”€â”€ results/
â”‚       â”œâ”€â”€ brms_fit_*.rds
â”‚       â”œâ”€â”€ gof_metrics.csv
â”‚       â”œâ”€â”€ parameters.csv
â”‚       â”œâ”€â”€ sensitivity_analysis.csv
â”‚       â”œâ”€â”€ plots/
â”‚       â””â”€â”€ plots_combined/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ README_data.md
â”‚   â”œâ”€â”€ drives_clean.csv
â”‚   â””â”€â”€ model_summary.csv
â”œâ”€â”€ data_exploration/
â”‚   â”œâ”€â”€ figure3_1.png
â”‚   â”œâ”€â”€ figure3_2.png
â”‚   â”œâ”€â”€ figure3_3.png
â”‚   â””â”€â”€ generate_figures.py
â””â”€â”€ python/
    â””â”€â”€ dashboard.py
```

## Running the Project

### Interactive Dashboard (Python / Streamlit)

The Streamlit dashboard executes all parameter estimations directly in Python via SciPy and NumPy, loading from the shipped dataset `data/drives_clean.csv`:

```bash
pip install -r requirements.txt
streamlit run python/dashboard.py
```

### Full R Analysis Pipeline

To run the complete statistical estimation and Bayesian Stan sampling:

```bash
Rscript packages.R
Rscript R/01_data_prep.R
Rscript R/02_estimation_methods.R
Rscript R/03_gof_metrics.R
Rscript R/04_plots.R
```

## Dashboard Preview

![Dashboard](assets/dashboard.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

