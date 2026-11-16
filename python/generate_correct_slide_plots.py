import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import expon, lognorm

# Set up the style to look somewhat like the presentation
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#1e293b')
fig.patch.set_facecolor('#1e293b')

# -------------------------------------------------------------
# GRAPH 1: The TRUE Exponential Prior used for MAP
# -------------------------------------------------------------
# In the code, prior(lam) = exp(-lam / mean_t_fail)
mean_t_fail = 50  # approximate from the data

x1 = np.linspace(0, 300, 1000)
# Exponential PDF: f(x) = (1/scale) * exp(-x/scale)
y1 = expon.pdf(x1, scale=mean_t_fail)

ax1.set_facecolor('#1e293b')
ax1.plot(x1, y1, color='#38bdf8', linewidth=3)
ax1.fill_between(x1, y1, color='#38bdf8', alpha=0.2)

ax1.set_title('Graph 1: The Actual Exponential Prior on $\lambda$', color='white', fontsize=14, pad=15)
ax1.set_xlabel('$\lambda$ (Days)', color='white')
ax1.set_ylabel('Probability Density', color='white')
ax1.grid(color='#334155', linestyle='-', linewidth=0.5)

# Annotation explaining the misspecification
ax1.annotate('Prior pulls strongly toward 0-50 days,\nignoring >99% of drives that survived 92 days',
             xy=(50, expon.pdf(50, scale=mean_t_fail)), xytext=(100, 0.015),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8),
             color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#38bdf8", lw=1))

# -------------------------------------------------------------
# GRAPH 2: The TRUE MCMC Posterior for lambda
# -------------------------------------------------------------
# From parameters.csv (TOSHIBA MG07ACA14TA):
# MCMC lambda_hat = 28,114
# 95% CI: [11,282 , 87,037]
# We approximate this with a log-normal distribution for visualization
shape = 0.5  # Approximate dispersion
scale = 28114 # Median
x2 = np.linspace(0, 120000, 1000)
y2 = lognorm.pdf(x2, s=shape, scale=scale)

ax2.set_facecolor('#1e293b')
ax2.plot(x2, y2, color='#34d399', linewidth=3)

# Highlight the 95% CI
x_ci = np.linspace(11282, 87037, 500)
y_ci = lognorm.pdf(x_ci, s=shape, scale=scale)
ax2.fill_between(x_ci, y_ci, color='#34d399', alpha=0.4, label='95% Credible Interval')

ax2.set_title('Graph 2: The Actual MCMC Posterior (Honest Uncertainty)', color='white', fontsize=14, pad=15)
ax2.set_xlabel('$\lambda$ (Scale Parameter in Days)', color='white')
ax2.set_ylabel('Posterior Density', color='white')
ax2.grid(color='#334155', linestyle='-', linewidth=0.5)

# Format x-axis with commas for thousands
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax2.annotate('95% Credible Interval\n[11,282 days to 87,037 days]',
             xy=(40000, lognorm.pdf(40000, s=shape, scale=scale)), xytext=(60000, 0.000015),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8),
             color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#34d399", lw=1))

plt.tight_layout(pad=3.0)
plt.savefig('c:/Users/Isra/Downloads/data_Q3_2025_final2/weibull_project/python/correct_bayesian_plots.png', dpi=300)
print("Saved correct plots to c:/Users/Isra/Downloads/data_Q3_2025_final2/weibull_project/python/correct_bayesian_plots.png")
