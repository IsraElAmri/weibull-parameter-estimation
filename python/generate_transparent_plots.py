import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import expon, lognorm

# Reset style
plt.style.use('default')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# Make figure background completely transparent
fig.patch.set_alpha(0.0)

# Colors for a light slide background
TEXT_COLOR = '#0f2945'  # Dark navy blue for text (matching slide title text)
AXIS_COLOR = '#47637d'  # Muted blue/gray for axes
LINE1_COLOR = '#0284c7' # Vivid blue for graph 1
LINE2_COLOR = '#0369a1' # Deeper blue for graph 2

# Helper to style axes
def style_axes(ax):
    ax.set_facecolor('none')  # Transparent axes background
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.5)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    # We remove the matplotlib grid because the slide already has a grid!
    ax.grid(False)

# -------------------------------------------------------------
# GRAPH 1: The TRUE Exponential Prior used for MAP
# -------------------------------------------------------------
mean_t_fail = 50 
x1 = np.linspace(0, 300, 1000)
y1 = expon.pdf(x1, scale=mean_t_fail)

style_axes(ax1)
ax1.plot(x1, y1, color=LINE1_COLOR, linewidth=3)
ax1.fill_between(x1, y1, color=LINE1_COLOR, alpha=0.25)

ax1.set_title('Graph 1: The Actual Exponential Prior on $\lambda$', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('$\lambda$ (Days)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')

ax1.annotate('Prior pulls strongly toward 0-50 days,\nignoring >99% of drives that survived 92 days',
             xy=(50, expon.pdf(50, scale=mean_t_fail)), xytext=(100, 0.015),
             arrowprops=dict(facecolor=TEXT_COLOR, shrink=0.05, width=1.5, headwidth=8, edgecolor=TEXT_COLOR),
             color='white', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc=TEXT_COLOR, ec="none", alpha=0.9))

# -------------------------------------------------------------
# GRAPH 2: The TRUE MCMC Posterior for lambda
# -------------------------------------------------------------
shape = 0.5  
scale = 28114 
x2 = np.linspace(0, 120000, 1000)
y2 = lognorm.pdf(x2, s=shape, scale=scale)

style_axes(ax2)
ax2.plot(x2, y2, color=LINE2_COLOR, linewidth=3)

# Highlight the 95% CI
x_ci = np.linspace(11282, 87037, 500)
y_ci = lognorm.pdf(x_ci, s=shape, scale=scale)
ax2.fill_between(x_ci, y_ci, color=LINE2_COLOR, alpha=0.35, label='95% Credible Interval')

ax2.set_title('Graph 2: The Actual MCMC Posterior (Honest Uncertainty)', fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel('$\lambda$ (Scale Parameter in Days)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Posterior Density', fontsize=11, fontweight='bold')

ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax2.annotate('95% Credible Interval\n[11,282 days to 87,037 days]',
             xy=(40000, lognorm.pdf(40000, s=shape, scale=scale)), xytext=(60000, 0.000015),
             arrowprops=dict(facecolor=TEXT_COLOR, shrink=0.05, width=1.5, headwidth=8, edgecolor=TEXT_COLOR),
             color='white', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc=TEXT_COLOR, ec="none", alpha=0.9))

plt.tight_layout(pad=3.0)
output_file = 'c:/Users/Isra/Downloads/data_Q3_2025_final2/weibull_project/python/correct_bayesian_plots_transparent.png'
plt.savefig(output_file, dpi=300, transparent=True)
print(f'Saved transparent plots to {output_file}')
