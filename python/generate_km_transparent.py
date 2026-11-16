import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up the figure
fig = plt.figure(figsize=(16, 9))
fig.patch.set_alpha(0.0)  # Transparent background

TEXT_COLOR = '#0f2945'
LINE_COLOR = '#1e3a8a'
BOX_BG = '#f8fafc'

# -------------------------
# HEADER
# -------------------------
fig.text(0.5, 0.92, 'The Kaplan-Meier Estimator: Assumption-Free Survival Analysis', 
         fontsize=28, fontweight='bold', color=TEXT_COLOR, ha='center', va='center')
fig.text(0.5, 0.86, 'A non-parametric product-limit framework for mapping empirical ground truth without theoretical curve-fitting.', 
         fontsize=16, color=TEXT_COLOR, ha='center', va='center')

# Vertical divider line
fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.8], color=TEXT_COLOR, linewidth=1.5, alpha=0.5))

# -------------------------
# LEFT SIDE: Math Engine
# -------------------------
fig.text(0.25, 0.78, 'The Mathematical Engine', fontsize=22, color=TEXT_COLOR, ha='center')

# Formula
# Matplotlib math rendering
formula = r"$\hat{S}(t) = \prod \left[ 1 - \frac{d_i}{n_i} \right] \quad \mathrm{for}\ t_i \leq t$"
fig.text(0.25, 0.45, formula, fontsize=38, color=TEXT_COLOR, ha='center', va='center')

# Annotations
# S(t)
fig.text(0.24, 0.65, r"$\mathbf{\hat{S}(t)}$ (Estimated Survival): The precise" + "\nprobability of survival at time $t$.", 
         fontsize=13, color=TEXT_COLOR, va='center')
# d_i
fig.text(0.36, 0.55, r"$\mathbf{d_i}$ (Failures): The exact" + "\nnumber of units that\nphysically fail at time $t_i$.", 
         fontsize=13, color=TEXT_COLOR, va='center')
# n_i
fig.text(0.34, 0.22, r"$\mathbf{n_i}$ (Risk Set): The total" + "\nnumber of units still alive\nand not yet censored\njust before $t_i$.", 
         fontsize=13, color=TEXT_COLOR, va='center', ha='left')
# Prod
fig.text(0.04, 0.18, r"$\mathbf{\prod}$ (Product Limit): Evaluates" + "\nsurvival as a continuous\nsequence of conditional\nprobabilities. The curve is a\nchain of independent events.", 
         fontsize=13, color=TEXT_COLOR, va='center', ha='left')

# Annotation lines
# S(t) line
fig.add_artist(plt.Line2D([0.15, 0.15, 0.23], [0.49, 0.65, 0.65], color=TEXT_COLOR, linewidth=1.5))
# d_i line
fig.add_artist(plt.Line2D([0.285, 0.285, 0.35], [0.49, 0.55, 0.55], color=TEXT_COLOR, linewidth=1.5))
# n_i line
fig.add_artist(plt.Line2D([0.285, 0.285, 0.33], [0.38, 0.22, 0.22], color=TEXT_COLOR, linewidth=1.5))
# Prod line
fig.add_artist(plt.Line2D([0.16, 0.16, 0.11], [0.37, 0.28, 0.28], color=TEXT_COLOR, linewidth=1.5))
fig.add_artist(plt.Line2D([0.11, 0.11], [0.28, 0.25], color=TEXT_COLOR, linewidth=1.5))

# -------------------------
# RIGHT SIDE: Plot
# -------------------------
fig.text(0.75, 0.78, 'The Step-Function Logic', fontsize=22, color=TEXT_COLOR, ha='center')

ax = fig.add_axes([0.58, 0.12, 0.38, 0.60])
ax.set_facecolor('none')

# The Step function data
t = [0, 1, 2, 3, 4, 5, 5.5]
S = [1.0, 1.0, 0.67, 0.53, 0.40, 0.27, 0.0]

ax.step(t, S, where='pre', color=LINE_COLOR, linewidth=4)
# Add drop lines
for i in range(1, len(t)-1):
    ax.plot([t[i], t[i]], [S[i-1], S[i]], color=LINE_COLOR, linewidth=4)

ax.set_xlim(0, 5.5)
ax.set_ylim(0, 1.05)

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['$t_1$', '$t_2$', '$t_3$', '$t_4$', '$t_5$'], fontsize=16)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.tick_params(colors=TEXT_COLOR, labelsize=14)
for spine in ax.spines.values():
    spine.set_color(TEXT_COLOR)
    spine.set_linewidth(1.5)

ax.set_xlabel('Time', fontsize=16, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel('Survival Probability', fontsize=16, color=TEXT_COLOR, labelpad=10)

# Grid
ax.grid(True, linestyle='-', color=TEXT_COLOR, alpha=0.3, linewidth=1)

# Text Box 1
box1_text = r"$\blacksquare$ No Parametric Assumptions: Unlike" + "\nWeibull distributions, Kaplan-Meier does\nnot force empirical data into a\npreconceived mathematical shape or\nassume an underlying hazard trend."
ax.text(1.5, 0.85, box1_text, fontsize=12, color=TEXT_COLOR, va='top',
        bbox=dict(facecolor=BOX_BG, edgecolor=TEXT_COLOR, boxstyle='square,pad=0.8', alpha=0.9))

# Text Box 2
box2_text = r"$\blacksquare$ The Empirical Staircase: The survival" + "\ncurve is a strict step function. It remains\nperfectly flat between intervals and\ndrops only when a physical failure occurs."
ax.text(0.5, 0.25, box2_text, fontsize=12, color=TEXT_COLOR, va='top',
        bbox=dict(facecolor=BOX_BG, edgecolor=TEXT_COLOR, boxstyle='square,pad=0.8', alpha=0.9))

plt.savefig('c:/Users/Isra/Downloads/data_Q3_2025_final2/weibull_project/python/kaplan_meier_transparent.png', dpi=300, transparent=True)
print('Saved Kaplan-Meier transparent slide to python/kaplan_meier_transparent.png')
