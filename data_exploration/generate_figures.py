import pandas as pd
import matplotlib.pyplot as plt
import io
import os

data_path = "../data/drives_clean.csv"
print(f"Loading data from: {data_path}")
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)
    
df = pd.read_csv(data_path)

# Figure 3.1: Raw Data Structure
print("Generating figure3_1.png (raw data structure)...")
buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
ax.text(0.05, 0.95, info_str, ha='left', va='top', fontname='monospace', fontsize=8, transform=ax.transAxes)
plt.tight_layout()
plt.savefig("figure3_1.png", dpi=300, bbox_inches='tight')
plt.close()

# Figure 3.2: Failure Counts Distribution
print("Generating figure3_2.png (failure counts distribution)...")
failures = df[df['failed'] == 1]
failure_counts = failures['model'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
failure_counts.plot(kind='bar', ax=ax, edgecolor='black', color='steelblue')
ax.set_title("Failure Counts Distribution by Hard Drive Model")
ax.set_xlabel("Drive Model")
ax.set_ylabel("Number of Failures")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("figure3_2.png", dpi=300, bbox_inches='tight')
plt.close()

# Figure 3.3: Histogram of Failure Lifetimes
print("Generating figure3_3.png (histogram of failure lifetimes)...")
lifetime_col = 'lifetime_days' 
if lifetime_col in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(failures[lifetime_col].dropna(), bins=40, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title("Histogram of Failure Lifetimes")
    ax.set_xlabel("Lifetime (days)")
    ax.set_ylabel("Frequency (Number of Failures)")
    plt.tight_layout()
    plt.savefig("figure3_3.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("Could not find lifetime column. Skipping Figure 3.3.")
    
print("All figures generated successfully.")
