import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

results_file = sys.argv[1] if len(sys.argv) > 1 else "results.txt"
output_file = sys.argv[2] if len(sys.argv) > 2 else "plot.png"

df = pd.read_csv(results_file, sep=r"\s+", skiprows=1)

K_values = sorted(df["k"].unique())
cols = 5
rows = math.ceil(len(K_values) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, k_val in enumerate(K_values):
    ax = axes[idx]
    df_k = df[df["k"] == k_val]
    
    for alg in ["A3_local", "A4_full", "A5_empirical_optimal"]:
        sub = df_k[df_k["algorithm"] == alg].sort_values("N")
        ax.errorbar(sub["N"], sub["bias"], yerr=sub["paired_MCSE"], marker="o", label=alg, capsize=3)

    ax.axhline(0, color="k", linestyle="--")
    ax.set_xscale("log")
    ax.set_title(f"k = {k_val}")
    
    if idx >= len(K_values) - cols:
        ax.set_xlabel("N")
    if idx % cols == 0:
        ax.set_ylabel("Bias")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.995))

plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(output_file, bbox_inches='tight')
plt.show()
