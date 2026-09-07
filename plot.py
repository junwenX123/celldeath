import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("results.txt", sep=r"\s+", skiprows=2)

# 获取所有不同的时间段 k (1 到 20)
K_values = sorted(df["k"].unique())
cols = 5
rows = math.ceil(len(K_values) / cols)

# 创建多子图网格 (4行5列)
fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, k_val in enumerate(K_values):
    ax = axes[idx]
    df_k = df[df["k"] == k_val]
    
    for alg in ["A3_local", "A4_full"]:
        sub = df_k[df_k["algorithm"] == alg].sort_values("N")
        ax.errorbar(sub["N"], sub["bias"], yerr=sub["paired_MCSE"], marker="o", label=alg, capsize=3)

    ax.axhline(0, color="k", linestyle="--")
    ax.set_xscale("log")
    ax.set_title(f"k = {k_val}")
    
    # 设置横纵轴标签
    if idx >= len(K_values) - cols:
        ax.set_xlabel("N")
    if idx % cols == 0:
        ax.set_ylabel("Bias")

# 添加全局图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig("plot_all_K.png", bbox_inches='tight')
plt.show()