import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2


n = 61
reps = 100
alpha = 0.10
t_crit = t.ppf(1 - alpha / 2, df=n - 1)

# 读取上一步生成的标准正态模拟总体 X

data = pd.read_csv(
    "simulated_X.csv",
    sep="\s+",
    skiprows=1,
    header=None,
    names=["X"],
)
X = data["X"].values


# 存储统计量和置信区间

means = []
stds = []
ci_mu = []
ci_std = []

# 100 次抽样 + 区间估计

for _ in range(reps):
    sample = np.random.choice(X, size=n, replace=False)
    xbar = np.mean(sample)
    s = np.std(sample, ddof=1)

    # 样本统计
    means.append(xbar)
    stds.append(s)

    # μ 的 90% CI（正态统计量）
    delta = t_crit * s / np.sqrt(n)
    ci_mu.append((xbar - delta, xbar + delta))

    # σ 的 90% CI（使用卡方分布，针对样本标准差 s）
    chi2_lower = chi2.ppf(1 - alpha / 2, df=n - 1)
    chi2_upper = chi2.ppf(alpha / 2, df=n - 1)
    lower = np.sqrt((n - 1) * s**2 / chi2_lower)
    upper = np.sqrt((n - 1) * s**2 / chi2_upper)
    ci_std.append((lower, upper))

means = np.array(means)
stds = np.array(stds)

# 直方图（均值 & 标准差）

plt.rcParams["font.sans-serif"] = ["STFangsong"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 均值
axes[0, 0].hist(means, bins=15, edgecolor="black")
axes[0, 0].set_title("样本均值的频数直方图")
axes[0, 1].hist(means, bins=15, density=True, edgecolor="black")
axes[0, 1].set_title("样本均值的频率直方图")

# 标准差
axes[1, 0].hist(stds, bins=15, edgecolor="black")
axes[1, 0].set_title("样本标准差的频数直方图")
axes[1, 1].hist(stds, bins=15, density=True, edgecolor="black")
axes[1, 1].set_title("样本标准差的频率直方图")

plt.tight_layout()
plt.show()

# 参数估计值
esti_mu = np.mean(sample)
esti_sigma = np.std(sample, ddof=0)

# μ 的置信区间图

fig, ax = plt.subplots(figsize=(10, 6))
true_mu = 0
ax.hlines(
    true_mu, xmin=0.5, xmax=reps + 0.5, colors="gray", linestyles="--", label="True μ"
)
ax.hlines(
    esti_mu,
    xmin=0.5,
    xmax=reps + 0.5,
    colors="blue",
    linestyles="--",
    label="Estimated μ",
)
for i, (low, high) in enumerate(ci_mu, start=1):
    _color = "black" if low <= true_mu <= high else "red"
    ax.vlines(i, low, high, colors=_color, linewidth=1)
ax.set_title(f"{reps} 次基于正态分布的 90% CI for μ (n={n})")
ax.set_xlabel("重复次数")
ax.set_ylabel("μ 的置信区间")
ax.set_xlim(0.5, reps + 0.5)
ax.legend()
plt.tight_layout()
plt.show()

# σ 的置信区间图

fig, ax = plt.subplots(figsize=(10, 6))
true_sigma = 1.0
ax.hlines(
    true_sigma,
    xmin=0.5,
    xmax=reps + 0.5,
    colors="gray",
    linestyles="--",
    label="True σ",
)
ax.hlines(
    esti_sigma,
    xmin=0.5,
    xmax=reps + 0.5,
    colors="blue",
    linestyles="--",
    label="Estimated σ",
)
for i, (low, high) in enumerate(ci_std, start=1):
    _color = "black" if low <= true_sigma <= high else "red"
    ax.vlines(i, low, high, colors=_color, linewidth=1)
ax.set_title(f"{reps} 次基于卡方分布的 90% CI for σ (n={n})")
ax.set_xlabel("重复次数")
ax.set_ylabel("σ 的置信区间")
ax.set_xlim(0.5, reps + 0.5)
ax.legend()
plt.tight_layout()
plt.show()
