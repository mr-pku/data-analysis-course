import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------
# 1. 读取上一步生成的标准正态模拟总体 X
# ------------------------------
# np.random.seed(42)
data = pd.read_csv(
    "simulated_X.csv",
    sep="\s+",
    skiprows=1,
    header=None,
    names=["X"],
)
X = data["X"].values

# ------------------------------
# 2. 从 X 中做一次简单随机抽样
# ------------------------------
n = 277
sample = np.random.choice(X, n, replace=False)

# ------------------------------
# 3. 样本统计量 & 图示
# ------------------------------
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)

print(f"样本均值 μ̂ = {sample_mean:.4f}")
print(f"样本标准差 σ̂ = {sample_std:.4f}")


plt.rcParams["font.sans-serif"] = ["STFangsong"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
# 3.1 频数直方图
axes[0].hist(sample, edgecolor="black", bins=25)
axes[0].set_title("样本频数直方图 (n=277)")
axes[0].set_xlabel("X 值")
axes[0].set_ylabel("频数")
# 3.2 频率直方图（密度）
axes[1].hist(sample, edgecolor="black", bins=25, density=True)
axes[1].set_title("样本频率直方图 (密度)")
axes[1].set_xlabel("X 值")
axes[1].set_ylabel("频率密度")
# 3.3 箱线图
axes[2].boxplot(sample, vert=False)
axes[2].set_title("样本箱线图")
axes[2].set_xlabel("X 值")
plt.tight_layout()
plt.show()

# ------------------------------
# 4. 参数估计
# ------------------------------

esti_mean = np.mean(sample)
esti_std = np.std(sample, ddof=0)

# ------------------------------
# 5. 自举法 (Bootstrap)
# ------------------------------
B = 1000
boot_means = np.array(
    [np.mean(np.random.choice(sample, n, replace=True)) for _ in range(B)]
)
boot_stds = np.array(
    [np.std(np.random.choice(sample, n, replace=True), ddof=1) for _ in range(B)]
)

# 6. 90% CI（上下5%分位数）
ci_mean = np.percentile(boot_means, [5, 95])
ci_std = np.percentile(boot_stds, [5, 95])

# ------------------------------
# 7. 可视化 Bootstrap 分布
# ------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 10))
# 均值分布
axes[0].hist(boot_means, bins=30, density=True, alpha=0.7)
axes[0].axvline(ci_mean[0], color="red", linestyle="--", label="5% 分位")
axes[0].axvline(ci_mean[1], color="red", linestyle="--", label="95% 分位")
axes[0].axvline(esti_mean, color="blue", linestyle="-", label="样本 μ̂")
axes[0].set_title("Bootstrap 样本均值分布 & 90% CI")
axes[0].legend()
# 标准差分布
axes[1].hist(boot_stds, bins=30, density=True, alpha=0.7)
axes[1].axvline(ci_std[0], color="red", linestyle="--", label="5% 分位")
axes[1].axvline(ci_std[1], color="red", linestyle="--", label="95% 分位")
axes[1].axvline(esti_std, color="blue", linestyle="-", label="样本 σ̂")
axes[1].set_title("Bootstrap 样本标准差分布 & 90% CI")
axes[1].legend()
plt.tight_layout()
plt.show()

# ------------------------------
# 8. 置信区间覆盖判断
# ------------------------------
mean_in = ci_mean[0] <= sample_mean <= ci_mean[1]
std_in = ci_std[0] <= sample_std <= ci_std[1]
print(f"样本均值在 90% CI 内: {mean_in}，CI = [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")
print(f"样本标准差在 90% CI 内: {std_in}，CI = [{ci_std[0]:.4f}, {ci_std[1]:.4f}]")
