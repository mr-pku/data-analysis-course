import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 1. 读取数据
data = pd.read_csv(
    "data.txt",
    sep="\s+",
    skiprows=1,
    header=None,
    names=["Index", "Length", "GC_Content"],
)

# 2. 分箱（可以调整 bins 的个数）
l_bins = pd.qcut(data["Length"], q=5, labels=False)
r_bins = pd.qcut(data["GC_Content"], q=5, labels=False)

# 3. 构建列联表
contingency_table = pd.crosstab(l_bins, r_bins)

# 4. Pearson 卡方独立性检验
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 5. 输出结果
alpha = 0.1
print("列联表：")
print(contingency_table)
print(f"\n卡方统计量 χ² = {chi2:.4f}")
print(f"自由度 = {dof}")
print(f"p 值 = {p:.4f}")
if p < alpha:
    print("结论：拒绝 H0，l 与 r 不独立")
else:
    print("结论：不能拒绝 H0，l 与 r 可以认为是独立的")


import seaborn as sns
import matplotlib.pyplot as plt

lengths = data["Length"]
gc_content = data["GC_Content"]

# 绘制散点图与线性回归拟合线
plt.rcParams["font.sans-serif"] = ["STFangsong"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(8, 6))
sns.regplot(
    x=lengths,
    y=gc_content,
    scatter_kws={"s": 10, "alpha": 0.5},
    line_kws={"color": "red"},
)
plt.title("基因长度与 GC 含量的关系")
plt.xlabel("基因长度 l (nt)")
plt.ylabel("GC 含量 r (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
