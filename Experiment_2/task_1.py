import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

# 1. 读取数据文件
file_path = "Data.txt"  # 修改为你的本地路径或相对路径
data = pd.read_csv(
    file_path,
    sep="\s+",
    skiprows=1,
    header=None,
    names=["Index", "Length", "GC_Content"],
)
gc_values = data["GC_Content"].values
N = len(gc_values)

# 2. 有放回抽样构造模拟总体 X
sample_size = 30  # 每个样本容量 n
num_samples = 4289  # 模拟总体 X 的大小
sample_means = [
    np.mean(np.random.choice(gc_values, sample_size, replace=True))
    for _ in range(num_samples)
]

sample_means_df = pd.DataFrame(sample_means, columns=["Sample_Mean_GC"])

# 3. 标准化
X = (
    sample_means_df["Sample_Mean_GC"] - sample_means_df["Sample_Mean_GC"].mean()
) / sample_means_df["Sample_Mean_GC"].std()

# 4. 保存抽样均值结果
X.to_csv("simulated_X.csv", index=False)

# 5. K-S检验
ks_stat, p_value = kstest(X, "norm")

# 输出检验结果
print(f"K-S检验统计量: {ks_stat:.4f}")
print(f"p值: {p_value:.4f}")
if p_value > 0.01:
    print("不能拒绝原假设：模拟总体 X 可认为服从标准正态分布。")
else:
    print("拒绝原假设：模拟总体 X 不服从标准正态分布。")


# 6. 图示化模拟结果
plt.rcParams["font.sans-serif"] = ["STFangsong"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 6))
plt.hist(
    X,
    bins=50,
    density=True,
    edgecolor="black",
    alpha=0.6,
    color="skyblue",
    label="标准化模拟总体 X",
)
x_vals = np.linspace(-4, 4, 1000)
plt.plot(x_vals, norm.pdf(x_vals), "r--", label="标准正态分布 $N(0, 1^2)$")
plt.title("模拟总体 X 与标准正态分布对比")
plt.xlabel("值")
plt.ylabel("概率密度")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("simulated_X_histogram.png")
plt.show()
