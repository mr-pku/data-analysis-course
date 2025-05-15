import pandas as pd
import numpy as np
from scipy import stats

"""
# 1. 构建示例DataFrame（实际应用时可用pd.read_csv()读取文件）
data = {
    "序号": [1, 2, 3, 4, 5],
    "长度": [66, 2463, 933, 1287, 297],
    "GC含量": [51.52, 53.07, 56.27, 52.84, 53.87],
}
df = pd.DataFrame(data)
"""

df = pd.read_csv(
    "data.txt", delim_whitespace=True, names=["序号", "长度", "GC含量"], header=0
)

# 2. 提取需要分析的列
gene_lengths = df["长度"].values
gc_contents = df["GC含量"].values


# 2. 定义计算统计量的函数
def calculate_statistics(series, name):
    stats_dict = {
        "均值": np.mean(series),
        "标准差": np.std(series, ddof=0),  # 总体标准差用ddof=0
        "极差": np.ptp(series),  # peak-to-peak
        "中位数": np.median(series),
        "众数": stats.mode(series, keepdims=True)[0][0],  # 取第一个众数
        "变异系数": np.std(series, ddof=0) / np.mean(series) * 100,  # 百分比形式
        "最大值": np.max(series),
        "最小值": np.min(series),
    }
    return pd.DataFrame(stats_dict.items(), columns=["统计量", name])


# 3. 分别计算长度和GC含量的统计量
length_stats = calculate_statistics(df["长度"], "基因长度(NT)")
gc_stats = calculate_statistics(df["GC含量"], "GC含量(%)")

# 4. 合并结果并格式化输出
result = pd.merge(length_stats, gc_stats, on="统计量")
pd.set_option("display.float_format", "{:.2f}".format)  # 统一保留2位小数

print("=" * 50)
print("基因特征统计分析结果")
print("=" * 50)
print(result.to_string(index=False))
print("\n注：变异系数已转换为百分比形式")
