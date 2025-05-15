import pandas as pd
import numpy as np


# 读取数据
def read_data(filename):
    df = pd.read_csv(
        filename,
        sep="\s+",
        skiprows=1,
        header=None,
        names=["序号", "长度", "GC含量"],
        usecols=[0, 1, 2],
    )
    return df


# 检验Chebyshev定理
def chebyshev_test(data, data_name):
    mu = np.mean(data)
    sigma = np.std(data)
    total = len(data)
    results = []

    for k in range(2, 11):
        lower = mu - k * sigma
        upper = mu + k * sigma
        count = np.sum((data >= lower) & (data <= upper))
        actual_ratio = count / total
        theory_ratio = 1 - 1 / (k**2)
        is_satisfied = actual_ratio >= theory_ratio
        results.append(
            {
                "k": k,
                "理论下限": theory_ratio,
                "实际比例": actual_ratio,
                "是否满足": is_satisfied,
            }
        )
    return pd.DataFrame(results), mu, sigma


# 主程序
df = read_data("data.txt")
length_results, mu_l, sigma_l = chebyshev_test(df["长度"], "基因长度")
gc_results, mu_r, sigma_r = chebyshev_test(df["GC含量"], "GC含量")
print(length_results)
print(gc_results)
# print(len(df["长度"]), mu_l, sigma_l)
# print(len(df["GC含量"]), mu_r, sigma_r)
