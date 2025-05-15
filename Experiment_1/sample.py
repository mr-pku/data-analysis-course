import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 数据读取与预处理
def load_data(filename):
    """读取基因数据文件"""
    df = pd.read_csv(
        filename,
        sep="\s+",
        skiprows=1,
        header=None,
        names=["index", "长度", "gc含量"],
        usecols=[0, 1, 2],
    )
    return df[["长度", "gc含量"]]


# 2. 定义总体参数
def calculate_population_parameters(df):
    """计算总体参数"""
    total_r = df["gc含量"].sum()
    total_l = df["长度"].sum()
    theta_ratio = total_r / total_l  # 总体比值参数
    return theta_ratio


# 3. 抽样与估计函数
def bootstrap_estimates(df, n_samples=1000, sample_size=100):
    """执行Bootstrap抽样并计算两种估计量"""
    ratio_estimates = []
    single_estimates = []

    for _ in range(n_samples):
        # 有放回抽样
        sample = df.sample(n=sample_size, replace=True)

        # 比值估计量：Σr_i / Σl_i
        ratio_est = sample["gc含量"].sum() / sample["长度"].sum()
        ratio_estimates.append(ratio_est)

        # 单变量估计量：mean(r_i/l_i)
        single_est = (sample["gc含量"] / sample["长度"]).mean()
        single_estimates.append(single_est)

    return np.array(ratio_estimates), np.array(single_estimates)


# 4. 效率分析函数
def compare_efficiency(ratio_est, single_est, true_ratio):
    """比较估计量效率"""
    metrics = {
        "Ratio Estimator": {
            "Bias": np.mean(ratio_est - true_ratio),
            "Variance": np.var(ratio_est),
            "MSE": np.mean((ratio_est - true_ratio) ** 2),
        },
        "Single Estimator": {
            "Bias": np.mean(single_est - true_ratio),
            "Variance": np.var(single_est),
            "MSE": np.mean((single_est - true_ratio) ** 2),
        },
    }
    return pd.DataFrame(metrics)


# 5. 可视化函数
def plot_comparison(ratio_est, single_est, true_ratio):
    """绘制比较图"""
    plt.figure(figsize=(12, 6))

    # 箱线图比较
    plt.subplot(121)
    plt.boxplot(
        [ratio_est, single_est],
        labels=["比值估计", "单变量估计"],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )
    plt.axhline(true_ratio, color="green", linestyle="--", label="真实比值")
    plt.title("估计量分布比较")
    plt.ylabel("估计值")
    plt.legend()

    # 核密度估计图
    plt.subplot(122)
    from scipy.stats import gaussian_kde

    kde_ratio = gaussian_kde(ratio_est)
    kde_single = gaussian_kde(single_est)
    x_min = min(np.min(ratio_est), np.min(single_est))
    x_max = max(np.max(ratio_est), np.max(single_est))
    x = np.linspace(x_min, x_max, 1000)

    plt.plot(x, kde_ratio(x), label="比值估计", color="blue")
    plt.plot(x, kde_single(x), label="单变量估计", color="orange")
    plt.axvline(true_ratio, color="green", linestyle="--", alpha=0.5)
    plt.title("概率密度分布")
    plt.xlabel("估计值")
    plt.ylabel("密度")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载数据
    df = load_data("data.txt")

    # 计算总体参数
    theta_ratio = calculate_population_parameters(df)
    print(f"总体比值参数: {theta_ratio:.6f}")

    # 执行抽样估计
    ratio_est, single_est = bootstrap_estimates(df, n_samples=1000, sample_size=200)

    # 效率分析
    efficiency_df = compare_efficiency(ratio_est, single_est, theta_ratio)
    print("\n效率比较:")
    print(efficiency_df)

    # 可视化结果
    plot_comparison(ratio_est, single_est, theta_ratio)
