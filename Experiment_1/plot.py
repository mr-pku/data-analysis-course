import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

# 设置中文字体
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False


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


def plot_freq_hist(data, title, xlabel, bins, xlim=None):
    # 计算频数
    counts, bin_edges = np.histogram(data, bins=bins)

    # 计算分箱中心位置
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    total = len(data)
    freqs = counts / total / bin_width  # 转换为比例

    plt.figure(figsize=(10, 6))
    plt.bar(
        bin_centers,
        freqs,
        width=bin_width,  # 控制柱宽
        edgecolor="black",
        color="#2c7fb8",
        alpha=0.8,
    )

    plt.title(f"{title}频率分布直方图", fontsize=14)  # 标题修改
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("频率/柱宽", fontsize=12)  # y轴标签修改
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if xlim:
        plt.xlim(xlim)

    # 动态调整刻度
    # plt.xticks(np.arange(min(bins), max(bins) + 1, (max(bins) - min(bins)) // 10))
    plt.tight_layout()
    plt.show()


# 绘制箱线图
def plot_box(data, title, vert=False):
    plt.figure(figsize=(10, 6))
    boxprops = dict(linestyle="-", linewidth=1.5, color="darkblue")
    medianprops = dict(linestyle="-", linewidth=2, color="firebrick")

    plt.boxplot(
        data,
        vert=vert,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        flierprops=dict(marker="o", markersize=4),
        widths=0.6,
    )

    plt.title(f"{title}箱线图", fontsize=14)
    plt.grid(axis="y" if vert else "x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = read_data("data.txt")

    # 基因长度分箱（保持原策略）
    length_bins = [
        0,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
        6500,
        7000,
        7500,
    ]

    # GC含量分箱（保持原策略）
    gc_bins = np.arange(25, 75, 2.5)

    # 绘制频率直方图
    plot_freq_hist(
        df["长度"], "基因长度", "核苷酸数量(NT)", bins=length_bins, xlim=(0, 7500)
    )
    plot_freq_hist(
        df["GC含量"], "GC含量", "GC含量百分比(%)", bins=gc_bins, xlim=(25, 75)
    )

    # 绘制箱线图
    plot_box(df["长度"], "基因长度")
    plot_box(df["GC含量"], "GC含量")
