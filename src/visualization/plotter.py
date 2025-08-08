"""
该模块负责结果的可视化
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def plot_results(result_file, figures_dir):
    """
    可视化实验结果

    Args:
        result_file (str): 结果文件的路径
        figures_dir (str): 保存图表的目录
    """
    # 加载结果
    with open(result_file, 'r') as f:
        results = json.load(f)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置图表样式
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 提取指标
    metrics = ['network_lifetime', 'packet_delivery_ratio', 'end_to_end_delay', 'prediction_accuracy', 'reliability']
    metric_names = ['网络生命周期', '数据包传递率', '端到端延迟', '预测准确性', '可靠性']

    # 创建比较图表
    for metric, metric_name in zip(metrics, metric_names):
        plt.figure(figsize=(10, 6))

        # 提取数据
        data = {protocol: results[protocol][metric] for protocol in results}

        # 绘制条形图
        ax = sns.barplot(x=list(data.keys()), y=list(data.values()))

        # 添加数值标签
        for i, v in enumerate(data.values()):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

        plt.title(f"不同路由协议的{metric_name}比较")
        plt.xlabel("路由协议")
        plt.ylabel(metric_name)

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{metric}_{timestamp}.png"))
        plt.close()

    # 创建能量消耗比较图表
    plt.figure(figsize=(10, 6))

    for protocol in results:
        if 'energy_consumption' in results[protocol] and results[protocol]['energy_consumption']:
            energy_data = results[protocol]['energy_consumption']
            plt.plot(energy_data, label=protocol)

    plt.title("不同路由协议的能量消耗比较")
    plt.xlabel("路由次数")
    plt.ylabel("能量消耗")
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"energy_consumption_{timestamp}.png"))
    plt.close()