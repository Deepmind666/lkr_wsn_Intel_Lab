#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WSN-Intel-Lab-Project 主脚本

该脚本是项目的入口点，用于下载数据集、预处理数据、运行实验和可视化结果。
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

# 将项目根目录添加到sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset(args):
    """
    下载Intel Berkeley Lab数据集
    
    Args:
        args: 命令行参数
    """
    from src.utils.download_dataset import download_intel_lab_dataset, verify_dataset, print_dataset_info
    
    logger.info("开始下载Intel Berkeley Lab数据集")
    
    # 下载数据集
    download_path = os.path.join(os.path.dirname(__file__), 'data')
    success = download_intel_lab_dataset(download_path)
    
    if success:
        # 验证数据集
        if verify_dataset(download_path):
            logger.info("数据集验证成功")
            # 打印数据集信息
            print_dataset_info(download_path)
        else:
            logger.error("数据集验证失败")
    else:
        logger.error("数据集下载失败")

def preprocess_data(args):
    """
    预处理Intel Berkeley Lab数据集
    
    Args:
        args: 命令行参数
    """
    from src.utils.preprocess_data import preprocess_intel_lab_dataset
    
    logger.info("开始预处理Intel Berkeley Lab数据集")
    
    # 数据路径
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    
    # 预处理数据
    preprocess_intel_lab_dataset(
        raw_dir=data_path,
        processed_dir=output_path
    )

def run_experiment(args):
    """
    运行实验
    
    Args:
        args: 命令行参数
    """
    # 动态导入综合评估器
    from experiments.comprehensive_algorithm_evaluation import ComprehensiveEvaluator
    
    logger.info("开始运行综合算法评估实验")
    
    # 创建评估器实例
    evaluator = ComprehensiveEvaluator()
    
    # 运行评估
    evaluator.run_all_evaluations()

    # 保存结果
    evaluator.save_results()

    # 可视化结果
    evaluator.visualize_results()
    
    # 打印结果摘要
    logger.info("实验完成，结果已保存并可视化。")

def visualize_results(args):
    """
    可视化结果
    
    Args:
        args: 命令行参数
    """
    from src.visualization.plotter import plot_results

    # 结果目录
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'data')
    figures_dir = os.path.join(os.path.dirname(__file__), 'results', 'figures')
    
    # 检查结果文件
    if args.result_file and os.path.exists(args.result_file):
        result_file = args.result_file
    else:
        # 查找最新的结果文件
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not result_files:
            logger.error("未找到结果文件")
            return
        
        # 按修改时间排序
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        result_file = os.path.join(results_dir, result_files[0])
    
    logger.info(f"使用结果文件: {result_file}")
    
    # 可视化结果
    plot_results(result_file, figures_dir)
    plt.figure(figsize=(10, 8))
    
    # 准备雷达图数据
    categories = ['网络生命周期', '数据包传递率', '端到端延迟', '预测准确性', '可靠性']
    N = len(categories)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 初始化雷达图
    ax = plt.subplot(111, polar=True)
    
    # 设置雷达图的角度，用于平分切开一个圆
    plt.xticks(angles[:-1], categories)
    
    # 设置雷达图的范围
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # 绘制雷达图
    for protocol in results:
        # 归一化数据
        values = []
        for metric in metrics:
            value = results[protocol][metric]
            
            # 对于延迟，值越小越好，需要反转
            if metric == 'end_to_end_delay':
                max_delay = max([results[p]['end_to_end_delay'] for p in results])
                if max_delay > 0:
                    value = 1 - (value / max_delay)
                else:
                    value = 1
            
            # 归一化到0-1之间
            if metric != 'end_to_end_delay':
                max_value = max([results[p][metric] for p in results])
                if max_value > 0:
                    value = value / max_value
                else:
                    value = 0
            
            values.append(value)
        
        # 闭合数据
        values += values[:1]
        
        # 绘制折线图
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=protocol)
        ax.fill(angles, values, alpha=0.1)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("不同路由协议的综合性能比较")
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"radar_chart_{timestamp}.png"))
    plt.close()
    
    logger.info(f"可视化结果已保存到: {figures_dir}")

def main():
    """
    主函数
    """
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='WSN-Intel-Lab-Project 主脚本')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 下载数据集命令
    download_parser = subparsers.add_parser('download', help='下载Intel Berkeley Lab数据集')
    
    # 预处理数据命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理Intel Berkeley Lab数据集')
    preprocess_parser.add_argument('--normalize', choices=['minmax', 'standard', 'none'], default='minmax', help='归一化方法')
    preprocess_parser.add_argument('--fill-missing', choices=['interpolate', 'mean', 'median', 'none'], default='interpolate', help='缺失值填充方法')
    preprocess_parser.add_argument('--generate-features', action='store_true', help='是否生成特征')
    preprocess_parser.add_argument('--visualize', action='store_true', help='是否可视化数据')
    
    # 运行实验命令
    experiment_parser = subparsers.add_parser('experiment', help='运行实验')
    experiment_parser.add_argument('--experiment', choices=['routing_comparison'], default='routing_comparison', help='实验类型')
    experiment_parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 可视化结果命令
    visualize_parser = subparsers.add_parser('visualize', help='可视化结果')
    visualize_parser.add_argument('--result-file', type=str, help='结果文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行命令
    if args.command == 'download':
        download_dataset(args)
    elif args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'visualize':
        visualize_results(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()