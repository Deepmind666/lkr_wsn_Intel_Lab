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
    # 基于参数选择实验
    exp = getattr(args, 'experiment', 'routing_comparison')
    if exp == 'routing_comparison':
        logger.info("运行路由协议比较实验 (ACO vs Baseline, SoD on/off)")
        from experiments.routing_comparison import main as rc_main
        # 传递配置文件参数（如果有的话）
        config_path = getattr(args, 'config', None)
        if config_path:
            logger.info(f"使用配置文件: {config_path}")
            # TODO: 在routing_comparison中实现配置文件读取
        rc_main()
        logger.info("routing_comparison 实验完成")
        return
    elif exp == 'comprehensive':
        from experiments.comprehensive_algorithm_evaluation import ComprehensiveEvaluator
        logger.info("开始运行综合算法评估实验 (AFW-RL / GNN-CTO / ILMR / EEHFR)")
        # Intel Lab真实规模：54节点，500轮评估
        evaluator = ComprehensiveEvaluator(network_size=54, area_size=(25, 25))
        evaluator.run_all_evaluations()
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(project_root, 'results', 'data', f'comprehensive_results_{timestamp}.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        evaluator.save_detailed_results(out_path)
        # 可视化结果
        evaluator.visualize_results('comprehensive_algorithm_comparison.png')
        logger.info("综合评估实验完成，结果已保存并可视化。")
        return
    elif exp == 'baseline_comparison':
        logger.info("运行基线协议对比实验 (LEACH vs HEED vs DirectTransmission)")
        from src.baseline_algorithms import run_protocol_comparison
        from src.visualize_baselines import main as visualize_main
        
        # 运行基线对比（Intel Lab真实规模：54节点，1000轮）
        results = run_protocol_comparison(num_nodes=54, num_rounds=1000)
        logger.info("基线协议对比实验完成")
        
        # 自动生成可视化
        logger.info("生成基线对比可视化图表...")
        visualize_main()
        logger.info("基线对比实验与可视化完成")
        return
    else:
        logger.error(f"未知实验类型: {exp}")
        raise SystemExit(2)

def visualize_results(args):
    """
    可视化结果
    
    Args:
        args: 命令行参数
    """
    from src.visualization.plotter import plot_results
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # 结果目录
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'data')
    figures_dir = os.path.join(os.path.dirname(__file__), 'experiments', 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 检查结果文件
    result_file = None
    if args.result_file and os.path.exists(args.result_file):
        result_file = args.result_file
    else:
        # 优先在 results/data 下找 JSON
        jsons_primary = sorted(glob.glob(os.path.join(results_dir, '*.json')), key=os.path.getmtime, reverse=True)
        if jsons_primary:
            result_file = jsons_primary[0]
        else:
            # 回退到 experiments/results/data 下找 JSON
            exp_data_dir = os.path.join(os.path.dirname(__file__), 'experiments', 'results', 'data')
            jsons_fallback = sorted(glob.glob(os.path.join(exp_data_dir, '*.json')), key=os.path.getmtime, reverse=True)
            if jsons_fallback:
                result_file = jsons_fallback[0]
            else:
                # 再次回退寻找 CSV（routing_comparison 产物）
                csvs = sorted(glob.glob(os.path.join(exp_data_dir, '*.csv')), key=os.path.getmtime, reverse=True)
                if csvs:
                    result_file = csvs[0]
                else:
                    logger.error("未找到结果文件 (JSON/CSV)")
                    return
    
    logger.info(f"使用结果文件: {result_file}")
    
    # 根据文件类型可视化
    if result_file.lower().endswith('.json'):
        plot_results(result_file, figures_dir)
        logger.info(f"JSON结果图表已生成到: {figures_dir}")
    elif result_file.lower().endswith('.csv'):
        # 生成 routing_comparison 的条形图（与脚本一致）
        df = pd.read_csv(result_file)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sns.set(style="whitegrid")
        # 聚合 mean±std
        agg = df.groupby(['router', 'sod']).agg(
            te_mean=('total_energy', 'mean'), te_std=('total_energy', 'std'),
            alive_mean=('final_alive', 'mean'), alive_std=('final_alive', 'std')
        ).reset_index()
        routers = ['aco', 'baseline']
        sods = ['off', 'on']
        x = np.arange(len(routers))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, metric in enumerate([('te_mean','te_std','Total Energy'), ('alive_mean','alive_std','Final Alive')]):
            mean_col, std_col, title = metric
            vals_off = [agg[(agg.router==r) & (agg.sod=='off')][mean_col].values[0] for r in routers]
            err_off  = [agg[(agg.router==r) & (agg.sod=='off')][std_col].values[0] for r in routers]
            vals_on  = [agg[(agg.router==r) & (agg.sod=='on')][mean_col].values[0]  for r in routers]
            err_on   = [agg[(agg.router==r) & (agg.sod=='on')][std_col].values[0]  for r in routers]
            axes[i].bar(x-width/2, vals_off, width, yerr=err_off, capsize=4, label='SoD Off')
            axes[i].bar(x+width/2, vals_on,  width, yerr=err_on,  capsize=4, label='SoD On')
            axes[i].set_xticks(x); axes[i].set_xticklabels([r.upper() for r in routers])
            axes[i].set_title(title)
            axes[i].legend()
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, f'routing_comparison_{ts}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"CSV结果图表已生成: {fig_path}")
    else:
        logger.error(f"未知结果文件类型: {result_file}")

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
    experiment_parser.add_argument('--experiment', choices=['routing_comparison', 'comprehensive', 'baseline_comparison'], default='routing_comparison', help='实验类型')
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