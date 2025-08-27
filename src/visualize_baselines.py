#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基线协议对比结果可视化
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_baseline_results(json_file: str) -> dict:
    """加载基线对比结果"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_comparison_plots(results: dict, output_dir: str):
    """创建对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    protocols = list(results.keys())
    
    # 1. 网络生命周期对比 (存活节点数随轮数变化)
    plt.figure(figsize=(12, 8))
    
    for protocol in protocols:
        rounds = [r['round'] for r in results[protocol]]
        alive_nodes = [r['alive_nodes'] for r in results[protocol]]
        plt.plot(rounds, alive_nodes, marker='o', linewidth=2, 
                label=f'{protocol}', markersize=4, alpha=0.8)
    
    plt.title('Network Lifetime Comparison\n(Alive Nodes over Rounds)', fontsize=16, fontweight='bold')
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Number of Alive Nodes', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'network_lifetime_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'network_lifetime_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 累积能耗对比
    plt.figure(figsize=(12, 8))
    
    for protocol in protocols:
        rounds = [r['round'] for r in results[protocol]]
        cumulative_energy = np.cumsum([r['energy_consumed'] for r in results[protocol]])
        plt.plot(rounds, cumulative_energy, marker='s', linewidth=2,
                label=f'{protocol}', markersize=4, alpha=0.8)
    
    plt.title('Cumulative Energy Consumption Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Cumulative Energy Consumed (J)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'cumulative_energy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cumulative_energy_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 关键指标汇总条形图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 计算关键指标
    metrics = {}
    for protocol in protocols:
        data = results[protocol]
        metrics[protocol] = {
            'total_energy': sum(r['energy_consumed'] for r in data),
            'total_rounds': len(data),
            'first_node_dead': next((r['round'] for r in data if r['alive_nodes'] < 100), len(data)),
            'avg_energy_per_round': sum(r['energy_consumed'] for r in data) / len(data)
        }
    
    # 绘制各项指标
    metric_names = ['Total Energy (J)', 'Network Lifetime (Rounds)', 
                   'First Node Death (Round)', 'Avg Energy/Round (J)']
    metric_keys = ['total_energy', 'total_rounds', 'first_node_dead', 'avg_energy_per_round']
    
    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[i // 2, i % 2]
        
        values = [metrics[protocol][metric_key] for protocol in protocols]
        bars = ax.bar(protocols, values, alpha=0.8, 
                     color=sns.color_palette("husl", len(protocols)))
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}' if value < 1 else f'{value:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Protocol Performance Metrics Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 能效分析 (每轮能耗分布)
    plt.figure(figsize=(12, 8))
    
    energy_data = []
    for protocol in protocols:
        energies = [r['energy_consumed'] for r in results[protocol]]
        energy_data.extend([(protocol, e) for e in energies])
    
    df = pd.DataFrame(energy_data, columns=['Protocol', 'Energy'])
    
    # 箱线图
    sns.boxplot(data=df, x='Protocol', y='Energy', palette="husl")
    plt.title('Energy Consumption Distribution per Round', fontsize=16, fontweight='bold')
    plt.xlabel('Protocol', fontsize=14)
    plt.ylabel('Energy Consumed per Round (J)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'energy_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'energy_distribution.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 网络剩余能量对比
    plt.figure(figsize=(12, 8))
    
    for protocol in protocols:
        rounds = [r['round'] for r in results[protocol]]
        remaining_energy = [r['remaining_energy'] for r in results[protocol]]
        plt.plot(rounds, remaining_energy, marker='^', linewidth=2,
                label=f'{protocol}', markersize=4, alpha=0.8)
    
    plt.title('Network Remaining Energy over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Remaining Energy (J)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'remaining_energy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'remaining_energy_comparison.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 基线对比图表已生成到: {output_dir}")
    
    # 打印关键指标汇总
    print("\n📊 关键指标汇总:")
    print("-" * 60)
    for protocol in protocols:
        m = metrics[protocol]
        print(f"{protocol:15} | 总能耗: {m['total_energy']:8.3f}J | "
              f"网络寿命: {m['total_rounds']:3d}轮 | "
              f"首节点死亡: {m['first_node_dead']:3d}轮")

def main():
    """主函数"""
    # 结果文件路径
    current_dir = Path(__file__).parent
    results_file = current_dir.parent / 'experiments' / 'results' / 'data' / 'baseline_protocols_comparison.json'
    output_dir = current_dir.parent / 'experiments' / 'results' / 'figures'
    
    if not results_file.exists():
        print(f"❌ 结果文件不存在: {results_file}")
        print("请先运行基线算法对比实验: python src/baseline_algorithms.py")
        return
    
    # 加载结果并生成图表
    results = load_baseline_results(str(results_file))
    create_comparison_plots(results, str(output_dir))

if __name__ == '__main__':
    main()
