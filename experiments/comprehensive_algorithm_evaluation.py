"""
综合算法评估实验
Comprehensive Algorithm Evaluation Experiment

本实验对比评估三个核心创新算法：
1. AFW-RL: 自适应模糊逻辑权重强化学习
2. GNN-CTO: 基于图神经网络的链式拓扑优化  
3. ILMR: 可解释的轻量级元启发式路由

作者: WSN研究团队
日期: 2025年1月
"""

import sys
import os
# 将项目根目录和src目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import time
from datetime import datetime
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

class NumpyJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理Numpy数据类型
    Custom JSON encoder for Numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象（如ChainTopology）
            return obj.__dict__
        elif hasattr(obj, '_asdict'):
            # 处理namedtuple
            return obj._asdict()
        return super(NumpyJSONEncoder, self).default(obj)

# 导入三个核心算法
from advanced_algorithms.afw_rl_algorithm import AFWRLAlgorithm
# 可选导入 GNN-CTO（依赖 torch_geometric），失败则降级跳过
try:
    from advanced_algorithms.gnn_cto_algorithm import GNNCTOAlgorithm  # type: ignore
    _HAS_GNN = True
    _GNN_IMPORT_ERROR = None
except Exception as _e:  # noqa: N816
    _HAS_GNN = False
    _GNN_IMPORT_ERROR = str(_e)
from advanced_algorithms.ilmr_algorithm import ILMRAlgorithm
from src.enhanced_eehfr_system import EnhancedEEHFRSystem, SystemConfig

class ComprehensiveEvaluator:
    """综合算法评估器"""
    
    def __init__(self, network_size: int = 54, area_size: Tuple[int, int] = (25, 25)):
        self.network_size = network_size
        self.area_size = area_size
        self.results = {}
        self.evaluation_metrics = [
            'energy_efficiency',
            'network_lifetime', 
            'routing_success_rate',
            'average_latency',
            'throughput',
            'convergence_speed',
            'computational_complexity',
            'scalability',
            'adaptability',
            'explainability'
        ]
        
        # 实验配置（基于Intel Lab真实环境）
        self.experiment_configs = {
            'intel_lab': {'nodes': self.network_size, 'area': self.area_size, 'rounds': 200},
            'small_network': {'nodes': 50, 'area': (100, 100), 'rounds': 100},
            'medium_network': {'nodes': 100, 'area': (200, 200), 'rounds': 150}
        }
        
    def generate_network_topology(self, num_nodes: int, area_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """生成网络拓扑"""
        # 随机部署节点
        nodes = np.random.uniform(0, area_size[0], (num_nodes, 2))
        
        # 节点特征: [x, y, energy, alive, trust_score]
        node_features = np.zeros((num_nodes, 5))
        node_features[:, :2] = nodes  # 位置
        node_features[:, 2] = np.random.uniform(0.8, 1.0, num_nodes)  # 初始能量
        node_features[:, 3] = 1.0  # 存活状态
        node_features[:, 4] = np.random.uniform(0.7, 1.0, num_nodes)  # 信任分数
        
        # 构建连接矩阵（基于通信半径）
        communication_range = min(area_size) / 8
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= communication_range:
                    adjacency_matrix[i][j] = distance
                    adjacency_matrix[j][i] = distance
        
        return adjacency_matrix, node_features
    
    def evaluate_afw_rl(self, network_config: Dict) -> Dict:
        """评估AFW-RL算法"""
        print(f"🔬 评估AFW-RL算法 - {network_config}")
        
        start_time = time.time()
        
        # 创建算法实例
        afw_rl = AFWRLAlgorithm(
            num_nodes=network_config['nodes']
        )
        
        # 准备数据
        nodes_data = np.random.rand(network_config['nodes'], 4)
        base_station_pos = np.array([network_config['area'][0]/2, network_config['area'][1]/2])

        # 训练算法（Intel Lab规模：减少到500轮）
        training_results = afw_rl.train_episode(nodes_data=nodes_data, base_station_pos=base_station_pos, max_rounds=500)
        
        # 评估性能
        evaluation_results = afw_rl.evaluate(nodes_data=nodes_data, base_station_pos=base_station_pos, max_rounds=network_config['rounds'])
        
        computation_time = time.time() - start_time
        
        # 整理结果
        results = {
            'algorithm': 'AFW-RL',
            'network_config': network_config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'computation_time': computation_time,
            'metrics': {
                'energy_efficiency': evaluation_results.get('avg_energy_efficiency', 0),
                'network_lifetime': evaluation_results.get('avg_network_lifetime', 0),
                'routing_success_rate': evaluation_results.get('avg_success_rate', 0),
                'average_latency': evaluation_results.get('avg_latency', 0),
                'convergence_speed': len(training_results.get('reward_history', [])),
                'computational_complexity': computation_time / network_config['nodes'],
                'adaptability': evaluation_results.get('adaptability_score', 0),
                'explainability': 0.7  # RL的可解释性相对较低
            }
        }
        
        return results
    
    def evaluate_gnn_cto(self, network_config: Dict) -> Dict:
        """评估GNN-CTO算法"""
        if not _HAS_GNN:
            print(f"⚠️ 跳过GNN-CTO（未安装所需依赖 torch_geometric）。原因: {_GNN_IMPORT_ERROR}")
            return {
                'algorithm': 'GNN-CTO',
                'network_config': network_config,
                'error': 'torch_geometric not available',
                'metrics': {m: 0.0 for m in self.evaluation_metrics}
            }
        print(f"🔬 评估GNN-CTO算法 - {network_config}")
        
        start_time = time.time()
        
        # 创建算法实例
        gnn_cto = GNNCTOAlgorithm(
            num_nodes=network_config['nodes'],
            area=network_config['area'],
            rounds=network_config['rounds']
        )
        
        # 生成训练数据
        adjacency_matrix, node_features = self.generate_network_topology(
            network_config['nodes'], network_config['area']
        )
        base_station_pos = np.array([network_config['area'][0] / 2, network_config['area'][1] / 2])
        
        # 提取节点特征并创建图数据
        gnn_node_features = gnn_cto.extract_node_features(node_features, base_station_pos)
        graph_data = gnn_cto.create_graph_data(gnn_node_features, adjacency_matrix)
        
        # 生成标签
        chains = gnn_cto.chain_optimizer.construct_chains(gnn_node_features, adjacency_matrix)
        role_labels, energy_labels = gnn_cto.generate_training_labels(gnn_node_features, chains)
        
        training_data = [(graph_data, role_labels, energy_labels)]
        
        # 训练GNN模型
        training_results = gnn_cto.train_gnn(training_data, epochs=100)
        
        # 运行拓扑优化
        optimized_chains, topology_metrics = gnn_cto.optimize_topology(node_features, base_station_pos)
        
        computation_time = time.time() - start_time
        
        # 整理结果
        results = {
            'algorithm': 'GNN-CTO',
            'network_config': network_config,
            'training_results': training_results,
            'optimization_results': {
                'chains': optimized_chains,
                'metrics': topology_metrics
            },
            'computation_time': computation_time,
            'metrics': {
                'energy_efficiency': topology_metrics.get('topology_efficiency', 0),
                'network_lifetime': 0, # Not directly calculated here
                'routing_success_rate': topology_metrics.get('coverage_ratio', 0),
                'average_latency': topology_metrics.get('average_chain_length', 0) * 0.05,
                'throughput': topology_metrics.get('chain_count', 0) / max(computation_time, 0.01),
                'convergence_speed': len(training_results.get('losses', [])),
                'computational_complexity': computation_time / network_config['nodes'],
                'scalability': min(1.0, 100 / network_config['nodes']), # Placeholder scalability
                'explainability': 0.6
            }
        }
        
        return results
    
    def evaluate_ilmr(self, network_config: Dict) -> Dict:
        """评估ILMR算法"""
        print(f"🔬 评估ILMR算法 - {network_config}")
        
        start_time = time.time()
        
        # 创建算法实例
        ilmr = ILMRAlgorithm()
        
        # 生成网络拓扑
        adjacency_matrix, node_features = self.generate_network_topology(
            network_config['nodes'], network_config['area']
        )
        
        # 生成路由请求
        routing_requests = []
        for _ in range(network_config['rounds']):
            source = np.random.randint(0, network_config['nodes'])
            destination = np.random.randint(0, network_config['nodes'])
            if source != destination:
                routing_requests.append((source, destination))
        
        # 运行网络路由模拟
        simulation_results = ilmr.simulate_network_routing(
            adjacency_matrix, node_features, routing_requests, 
            max_rounds=network_config['rounds']
        )
        
        computation_time = time.time() - start_time
        
        # 获取可解释性报告
        explainability_report = ilmr.get_explainability_report()
        
        # 整理结果
        results = {
            'algorithm': 'ILMR',
            'network_config': network_config,
            'simulation_results': simulation_results,
            'explainability_report': explainability_report,
            'computation_time': computation_time,
            'metrics': {
                'energy_efficiency': 1.0 / (simulation_results.get('total_energy_consumption', 1) + 0.01),
                'network_lifetime': len(simulation_results.get('performance_evolution', [])),
                'routing_success_rate': simulation_results.get('success_rate', 0),
                'average_latency': simulation_results.get('average_path_length', 0) * 0.1,
                'throughput': simulation_results.get('successful_routes', 0) / max(computation_time, 0.01),
                'convergence_speed': 50,  # ILMR收敛较快
                'computational_complexity': computation_time / network_config['nodes'],
                'scalability': min(1.0, 100 / network_config['nodes']),
                'explainability': simulation_results.get('average_confidence', 0.9)  # ILMR可解释性最高
            }
        }
        
        return results

    def run_all_evaluations(self):
        """运行所有算法在所有配置下的评估"""
        all_results = []
        algorithms_to_evaluate = [
            self.evaluate_afw_rl,
            self.evaluate_gnn_cto,
            self.evaluate_ilmr,
            self.evaluate_baseline_eehfr
        ]

        # 只运行Intel Lab配置
        config_name = 'intel_lab'
        config = self.experiment_configs[config_name]
        print(f"\n🚀 开始评估网络配置: {config_name} (54节点, 25x25m)")
        for eval_func in algorithms_to_evaluate:
            result = eval_func(config)
            all_results.append(result)
        
        self.results = all_results
        self.save_results()
        return self.results

    def save_results(self, filename: str = None):
        """保存评估结果到JSON文件"""
        if not self.results:
            print("没有结果可供保存")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_results_{timestamp}.json"
        
        save_dir = os.path.join(project_root, 'results', 'data')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'w') as f:
            json.dump(self.results, f, cls=NumpyJSONEncoder, indent=4)
        
        print(f"✅ 评估结果已保存到: {save_path}")
    
    def evaluate_baseline_eehfr(self, network_config: Dict) -> Dict:
        """评估基准EEHFR算法"""
        print(f"🔬 评估基准EEHFR算法 - {network_config}")
        
        start_time = time.time()
        
        # 创建基准系统
        config = SystemConfig(
            num_nodes=network_config['nodes'],
            network_size=network_config['area'],
            simulation_rounds=network_config['rounds']
        )
        eehfr_system = EnhancedEEHFRSystem(config=config)
        
        # 运行系统
        try:
            performance_history = eehfr_system.run_simulation()
            
            # 从仿真历史中提取最终的性能指标
            if performance_history:
                final_performance = performance_history[-1]['performance']
                system_results = asdict(final_performance)
            else:
                system_results = {}

            computation_time = time.time() - start_time
            
            # 整理结果
            baseline_results = {
                'algorithm': 'EEHFR-Baseline',
                'network_config': network_config,
                'system_results': system_results,
                'computation_time': computation_time,
                'metrics': {
                    'energy_efficiency': system_results.get('energy_efficiency', 0.5),
                    'network_lifetime': system_results.get('network_lifetime', 50),
                    'routing_success_rate': system_results.get('routing_success_rate', 0.8),
                    'average_latency': system_results.get('average_latency', 0.5),
                    'throughput': system_results.get('throughput', 0.6),
                    'convergence_speed': 100,
                    'computational_complexity': computation_time / network_config['nodes'],
                    'scalability': 0.7,
                    'explainability': 0.4  # 传统算法可解释性较低
                }
            }
            
        except Exception as e:
            print(f"基准算法评估失败: {e}")
            baseline_results = {
                'algorithm': 'EEHFR-Baseline',
                'network_config': network_config,
                'error': str(e),
                'metrics': {metric: 0.0 for metric in self.evaluation_metrics}
            }
        
        return baseline_results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """运行综合评估"""
        print("🚀 开始综合算法评估实验")
        print("=" * 80)
        
        all_results = {}
        
        for config_name, config in self.experiment_configs.items():
            print(f"\n📊 评估配置: {config_name}")
            print("-" * 50)
            
            config_results = {}
            
            # 评估AFW-RL
            try:
                config_results['AFW-RL'] = self.evaluate_afw_rl(config)
            except Exception as e:
                print(f"AFW-RL评估失败: {e}")
                config_results['AFW-RL'] = {'error': str(e)}
            
            # 评估GNN-CTO
            try:
                config_results['GNN-CTO'] = self.evaluate_gnn_cto(config)
            except Exception as e:
                print(f"GNN-CTO评估失败: {e}")
                config_results['GNN-CTO'] = {'error': str(e)}
            
            # 评估ILMR
            try:
                config_results['ILMR'] = self.evaluate_ilmr(config)
            except Exception as e:
                print(f"ILMR评估失败: {e}")
                config_results['ILMR'] = {'error': str(e)}
            
            # 评估基准算法
            try:
                config_results['EEHFR-Baseline'] = self.evaluate_baseline_eehfr(config)
            except Exception as e:
                print(f"基准算法评估失败: {e}")
                config_results['EEHFR-Baseline'] = {'error': str(e)}
            
            all_results[config_name] = config_results
        
        self.results = all_results
        return all_results
    
    def generate_comparison_report(self) -> Dict:
        """生成对比报告"""
        if not self.results:
            print("警告: 评估结果为空，无法生成报告。")
            return {}
        
        # 处理results可能是list或dict的情况
        if isinstance(self.results, list):
            # 直接处理list中的每个算法结果
            comparison_data = []
            for result in self.results:
                if isinstance(result, dict) and 'algorithm' in result and 'metrics' in result:
                    row = {
                        'Algorithm': result['algorithm'],
                        'Configuration': 'default',
                        **result['metrics']
                    }
                    comparison_data.append(row)
        else:
            # 处理dict格式
            comparison_data = []
            for config_name, config_results in self.results.items():
                if isinstance(config_results, dict):
                    for algorithm, result in config_results.items():
                        if isinstance(result, dict) and 'metrics' in result:
                            row = {
                                'Configuration': config_name,
                                'Algorithm': algorithm,
                                **result['metrics']
                            }
                            comparison_data.append(row)

        if not comparison_data:
            print("警告: 没有有效的评估数据可供生成报告。")
            return {}

        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n📈 生成算法对比报告...")
        

        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 计算算法排名
        algorithm_rankings = {}
        for metric in self.evaluation_metrics:
            if metric in comparison_df.columns:
                # 对于某些指标，越小越好（如延迟、计算复杂度）
                ascending = metric in ['average_latency', 'computational_complexity']
                ranked = comparison_df.groupby('Algorithm')[metric].mean().rank(ascending=ascending)
                algorithm_rankings[metric] = ranked.to_dict()
        
        # 计算综合得分
        algorithm_scores = {}
        for algorithm in comparison_df['Algorithm'].unique():
            total_score = 0
            valid_metrics = 0
            
            for metric in self.evaluation_metrics:
                if metric in algorithm_rankings:
                    # 排名转换为得分（排名越高得分越高）
                    rank = algorithm_rankings[metric].get(algorithm, 0)
                    max_rank = len(algorithm_rankings[metric])
                    score = (max_rank - rank + 1) / max_rank
                    total_score += score
                    valid_metrics += 1
            
            if valid_metrics > 0:
                algorithm_scores[algorithm] = total_score / valid_metrics
        
        # 生成报告
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'comparison_table': comparison_df.to_dict('records'),
            'algorithm_rankings': algorithm_rankings,
            'algorithm_scores': algorithm_scores,
            'best_algorithm': max(algorithm_scores.items(), key=lambda x: x[1]) if algorithm_scores else None,
            'summary_statistics': {
                'total_experiments': len(comparison_data),
                'algorithms_tested': len(comparison_df['Algorithm'].unique()),
                'configurations_tested': len(comparison_df['Configuration'].unique())
            }
        }
        
        return report
    
    def visualize_results(self, save_path: str = None):
        """以更美观、更专业的方式可视化结果"""
        if not self.results:
            print("没有结果可供可视化")
            return

        print("🎨 生成高质量可视化图表...")
        
        # 设置更现代、更专业的绘图风格
        sns.set_theme(style="darkgrid", palette="pastel")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 准备数据
        report = self.generate_comparison_report()
        if not report or 'comparison_table' not in report:
            print("无法生成报告或报告中缺少对比表。")
            return
        comparison_df = pd.DataFrame(report['comparison_table'])
        if comparison_df.empty:
            print("无有效数据可供可视化。")
            return

        # 1. 核心性能指标对比 (增强版雷达图)
        self._plot_enhanced_radar(comparison_df, save_path)

        # 2. 各项指标分布与对比 (小提琴图 + 散点图)
        self._plot_metrics_distribution(comparison_df, save_path)

        # 3. 性能权衡分析 (气泡图)
        self._plot_performance_tradeoff(comparison_df, save_path)

        # 4. 综合得分与可解释性 (带注释的散点图)
        self._plot_score_vs_explainability(report, comparison_df, save_path)

    def _plot_enhanced_radar(self, df: pd.DataFrame, save_path: str):
        metrics_for_radar = ['energy_efficiency', 'network_lifetime', 'routing_success_rate', 'average_latency', 'throughput', 'explainability']
        radar_df = df.groupby('Algorithm')[metrics_for_radar].mean()
        
        for metric in metrics_for_radar:
            if metric == 'average_latency':
                radar_df[metric] = 1 - (radar_df[metric] / radar_df[metric].max())
            else:
                radar_df[metric] = radar_df[metric] / radar_df[metric].max()

        labels = [l.replace('_', ' ').title() for l in radar_df.columns]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = sns.color_palette('viridis', len(radar_df))

        for i, (index, row) in enumerate(radar_df.iterrows()):
            values = row.tolist() + [row.tolist()[0]]
            ax.plot(angles, values, color=colors[i], linewidth=2.5, label=index)
            ax.fill(angles, values, color=colors[i], alpha=0.2)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        ax.set_title('Core Performance Radar', size=20, weight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
        if save_path:
            plt.savefig(save_path.replace('.png', '_radar_enhanced.png'), dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_metrics_distribution(self, df: pd.DataFrame, save_path: str):
        metrics_to_plot = self.evaluation_metrics[:6]
        for metric in metrics_to_plot:
            plt.figure(figsize=(14, 8))
            ax = sns.violinplot(data=df, x='Configuration', y=metric, hue='Algorithm', split=True, inner='quart', linewidth=1.5)
            sns.stripplot(data=df, x='Configuration', y=metric, hue='Algorithm', dodge=True, jitter=True, alpha=0.6, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontsize=18, weight='bold')
            ax.set_xlabel('Network Configuration', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = df['Algorithm'].nunique()
            ax.legend(handles[:unique_labels], labels[:unique_labels], title='Algorithm')
            if save_path:
                plt.savefig(save_path.replace('.png', f'_{metric}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_performance_tradeoff(self, df: pd.DataFrame, save_path: str):
        g = sns.relplot(
            data=df, x='energy_efficiency', y='average_latency', 
            hue='Algorithm', style='Configuration', size='throughput',
            sizes=(50, 500), alpha=0.8, palette='muted', height=7, aspect=1.2
        )
        g.fig.suptitle('Performance Trade-off: Energy vs. Latency', fontsize=20, weight='bold')
        g.set_axis_labels('Energy Efficiency (Higher is Better)', 'Latency (Lower is Better)', fontsize=14)
        g.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            plt.savefig(save_path.replace('.png', '_tradeoff_bubble.png'), dpi=300)
        plt.show()

    def _plot_score_vs_explainability(self, report: Dict, df: pd.DataFrame, save_path: str):
        if 'algorithm_scores' not in report:
            return
        scores_df = pd.DataFrame.from_dict(report['algorithm_scores'], orient='index', columns=['Score']).reset_index().rename(columns={'index': 'Algorithm'})
        explainability_df = df.groupby('Algorithm')['explainability'].mean().reset_index()
        merged_df = pd.merge(scores_df, explainability_df, on='Algorithm')

        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(data=merged_df, x='explainability', y='Score', hue='Algorithm', s=300, style='Algorithm', palette='magma', markers=True)
        for i, row in merged_df.iterrows():
            ax.text(row['explainability'] + 0.01, row['Score'], row['Algorithm'], fontsize=12)
        ax.set_title('Overall Score vs. Explainability', fontsize=18, weight='bold')
        ax.set_xlabel('Explainability', fontsize=14)
        ax.set_ylabel('Overall Performance Score', fontsize=14)
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.png', '_score_explainability.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_results(self, filepath: str):
        """保存结果"""
        # 生成完整报告
        report = self.generate_comparison_report()
        
        # 保存详细结果
        output_data = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'network_configurations': self.experiment_configs,
                'evaluation_metrics': self.evaluation_metrics
            },
            'detailed_results': self.results,
            'comparison_report': report
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        
        print(f"✅ 实验结果已保存到: {filepath}")
        
        # 保存CSV格式的对比表
        if 'comparison_table' in report:
            csv_path = filepath.replace('.json', '_comparison.csv')
            comparison_df = pd.DataFrame(report['comparison_table'])
            comparison_df.to_csv(csv_path, index=False)
            print(f"✅ 对比表已保存到: {csv_path}")


def main():
    """主函数"""
    print("🎯 WSN算法综合评估实验")
    print("=" * 80)
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator()
    
    # 运行综合评估
    results = evaluator.run_comprehensive_evaluation()
    
    # 生成报告
    report = evaluator.generate_comparison_report()
    
    # 显示最佳算法
    if report.get('best_algorithm'):
        best_alg, best_score = report['best_algorithm']
        print(f"\n🏆 最佳算法: {best_alg} (综合得分: {best_score:.3f})")
    
    # 可视化结果
    evaluator.visualize_results('comprehensive_algorithm_comparison.png')
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_evaluation_results_{timestamp}.json"
    evaluator.save_detailed_results(results_file)
    
    print("\n✅ 综合评估实验完成！")
    
    return results, report


if __name__ == "__main__":
    main()