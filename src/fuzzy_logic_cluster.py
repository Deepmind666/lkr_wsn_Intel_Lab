"""
Enhanced EEHFR WSN系统 - 模糊逻辑簇头选择模块
基于用户调研文件中的EEHFR协议设计
融合能量、距离、邻居数量的多维模糊推理
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class FuzzyMembershipFunction:
    """模糊隶属度函数类"""
    
    @staticmethod
    def triangular(x, a, b, c):
        """三角形隶属度函数"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x, a, b, c, d):
        """梯形隶属度函数"""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x, mean, sigma):
        """高斯隶属度函数"""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

class FuzzyLogicClusterHead:
    """
    模糊逻辑簇头选择算法
    基于EEHFR协议的三维模糊推理系统
    """
    
    def __init__(self):
        self.membership = FuzzyMembershipFunction()
        
        # 定义模糊集合的参数
        self.energy_params = {
            'low': (0.0, 0.0, 0.3, 0.5),      # 梯形
            'medium': (0.3, 0.5, 0.7),        # 三角形
            'high': (0.5, 0.7, 1.0, 1.0)      # 梯形
        }
        
        self.distance_params = {
            'near': (0.0, 0.0, 0.3, 0.5),     # 梯形
            'medium': (0.3, 0.5, 0.7),        # 三角形
            'far': (0.5, 0.7, 1.0, 1.0)       # 梯形
        }
        
        self.neighbor_params = {
            'few': (0, 3, 6),                 # 三角形
            'moderate': (4, 7, 10),           # 三角形
            'many': (8, 12, 15)               # 三角形
        }
        
        # 模糊规则库
        self.fuzzy_rules = self._initialize_fuzzy_rules()
        
        # 性能统计
        self.selection_history = []
        self.performance_metrics = {}
    
    def _initialize_fuzzy_rules(self) -> List[Dict]:
        """初始化模糊规则库"""
        rules = [
            # 高能量节点优先规则
            {'energy': 'high', 'distance': 'near', 'neighbors': 'many', 'output': 0.95},
            {'energy': 'high', 'distance': 'near', 'neighbors': 'moderate', 'output': 0.90},
            {'energy': 'high', 'distance': 'near', 'neighbors': 'few', 'output': 0.80},
            
            {'energy': 'high', 'distance': 'medium', 'neighbors': 'many', 'output': 0.85},
            {'energy': 'high', 'distance': 'medium', 'neighbors': 'moderate', 'output': 0.80},
            {'energy': 'high', 'distance': 'medium', 'neighbors': 'few', 'output': 0.70},
            
            {'energy': 'high', 'distance': 'far', 'neighbors': 'many', 'output': 0.75},
            {'energy': 'high', 'distance': 'far', 'neighbors': 'moderate', 'output': 0.65},
            {'energy': 'high', 'distance': 'far', 'neighbors': 'few', 'output': 0.55},
            
            # 中等能量节点规则
            {'energy': 'medium', 'distance': 'near', 'neighbors': 'many', 'output': 0.70},
            {'energy': 'medium', 'distance': 'near', 'neighbors': 'moderate', 'output': 0.65},
            {'energy': 'medium', 'distance': 'near', 'neighbors': 'few', 'output': 0.55},
            
            {'energy': 'medium', 'distance': 'medium', 'neighbors': 'many', 'output': 0.60},
            {'energy': 'medium', 'distance': 'medium', 'neighbors': 'moderate', 'output': 0.50},
            {'energy': 'medium', 'distance': 'medium', 'neighbors': 'few', 'output': 0.40},
            
            {'energy': 'medium', 'distance': 'far', 'neighbors': 'many', 'output': 0.45},
            {'energy': 'medium', 'distance': 'far', 'neighbors': 'moderate', 'output': 0.35},
            {'energy': 'medium', 'distance': 'far', 'neighbors': 'few', 'output': 0.25},
            
            # 低能量节点规则（避免选择）
            {'energy': 'low', 'distance': 'near', 'neighbors': 'many', 'output': 0.30},
            {'energy': 'low', 'distance': 'near', 'neighbors': 'moderate', 'output': 0.25},
            {'energy': 'low', 'distance': 'near', 'neighbors': 'few', 'output': 0.15},
            
            {'energy': 'low', 'distance': 'medium', 'neighbors': 'many', 'output': 0.20},
            {'energy': 'low', 'distance': 'medium', 'neighbors': 'moderate', 'output': 0.15},
            {'energy': 'low', 'distance': 'medium', 'neighbors': 'few', 'output': 0.10},
            
            {'energy': 'low', 'distance': 'far', 'neighbors': 'many', 'output': 0.10},
            {'energy': 'low', 'distance': 'far', 'neighbors': 'moderate', 'output': 0.05},
            {'energy': 'low', 'distance': 'far', 'neighbors': 'few', 'output': 0.01},
        ]
        return rules
    
    def calculate_energy_membership(self, energy_ratio: float) -> Dict[str, float]:
        """计算能量的模糊隶属度"""
        memberships = {}
        
        # 低能量
        memberships['low'] = self.membership.trapezoidal(
            energy_ratio, *self.energy_params['low'])
        
        # 中等能量
        memberships['medium'] = self.membership.triangular(
            energy_ratio, *self.energy_params['medium'])
        
        # 高能量
        memberships['high'] = self.membership.trapezoidal(
            energy_ratio, *self.energy_params['high'])
        
        return memberships
    
    def calculate_distance_membership(self, distance_ratio: float) -> Dict[str, float]:
        """计算距离的模糊隶属度"""
        memberships = {}
        
        # 近距离
        memberships['near'] = self.membership.trapezoidal(
            distance_ratio, *self.distance_params['near'])
        
        # 中等距离
        memberships['medium'] = self.membership.triangular(
            distance_ratio, *self.distance_params['medium'])
        
        # 远距离
        memberships['far'] = self.membership.trapezoidal(
            distance_ratio, *self.distance_params['far'])
        
        return memberships
    
    def calculate_neighbor_membership(self, neighbor_count: int) -> Dict[str, float]:
        """计算邻居数量的模糊隶属度"""
        memberships = {}
        
        # 少邻居
        memberships['few'] = self.membership.triangular(
            neighbor_count, *self.neighbor_params['few'])
        
        # 中等邻居
        memberships['moderate'] = self.membership.triangular(
            neighbor_count, *self.neighbor_params['moderate'])
        
        # 多邻居
        memberships['many'] = self.membership.triangular(
            neighbor_count, *self.neighbor_params['many'])
        
        return memberships
    
    def fuzzy_inference(self, energy_ratio: float, distance_ratio: float, 
                      neighbor_count: int) -> float:
        """模糊推理计算簇头适合度"""
        
        # 计算各输入变量的隶属度
        energy_memberships = self.calculate_energy_membership(energy_ratio)
        distance_memberships = self.calculate_distance_membership(distance_ratio)
        neighbor_memberships = self.calculate_neighbor_membership(neighbor_count)
        
        # 应用模糊规则
        rule_outputs = []
        rule_weights = []
        
        for rule in self.fuzzy_rules:
            # 计算规则的激活强度（使用最小值操作）
            activation = min(
                energy_memberships[rule['energy']],
                distance_memberships[rule['distance']],
                neighbor_memberships[rule['neighbors']]
            )
            
            if activation > 0:
                rule_outputs.append(rule['output'])
                rule_weights.append(activation)
        
        # 去模糊化（加权平均法）
        if sum(rule_weights) > 0:
            fuzzy_score = sum(w * o for w, o in zip(rule_weights, rule_outputs)) / sum(rule_weights)
        else:
            fuzzy_score = 0.0
        
        return fuzzy_score
    
    def calculate_fuzzy_score(self, energy_ratio: float, distance_ratio: float, 
                            neighbor_count: int) -> float:
        """计算节点的模糊评分（主要接口）"""
        return self.fuzzy_inference(energy_ratio, distance_ratio, neighbor_count)
    
    def select_cluster_heads(self, nodes_data: pd.DataFrame, n_clusters: int = 6) -> List[int]:
        """
        选择簇头节点
        
        Args:
            nodes_data: 节点数据DataFrame，包含node_id, energy, distance, neighbors等列
            n_clusters: 需要选择的簇头数量
            
        Returns:
            选中的簇头节点ID列表
        """
        
        node_scores = []
        
        for _, node in nodes_data.iterrows():
            # 计算模糊评分
            fuzzy_score = self.calculate_fuzzy_score(
                node['energy_ratio'],
                node['distance_ratio'], 
                node['neighbor_count']
            )
            
            node_scores.append({
                'node_id': node['node_id'],
                'fuzzy_score': fuzzy_score,
                'energy_ratio': node['energy_ratio'],
                'distance_ratio': node['distance_ratio'],
                'neighbor_count': node['neighbor_count']
            })
        
        # 按模糊评分排序
        node_scores.sort(key=lambda x: x['fuzzy_score'], reverse=True)
        
        # 选择前n_clusters个节点作为簇头
        selected_cluster_heads = [score['node_id'] for score in node_scores[:n_clusters]]
        
        # 记录选择历史
        self.selection_history.append({
            'cluster_heads': selected_cluster_heads,
            'scores': node_scores[:n_clusters],
            'timestamp': pd.Timestamp.now()
        })
        
        return selected_cluster_heads
    
    def visualize_membership_functions(self, save_path: str = None):
        """可视化模糊隶属度函数"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 能量隶属度函数
        x_energy = np.linspace(0, 1, 100)
        y_energy_low = [self.membership.trapezoidal(x, *self.energy_params['low']) for x in x_energy]
        y_energy_med = [self.membership.triangular(x, *self.energy_params['medium']) for x in x_energy]
        y_energy_high = [self.membership.trapezoidal(x, *self.energy_params['high']) for x in x_energy]
        
        axes[0].plot(x_energy, y_energy_low, 'r-', label='低能量', linewidth=2)
        axes[0].plot(x_energy, y_energy_med, 'g-', label='中等能量', linewidth=2)
        axes[0].plot(x_energy, y_energy_high, 'b-', label='高能量', linewidth=2)
        axes[0].set_title('能量隶属度函数')
        axes[0].set_xlabel('能量比例')
        axes[0].set_ylabel('隶属度')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 距离隶属度函数
        x_distance = np.linspace(0, 1, 100)
        y_distance_near = [self.membership.trapezoidal(x, *self.distance_params['near']) for x in x_distance]
        y_distance_med = [self.membership.triangular(x, *self.distance_params['medium']) for x in x_distance]
        y_distance_far = [self.membership.trapezoidal(x, *self.distance_params['far']) for x in x_distance]
        
        axes[1].plot(x_distance, y_distance_near, 'r-', label='近距离', linewidth=2)
        axes[1].plot(x_distance, y_distance_med, 'g-', label='中等距离', linewidth=2)
        axes[1].plot(x_distance, y_distance_far, 'b-', label='远距离', linewidth=2)
        axes[1].set_title('距离隶属度函数')
        axes[1].set_xlabel('距离比例')
        axes[1].set_ylabel('隶属度')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 邻居数量隶属度函数
        x_neighbor = np.linspace(0, 15, 100)
        y_neighbor_few = [self.membership.triangular(x, *self.neighbor_params['few']) for x in x_neighbor]
        y_neighbor_mod = [self.membership.triangular(x, *self.neighbor_params['moderate']) for x in x_neighbor]
        y_neighbor_many = [self.membership.triangular(x, *self.neighbor_params['many']) for x in x_neighbor]
        
        axes[2].plot(x_neighbor, y_neighbor_few, 'r-', label='少邻居', linewidth=2)
        axes[2].plot(x_neighbor, y_neighbor_mod, 'g-', label='中等邻居', linewidth=2)
        axes[2].plot(x_neighbor, y_neighbor_many, 'b-', label='多邻居', linewidth=2)
        axes[2].set_title('邻居数量隶属度函数')
        axes[2].set_xlabel('邻居数量')
        axes[2].set_ylabel('隶属度')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 隶属度函数图表已保存到: {save_path}")
        
        return fig
    
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        if not self.selection_history:
            return {"message": "暂无选择历史"}
        
        latest_selection = self.selection_history[-1]
        
        summary = {
            "total_selections": len(self.selection_history),
            "latest_cluster_heads": latest_selection['cluster_heads'],
            "latest_scores": [score['fuzzy_score'] for score in latest_selection['scores']],
            "average_score": np.mean([score['fuzzy_score'] for score in latest_selection['scores']]),
            "score_std": np.std([score['fuzzy_score'] for score in latest_selection['scores']]),
            "selection_timestamp": latest_selection['timestamp']
        }
        
        return summary

if __name__ == "__main__":
    # 测试模糊逻辑簇头选择
    fuzzy_selector = FuzzyLogicClusterHead()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'node_id': range(1, 21),
        'energy_ratio': np.random.uniform(0.2, 1.0, 20),
        'distance_ratio': np.random.uniform(0.1, 0.9, 20),
        'neighbor_count': np.random.randint(3, 12, 20)
    })
    
    # 选择簇头
    cluster_heads = fuzzy_selector.select_cluster_heads(test_data, n_clusters=6)
    print(f"选择的簇头: {cluster_heads}")
    
    # 获取性能总结
    summary = fuzzy_selector.get_performance_summary()
    print(f"性能总结: {summary}")
    
    # 可视化隶属度函数
    fuzzy_selector.visualize_membership_functions("fuzzy_membership_functions.png")