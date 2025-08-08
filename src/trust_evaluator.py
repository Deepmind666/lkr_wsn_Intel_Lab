"""
Enhanced EEHFR WSN System - Trust Evaluation Module
信任评估模块：专门处理WSN中的数据可靠性和节点信任度评估

主要功能：
1. 多维度信任评估（数据信任、通信信任、行为信任）
2. 动态信任更新机制
3. 恶意节点检测与隔离
4. 数据可靠性验证
5. 信任传播算法
6. 异常行为检测

作者：Enhanced EEHFR Team
日期：2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TrustType(Enum):
    """信任类型枚举"""
    DATA_TRUST = "data_trust"           # 数据信任
    COMMUNICATION_TRUST = "comm_trust"  # 通信信任
    BEHAVIOR_TRUST = "behavior_trust"   # 行为信任
    COMPOSITE_TRUST = "composite_trust" # 综合信任

class NodeStatus(Enum):
    """节点状态枚举"""
    TRUSTED = "trusted"         # 可信
    SUSPICIOUS = "suspicious"   # 可疑
    MALICIOUS = "malicious"     # 恶意
    UNKNOWN = "unknown"         # 未知

@dataclass
class TrustMetrics:
    """信任度量指标"""
    data_consistency: float = 0.0      # 数据一致性
    communication_reliability: float = 0.0  # 通信可靠性
    packet_delivery_ratio: float = 0.0      # 包投递率
    response_time: float = 0.0              # 响应时间
    energy_efficiency: float = 0.0          # 能效比
    neighbor_recommendations: float = 0.0   # 邻居推荐度

@dataclass
class TrustRecord:
    """信任记录"""
    node_id: int
    timestamp: float
    trust_type: TrustType
    trust_value: float
    evidence: Dict[str, Any]
    evaluator_id: int

class DataReliabilityAnalyzer:
    """数据可靠性分析器"""
    
    def __init__(self, window_size: int = 10, threshold: float = 0.8):
        self.window_size = window_size
        self.threshold = threshold
        self.data_history = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_data_consistency(self, node_id: int, data_values: List[float], 
                               neighbor_data: Dict[int, List[float]]) -> float:
        """分析数据一致性"""
        if len(data_values) < 2:
            return 0.5
            
        # 计算数据的统计特征
        node_mean = np.mean(data_values)
        node_std = np.std(data_values)
        
        # 与邻居节点数据比较
        consistency_scores = []
        for neighbor_id, neighbor_values in neighbor_data.items():
            if len(neighbor_values) >= 2:
                neighbor_mean = np.mean(neighbor_values)
                neighbor_std = np.std(neighbor_values)
                
                # 计算均值差异
                mean_diff = abs(node_mean - neighbor_mean)
                max_mean = max(abs(node_mean), abs(neighbor_mean), 1e-6)
                mean_consistency = 1.0 - min(mean_diff / max_mean, 1.0)
                
                # 计算方差一致性
                std_diff = abs(node_std - neighbor_std)
                max_std = max(node_std, neighbor_std, 1e-6)
                std_consistency = 1.0 - min(std_diff / max_std, 1.0)
                
                consistency_scores.append((mean_consistency + std_consistency) / 2)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def detect_anomalies(self, node_data: Dict[int, List[float]]) -> Dict[int, bool]:
        """检测异常数据"""
        anomaly_results = {}
        
        if len(node_data) < 2:
            return {node_id: False for node_id in node_data.keys()}
        
        # 准备数据
        all_data = []
        node_ids = []
        for node_id, values in node_data.items():
            if len(values) > 0:
                all_data.extend(values)
                node_ids.extend([node_id] * len(values))
        
        if len(all_data) < 10:
            return {node_id: False for node_id in node_data.keys()}
        
        # 标准化数据
        data_array = np.array(all_data).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data_array)
        
        # 异常检测
        anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
        
        # 统计每个节点的异常比例
        for node_id in node_data.keys():
            node_indices = [i for i, nid in enumerate(node_ids) if nid == node_id]
            if node_indices:
                node_anomalies = [anomaly_labels[i] for i in node_indices]
                anomaly_ratio = sum(1 for x in node_anomalies if x == -1) / len(node_anomalies)
                anomaly_results[node_id] = anomaly_ratio > 0.3
            else:
                anomaly_results[node_id] = False
        
        return anomaly_results

class TrustEvaluator:
    """信任评估器 - 核心信任评估算法"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1):
        """
        初始化信任评估器
        
        Args:
            alpha: 数据信任权重
            beta: 通信信任权重  
            gamma: 行为信任权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 信任值存储
        self.trust_values = {}  # {node_id: {trust_type: value}}
        self.trust_history = []  # List[TrustRecord]
        self.node_status = {}   # {node_id: NodeStatus}
        
        # 信任评估参数
        self.trust_decay_factor = 0.95  # 信任衰减因子
        self.min_trust_threshold = 0.3  # 最小信任阈值
        self.max_trust_threshold = 0.8  # 最大信任阈值
        
        # 数据可靠性分析器
        self.reliability_analyzer = DataReliabilityAnalyzer()
        
        # 性能统计
        self.evaluation_stats = {
            'total_evaluations': 0,
            'malicious_detected': 0,
            'false_positives': 0,
            'trust_updates': 0
        }
    
    def initialize_trust(self, node_ids: List[int], initial_trust: float = 0.5):
        """初始化节点信任值"""
        for node_id in node_ids:
            self.trust_values[node_id] = {
                TrustType.DATA_TRUST: initial_trust,
                TrustType.COMMUNICATION_TRUST: initial_trust,
                TrustType.BEHAVIOR_TRUST: initial_trust,
                TrustType.COMPOSITE_TRUST: initial_trust
            }
            self.node_status[node_id] = NodeStatus.UNKNOWN
    
    def evaluate_data_trust(self, node_id: int, metrics: TrustMetrics, 
                           neighbor_data: Dict[int, List[float]]) -> float:
        """评估数据信任度"""
        # 数据一致性评估
        consistency_score = metrics.data_consistency
        
        # 数据质量评估（基于统计特征）
        quality_score = min(metrics.energy_efficiency, 1.0)
        
        # 邻居推荐度
        recommendation_score = metrics.neighbor_recommendations
        
        # 综合数据信任度
        data_trust = (0.4 * consistency_score + 
                     0.3 * quality_score + 
                     0.3 * recommendation_score)
        
        return np.clip(data_trust, 0.0, 1.0)
    
    def evaluate_communication_trust(self, node_id: int, metrics: TrustMetrics) -> float:
        """评估通信信任度"""
        # 包投递率
        delivery_score = metrics.packet_delivery_ratio
        
        # 通信可靠性
        reliability_score = metrics.communication_reliability
        
        # 响应时间（越短越好）
        response_score = max(0, 1.0 - metrics.response_time / 1000.0)  # 假设1000ms为最大可接受时间
        
        # 综合通信信任度
        comm_trust = (0.4 * delivery_score + 
                     0.4 * reliability_score + 
                     0.2 * response_score)
        
        return np.clip(comm_trust, 0.0, 1.0)
    
    def evaluate_behavior_trust(self, node_id: int, metrics: TrustMetrics) -> float:
        """评估行为信任度"""
        # 能效比评估
        efficiency_score = metrics.energy_efficiency
        
        # 协作度评估（基于邻居推荐）
        cooperation_score = metrics.neighbor_recommendations
        
        # 稳定性评估（基于历史表现）
        stability_score = self._calculate_stability_score(node_id)
        
        # 综合行为信任度
        behavior_trust = (0.4 * efficiency_score + 
                         0.3 * cooperation_score + 
                         0.3 * stability_score)
        
        return np.clip(behavior_trust, 0.0, 1.0)
    
    def _calculate_stability_score(self, node_id: int) -> float:
        """计算节点稳定性得分"""
        if node_id not in self.trust_values:
            return 0.5
        
        # 获取历史信任记录
        node_records = [r for r in self.trust_history if r.node_id == node_id]
        
        if len(node_records) < 3:
            return 0.5
        
        # 计算信任值的方差（稳定性）
        recent_trusts = [r.trust_value for r in node_records[-10:]]
        trust_variance = np.var(recent_trusts)
        
        # 方差越小，稳定性越高
        stability = max(0, 1.0 - trust_variance * 2)
        
        return stability
    
    def calculate_composite_trust(self, node_id: int) -> float:
        """计算综合信任度"""
        if node_id not in self.trust_values:
            return 0.0
        
        trust_vals = self.trust_values[node_id]
        
        composite_trust = (self.alpha * trust_vals[TrustType.DATA_TRUST] +
                          self.beta * trust_vals[TrustType.COMMUNICATION_TRUST] +
                          self.gamma * trust_vals[TrustType.BEHAVIOR_TRUST])
        
        return np.clip(composite_trust, 0.0, 1.0)
    
    def update_trust(self, node_id: int, metrics: TrustMetrics, 
                    neighbor_data: Dict[int, List[float]], timestamp: float):
        """更新节点信任度"""
        if node_id not in self.trust_values:
            self.initialize_trust([node_id])
        
        # 评估各维度信任度
        data_trust = self.evaluate_data_trust(node_id, metrics, neighbor_data)
        comm_trust = self.evaluate_communication_trust(node_id, metrics)
        behavior_trust = self.evaluate_behavior_trust(node_id, metrics)
        
        # 应用信任衰减
        current_trusts = self.trust_values[node_id]
        for trust_type in [TrustType.DATA_TRUST, TrustType.COMMUNICATION_TRUST, TrustType.BEHAVIOR_TRUST]:
            current_trusts[trust_type] *= self.trust_decay_factor
        
        # 更新信任值（加权平均）
        learning_rate = 0.3
        self.trust_values[node_id][TrustType.DATA_TRUST] = (
            (1 - learning_rate) * current_trusts[TrustType.DATA_TRUST] + 
            learning_rate * data_trust
        )
        self.trust_values[node_id][TrustType.COMMUNICATION_TRUST] = (
            (1 - learning_rate) * current_trusts[TrustType.COMMUNICATION_TRUST] + 
            learning_rate * comm_trust
        )
        self.trust_values[node_id][TrustType.BEHAVIOR_TRUST] = (
            (1 - learning_rate) * current_trusts[TrustType.BEHAVIOR_TRUST] + 
            learning_rate * behavior_trust
        )
        
        # 计算综合信任度
        composite_trust = self.calculate_composite_trust(node_id)
        self.trust_values[node_id][TrustType.COMPOSITE_TRUST] = composite_trust
        
        # 更新节点状态
        self._update_node_status(node_id, composite_trust)
        
        # 记录信任更新
        self.trust_history.append(TrustRecord(
            node_id=node_id,
            timestamp=timestamp,
            trust_type=TrustType.COMPOSITE_TRUST,
            trust_value=composite_trust,
            evidence={'metrics': metrics.__dict__},
            evaluator_id=0
        ))
        
        self.evaluation_stats['trust_updates'] += 1
    
    def _update_node_status(self, node_id: int, composite_trust: float):
        """更新节点状态"""
        if composite_trust < self.min_trust_threshold:
            if self.node_status[node_id] == NodeStatus.SUSPICIOUS:
                self.node_status[node_id] = NodeStatus.MALICIOUS
                self.evaluation_stats['malicious_detected'] += 1
            else:
                self.node_status[node_id] = NodeStatus.SUSPICIOUS
        elif composite_trust > self.max_trust_threshold:
            self.node_status[node_id] = NodeStatus.TRUSTED
        else:
            if self.node_status[node_id] == NodeStatus.MALICIOUS:
                self.node_status[node_id] = NodeStatus.SUSPICIOUS
    
    def detect_malicious_nodes(self) -> List[int]:
        """检测恶意节点"""
        malicious_nodes = []
        for node_id, status in self.node_status.items():
            if status == NodeStatus.MALICIOUS:
                malicious_nodes.append(node_id)
        return malicious_nodes
    
    def get_trusted_nodes(self, min_trust: float = 0.6) -> List[int]:
        """获取可信节点列表"""
        trusted_nodes = []
        for node_id, trust_vals in self.trust_values.items():
            if trust_vals[TrustType.COMPOSITE_TRUST] >= min_trust:
                trusted_nodes.append(node_id)
        return trusted_nodes
    
    def propagate_trust(self, network_topology: Dict[int, List[int]]):
        """信任传播算法"""
        # 基于网络拓扑进行信任传播
        propagation_factor = 0.1
        
        for node_id, neighbors in network_topology.items():
            if node_id not in self.trust_values:
                continue
                
            node_trust = self.trust_values[node_id][TrustType.COMPOSITE_TRUST]
            
            # 向邻居传播信任
            for neighbor_id in neighbors:
                if neighbor_id in self.trust_values:
                    neighbor_trust = self.trust_values[neighbor_id][TrustType.COMPOSITE_TRUST]
                    
                    # 信任传播公式
                    trust_influence = propagation_factor * node_trust * (1 - neighbor_trust)
                    new_trust = neighbor_trust + trust_influence
                    
                    self.trust_values[neighbor_id][TrustType.COMPOSITE_TRUST] = np.clip(new_trust, 0.0, 1.0)
    
    def visualize_trust_evolution(self, save_path: str = None):
        """可视化信任演化过程"""
        if not self.trust_history:
            print("没有信任历史数据可供可视化")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 提取数据
        df = pd.DataFrame([{
            'node_id': r.node_id,
            'timestamp': r.timestamp,
            'trust_value': r.trust_value
        } for r in self.trust_history])
        
        # 1. 信任值时间序列
        plt.subplot(2, 2, 1)
        for node_id in df['node_id'].unique()[:10]:  # 只显示前10个节点
            node_data = df[df['node_id'] == node_id]
            plt.plot(node_data['timestamp'], node_data['trust_value'], 
                    label=f'Node {node_id}', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('信任值')
        plt.title('节点信任值演化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. 信任值分布
        plt.subplot(2, 2, 2)
        current_trusts = [trust_vals[TrustType.COMPOSITE_TRUST] 
                         for trust_vals in self.trust_values.values()]
        plt.hist(current_trusts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.min_trust_threshold, color='red', linestyle='--', label='恶意阈值')
        plt.axvline(self.max_trust_threshold, color='green', linestyle='--', label='可信阈值')
        plt.xlabel('信任值')
        plt.ylabel('节点数量')
        plt.title('当前信任值分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 节点状态统计
        plt.subplot(2, 2, 3)
        status_counts = {}
        for status in self.node_status.values():
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        plt.title('节点状态分布')
        
        # 4. 信任评估统计
        plt.subplot(2, 2, 4)
        stats_data = self.evaluation_stats
        categories = list(stats_data.keys())
        values = list(stats_data.values())
        
        plt.bar(categories, values, color=['blue', 'red', 'orange', 'green'])
        plt.xlabel('统计类别')
        plt.ylabel('数量')
        plt.title('信任评估统计')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"信任演化图表已保存到: {save_path}")
        
        plt.show()
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """获取信任评估总结"""
        if not self.trust_values:
            return {"error": "没有信任数据"}
        
        current_trusts = [trust_vals[TrustType.COMPOSITE_TRUST] 
                         for trust_vals in self.trust_values.values()]
        
        summary = {
            "总节点数": len(self.trust_values),
            "平均信任值": np.mean(current_trusts),
            "信任值标准差": np.std(current_trusts),
            "最高信任值": np.max(current_trusts),
            "最低信任值": np.min(current_trusts),
            "可信节点数": len(self.get_trusted_nodes()),
            "恶意节点数": len(self.detect_malicious_nodes()),
            "信任更新次数": self.evaluation_stats['trust_updates'],
            "检测到的恶意节点": self.evaluation_stats['malicious_detected'],
            "节点状态分布": {status.value: sum(1 for s in self.node_status.values() if s == status) 
                          for status in NodeStatus}
        }
        
        return summary
    
    def save_trust_data(self, filepath: str):
        """保存信任数据"""
        trust_data = {
            'trust_values': {str(k): {tt.value: v for tt, v in tv.items()} 
                           for k, tv in self.trust_values.items()},
            'node_status': {str(k): v.value for k, v in self.node_status.items()},
            'evaluation_stats': self.evaluation_stats,
            'trust_history': [{
                'node_id': r.node_id,
                'timestamp': r.timestamp,
                'trust_type': r.trust_type.value,
                'trust_value': r.trust_value,
                'evaluator_id': r.evaluator_id
            } for r in self.trust_history]
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trust_data, f, indent=2, ensure_ascii=False)
        
        print(f"信任数据已保存到: {filepath}")

# 测试和演示代码
if __name__ == "__main__":
    print("=== Enhanced EEHFR WSN Trust Evaluation System ===")
    print("正在初始化信任评估系统...")
    
    # 创建信任评估器
    trust_evaluator = TrustEvaluator()
    
    # 模拟网络节点
    node_ids = list(range(1, 21))  # 20个节点
    trust_evaluator.initialize_trust(node_ids)
    
    print(f"已初始化 {len(node_ids)} 个节点的信任评估")
    
    # 模拟信任评估过程
    np.random.seed(42)
    for round_num in range(50):
        for node_id in node_ids:
            # 模拟节点指标
            metrics = TrustMetrics(
                data_consistency=np.random.beta(2, 2),
                communication_reliability=np.random.beta(3, 1),
                packet_delivery_ratio=np.random.beta(4, 1),
                response_time=np.random.exponential(100),
                energy_efficiency=np.random.beta(2, 1),
                neighbor_recommendations=np.random.beta(2, 2)
            )
            
            # 模拟邻居数据
            neighbor_data = {
                nid: [np.random.normal(25, 2) for _ in range(5)]
                for nid in np.random.choice(node_ids, size=3, replace=False)
                if nid != node_id
            }
            
            # 更新信任度
            trust_evaluator.update_trust(node_id, metrics, neighbor_data, round_num)
    
    # 显示结果
    print("\n=== 信任评估结果 ===")
    summary = trust_evaluator.get_trust_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\n可信节点: {trust_evaluator.get_trusted_nodes()}")
    print(f"恶意节点: {trust_evaluator.detect_malicious_nodes()}")
    
    # 可视化结果
    trust_evaluator.visualize_trust_evolution("trust_evolution.png")
    
    print("\n信任评估模块测试完成！")