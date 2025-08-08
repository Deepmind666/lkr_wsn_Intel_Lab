"""
GNN-CTO: Graph Neural Network-based Chain Topology Optimization
基于图神经网络的链式拓扑优化算法

核心创新：
1. 将WSN建模为动态图结构
2. 使用图注意力网络(GAT)学习节点特征
3. 链式拓扑优化与能量均衡
4. 自适应拓扑重构机制

作者: WSN研究团队
日期: 2025年1月
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NodeFeatures:
    """节点特征表示"""
    node_id: int
    position: np.ndarray      # [x, y] 位置坐标
    energy: float            # 剩余能量
    degree: int              # 节点度
    centrality: float        # 中心性
    cluster_id: int          # 所属簇ID
    is_cluster_head: bool    # 是否为簇头
    distance_to_bs: float    # 到基站距离
    neighbor_energy_avg: float  # 邻居平均能量
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.position[0], self.position[1],
            self.energy, self.degree, self.centrality,
            float(self.is_cluster_head), self.distance_to_bs,
            self.neighbor_energy_avg
        ])

@dataclass
class ChainTopology:
    """链式拓扑结构"""
    chain_id: int
    nodes: List[int]          # 链中的节点序列
    head_node: int           # 链头节点
    energy_cost: float       # 链的总能耗
    reliability: float       # 链的可靠性
    length: int              # 链长度
    
    def get_chain_efficiency(self) -> float:
        """计算链效率"""
        if self.energy_cost == 0:
            return 0
        return self.reliability / (self.energy_cost * self.length)

class GraphAttentionNetwork(nn.Module):
    """图注意力网络模型"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, 
                 output_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # 第一层GAT
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # 第二层GAT
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3类：普通节点、链头、中继节点
        )
        
        # 回归层（预测能耗）
        self.regressor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 第一层GAT
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层GAT
        node_embeddings = self.gat2(x, edge_index)
        node_embeddings = self.bn2(node_embeddings)
        node_embeddings = F.relu(node_embeddings)
        
        # 分类输出（节点角色）
        node_roles = self.classifier(node_embeddings)
        
        # 回归输出（能耗预测）
        energy_pred = self.regressor(node_embeddings)
        
        return node_embeddings, node_roles, energy_pred

class ChainOptimizer:
    """链式拓扑优化器"""
    
    def __init__(self, max_chain_length: int = 5, energy_threshold: float = 0.1):
        self.max_chain_length = max_chain_length
        self.energy_threshold = energy_threshold
    
    def construct_chains(self, nodes_features: List[NodeFeatures], 
                        adjacency_matrix: np.ndarray) -> List[ChainTopology]:
        """构建链式拓扑"""
        num_nodes = len(nodes_features)
        visited = set()
        chains = []
        chain_id = 0
        
        # 按能量排序，优先选择高能量节点作为链头
        sorted_nodes = sorted(
            enumerate(nodes_features), 
            key=lambda x: x[1].energy, 
            reverse=True
        )
        
        for node_idx, node_feature in sorted_nodes:
            if node_idx in visited:
                continue
            
            # 构建以当前节点为起点的链
            chain = self._build_chain_from_node(
                start_node=node_idx,
                nodes_features=nodes_features,
                adjacency_matrix=adjacency_matrix,
                visited=visited
            )
            
            if len(chain) > 1:  # 至少包含2个节点
                chain_topology = self._create_chain_topology(
                    chain_id=chain_id,
                    chain_nodes=chain,
                    nodes_features=nodes_features
                )
                chains.append(chain_topology)
                chain_id += 1
                
                # 标记已访问的节点
                visited.update(chain)
        
        return chains
    
    def _build_chain_from_node(self, start_node: int, nodes_features: List[NodeFeatures],
                              adjacency_matrix: np.ndarray, visited: set) -> List[int]:
        """从指定节点开始构建链"""
        chain = [start_node]
        current_node = start_node
        
        while len(chain) < self.max_chain_length:
            # 找到最佳下一个节点
            next_node = self._find_best_next_node(
                current_node=current_node,
                chain=chain,
                nodes_features=nodes_features,
                adjacency_matrix=adjacency_matrix,
                visited=visited
            )
            
            if next_node is None:
                break
            
            chain.append(next_node)
            current_node = next_node
        
        return chain
    
    def _find_best_next_node(self, current_node: int, chain: List[int],
                           nodes_features: List[NodeFeatures],
                           adjacency_matrix: np.ndarray, visited: set) -> Optional[int]:
        """找到最佳的下一个节点"""
        candidates = []
        
        # 遍历当前节点的邻居
        for neighbor in range(len(adjacency_matrix)):
            if (adjacency_matrix[current_node][neighbor] > 0 and 
                neighbor not in visited and 
                neighbor not in chain and
                nodes_features[neighbor].energy > self.energy_threshold):
                
                # 计算候选节点的评分
                score = self._calculate_node_score(
                    node_idx=neighbor,
                    current_chain=chain,
                    nodes_features=nodes_features
                )
                candidates.append((neighbor, score))
        
        if not candidates:
            return None
        
        # 选择评分最高的节点
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_node_score(self, node_idx: int, current_chain: List[int],
                            nodes_features: List[NodeFeatures]) -> float:
        """计算节点评分"""
        node = nodes_features[node_idx]
        
        # 能量权重
        energy_score = node.energy * 0.4
        
        # 距离权重（距离基站越近越好）
        distance_score = (1.0 / (1.0 + node.distance_to_bs)) * 0.3
        
        # 度权重（适中的度最好）
        optimal_degree = 3
        degree_score = 1.0 / (1.0 + abs(node.degree - optimal_degree)) * 0.2
        
        # 中心性权重
        centrality_score = node.centrality * 0.1
        
        return energy_score + distance_score + degree_score + centrality_score
    
    def _create_chain_topology(self, chain_id: int, chain_nodes: List[int],
                             nodes_features: List[NodeFeatures]) -> ChainTopology:
        """创建链拓扑对象"""
        head_node = chain_nodes[0]
        
        # 计算链的总能耗
        total_energy_cost = 0
        for i, node_idx in enumerate(chain_nodes):
            node = nodes_features[node_idx]
            # 链头额外能耗
            if i == 0:
                total_energy_cost += 0.01
            # 传输能耗（距离相关）
            if i < len(chain_nodes) - 1:
                next_node = nodes_features[chain_nodes[i + 1]]
                distance = np.linalg.norm(node.position - next_node.position)
                total_energy_cost += 0.001 * (distance ** 2)
        
        # 计算链的可靠性（基于节点能量）
        min_energy = min(nodes_features[idx].energy for idx in chain_nodes)
        avg_energy = np.mean([nodes_features[idx].energy for idx in chain_nodes])
        reliability = (min_energy * 0.6 + avg_energy * 0.4)
        
        return ChainTopology(
            chain_id=chain_id,
            nodes=chain_nodes,
            head_node=head_node,
            energy_cost=total_energy_cost,
            reliability=reliability,
            length=len(chain_nodes)
        )

class GNNCTOAlgorithm:
    """GNN-CTO主算法类"""
    
    def __init__(self, num_nodes: int, area: Tuple[int, int], rounds: int, 
                 transmission_range: float = 50, initial_energy: float = 1.0, 
                 packet_size: int = 2000, E_elec: float = 50e-9, 
                 E_amp: float = 100e-12, max_chain_length: int = 6, 
                 gnn_hidden_dim: int = 128, gnn_output_dim: int = 64, 
                 gnn_heads: int = 4, gnn_epochs: int = 50, 
                 learning_rate: float = 0.005, topology_update_interval: int = 10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_nodes = num_nodes
        self.area = area
        self.rounds = rounds
        self.transmission_range = transmission_range
        self.initial_energy = initial_energy
        self.packet_size = packet_size
        self.E_elec = E_elec
        self.E_amp = E_amp
        self.max_chain_length = max_chain_length
        self.topology_update_interval = topology_update_interval
        
        # 初始化GNN模型
        self.gnn_model = GraphAttentionNetwork(
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_heads=gnn_heads
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.gnn_model.parameters(), 
            lr=learning_rate
        )
        
        # 损失函数
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = nn.MSELoss()
        
        # 链优化器
        self.chain_optimizer = ChainOptimizer(
            max_chain_length=max_chain_length
        )
        
        # 特征标准化
        self.feature_scaler = StandardScaler()
        
        # 仿真状态
        self.network_graph: Optional[nx.Graph] = None
        self.nodes_features: List[NodeFeatures] = []
        self.chains: List[ChainTopology] = []
        self.performance_metrics: Dict[str, List] = {
            'energy_efficiency': [],
            'network_lifetime': [],
            'routing_success_rate': [],
            'average_latency': [],
            'throughput': [],
            'dead_nodes': []
        }
        self.current_round = 0

        self.training_history: Dict[str, List] = {
            'losses': [],
            'classification_losses': [],
            'regression_losses': [],
            'accuracies': []
        }
    
    def extract_node_features(self, nodes_data: np.ndarray, 
                            base_station_pos: np.ndarray) -> List[NodeFeatures]:
        """提取节点特征"""
        num_nodes = len(nodes_data)
        node_features = []
        
        # 构建邻接矩阵
        adjacency_matrix = self._build_adjacency_matrix(nodes_data)
        
        # 计算网络图
        G = nx.from_numpy_array(adjacency_matrix)
        
        # 计算中心性
        try:
            centrality = nx.betweenness_centrality(G)
        except:
            centrality = {i: 0.0 for i in range(num_nodes)}
        
        for i, node_data in enumerate(nodes_data):
            if node_data[3] <= 0:  # 死亡节点
                continue
            
            # 计算邻居平均能量
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            neighbor_energies = [nodes_data[j][2] for j in neighbors if nodes_data[j][3] > 0]
            neighbor_energy_avg = np.mean(neighbor_energies) if neighbor_energies else 0
            
            # 计算到基站距离
            distance_to_bs = np.linalg.norm(node_data[:2] - base_station_pos)
            
            node_feature = NodeFeatures(
                node_id=i,
                position=node_data[:2],
                energy=node_data[2],
                degree=len(neighbors),
                centrality=centrality.get(i, 0.0),
                cluster_id=-1,  # 待分配
                is_cluster_head=False,  # 待确定
                distance_to_bs=distance_to_bs,
                neighbor_energy_avg=neighbor_energy_avg
            )
            
            node_features.append(node_feature)
        
        return node_features
    
    def _build_adjacency_matrix(self, nodes_data: np.ndarray) -> np.ndarray:
        """构建邻接矩阵"""
        num_nodes = len(nodes_data)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            if nodes_data[i][3] <= 0:  # 死亡节点
                continue
            for j in range(i + 1, num_nodes):
                if nodes_data[j][3] <= 0:  # 死亡节点
                    continue
                
                distance = np.linalg.norm(nodes_data[i][:2] - nodes_data[j][:2])
                if distance <= self.transmission_range:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
        
        return adjacency_matrix
    
    def create_graph_data(self, node_features: List[NodeFeatures],
                         adjacency_matrix: np.ndarray) -> Data:
        """创建PyTorch Geometric图数据"""
        # 节点特征矩阵
        node_feature_matrix = np.array([nf.to_vector() for nf in node_features])
        
        # 标准化特征
        if hasattr(self.feature_scaler, 'mean_'):
            node_feature_matrix = self.feature_scaler.transform(node_feature_matrix)
        else:
            node_feature_matrix = self.feature_scaler.fit_transform(node_feature_matrix)
        
        # 边索引
        edge_indices = np.where(adjacency_matrix > 0)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # 转换为张量
        x = torch.tensor(node_feature_matrix, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)
    
    def generate_training_labels(self, node_features: List[NodeFeatures],
                               chains: List[ChainTopology]) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练标签"""
        num_nodes = len(node_features)
        
        # 节点角色标签 (0: 普通节点, 1: 链头, 2: 中继节点)
        role_labels = torch.zeros(num_nodes, dtype=torch.long)
        
        # 能耗标签（模拟真实能耗）
        energy_labels = torch.zeros(num_nodes, dtype=torch.float)
        
        node_id_to_idx = {nf.node_id: i for i, nf in enumerate(node_features)}
        
        for chain in chains:
            for i, node_id in enumerate(chain.nodes):
                if node_id in node_id_to_idx:
                    idx = node_id_to_idx[node_id]
                    
                    if i == 0:  # 链头
                        role_labels[idx] = 1
                        energy_labels[idx] = 0.01  # 链头基础能耗
                    else:  # 中继节点
                        role_labels[idx] = 2
                        energy_labels[idx] = 0.005  # 中继节点能耗
                    
                    # 添加传输能耗
                    if i < len(chain.nodes) - 1:
                        next_node_id = chain.nodes[i + 1]
                        if next_node_id in node_id_to_idx:
                            current_pos = node_features[idx].position
                            next_idx = node_id_to_idx[next_node_id]
                            next_pos = node_features[next_idx].position
                            distance = np.linalg.norm(current_pos - next_pos)
                            energy_labels[idx] += 0.001 * (distance ** 2)
        
        return role_labels, energy_labels
    
    def train_gnn(self, training_data: List[Tuple[Data, torch.Tensor, torch.Tensor]],
                  epochs: int = 100) -> Dict:
        """训练GNN模型"""
        print(f"🚀 开始训练GNN模型，共{epochs}个epochs")
        
        self.gnn_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_classification_loss = 0
            total_regression_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for graph_data, role_labels, energy_labels in training_data:
                graph_data = graph_data.to(self.device)
                role_labels = role_labels.to(self.device)
                energy_labels = energy_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                node_embeddings, role_pred, energy_pred = self.gnn_model(
                    graph_data.x, graph_data.edge_index
                )
                
                # 计算损失
                classification_loss = self.criterion_classification(role_pred, role_labels)
                regression_loss = self.criterion_regression(energy_pred.squeeze(), energy_labels)
                
                # 总损失（加权组合）
                loss = classification_loss + 0.5 * regression_loss
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_regression_loss += regression_loss.item()
                
                # 计算准确率
                _, predicted = torch.max(role_pred.data, 1)
                total_predictions += role_labels.size(0)
                correct_predictions += (predicted == role_labels).sum().item()
            
            # 记录训练历史
            avg_loss = total_loss / len(training_data)
            avg_classification_loss = total_classification_loss / len(training_data)
            avg_regression_loss = total_regression_loss / len(training_data)
            accuracy = correct_predictions / total_predictions
            
            self.training_history['losses'].append(avg_loss)
            self.training_history['classification_losses'].append(avg_classification_loss)
            self.training_history['regression_losses'].append(avg_regression_loss)
            self.training_history['accuracies'].append(accuracy)
            
            # 进度报告
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Classification Loss={avg_classification_loss:.4f}, "
                      f"Regression Loss={avg_regression_loss:.4f}, "
                      f"Accuracy={accuracy:.3f}")
        
        print("✅ GNN模型训练完成！")
        return self.training_history
    
    def optimize_topology(self, nodes_data: np.ndarray, 
                         base_station_pos: np.ndarray) -> Tuple[List[ChainTopology], Dict]:
        """优化网络拓扑"""
        # 提取节点特征
        node_features = self.extract_node_features(nodes_data, base_station_pos)
        
        if len(node_features) == 0:
            return [], {'error': 'No alive nodes'}
        
        # 构建邻接矩阵
        adjacency_matrix = self._build_adjacency_matrix(nodes_data)
        
        # 创建图数据
        graph_data = self.create_graph_data(node_features, adjacency_matrix)
        
        # GNN预测
        self.gnn_model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            node_embeddings, role_pred, energy_pred = self.gnn_model(
                graph_data.x, graph_data.edge_index
            )
            
            # 获取预测结果
            predicted_roles = torch.argmax(role_pred, dim=1).cpu().numpy()
            predicted_energies = energy_pred.squeeze().cpu().numpy()
        
        # 更新节点特征（基于GNN预测）
        for i, node_feature in enumerate(node_features):
            if predicted_roles[i] == 1:  # 链头
                node_feature.is_cluster_head = True
        
        # 构建优化的链式拓扑
        optimized_chains = self.chain_optimizer.construct_chains(
            node_features, adjacency_matrix
        )
        
        # 计算拓扑质量指标
        topology_metrics = self._calculate_topology_metrics(
            optimized_chains, node_features, predicted_energies
        )
        
        return optimized_chains, topology_metrics
    
    def _calculate_topology_metrics(self, chains: List[ChainTopology],
                                  node_features: List[NodeFeatures],
                                  predicted_energies: np.ndarray) -> Dict:
        """计算拓扑质量指标"""
        if not chains:
            return {
                'chain_count': 0,
                'average_chain_length': 0,
                'total_energy_cost': 0,
                'average_reliability': 0,
                'topology_efficiency': 0,
                'coverage_ratio': 0
            }
        
        # 基本统计
        chain_count = len(chains)
        chain_lengths = [chain.length for chain in chains]
        average_chain_length = np.mean(chain_lengths)
        
        # 能耗统计
        total_energy_cost = sum(chain.energy_cost for chain in chains)
        
        # 可靠性统计
        reliabilities = [chain.reliability for chain in chains]
        average_reliability = np.mean(reliabilities)
        
        # 拓扑效率
        efficiencies = [chain.get_chain_efficiency() for chain in chains]
        topology_efficiency = np.mean(efficiencies) if efficiencies else 0
        
        # 覆盖率（链中节点占总存活节点的比例）
        nodes_in_chains = set()
        for chain in chains:
            nodes_in_chains.update(chain.nodes)
        coverage_ratio = len(nodes_in_chains) / len(node_features) if node_features else 0
        
        return {
            'chain_count': chain_count,
            'average_chain_length': average_chain_length,
            'total_energy_cost': total_energy_cost,
            'average_reliability': average_reliability,
            'topology_efficiency': topology_efficiency,
            'coverage_ratio': coverage_ratio,
            'chain_details': [
                {
                    'chain_id': chain.chain_id,
                    'nodes': chain.nodes,
                    'head_node': chain.head_node,
                    'length': chain.length,
                    'energy_cost': chain.energy_cost,
                    'reliability': chain.reliability,
                    'efficiency': chain.get_chain_efficiency()
                }
                for chain in chains
            ]
        }
    
    def simulate_network_operation(self, initial_nodes_data: np.ndarray,
                                 base_station_pos: np.ndarray,
                                 max_rounds: int = 200) -> Dict:
        """模拟网络运行"""
        print("🔍 开始GNN-CTO网络运行模拟...")
        
        simulation_results = {
            'network_lifetime': 0,
            'total_energy_consumption': 0,
            'topology_evolution': [],
            'performance_metrics': []
        }
        
        nodes_data = initial_nodes_data.copy()
        initial_energy = np.sum(nodes_data[:, 2])
        
        for round_num in range(max_rounds):
            # 优化拓扑
            chains, topology_metrics = self.optimize_topology(nodes_data, base_station_pos)
            
            if not chains:
                simulation_results['network_lifetime'] = round_num
                break
            
            # 模拟能量消耗
            energy_consumption = self._simulate_energy_consumption(
                nodes_data, chains, base_station_pos
            )
            
            # 更新节点状态
            nodes_data[:, 2] -= energy_consumption
            nodes_data[nodes_data[:, 2] <= 0, 3] = 0  # 标记死亡节点
            
            # 记录轮次信息
            alive_count = np.sum(nodes_data[:, 3] > 0)
            remaining_energy = np.sum(nodes_data[:, 2])
            
            round_info = {
                'round': round_num + 1,
                'alive_nodes': alive_count,
                'remaining_energy': remaining_energy,
                'chain_count': len(chains),
                'topology_metrics': topology_metrics
            }
            
            simulation_results['topology_evolution'].append(round_info)
            
            # 检查网络死亡条件
            if alive_count <= len(initial_nodes_data) * 0.1:  # 90%节点死亡
                simulation_results['network_lifetime'] = round_num + 1
                break
        
        # 计算最终指标
        final_energy = np.sum(nodes_data[:, 2])
        simulation_results['total_energy_consumption'] = initial_energy - final_energy
        
        # 计算平均性能指标
        if simulation_results['topology_evolution']:
            avg_metrics = {}
            for key in ['chain_count', 'topology_metrics']:
                if key == 'topology_metrics':
                    # 计算拓扑指标的平均值
                    metric_keys = ['topology_efficiency', 'average_reliability', 'coverage_ratio']
                    for metric_key in metric_keys:
                        values = [round_info[key].get(metric_key, 0) 
                                for round_info in simulation_results['topology_evolution']]
                        avg_metrics[f'avg_{metric_key}'] = np.mean(values)
                else:
                    values = [round_info[key] for round_info in simulation_results['topology_evolution']]
                    avg_metrics[f'avg_{key}'] = np.mean(values)
            
            simulation_results['performance_metrics'] = avg_metrics
        
        print(f"✅ 模拟完成！网络寿命: {simulation_results['network_lifetime']} 轮")
        return simulation_results
    
    def _simulate_energy_consumption(self, nodes_data: np.ndarray,
                                   chains: List[ChainTopology],
                                   base_station_pos: np.ndarray) -> np.ndarray:
        """模拟能量消耗"""
        energy_consumption = np.zeros(len(nodes_data))
        
        # 基础感知能耗
        base_sensing_energy = 0.001
        energy_consumption[nodes_data[:, 3] > 0] += base_sensing_energy
        
        # 链式传输能耗
        for chain in chains:
            for i, node_id in enumerate(chain.nodes):
                if node_id < len(nodes_data) and nodes_data[node_id][3] > 0:
                    # 链头额外能耗
                    if i == 0:
                        energy_consumption[node_id] += 0.01
                    
                    # 传输能耗
                    if i < len(chain.nodes) - 1:
                        next_node_id = chain.nodes[i + 1]
                        if next_node_id < len(nodes_data):
                            current_pos = nodes_data[node_id][:2]
                            next_pos = nodes_data[next_node_id][:2]
                            distance = np.linalg.norm(current_pos - next_pos)
                            energy_consumption[node_id] += 0.001 * (distance ** 2)
                    else:
                        # 最后一个节点传输到基站
                        distance = np.linalg.norm(nodes_data[node_id][:2] - base_station_pos)
                        energy_consumption[node_id] += 0.001 * (distance ** 2)
        
        return energy_consumption
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        model_data = {
            'model_state_dict': self.gnn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_scaler': {
                'mean_': self.feature_scaler.mean_.tolist() if hasattr(self.feature_scaler, 'mean_') else None,
                'scale_': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None
            },
            'training_history': self.training_history,
            'performance_history': self.performance_history,
            'hyperparameters': {
                'communication_range': self.communication_range,
                'learning_rate': self.learning_rate,
                'max_chain_length': self.chain_optimizer.max_chain_length,
                'energy_threshold': self.chain_optimizer.energy_threshold
            }
        }
        
        torch.save(model_data, filepath)
        print(f"✅ GNN-CTO模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载训练好的模型"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # 恢复模型状态
        self.gnn_model.load_state_dict(model_data['model_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        # 恢复特征标准化器
        if model_data['feature_scaler']['mean_'] is not None:
            self.feature_scaler.mean_ = np.array(model_data['feature_scaler']['mean_'])
            self.feature_scaler.scale_ = np.array(model_data['feature_scaler']['scale_'])
        
        # 恢复历史记录
        self.training_history = model_data['training_history']
        self.performance_history = model_data['performance_history']
        
        print(f"✅ GNN-CTO模型已从 {filepath} 加载")

def demonstrate_gnn_cto():
    """演示GNN-CTO算法"""
    print("🎯 GNN-CTO算法演示")
    print("=" * 60)
    
    # 生成模拟网络数据
    np.random.seed(42)
    num_nodes = 54
    
    # 节点位置（随机分布在50x50区域）
    positions = np.random.uniform(0, 50, (num_nodes, 2))
    
    # 初始能量（归一化到0-1）
    initial_energy = np.ones(num_nodes)
    
    # 存活状态（1为存活，0为死亡）
    alive_status = np.ones(num_nodes)
    
    # 组合节点数据 [x, y, energy, alive]
    nodes_data = np.column_stack([positions, initial_energy, alive_status])
    
    # 基站位置
    base_station_pos = np.array([25.0, 25.0])
    
    # 创建GNN-CTO算法实例
    gnn_cto = GNNCTOAlgorithm(communication_range=15.0, learning_rate=0.001)
    
    # 生成训练数据
    print("📊 生成训练数据...")
    training_data = []
    
    for _ in range(10):  # 生成10个训练样本
        # 随机扰动节点能量
        perturbed_data = nodes_data.copy()
        perturbed_data[:, 2] += np.random.normal(0, 0.1, num_nodes)
        perturbed_data[:, 2] = np.clip(perturbed_data[:, 2], 0, 1)
        
        # 提取特征
        node_features = gnn_cto.extract_node_features(perturbed_data, base_station_pos)
        adjacency_matrix = gnn_cto._build_adjacency_matrix(perturbed_data)
        
        # 构建链（用于生成标签）
        chains = gnn_cto.chain_optimizer.construct_chains(node_features, adjacency_matrix)
        
        # 创建图数据
        graph_data = gnn_cto.create_graph_data(node_features, adjacency_matrix)
        
        # 生成标签
        role_labels, energy_labels = gnn_cto.generate_training_labels(node_features, chains)
        
        training_data.append((graph_data, role_labels, energy_labels))
    
    # 训练GNN模型
    training_history = gnn_cto.train_gnn(training_data, num_epochs=50)
    
    # 运行网络模拟
    simulation_results = gnn_cto.simulate_network_operation(
        initial_nodes_data=nodes_data,
        base_station_pos=base_station_pos,
        max_rounds=200
    )
    
    # 保存结果
    results_dir = "results/gnn_cto"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存模型
    gnn_cto.save_model(f"{results_dir}/gnn_cto_model.pth")
    
    # 保存模拟结果
    with open(f"{results_dir}/simulation_results.json", 'w') as f:
        json.dump(simulation_results, f, indent=2)
    
    # 打印结果摘要
    print("\n📊 GNN-CTO算法性能摘要")
    print("=" * 60)
    print(f"🔋 网络寿命: {simulation_results['network_lifetime']} 轮")
    print(f"⚡ 总能耗: {simulation_results['total_energy_consumption']:.3f}")
    
    if simulation_results['performance_metrics']:
        metrics = simulation_results['performance_metrics']
        print(f"🔗 平均链数: {metrics.get('avg_chain_count', 0):.1f}")
        print(f"📈 平均拓扑效率: {metrics.get('avg_topology_efficiency', 0):.3f}")
        print(f"🎯 平均可靠性: {metrics.get('avg_average_reliability', 0):.3f}")
        print(f"📊 平均覆盖率: {metrics.get('avg_coverage_ratio', 0):.3f}")
    
    print(f"🧠 GNN训练准确率: {training_history['accuracies'][-1]:.3f}")
    print(f"📉 最终训练损失: {training_history['losses'][-1]:.4f}")
    
    return gnn_cto, training_history, simulation_results

if __name__ == "__main__":
    # 运行演示
    gnn_cto, training_history, simulation_results = demonstrate_gnn_cto()