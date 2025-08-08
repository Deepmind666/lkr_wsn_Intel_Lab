"""
AFW-RL: Adaptive Fuzzy Weight Reinforcement Learning Algorithm
自适应模糊逻辑权重强化学习算法

核心创新：
1. 将静态模糊逻辑权重转化为动态学习策略
2. Q-Learning与模糊逻辑的深度融合
3. 实时网络状态感知和权重自适应调整

作者: WSN研究团队
日期: 2025年1月
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import pickle
from dataclasses import dataclass
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class NetworkState:
    """网络状态表示"""
    remaining_energy_ratio: float  # 剩余能量比例
    alive_nodes_ratio: float      # 存活节点比例
    network_connectivity: float   # 网络连通性
    energy_variance: float        # 能量方差（均衡性）
    avg_node_degree: float        # 平均节点度
    
    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.remaining_energy_ratio,
            self.alive_nodes_ratio,
            self.network_connectivity,
            self.energy_variance,
            self.avg_node_degree
        ])
    
    def discretize(self, bins: int = 5) -> int:
        """状态离散化"""
        vector = self.to_vector()
        # 将连续状态映射到离散状态空间
        discrete_values = []
        for val in vector:
            discrete_val = min(int(val * bins), bins - 1)
            discrete_values.append(discrete_val)
        
        # 组合成单一状态ID
        state_id = 0
        for i, val in enumerate(discrete_values):
            state_id += val * (bins ** i)
        
        return state_id

@dataclass
class FuzzyWeightAction:
    """模糊权重动作"""
    energy_weight: float     # 能量权重
    location_weight: float   # 位置权重
    connectivity_weight: float  # 连通性权重
    
    def __post_init__(self):
        """确保权重和为1"""
        total = self.energy_weight + self.location_weight + self.connectivity_weight
        if total > 0:
            self.energy_weight /= total
            self.location_weight /= total
            self.connectivity_weight /= total
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        return np.array([self.energy_weight, self.location_weight, self.connectivity_weight])

class FuzzyLogicClusterHead:
    """改进的模糊逻辑簇头选择"""
    
    def __init__(self, weights: FuzzyWeightAction):
        self.weights = weights
    
    def fuzzy_membership_energy(self, energy_ratio: float) -> float:
        """能量模糊隶属度函数 - 改进版"""
        if energy_ratio >= 0.8:
            return 1.0
        elif energy_ratio >= 0.6:
            return 0.8 + 0.2 * (energy_ratio - 0.6) / 0.2
        elif energy_ratio >= 0.4:
            return 0.5 + 0.3 * (energy_ratio - 0.4) / 0.2
        elif energy_ratio >= 0.2:
            return 0.2 + 0.3 * (energy_ratio - 0.2) / 0.2
        else:
            return 0.1 * energy_ratio / 0.2
    
    def fuzzy_membership_location(self, distance_ratio: float) -> float:
        """位置模糊隶属度函数 - 改进版"""
        # 距离基站越近越好
        if distance_ratio <= 0.2:
            return 1.0
        elif distance_ratio <= 0.4:
            return 0.8 + 0.2 * (0.4 - distance_ratio) / 0.2
        elif distance_ratio <= 0.6:
            return 0.5 + 0.3 * (0.6 - distance_ratio) / 0.2
        elif distance_ratio <= 0.8:
            return 0.2 + 0.3 * (0.8 - distance_ratio) / 0.2
        else:
            return 0.1 * (1.0 - distance_ratio) / 0.2
    
    def fuzzy_membership_connectivity(self, neighbor_count: int, max_neighbors: int = 10) -> float:
        """连通性模糊隶属度函数 - 改进版"""
        ratio = min(neighbor_count / max_neighbors, 1.0)
        if ratio >= 0.8:
            return 1.0
        elif ratio >= 0.6:
            return 0.8 + 0.2 * (ratio - 0.6) / 0.2
        elif ratio >= 0.4:
            return 0.5 + 0.3 * (ratio - 0.4) / 0.2
        elif ratio >= 0.2:
            return 0.2 + 0.3 * (ratio - 0.2) / 0.2
        else:
            return 0.1 * ratio / 0.2
    
    def calculate_fuzzy_score(self, energy_ratio: float, distance_ratio: float, 
                            neighbor_count: int) -> float:
        """计算模糊综合评分"""
        energy_score = self.fuzzy_membership_energy(energy_ratio)
        location_score = self.fuzzy_membership_location(distance_ratio)
        connectivity_score = self.fuzzy_membership_connectivity(neighbor_count)
        
        # 使用动态权重
        fuzzy_score = (self.weights.energy_weight * energy_score + 
                      self.weights.location_weight * location_score + 
                      self.weights.connectivity_weight * connectivity_score)
        
        return fuzzy_score

class QLearningAgent:
    """Q-Learning智能体"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-Table初始化
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # 训练历史
        self.training_history = {
            'rewards': [],
            'epsilon_values': [],
            'q_values_mean': [],
            'actions_taken': []
        }
    
    def get_action(self, state: int) -> int:
        """ε-贪婪策略选择动作"""
        if np.random.random() <= self.epsilon:
            # 探索：随机选择动作
            action = np.random.choice(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state]
            action = np.argmax(q_values)
        
        self.training_history['actions_taken'].append(action)
        return action
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int):
        """更新Q表"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-Learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # 记录训练历史
        self.training_history['rewards'].append(reward)
        self.training_history['epsilon_values'].append(self.epsilon)
        
        # 计算Q值均值
        all_q_values = []
        for state_q in self.q_table.values():
            all_q_values.extend(state_q)
        self.training_history['q_values_mean'].append(np.mean(all_q_values))
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AFWRLAlgorithm:
    """AFW-RL主算法类"""
    
    def __init__(self, num_nodes: int = 54, num_clusters: int = 6):
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        
        # 预定义的权重动作空间
        self.action_space = self._create_action_space()
        self.action_size = len(self.action_space)
        
        # Q-Learning智能体
        self.q_agent = QLearningAgent(
            state_size=5**5,  # 5个状态特征，每个5个离散值
            action_size=self.action_size,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # 性能历史
        self.performance_history = {
            'network_lifetime': [],
            'energy_consumption': [],
            'cluster_quality': [],
            'convergence_rounds': []
        }
        
        # 当前网络状态
        self.current_state = None
        self.current_weights = None
        
    def _create_action_space(self) -> List[FuzzyWeightAction]:
        """创建离散的权重动作空间"""
        actions = []
        
        # 预定义的权重组合（确保多样性和合理性）
        weight_combinations = [
            (0.7, 0.2, 0.1),  # 高能量权重
            (0.6, 0.3, 0.1),  # 平衡能量-位置
            (0.5, 0.3, 0.2),  # 均衡权重
            (0.4, 0.4, 0.2),  # 位置优先
            (0.3, 0.5, 0.2),  # 高位置权重
            (0.4, 0.2, 0.4),  # 连通性优先
            (0.3, 0.3, 0.4),  # 高连通性权重
            (0.5, 0.2, 0.3),  # 能量-连通性平衡
            (0.2, 0.4, 0.4),  # 位置-连通性平衡
            (0.8, 0.1, 0.1),  # 极高能量权重
        ]
        
        for w_e, w_l, w_c in weight_combinations:
            actions.append(FuzzyWeightAction(w_e, w_l, w_c))
        
        return actions
    
    def calculate_network_state(self, nodes_data: np.ndarray, 
                              base_station_pos: np.ndarray) -> NetworkState:
        """计算当前网络状态"""
        # 假设nodes_data格式: [x, y, energy, alive]
        alive_nodes = nodes_data[nodes_data[:, 3] > 0]
        
        if len(alive_nodes) == 0:
            return NetworkState(0, 0, 0, 0, 0)
        
        # 剩余能量比例
        total_energy = np.sum(alive_nodes[:, 2])
        initial_energy = len(alive_nodes) * 1.0  # 假设初始能量为1
        remaining_energy_ratio = min(total_energy / initial_energy, 1.0)
        
        # 存活节点比例
        alive_nodes_ratio = len(alive_nodes) / len(nodes_data)
        
        # 网络连通性（简化计算）
        distances = np.linalg.norm(
            alive_nodes[:, :2][:, np.newaxis] - alive_nodes[:, :2], axis=2
        )
        communication_range = 15.0  # 通信范围
        connectivity_matrix = distances <= communication_range
        avg_connectivity = np.mean(np.sum(connectivity_matrix, axis=1) - 1)  # 减去自己
        network_connectivity = min(avg_connectivity / 10.0, 1.0)  # 归一化
        
        # 能量方差（均衡性）
        energy_variance = np.var(alive_nodes[:, 2]) if len(alive_nodes) > 1 else 0
        energy_variance = min(energy_variance, 1.0)
        
        # 平均节点度
        avg_node_degree = avg_connectivity
        
        return NetworkState(
            remaining_energy_ratio=remaining_energy_ratio,
            alive_nodes_ratio=alive_nodes_ratio,
            network_connectivity=network_connectivity,
            energy_variance=energy_variance,
            avg_node_degree=avg_node_degree
        )
    
    def calculate_reward(self, prev_state: NetworkState, current_state: NetworkState,
                        cluster_quality: float) -> float:
        """计算奖励函数"""
        # 多目标奖励设计
        
        # 1. 网络寿命奖励（存活节点比例变化）
        lifetime_reward = (current_state.alive_nodes_ratio - prev_state.alive_nodes_ratio) * 10
        
        # 2. 能量效率奖励（剩余能量比例变化）
        energy_reward = (current_state.remaining_energy_ratio - prev_state.remaining_energy_ratio) * 5
        
        # 3. 网络连通性奖励
        connectivity_reward = (current_state.network_connectivity - prev_state.network_connectivity) * 3
        
        # 4. 能量均衡奖励（方差越小越好）
        balance_reward = (prev_state.energy_variance - current_state.energy_variance) * 2
        
        # 5. 簇质量奖励
        cluster_reward = cluster_quality * 2
        
        # 综合奖励
        total_reward = (lifetime_reward + energy_reward + connectivity_reward + 
                       balance_reward + cluster_reward)
        
        return total_reward
    
    def select_cluster_heads(self, nodes_data: np.ndarray, 
                           base_station_pos: np.ndarray) -> Tuple[List[int], float]:
        """使用当前权重选择簇头"""
        if self.current_weights is None:
            # 使用默认权重
            self.current_weights = FuzzyWeightAction(0.5, 0.3, 0.2)
        
        fuzzy_logic = FuzzyLogicClusterHead(self.current_weights)
        alive_nodes = nodes_data[nodes_data[:, 3] > 0]
        
        if len(alive_nodes) < self.num_clusters:
            # 如果存活节点少于簇数，全部作为簇头
            return list(range(len(alive_nodes))), 0.5
        
        # 计算每个节点的模糊评分
        scores = []
        for i, node in enumerate(alive_nodes):
            # 能量比例
            energy_ratio = node[2]  # 假设已归一化
            
            # 距离比例
            distance = np.linalg.norm(node[:2] - base_station_pos)
            max_distance = 50.0  # 假设最大距离
            distance_ratio = min(distance / max_distance, 1.0)
            
            # 邻居数量
            distances = np.linalg.norm(alive_nodes[:, :2] - node[:2], axis=1)
            neighbor_count = np.sum(distances <= 15.0) - 1  # 减去自己
            
            score = fuzzy_logic.calculate_fuzzy_score(
                energy_ratio, distance_ratio, neighbor_count
            )
            scores.append((i, score))
        
        # 选择评分最高的节点作为簇头
        scores.sort(key=lambda x: x[1], reverse=True)
        cluster_heads = [scores[i][0] for i in range(min(self.num_clusters, len(scores)))]
        
        # 计算簇质量
        cluster_quality = np.mean([scores[i][1] for i in range(len(cluster_heads))])
        
        return cluster_heads, cluster_quality
    
    def train_episode(self, nodes_data: np.ndarray, base_station_pos: np.ndarray,
                     max_rounds: int = 100) -> Dict:
        """训练一个episode"""
        episode_results = {
            'total_reward': 0,
            'rounds_survived': 0,
            'final_energy': 0,
            'cluster_quality_history': []
        }
        
        # 初始化网络状态
        prev_state = self.calculate_network_state(nodes_data, base_station_pos)
        prev_state_id = prev_state.discretize()
        
        for round_num in range(max_rounds):
            # 选择动作（权重组合）
            action_id = self.q_agent.get_action(prev_state_id)
            self.current_weights = self.action_space[action_id]
            
            # 执行簇头选择
            cluster_heads, cluster_quality = self.select_cluster_heads(
                nodes_data, base_station_pos
            )
            
            # 模拟能量消耗（简化）
            energy_consumption = self._simulate_energy_consumption(
                nodes_data, cluster_heads, base_station_pos
            )
            
            # 更新节点能量
            nodes_data[:, 2] -= energy_consumption
            nodes_data[nodes_data[:, 2] <= 0, 3] = 0  # 标记死亡节点
            
            # 计算新状态
            current_state = self.calculate_network_state(nodes_data, base_station_pos)
            current_state_id = current_state.discretize()
            
            # 计算奖励
            reward = self.calculate_reward(prev_state, current_state, cluster_quality)
            
            # 更新Q表
            self.q_agent.update_q_table(prev_state_id, action_id, reward, current_state_id)
            
            # 记录结果
            episode_results['total_reward'] += reward
            episode_results['rounds_survived'] = round_num + 1
            episode_results['cluster_quality_history'].append(cluster_quality)
            
            # 检查网络是否死亡
            if current_state.alive_nodes_ratio <= 0.1:  # 90%节点死亡
                break
            
            # 更新状态
            prev_state = current_state
            prev_state_id = current_state_id
        
        # 衰减探索率
        self.q_agent.decay_epsilon()
        
        # 记录最终能量
        episode_results['final_energy'] = np.sum(nodes_data[:, 2])
        
        return episode_results
    
    def _simulate_energy_consumption(self, nodes_data: np.ndarray, 
                                   cluster_heads: List[int], 
                                   base_station_pos: np.ndarray) -> np.ndarray:
        """模拟能量消耗"""
        energy_consumption = np.zeros(len(nodes_data))
        alive_nodes = nodes_data[nodes_data[:, 3] > 0]
        
        # 基础感知能耗
        base_sensing_energy = 0.001
        energy_consumption[nodes_data[:, 3] > 0] += base_sensing_energy
        
        # 簇头额外能耗
        cluster_head_energy = 0.005
        for ch_idx in cluster_heads:
            if ch_idx < len(alive_nodes):
                original_idx = np.where(nodes_data[:, 3] > 0)[0][ch_idx]
                energy_consumption[original_idx] += cluster_head_energy
        
        # 传输能耗（距离相关）
        transmission_energy_factor = 0.0001
        for i, node in enumerate(nodes_data):
            if node[3] > 0:  # 存活节点
                distance = np.linalg.norm(node[:2] - base_station_pos)
                transmission_energy = transmission_energy_factor * (distance ** 2)
                energy_consumption[i] += transmission_energy
        
        return energy_consumption
    
    def train(self, initial_nodes_data: np.ndarray, base_station_pos: np.ndarray,
              num_episodes: int = 100) -> Dict:
        """训练AFW-RL算法"""
        print(f"🚀 开始AFW-RL算法训练，共{num_episodes}个episodes")
        
        training_results = {
            'episode_rewards': [],
            'episode_lifetimes': [],
            'convergence_episode': None,
            'best_weights': None,
            'best_performance': 0
        }
        
        best_performance = -float('inf')
        convergence_threshold = 0.01
        recent_rewards = deque(maxlen=10)
        
        for episode in range(num_episodes):
            # 重置网络状态
            nodes_data = initial_nodes_data.copy()
            
            # 训练一个episode
            episode_result = self.train_episode(nodes_data, base_station_pos)
            
            # 记录结果
            training_results['episode_rewards'].append(episode_result['total_reward'])
            training_results['episode_lifetimes'].append(episode_result['rounds_survived'])
            
            recent_rewards.append(episode_result['total_reward'])
            
            # 检查是否为最佳性能
            if episode_result['total_reward'] > best_performance:
                best_performance = episode_result['total_reward']
                training_results['best_weights'] = self.current_weights
                training_results['best_performance'] = best_performance
            
            # 检查收敛
            if len(recent_rewards) == 10:
                reward_std = np.std(recent_rewards)
                if reward_std < convergence_threshold and training_results['convergence_episode'] is None:
                    training_results['convergence_episode'] = episode
                    print(f"✅ 算法在第{episode}个episode收敛")
            
            # 进度报告
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(training_results['episode_rewards'][-20:])
                avg_lifetime = np.mean(training_results['episode_lifetimes'][-20:])
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"平均奖励={avg_reward:.2f}, 平均寿命={avg_lifetime:.1f}, "
                      f"ε={self.q_agent.epsilon:.3f}")
        
        print(f"✅ AFW-RL训练完成！最佳性能: {best_performance:.2f}")
        return training_results
    
    def evaluate(self, nodes_data: np.ndarray, base_station_pos: np.ndarray,
                max_rounds: int = 200) -> Dict:
        """评估训练后的算法性能"""
        print("🔍 评估AFW-RL算法性能...")
        
        # 设置为评估模式（不探索）
        original_epsilon = self.q_agent.epsilon
        self.q_agent.epsilon = 0.0
        
        evaluation_results = {
            'network_lifetime': 0,
            'total_energy_consumption': 0,
            'average_cluster_quality': 0,
            'energy_efficiency': 0,
            'round_details': []
        }
        
        initial_energy = np.sum(nodes_data[:, 2])
        cluster_qualities = []
        
        for round_num in range(max_rounds):
            # 计算网络状态
            current_state = self.calculate_network_state(nodes_data, base_station_pos)
            state_id = current_state.discretize()
            
            # 选择最优动作
            action_id = self.q_agent.get_action(state_id)
            self.current_weights = self.action_space[action_id]
            
            # 执行簇头选择
            cluster_heads, cluster_quality = self.select_cluster_heads(
                nodes_data, base_station_pos
            )
            cluster_qualities.append(cluster_quality)
            
            # 模拟能量消耗
            energy_consumption = self._simulate_energy_consumption(
                nodes_data, cluster_heads, base_station_pos
            )
            
            # 更新节点状态
            nodes_data[:, 2] -= energy_consumption
            nodes_data[nodes_data[:, 2] <= 0, 3] = 0
            
            # 记录轮次详情
            alive_count = np.sum(nodes_data[:, 3] > 0)
            remaining_energy = np.sum(nodes_data[:, 2])
            
            evaluation_results['round_details'].append({
                'round': round_num + 1,
                'alive_nodes': alive_count,
                'remaining_energy': remaining_energy,
                'cluster_quality': cluster_quality,
                'selected_weights': self.current_weights.to_vector().tolist()
            })
            
            # 检查网络死亡
            if alive_count <= len(nodes_data) * 0.1:  # 90%节点死亡
                evaluation_results['network_lifetime'] = round_num + 1
                break
        
        # 计算最终指标
        final_energy = np.sum(nodes_data[:, 2])
        evaluation_results['total_energy_consumption'] = initial_energy - final_energy
        evaluation_results['average_cluster_quality'] = np.mean(cluster_qualities)
        evaluation_results['energy_efficiency'] = (
            evaluation_results['network_lifetime'] / evaluation_results['total_energy_consumption']
            if evaluation_results['total_energy_consumption'] > 0 else 0
        )
        
        # 恢复原始epsilon
        self.q_agent.epsilon = original_epsilon
        
        print(f"✅ 评估完成！网络寿命: {evaluation_results['network_lifetime']} 轮")
        return evaluation_results
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        model_data = {
            'q_table': dict(self.q_agent.q_table),
            'action_space': [action.to_vector().tolist() for action in self.action_space],
            'training_history': self.q_agent.training_history,
            'performance_history': self.performance_history,
            'hyperparameters': {
                'learning_rate': self.q_agent.learning_rate,
                'discount_factor': self.q_agent.discount_factor,
                'epsilon': self.q_agent.epsilon,
                'num_nodes': self.num_nodes,
                'num_clusters': self.num_clusters
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载训练好的模型"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # 恢复Q表
        self.q_agent.q_table = defaultdict(lambda: np.zeros(self.action_size))
        for state_str, q_values in model_data['q_table'].items():
            self.q_agent.q_table[int(state_str)] = np.array(q_values)
        
        # 恢复其他参数
        self.q_agent.training_history = model_data['training_history']
        self.performance_history = model_data['performance_history']
        
        hyperparams = model_data['hyperparameters']
        self.q_agent.learning_rate = hyperparams['learning_rate']
        self.q_agent.discount_factor = hyperparams['discount_factor']
        self.q_agent.epsilon = hyperparams['epsilon']
        
        print(f"✅ 模型已从 {filepath} 加载")

def demonstrate_afw_rl():
    """演示AFW-RL算法"""
    print("🎯 AFW-RL算法演示")
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
    
    # 创建AFW-RL算法实例
    afw_rl = AFWRLAlgorithm(num_nodes=num_nodes, num_clusters=6)
    
    # 训练算法
    training_results = afw_rl.train(
        initial_nodes_data=nodes_data,
        base_station_pos=base_station_pos,
        num_episodes=50  # 演示用较少episodes
    )
    
    # 评估算法
    evaluation_results = afw_rl.evaluate(
        nodes_data=nodes_data.copy(),
        base_station_pos=base_station_pos,
        max_rounds=200
    )
    
    # 保存结果
    results_dir = "results/afw_rl"
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存模型
    afw_rl.save_model(f"{results_dir}/afw_rl_model.json")
    
    # 保存评估结果
    with open(f"{results_dir}/evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # 打印结果摘要
    print("\n📊 AFW-RL算法性能摘要")
    print("=" * 60)
    print(f"🔋 网络寿命: {evaluation_results['network_lifetime']} 轮")
    print(f"⚡ 总能耗: {evaluation_results['total_energy_consumption']:.3f}")
    print(f"🎯 平均簇质量: {evaluation_results['average_cluster_quality']:.3f}")
    print(f"📈 能量效率: {evaluation_results['energy_efficiency']:.3f}")
    print(f"🏆 最佳权重: {training_results['best_weights'].to_vector()}")
    print(f"🎖️ 收敛轮次: {training_results['convergence_episode']}")
    
    return afw_rl, training_results, evaluation_results

if __name__ == "__main__":
    # 运行演示
    afw_rl, training_results, evaluation_results = demonstrate_afw_rl()