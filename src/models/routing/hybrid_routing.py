#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能节能路由协议

该模块实现了融合元启发式优化、时序预测和模糊逻辑可靠性的WSN混合智能节能路由协议。
该协议通过综合考虑能量消耗、预测误差和数据可靠性，优化路由决策，延长网络生命周期。
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 导入相关模块
from src.models.metaheuristic.pso import PSO
from src.models.time_series.lstm import LSTMPredictor
from src.models.reliability.fuzzy_logic import FuzzyReliabilityModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridEnergyEfficientRouting:
    """
    混合智能节能路由协议
    
    该类实现了融合元启发式优化、时序预测和模糊逻辑可靠性的WSN混合智能节能路由协议。
    """
    
    def __init__(self, network, alpha=0.6, beta=0.3, gamma=0.1, pso_params=None, lstm_params=None):
        """
        初始化混合智能节能路由协议
        
        Args:
            network: WSN网络对象
            alpha: 能耗权重
            beta: 预测误差权重
            gamma: 数据可靠性权重
            pso_params: PSO参数字典
            lstm_params: LSTM参数字典
        """
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 初始化PSO参数
        self.pso_params = {
            'n_particles': 30,
            'dimensions': 10,  # 根据网络规模调整
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'max_iter': 50
        }
        if pso_params:
            self.pso_params.update(pso_params)
        
        # 初始化LSTM参数
        self.lstm_params = {
            'input_size': 4,  # 温度、湿度、光照、电压
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 4,
            'seq_length': 24
        }
        if lstm_params:
            self.lstm_params.update(lstm_params)
        
        # 初始化模型
        self._init_models()
        
        # 初始化路由表
        self.routing_table = {}
        
        # 初始化性能指标
        self.metrics = {
            'energy_consumption': [],
            'network_lifetime': 0,
            'packet_delivery_ratio': 0,
            'end_to_end_delay': 0,
            'prediction_accuracy': 0,
            'reliability': 0
        }
        
        logger.info("混合智能节能路由协议初始化完成")
        logger.info(f"能耗权重: {alpha}, 预测误差权重: {beta}, 数据可靠性权重: {gamma}")
    
    def _init_models(self):
        """
        初始化模型
        """
        # 初始化PSO模型
        self.pso_model = None  # 将在优化路由时动态创建
        
        # 初始化LSTM预测模型
        self.lstm_predictors = {}
        for node_id in self.network.get_node_ids():
            self.lstm_predictors[node_id] = LSTMPredictor(
                input_size=self.lstm_params['input_size'],
                hidden_size=self.lstm_params['hidden_size'],
                num_layers=self.lstm_params['num_layers'],
                output_size=self.lstm_params['output_size'],
                seq_length=self.lstm_params['seq_length']
            )
        
        # 初始化模糊逻辑可靠性模型
        self.reliability_model = FuzzyReliabilityModel()
        
        logger.info("模型初始化完成")
    
    def train_prediction_models(self, training_data):
        """
        训练预测模型
        
        Args:
            training_data: 训练数据，格式为 {node_id: data_array}
            
        Returns:
            dict: 训练指标
        """
        logger.info("开始训练预测模型")
        
        training_metrics = {}
        
        for node_id, data in tqdm(training_data.items(), desc="训练节点预测模型"):
            if node_id not in self.lstm_predictors:
                logger.warning(f"节点 {node_id} 不在预测器列表中，跳过训练")
                continue
            
            # 划分训练集和验证集
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            val_data = data[train_size:]
            
            # 训练模型
            history = self.lstm_predictors[node_id].train(
                train_data=train_data,
                val_data=val_data,
                epochs=50,
                batch_size=32,
                patience=5,
                verbose=False
            )
            
            # 记录训练指标
            training_metrics[node_id] = {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1] if 'val_loss' in history else None
            }
            
            logger.info(f"节点 {node_id} 预测模型训练完成，训练损失: {training_metrics[node_id]['train_loss']:.6f}")
        
        logger.info("预测模型训练完成")
        return training_metrics
    
    def _create_fitness_function(self, source_node, sink_node):
        """
        创建适应度函数
        
        Args:
            source_node: 源节点ID
            sink_node: 汇聚节点ID
            
        Returns:
            function: 适应度函数
        """
        def fitness_function(solution):
            # 解码路由路径
            path = self._decode_routing_path(solution, source_node, sink_node)
            
            # 如果路径无效，返回极大值
            if not path or path[-1] != sink_node:
                return float('inf')
            
            # 计算能耗
            energy_consumption = self._calculate_energy_consumption(path)
            
            # 计算预测误差
            prediction_error = self._calculate_prediction_error(path)
            
            # 计算数据可靠性
            data_reliability = self._calculate_data_reliability(path)
            
            # 计算加权适应度
            fitness = self.alpha * energy_consumption + \
                     self.beta * prediction_error + \
                     self.gamma * (1 - data_reliability)
            
            return fitness
        
        return fitness_function
    
    def _decode_routing_path(self, solution, source_node, sink_node):
        """
        将PSO解决方案解码为路由路径
        
        Args:
            solution: PSO解决方案
            source_node: 源节点ID
            sink_node: 汇聚节点ID
            
        Returns:
            list: 路由路径（节点ID列表）
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id in self.network.get_node_ids():
            G.add_node(node_id)
        
        # 添加边和权重
        for i, node_id in enumerate(self.network.get_node_ids()):
            neighbors = self.network.get_neighbors(node_id)
            for j, neighbor in enumerate(neighbors):
                # 使用solution中的值作为边的权重
                if i < len(solution) and j < len(neighbors):
                    weight = abs(solution[i]) + 1e-6  # 确保权重为正
                    G.add_edge(node_id, neighbor, weight=weight)
        
        # 使用Dijkstra算法找到最短路径
        try:
            path = nx.dijkstra_path(G, source=source_node, target=sink_node, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def _calculate_energy_consumption(self, path):
        """
        计算路由路径的能量消耗
        
        Args:
            path: 路由路径
            
        Returns:
            float: 能量消耗
        """
        energy = 0.0
        for i in range(len(path) - 1):
            energy += self.network.get_energy_cost(path[i], path[i+1])
        return energy
    
    def _calculate_prediction_error(self, path):
        """
        计算路由路径的预测误差
        
        Args:
            path: 路由路径
            
        Returns:
            float: 预测误差
        """
        error = 0.0
        for node_id in path:
            error += self.network.get_prediction_error(node_id)
        return error / len(path) if path else float('inf')
    
    def _calculate_data_reliability(self, path):
        """
        计算路由路径的数据可靠性
        
        Args:
            path: 路由路径
            
        Returns:
            float: 数据可靠性（0-1之间，越大越好）
        """
        return self.reliability_model.evaluate_path_reliability(path, self.network)
    
    def optimize_route(self, source_node, sink_node):
        """
        优化从源节点到汇聚节点的路由
        
        Args:
            source_node: 源节点ID
            sink_node: 汇聚节点ID
            
        Returns:
            list: 优化后的路由路径
        """
        logger.info(f"优化从节点 {source_node} 到节点 {sink_node} 的路由")
        
        # 创建适应度函数
        fitness_func = self._create_fitness_function(source_node, sink_node)
        
        # 设置PSO参数
        dimensions = min(self.pso_params['dimensions'], len(self.network.get_node_ids()))
        bounds = [(-1, 1)] * dimensions  # 使用归一化的边权重
        
        # 创建PSO实例
        self.pso_model = PSO(
            n_particles=self.pso_params['n_particles'],
            dimensions=dimensions,
            bounds=bounds,
            w=self.pso_params['w'],
            c1=self.pso_params['c1'],
            c2=self.pso_params['c2'],
            max_iter=self.pso_params['max_iter'],
            fitness_func=fitness_func,
            minimize=True
        )
        
        # 运行优化
        best_position, best_fitness, _ = self.pso_model.optimize(verbose=False)
        
        # 解码最优路径
        best_path = self._decode_routing_path(best_position, source_node, sink_node)
        
        if best_path:
            logger.info(f"优化完成，最优路径: {best_path}，适应度: {best_fitness:.6f}")
            
            # 更新路由表
            self.routing_table[(source_node, sink_node)] = best_path
            
            # 计算性能指标
            energy = self._calculate_energy_consumption(best_path)
            self.metrics['energy_consumption'].append(energy)
            
            return best_path
        else:
            logger.warning(f"未找到从节点 {source_node} 到节点 {sink_node} 的有效路径")
            return None
    
    def predict_sensor_data(self, node_id, sequence, horizon=1):
        """
        预测传感器数据
        
        Args:
            node_id: 节点ID
            sequence: 历史序列数据
            horizon: 预测步数
            
        Returns:
            np.ndarray: 预测结果
        """
        if node_id not in self.lstm_predictors:
            logger.warning(f"节点 {node_id} 不在预测器列表中")
            return None
        
        # 预测未来数据
        predictions = self.lstm_predictors[node_id].predict_next_n_steps(sequence, horizon)
        
        return predictions
    
    def evaluate_reliability(self, path):
        """
        评估路径可靠性
        
        Args:
            path: 路由路径
            
        Returns:
            float: 路径可靠性
        """
        return self.reliability_model.evaluate_path_reliability(path, self.network)
    
    def update_network_state(self):
        """
        更新网络状态
        """
        # 更新节点能量
        for node_id in self.network.get_node_ids():
            self.network.update_node_energy(node_id)
        
        # 更新链路状态
        for node_id in self.network.get_node_ids():
            neighbors = self.network.get_neighbors(node_id)
            for neighbor in neighbors:
                self.network.update_link_state(node_id, neighbor)
        
        logger.info("网络状态已更新")
    
    def route_packet(self, source_node, sink_node, data=None):
        """
        路由数据包
        
        Args:
            source_node: 源节点ID
            sink_node: 汇聚节点ID
            data: 数据包内容
            
        Returns:
            bool: 路由是否成功
        """
        # 检查路由表中是否有缓存的路径
        if (source_node, sink_node) in self.routing_table:
            path = self.routing_table[(source_node, sink_node)]
            
            # 检查路径是否仍然有效
            valid_path = True
            for i in range(len(path) - 1):
                if not self.network.is_link_active(path[i], path[i+1]):
                    valid_path = False
                    break
            
            if not valid_path:
                # 重新优化路由
                logger.info(f"路径 {path} 不再有效，重新优化路由")
                path = self.optimize_route(source_node, sink_node)
        else:
            # 优化路由
            path = self.optimize_route(source_node, sink_node)
        
        if not path:
            logger.warning(f"无法找到从节点 {source_node} 到节点 {sink_node} 的路由路径")
            return False
        
        # 模拟数据包传输
        success = self.network.transmit_packet(path, data)
        
        if success:
            logger.info(f"数据包从节点 {source_node} 成功路由到节点 {sink_node}")
        else:
            logger.warning(f"数据包从节点 {source_node} 到节点 {sink_node} 的路由失败")
        
        return success
    
    def calculate_performance_metrics(self):
        """
        计算性能指标
        
        Returns:
            dict: 性能指标
        """
        # 计算网络生命周期
        self.metrics['network_lifetime'] = self.network.calculate_network_lifetime()
        
        # 计算数据包传递率
        self.metrics['packet_delivery_ratio'] = self.network.calculate_packet_delivery_ratio()
        
        # 计算端到端延迟
        self.metrics['end_to_end_delay'] = self.network.calculate_end_to_end_delay()
        
        # 计算预测准确性
        prediction_accuracy = 0.0
        for node_id in self.lstm_predictors:
            accuracy = self.network.calculate_prediction_accuracy(node_id)
            prediction_accuracy += accuracy
        self.metrics['prediction_accuracy'] = prediction_accuracy / len(self.lstm_predictors) if self.lstm_predictors else 0
        
        # 计算平均可靠性
        reliability = 0.0
        count = 0
        for (source, sink), path in self.routing_table.items():
            path_reliability = self.evaluate_reliability(path)
            reliability += path_reliability
            count += 1
        self.metrics['reliability'] = reliability / count if count > 0 else 0
        
        logger.info("性能指标计算完成")
        logger.info(f"网络生命周期: {self.metrics['network_lifetime']:.2f}")
        logger.info(f"数据包传递率: {self.metrics['packet_delivery_ratio']:.2f}")
        logger.info(f"端到端延迟: {self.metrics['end_to_end_delay']:.2f}")
        logger.info(f"预测准确性: {self.metrics['prediction_accuracy']:.2f}")
        logger.info(f"平均可靠性: {self.metrics['reliability']:.2f}")
        
        return self.metrics
    
    def visualize_routes(self, save_path=None):
        """
        可视化路由
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 获取节点位置
        pos = {}
        for node_id in self.network.get_node_ids():
            pos[node_id] = self.network.get_node_position(node_id)
        
        # 绘制网络拓扑
        G = nx.Graph()
        
        # 添加节点
        for node_id in self.network.get_node_ids():
            G.add_node(node_id)
        
        # 添加边
        for node_id in self.network.get_node_ids():
            neighbors = self.network.get_neighbors(node_id)
            for neighbor in neighbors:
                G.add_edge(node_id, neighbor)
        
        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # 绘制路由路径
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, ((source, sink), path) in enumerate(self.routing_table.items()):
            color = colors[i % len(colors)]
            
            # 创建路径边
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            
            # 绘制路径
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, alpha=0.7, edge_color=color)
            
            # 添加路径标签
            plt.text(pos[source][0], pos[source][1] + 0.1, f"S{source}", fontsize=12, color=color)
            plt.text(pos[sink][0], pos[sink][1] + 0.1, f"D{sink}", fontsize=12, color=color)
        
        plt.title('WSN路由可视化')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"路由可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_energy_consumption(self, save_path=None):
        """
        绘制能量消耗
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['energy_consumption'], 'b-', linewidth=2)
        plt.xlabel('路由次数')
        plt.ylabel('能量消耗')
        plt.title('路由能量消耗')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"能量消耗图已保存到: {save_path}")
        else:
            plt.show()

# 模拟网络类（用于测试）
class MockWSNNetwork:
    """
    模拟WSN网络类，用于测试混合智能节能路由协议
    """
    
    def __init__(self, node_count=10, area_size=(100, 100), transmission_range=30):
        """
        初始化模拟WSN网络
        
        Args:
            node_count: 节点数量
            area_size: 区域大小
            transmission_range: 传输范围
        """
        self.node_count = node_count
        self.area_size = area_size
        self.transmission_range = transmission_range
        
        # 初始化节点
        self.nodes = {}
        for i in range(node_count):
            self.nodes[i] = {
                'position': (np.random.uniform(0, area_size[0]), np.random.uniform(0, area_size[1])),
                'energy': np.random.uniform(80, 100),  # 初始能量
                'data': [],  # 历史数据
                'prediction_error': np.random.uniform(0, 20)  # 预测误差
            }
        
        # 计算节点间距离和邻居
        self.distances = {}
        self.neighbors = {}
        for i in range(node_count):
            self.neighbors[i] = []
            for j in range(node_count):
                if i != j:
                    dist = self._calculate_distance(self.nodes[i]['position'], self.nodes[j]['position'])
                    self.distances[(i, j)] = dist
                    if dist <= transmission_range:
                        self.neighbors[i].append(j)
        
        # 初始化链路状态
        self.links = {}
        for i in range(node_count):
            for j in self.neighbors[i]:
                self.links[(i, j)] = {
                    'quality': np.random.uniform(70, 100),  # 链路质量
                    'interference': np.random.uniform(0, 30),  # 干扰
                    'active': True  # 链路是否活跃
                }
        
        # 初始化性能指标
        self.packets_sent = 0
        self.packets_received = 0
        self.total_delay = 0
        
        logger.info(f"模拟WSN网络初始化完成，{node_count}个节点，传输范围{transmission_range}")
    
    def _calculate_distance(self, pos1, pos2):
        """
        计算两点之间的欧几里得距离
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_node_ids(self):
        """
        获取所有节点ID
        """
        return list(self.nodes.keys())
    
    def get_neighbors(self, node_id):
        """
        获取节点的邻居
        """
        return self.neighbors[node_id]
    
    def get_node_position(self, node_id):
        """
        获取节点位置
        """
        return self.nodes[node_id]['position']
    
    def get_node_energy(self, node_id):
        """
        获取节点能量
        """
        return self.nodes[node_id]['energy']
    
    def get_node_data_consistency(self, node_id):
        """
        获取节点数据一致性（模拟值）
        """
        return np.random.uniform(70, 100)  # 模拟值
    
    def get_node_prediction_error(self, node_id):
        """
        获取节点预测误差
        """
        return self.nodes[node_id]['prediction_error']
    
    def get_link_quality(self, node1, node2):
        """
        获取链路质量
        """
        if (node1, node2) in self.links:
            return self.links[(node1, node2)]['quality']
        elif (node2, node1) in self.links:
            return self.links[(node2, node1)]['quality']
        else:
            return 0
    
    def get_link_distance(self, node1, node2):
        """
        获取链路距离（归一化到0-100）
        """
        if (node1, node2) in self.distances:
            dist = self.distances[(node1, node2)]
        elif (node2, node1) in self.distances:
            dist = self.distances[(node2, node1)]
        else:
            return 100  # 最大距离
        
        # 归一化到0-100，距离越远，值越大
        normalized_dist = min(100, dist / self.transmission_range * 100)
        return normalized_dist
    
    def get_link_interference(self, node1, node2):
        """
        获取链路干扰
        """
        if (node1, node2) in self.links:
            return self.links[(node1, node2)]['interference']
        elif (node2, node1) in self.links:
            return self.links[(node2, node1)]['interference']
        else:
            return 100  # 最大干扰
    
    def is_link_active(self, node1, node2):
        """
        检查链路是否活跃
        """
        if (node1, node2) in self.links:
            return self.links[(node1, node2)]['active']
        elif (node2, node1) in self.links:
            return self.links[(node2, node1)]['active']
        else:
            return False
    
    def get_energy_cost(self, node1, node2):
        """
        计算从node1到node2的能量消耗
        """
        if (node1, node2) in self.distances:
            dist = self.distances[(node1, node2)]
        else:
            return float('inf')  # 不可达
        
        # 简化的能量模型：能量消耗与距离的平方成正比
        return 0.01 * (dist ** 2)
    
    def update_node_energy(self, node_id, energy_consumed=0):
        """
        更新节点能量
        
        Args:
            node_id: 节点ID
            energy_consumed: 消耗的能量，如果为0则使用随机衰减
        """
        if energy_consumed == 0:
            # 随机能量衰减
            energy_consumed = np.random.uniform(0, 0.5)
        
        self.nodes[node_id]['energy'] = max(0, self.nodes[node_id]['energy'] - energy_consumed)
    
    def update_link_state(self, node1, node2):
        """
        更新链路状态
        """
        if (node1, node2) in self.links:
            # 随机更新链路质量和干扰
            self.links[(node1, node2)]['quality'] = max(0, min(100, self.links[(node1, node2)]['quality'] + np.random.uniform(-5, 5)))
            self.links[(node1, node2)]['interference'] = max(0, min(100, self.links[(node1, node2)]['interference'] + np.random.uniform(-5, 5)))
            
            # 如果节点能量太低或链路质量太差，链路可能失效
            if self.nodes[node1]['energy'] < 10 or self.nodes[node2]['energy'] < 10 or self.links[(node1, node2)]['quality'] < 20:
                self.links[(node1, node2)]['active'] = False
    
    def transmit_packet(self, path, data=None):
        """
        模拟数据包传输
        
        Args:
            path: 路由路径
            data: 数据包内容
            
        Returns:
            bool: 传输是否成功
        """
        if len(path) < 2:
            return False
        
        self.packets_sent += 1
        delay = 0
        success = True
        
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i+1]
            
            # 检查链路是否活跃
            if not self.is_link_active(node1, node2):
                success = False
                break
            
            # 计算传输延迟（与距离和链路质量相关）
            dist = self.distances.get((node1, node2), self.distances.get((node2, node1), 0))
            quality = self.get_link_quality(node1, node2)
            link_delay = dist * (100 - quality) / 1000  # 简化的延迟模型
            delay += link_delay
            
            # 消耗能量
            energy_cost = self.get_energy_cost(node1, node2)
            self.update_node_energy(node1, energy_cost)
        
        if success:
            self.packets_received += 1
            self.total_delay += delay
        
        return success
    
    def calculate_network_lifetime(self):
        """
        计算网络生命周期（以最低节点能量为指标）
        """
        min_energy = min([node['energy'] for node in self.nodes.values()])
        return min_energy
    
    def calculate_packet_delivery_ratio(self):
        """
        计算数据包传递率
        """
        if self.packets_sent == 0:
            return 0
        return self.packets_received / self.packets_sent
    
    def calculate_end_to_end_delay(self):
        """
        计算平均端到端延迟
        """
        if self.packets_received == 0:
            return 0
        return self.total_delay / self.packets_received
    
    def calculate_prediction_accuracy(self, node_id):
        """
        计算预测准确性（模拟值）
        """
        # 预测误差越小，准确性越高
        error = self.nodes[node_id]['prediction_error']
        accuracy = max(0, 100 - error) / 100
        return accuracy

# 测试函数
def test_hybrid_routing():
    """
    测试混合智能节能路由协议
    """
    # 创建模拟网络
    network = MockWSNNetwork(node_count=10, area_size=(100, 100), transmission_range=30)
    
    # 创建路由协议
    routing = HybridEnergyEfficientRouting(
        network=network,
        alpha=0.6,  # 能耗权重
        beta=0.3,   # 预测误差权重
        gamma=0.1   # 数据可靠性权重
    )
    
    # 创建模拟训练数据
    training_data = {}
    for node_id in network.get_node_ids():
        # 创建模拟传感器数据：温度、湿度、光照、电压
        n_samples = 1000
        time = np.arange(n_samples)
        
        # 温度：正弦波 + 噪声 + 趋势
        temperature = 25 + 5 * np.sin(0.01 * time) + 0.1 * np.random.randn(n_samples) + 0.001 * time
        
        # 湿度：余弦波 + 噪声
        humidity = 60 + 10 * np.cos(0.01 * time) + 0.1 * np.random.randn(n_samples)
        
        # 光照：白天/夜间模式 + 噪声
        light = 500 + 400 * np.sin(0.005 * time) + 10 * np.random.randn(n_samples)
        light = np.maximum(0, light)  # 光照不能为负
        
        # 电压：缓慢下降 + 噪声
        voltage = 3.0 - 0.0005 * time + 0.01 * np.random.randn(n_samples)
        
        # 组合数据
        data = np.column_stack([temperature, humidity, light, voltage])
        training_data[node_id] = data
    
    # 训练预测模型
    training_metrics = routing.train_prediction_models(training_data)
    
    # 测试路由优化
    source_node = 0
    sink_node = 9
    path = routing.optimize_route(source_node, sink_node)
    
    if path:
        logger.info(f"优化路径: {path}")
        
        # 测试数据包路由
        for _ in range(10):
            success = routing.route_packet(source_node, sink_node, data="测试数据")
            logger.info(f"路由结果: {'成功' if success else '失败'}")
            
            # 更新网络状态
            routing.update_network_state()
        
        # 计算性能指标
        metrics = routing.calculate_performance_metrics()
        
        # 可视化路由
        routing.visualize_routes()
        
        # 绘制能量消耗
        routing.plot_energy_consumption()
    else:
        logger.warning("未找到有效路径")

if __name__ == '__main__':
    test_hybrid_routing()