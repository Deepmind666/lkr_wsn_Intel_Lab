#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WSN经典基线算法实现
包含LEACH、HEED、Direct Transmission等经典路由协议
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)

class Node:
    """WSN节点类"""
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 1.0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.is_alive = True
        self.is_cluster_head = False
        self.cluster_head_id = None
        self.round_as_ch = 0  # 担任簇头的轮数

    def distance_to(self, other) -> float:
        """计算到另一个节点的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def consume_energy(self, amount: float):
        """消耗能量"""
        self.current_energy = max(0, self.current_energy - amount)
        if self.current_energy <= 0:
            self.is_alive = False

class LEACHProtocol:
    """LEACH (Low-Energy Adaptive Clustering Hierarchy) 协议实现"""
    
    def __init__(self, nodes: List[Node], base_station: Tuple[float, float], 
                 p: float = 0.1, energy_model: Dict = None):
        """
        初始化LEACH协议
        
        Args:
            nodes: 节点列表
            base_station: 基站坐标 (x, y)
            p: 期望簇头比例
            energy_model: 能耗模型参数
        """
        self.nodes = nodes
        self.base_station = base_station
        self.p = p
        self.round_number = 0
        
        # 默认能耗模型参数
        self.energy_model = energy_model or {
            'E_elec': 50e-9,      # 发送/接收电路能耗 (J/bit)
            'E_fs': 10e-12,       # 自由空间模型放大器能耗 (J/bit/m²)
            'E_mp': 0.0013e-12,   # 多径衰落模型放大器能耗 (J/bit/m⁴)
            'd_crossover': 87.7,  # 自由空间和多径模型的交叉距离 (m)
            'E_DA': 5e-9,         # 数据聚合能耗 (J/bit/signal)
            'packet_size': 4000   # 数据包大小 (bits)
        }

    def calculate_transmission_energy(self, distance: float, packet_size: int = None) -> float:
        """计算传输能耗"""
        if packet_size is None:
            packet_size = self.energy_model['packet_size']
            
        E_elec = self.energy_model['E_elec']
        E_fs = self.energy_model['E_fs']
        E_mp = self.energy_model['E_mp']
        d_crossover = self.energy_model['d_crossover']
        
        if distance < d_crossover:
            # 自由空间模型
            energy = packet_size * (E_elec + E_fs * distance**2)
        else:
            # 多径衰落模型
            energy = packet_size * (E_elec + E_mp * distance**4)
            
        return energy

    def calculate_reception_energy(self, packet_size: int = None) -> float:
        """计算接收能耗"""
        if packet_size is None:
            packet_size = self.energy_model['packet_size']
        return packet_size * self.energy_model['E_elec']

    def select_cluster_heads(self) -> List[int]:
        """LEACH簇头选择算法"""
        cluster_heads = []
        
        # LEACH阈值计算
        for node in self.nodes:
            if not node.is_alive:
                continue
                
            # 计算阈值T(n)
            if node.round_as_ch < (1 / self.p):
                threshold = self.p / (1 - self.p * (self.round_number % (1/self.p)))
            else:
                threshold = 0
                
            # 随机数判断是否成为簇头
            if random.random() < threshold:
                cluster_heads.append(node.node_id)
                node.is_cluster_head = True
                node.round_as_ch += 1
            else:
                node.is_cluster_head = False
                
        # 确保至少有一个簇头
        if not cluster_heads:
            alive_nodes = [n for n in self.nodes if n.is_alive]
            if alive_nodes:
                ch_node = random.choice(alive_nodes)
                cluster_heads.append(ch_node.node_id)
                ch_node.is_cluster_head = True
                
        logger.info(f"LEACH第{self.round_number}轮选出簇头: {cluster_heads}")
        return cluster_heads

    def form_clusters(self, cluster_heads: List[int]) -> Dict[int, List[int]]:
        """形成簇结构"""
        clusters = {ch_id: [] for ch_id in cluster_heads}
        
        # 为每个非簇头节点分配最近的簇头
        for node in self.nodes:
            if not node.is_alive or node.is_cluster_head:
                continue
                
            best_ch = None
            min_distance = float('inf')
            
            for ch_id in cluster_heads:
                ch_node = self.nodes[ch_id]
                if ch_node.is_alive:
                    distance = node.distance_to(ch_node)
                    if distance < min_distance:
                        min_distance = distance
                        best_ch = ch_id
                        
            if best_ch is not None:
                clusters[best_ch].append(node.node_id)
                node.cluster_head_id = best_ch
                
        return clusters

    def run_round(self) -> Dict:
        """运行一轮LEACH协议"""
        self.round_number += 1
        
        # 1. 簇头选择
        cluster_heads = self.select_cluster_heads()
        
        # 2. 簇形成
        clusters = self.form_clusters(cluster_heads)
        
        # 3. 计算能耗
        total_energy_consumed = 0
        
        # 簇内通信能耗
        for ch_id, members in clusters.items():
            ch_node = self.nodes[ch_id]
            if not ch_node.is_alive:
                continue
                
            # 簇头接收成员数据的能耗
            reception_energy = len(members) * self.calculate_reception_energy()
            ch_node.consume_energy(reception_energy)
            total_energy_consumed += reception_energy
            
            # 簇头数据聚合能耗
            if members:
                aggregation_energy = len(members) * self.energy_model['E_DA'] * self.energy_model['packet_size']
                ch_node.consume_energy(aggregation_energy)
                total_energy_consumed += aggregation_energy
            
            # 成员节点发送数据给簇头的能耗
            for member_id in members:
                member_node = self.nodes[member_id]
                if member_node.is_alive:
                    distance = member_node.distance_to(ch_node)
                    transmission_energy = self.calculate_transmission_energy(distance)
                    member_node.consume_energy(transmission_energy)
                    total_energy_consumed += transmission_energy
            
            # 簇头发送聚合数据给基站的能耗
            bs_distance = np.sqrt((ch_node.x - self.base_station[0])**2 + 
                                (ch_node.y - self.base_station[1])**2)
            bs_transmission_energy = self.calculate_transmission_energy(bs_distance)
            ch_node.consume_energy(bs_transmission_energy)
            total_energy_consumed += bs_transmission_energy
        
        # 统计存活节点
        alive_nodes = sum(1 for node in self.nodes if node.is_alive)
        
        # 计算网络剩余能量
        remaining_energy = sum(node.current_energy for node in self.nodes if node.is_alive)
        
        return {
            'round': self.round_number,
            'cluster_heads': cluster_heads,
            'clusters': clusters,
            'energy_consumed': total_energy_consumed,
            'alive_nodes': alive_nodes,
            'remaining_energy': remaining_energy,
            'first_node_dead': alive_nodes < len(self.nodes)
        }

class HEEDProtocol:
    """HEED (Hybrid Energy-Efficient Distributed clustering) 协议实现"""
    
    def __init__(self, nodes: List[Node], base_station: Tuple[float, float], 
                 c_prob: float = 0.1, energy_model: Dict = None):
        """
        初始化HEED协议
        
        Args:
            nodes: 节点列表
            base_station: 基站坐标
            c_prob: 初始簇头概率
            energy_model: 能耗模型参数
        """
        self.nodes = nodes
        self.base_station = base_station
        self.c_prob = c_prob
        self.round_number = 0
        self.energy_model = energy_model or LEACHProtocol(nodes, base_station).energy_model

    def calculate_residual_energy_ratio(self, node: Node) -> float:
        """计算节点剩余能量比例"""
        if node.initial_energy == 0:
            return 0
        return node.current_energy / node.initial_energy

    def select_cluster_heads(self) -> List[int]:
        """HEED簇头选择算法"""
        cluster_heads = []
        
        for node in self.nodes:
            if not node.is_alive:
                continue
                
            # 基于剩余能量的簇头选择概率
            energy_ratio = self.calculate_residual_energy_ratio(node)
            ch_probability = self.c_prob * energy_ratio
            
            # 随机选择
            if random.random() < ch_probability:
                cluster_heads.append(node.node_id)
                node.is_cluster_head = True
            else:
                node.is_cluster_head = False
                
        # 确保至少有一个簇头
        if not cluster_heads:
            alive_nodes = [n for n in self.nodes if n.is_alive]
            if alive_nodes:
                # 选择剩余能量最高的节点作为簇头
                best_node = max(alive_nodes, key=lambda n: n.current_energy)
                cluster_heads.append(best_node.node_id)
                best_node.is_cluster_head = True
                
        logger.info(f"HEED第{self.round_number}轮选出簇头: {cluster_heads}")
        return cluster_heads

    def run_round(self) -> Dict:
        """运行一轮HEED协议"""
        self.round_number += 1
        
        # 使用类似LEACH的簇形成和通信过程
        leach_temp = LEACHProtocol(self.nodes, self.base_station, energy_model=self.energy_model)
        leach_temp.round_number = self.round_number
        
        # 使用HEED的簇头选择
        cluster_heads = self.select_cluster_heads()
        clusters = leach_temp.form_clusters(cluster_heads)
        
        # 计算能耗（使用LEACH的能耗模型）
        total_energy_consumed = 0
        
        for ch_id, members in clusters.items():
            ch_node = self.nodes[ch_id]
            if not ch_node.is_alive:
                continue
                
            reception_energy = len(members) * leach_temp.calculate_reception_energy()
            ch_node.consume_energy(reception_energy)
            total_energy_consumed += reception_energy
            
            if members:
                aggregation_energy = len(members) * self.energy_model['E_DA'] * self.energy_model['packet_size']
                ch_node.consume_energy(aggregation_energy)
                total_energy_consumed += aggregation_energy
            
            for member_id in members:
                member_node = self.nodes[member_id]
                if member_node.is_alive:
                    distance = member_node.distance_to(ch_node)
                    transmission_energy = leach_temp.calculate_transmission_energy(distance)
                    member_node.consume_energy(transmission_energy)
                    total_energy_consumed += transmission_energy
            
            bs_distance = np.sqrt((ch_node.x - self.base_station[0])**2 + 
                                (ch_node.y - self.base_station[1])**2)
            bs_transmission_energy = leach_temp.calculate_transmission_energy(bs_distance)
            ch_node.consume_energy(bs_transmission_energy)
            total_energy_consumed += bs_transmission_energy
        
        alive_nodes = sum(1 for node in self.nodes if node.is_alive)
        remaining_energy = sum(node.current_energy for node in self.nodes if node.is_alive)
        
        return {
            'round': self.round_number,
            'cluster_heads': cluster_heads,
            'clusters': clusters,
            'energy_consumed': total_energy_consumed,
            'alive_nodes': alive_nodes,
            'remaining_energy': remaining_energy,
            'first_node_dead': alive_nodes < len(self.nodes)
        }

class DirectTransmissionProtocol:
    """直接传输协议（基线对比）"""
    
    def __init__(self, nodes: List[Node], base_station: Tuple[float, float], energy_model: Dict = None):
        self.nodes = nodes
        self.base_station = base_station
        self.round_number = 0
        self.energy_model = energy_model or LEACHProtocol(nodes, base_station).energy_model

    def run_round(self) -> Dict:
        """运行一轮直接传输"""
        self.round_number += 1
        
        total_energy_consumed = 0
        
        # 每个节点直接向基站传输
        for node in self.nodes:
            if not node.is_alive:
                continue
                
            # 计算到基站的距离
            bs_distance = np.sqrt((node.x - self.base_station[0])**2 + 
                                (node.y - self.base_station[1])**2)
            
            # 计算传输能耗
            leach_temp = LEACHProtocol(self.nodes, self.base_station, energy_model=self.energy_model)
            transmission_energy = leach_temp.calculate_transmission_energy(bs_distance)
            
            node.consume_energy(transmission_energy)
            total_energy_consumed += transmission_energy
        
        alive_nodes = sum(1 for node in self.nodes if node.is_alive)
        remaining_energy = sum(node.current_energy for node in self.nodes if node.is_alive)
        
        return {
            'round': self.round_number,
            'cluster_heads': [],
            'clusters': {},
            'energy_consumed': total_energy_consumed,
            'alive_nodes': alive_nodes,
            'remaining_energy': remaining_energy,
            'first_node_dead': alive_nodes < len(self.nodes)
        }

def create_random_topology(num_nodes: int, field_size: Tuple[float, float] = (100, 100), 
                          initial_energy: float = 1.0, random_seed: int = None) -> List[Node]:
    """创建随机网络拓扑"""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    nodes = []
    for i in range(num_nodes):
        x = random.uniform(0, field_size[0])
        y = random.uniform(0, field_size[1])
        node = Node(i, x, y, initial_energy)
        nodes.append(node)
    
    return nodes

def run_protocol_comparison(num_nodes: int = 54, num_rounds: int = 1000, 
                           field_size: Tuple[float, float] = (25, 25),  # Intel Lab实际场地尺寸
                           base_station: Tuple[float, float] = (12.5, 30),
                           random_seed: int = 42) -> Dict:
    """运行协议对比实验"""
    
    protocols = ['LEACH', 'HEED', 'DirectTransmission']
    results = {}
    
    for protocol_name in protocols:
        logger.info(f"运行 {protocol_name} 协议...")
        
        # 创建相同的网络拓扑
        nodes = create_random_topology(num_nodes, field_size, random_seed=random_seed)
        
        # 初始化协议
        if protocol_name == 'LEACH':
            protocol = LEACHProtocol(nodes, base_station)
        elif protocol_name == 'HEED':
            protocol = HEEDProtocol(nodes, base_station)
        elif protocol_name == 'DirectTransmission':
            protocol = DirectTransmissionProtocol(nodes, base_station)
        
        # 运行实验
        round_results = []
        for round_num in range(1, num_rounds + 1):
            round_result = protocol.run_round()
            round_results.append(round_result)
            
            # 检查是否所有节点都死亡
            if round_result['alive_nodes'] == 0:
                logger.info(f"{protocol_name}: 所有节点在第{round_num}轮死亡")
                break
        
        results[protocol_name] = round_results
        
        # 计算关键指标
        total_rounds = len(round_results)
        first_node_dead_round = next((r['round'] for r in round_results if r['first_node_dead']), total_rounds)
        total_energy = sum(r['energy_consumed'] for r in round_results)
        
        logger.info(f"{protocol_name} 结果: 总轮数={total_rounds}, 首个节点死亡轮数={first_node_dead_round}, 总能耗={total_energy:.6f}")
    
    return results

if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行对比实验
    results = run_protocol_comparison(num_nodes=100, num_rounds=200)
    
    # 保存结果
    import json
    import os
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'baseline_protocols_comparison.json')
    with open(output_file, 'w') as f:
        # 转换为可序列化格式
        serializable_results = {}
        for protocol, rounds in results.items():
            serializable_results[protocol] = []
            for round_data in rounds:
                serializable_round = {k: v for k, v in round_data.items() if k != 'clusters'}
                serializable_round['num_clusters'] = len(round_data['clusters'])
                serializable_results[protocol].append(serializable_round)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ 基线协议对比结果保存到: {output_file}")
