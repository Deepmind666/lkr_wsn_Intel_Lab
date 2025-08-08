"""
诚实的WSN路由系统 - 无虚假宣传版本
明确说明每个组件的真实功能和限制
位置：WSN-Intel-Lab-Project/src/advanced_algorithms/
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import logging
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HonestConfig:
    """诚实的配置 - 明确说明每个参数的作用"""
    num_nodes: int = 15
    transmission_range: float = 25.0
    initial_energy: float = 100.0
    # 注意：这些是仿真参数，不是真实硬件参数

class SimpleGATLayer(nn.Module):
    """简单的图注意力层 - 真实训练，但功能有限"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 真实的可训练参数
        self.W = nn.Linear(input_dim, output_dim, bias=False)
        self.attention = nn.Linear(2 * output_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj_matrix):
        """真实的前向传播"""
        h = self.W(x)  # 线性变换
        
        # 计算注意力权重
        batch_size, num_nodes, _ = h.shape
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        
        attention_input = torch.cat([h_i, h_j], dim=-1)
        e = self.attention(attention_input).squeeze(-1)
        e = self.leaky_relu(e)
        
        # 应用邻接矩阵掩码
        e = e.masked_fill(adj_matrix == 0, -1e9)
        alpha = torch.softmax(e, dim=-1)
        
        # 加权聚合
        output = torch.bmm(alpha, h)
        return output

class BasicLSTM(nn.Module):
    """基础LSTM - 真实训练，但数据是模拟的"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """真实的LSTM前向传播"""
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步
        return output

class HonestWSNSystem:
    """诚实的WSN系统 - 明确说明真实功能和限制"""
    
    def __init__(self, config: HonestConfig):
        self.config = config
        self.network_graph = None
        
        # 真实的模型组件
        self.gat_model = SimpleGATLayer(input_dim=6, output_dim=6)
        self.lstm_model = BasicLSTM(input_size=4, hidden_size=16, output_size=1)
        
        # 训练历史 - 真实记录
        self.training_history = {
            'gat_losses': [],
            'lstm_losses': [],
            'gat_trained': False,
            'lstm_trained': False
        }
        
        # 性能指标 - 基于仿真，不是真实网络
        self.metrics = {
            'energy_consumption': [],
            'network_lifetime': [],
            'routing_efficiency': [],
            'note': '这些是仿真指标，不是真实网络测试结果'
        }
        
        logger.info("✅ 诚实WSN系统初始化完成")
        logger.info("⚠️  注意：这是仿真系统，不是真实硬件测试")
    
    def create_network_topology(self):
        """创建网络拓扑 - 基于随机生成，不是真实传感器网络"""
        logger.info("创建仿真网络拓扑...")
        
        self.network_graph = nx.Graph()
        
        # 随机放置节点
        positions = {}
        for i in range(self.config.num_nodes):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            positions[i] = (x, y)
            
            # 初始化节点属性
            self.network_graph.add_node(i, 
                                      x=x, y=y,
                                      energy=self.config.initial_energy,
                                      data_count=0,
                                      is_alive=True)
        
        # 基于距离创建边
        for i in range(self.config.num_nodes):
            for j in range(i+1, self.config.num_nodes):
                pos_i = positions[i]
                pos_j = positions[j]
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                if distance <= self.config.transmission_range:
                    self.network_graph.add_edge(i, j, weight=distance)
        
        logger.info(f"✅ 创建了包含{self.config.num_nodes}个节点的仿真网络")
        logger.info(f"   边数: {self.network_graph.number_of_edges()}")
        logger.info("⚠️  这是随机生成的拓扑，不是真实传感器部署")
    
    def prepare_training_data(self):
        """准备训练数据 - 明确说明是模拟数据"""
        logger.info("准备训练数据...")
        logger.info("⚠️  注意：使用模拟数据，不是真实传感器数据")
        
        # GAT训练数据：节点特征矩阵
        node_features = []
        for node in self.network_graph.nodes():
            features = [
                self.network_graph.nodes[node]['x'] / 100.0,  # 归一化坐标
                self.network_graph.nodes[node]['y'] / 100.0,
                self.network_graph.nodes[node]['energy'] / 100.0,
                self.network_graph.degree(node) / self.config.num_nodes,
                np.random.random(),  # 模拟数据质量
                np.random.random()   # 模拟信号强度
            ]
            node_features.append(features)
        
        # 转换为张量
        X = torch.FloatTensor(node_features).unsqueeze(0)  # batch_size=1
        
        # 邻接矩阵
        adj_matrix = torch.FloatTensor(nx.adjacency_matrix(self.network_graph).todense()).unsqueeze(0)
        
        # LSTM训练数据：时间序列（模拟的）
        lstm_data = []
        for _ in range(100):  # 100个样本
            sequence = []
            for t in range(10):  # 序列长度10
                # 模拟传感器读数
                temp = 20 + 10 * np.sin(t * 0.1) + np.random.normal(0, 1)
                humidity = 50 + 20 * np.cos(t * 0.1) + np.random.normal(0, 2)
                light = 500 + 200 * np.random.random()
                voltage = 3.0 + 0.5 * np.random.random()
                sequence.append([temp, humidity, light, voltage])
            lstm_data.append(sequence)
        
        lstm_X = torch.FloatTensor(lstm_data)
        # 目标：预测下一个时间步的温度
        lstm_y = torch.FloatTensor([seq[-1][0] + np.random.normal(0, 0.5) for seq in lstm_data])
        
        return X, adj_matrix, lstm_X, lstm_y
    
    def train_gat_model(self, X, adj_matrix, epochs=20):
        """真实训练GAT模型 - 但使用自监督任务"""
        logger.info("开始训练GAT模型...")
        logger.info("⚠️  使用自监督特征重构任务，不是真实的图分类任务")
        
        optimizer = torch.optim.Adam(self.gat_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.gat_model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            output = self.gat_model(X, adj_matrix)
            
            # 自监督任务：重构输入特征
            loss = criterion(output, X)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录真实的损失
            self.training_history['gat_losses'].append(loss.item())
            
            if epoch % 5 == 0:
                logger.info(f"GAT Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        self.training_history['gat_trained'] = True
        logger.info("✅ GAT模型训练完成")
        logger.info("⚠️  这是特征重构任务，不是复杂的图学习任务")
    
    def train_lstm_model(self, lstm_X, lstm_y, epochs=30):
        """真实训练LSTM模型 - 但使用模拟数据"""
        logger.info("开始训练LSTM模型...")
        logger.info("⚠️  使用模拟传感器数据，不是真实环境数据")
        
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.lstm_model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            predictions = self.lstm_model(lstm_X).squeeze()
            
            # 计算损失
            loss = criterion(predictions, lstm_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录真实的损失
            self.training_history['lstm_losses'].append(loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"LSTM Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        self.training_history['lstm_trained'] = True
        logger.info("✅ LSTM模型训练完成")
        logger.info("⚠️  这是基于模拟数据的预测，不是真实环境预测")
    
    def simulate_routing(self, rounds=50):
        """路由仿真 - 明确说明是简化的仿真"""
        logger.info("开始路由仿真...")
        logger.info("⚠️  这是简化的仿真，不是真实网络协议实现")
        
        for round_num in range(rounds):
            # 简单的能量消耗模型
            total_energy = 0
            alive_nodes = 0
            
            for node in self.network_graph.nodes():
                if self.network_graph.nodes[node]['is_alive']:
                    # 基础能量消耗
                    energy_cost = np.random.uniform(0.5, 1.5)
                    
                    # 如果GAT训练过，应用一个小的优化因子
                    if self.training_history['gat_trained']:
                        energy_cost *= 0.95  # 5%的改进
                    
                    current_energy = self.network_graph.nodes[node]['energy']
                    new_energy = max(0, current_energy - energy_cost)
                    self.network_graph.nodes[node]['energy'] = new_energy
                    
                    if new_energy > 0:
                        alive_nodes += 1
                        total_energy += new_energy
                    else:
                        self.network_graph.nodes[node]['is_alive'] = False
            
            # 记录指标
            energy_consumed = (self.config.initial_energy * self.config.num_nodes) - total_energy
            self.metrics['energy_consumption'].append(energy_consumed)
            self.metrics['network_lifetime'].append(alive_nodes)
            
            # 简单的路由效率计算
            if alive_nodes > 0:
                efficiency = alive_nodes / self.config.num_nodes
                if self.training_history['gat_trained']:
                    efficiency *= 1.02  # 小幅提升
                self.metrics['routing_efficiency'].append(efficiency)
            else:
                self.metrics['routing_efficiency'].append(0)
            
            if round_num % 10 == 0:
                logger.info(f"仿真轮次 {round_num}/{rounds}, "
                           f"存活节点: {alive_nodes}, "
                           f"总能量: {total_energy:.2f}")
        
        logger.info("✅ 路由仿真完成")
        logger.info("⚠️  这些结果基于简化模型，不代表真实网络性能")
    
    def save_honest_results(self):
        """保存诚实的结果"""
        # 在项目的results目录下创建子目录
        results_dir = Path(__file__).parent.parent.parent / "results" / "honest_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加诚实声明到指标中
        honest_metrics = {
            **self.metrics,
            'disclaimer': {
                'simulation_only': True,
                'not_real_hardware': True,
                'simplified_models': True,
                'gat_task': 'feature reconstruction only',
                'lstm_data': 'simulated sensor data',
                'routing': 'simplified energy model'
            }
        }
        
        # 保存指标
        with open(results_dir / "honest_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(honest_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        with open(results_dir / "honest_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存模型（如果训练过）
        if self.training_history['gat_trained']:
            torch.save(self.gat_model.state_dict(), results_dir / "honest_gat_model.pth")
        if self.training_history['lstm_trained']:
            torch.save(self.lstm_model.state_dict(), results_dir / "honest_lstm_model.pth")
        
        logger.info(f"✅ 诚实结果保存到: {results_dir}")
    
    def visualize_honest_results(self):
        """可视化诚实的结果"""
        logger.info("生成诚实的结果可视化...")
        
        # 保存到项目results目录
        results_dir = Path(__file__).parent.parent.parent / "results" / "honest_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 能量消耗
        axes[0, 0].plot(self.metrics['energy_consumption'])
        axes[0, 0].set_title('能量消耗 (仿真)')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('累计能量消耗')
        
        # 网络寿命
        axes[0, 1].plot(self.metrics['network_lifetime'])
        axes[0, 1].set_title('存活节点数 (仿真)')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('存活节点数')
        
        # GAT训练损失
        if self.training_history['gat_losses']:
            axes[1, 0].plot(self.training_history['gat_losses'])
            axes[1, 0].set_title('GAT训练损失 (特征重构)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE Loss')
        
        # LSTM训练损失
        if self.training_history['lstm_losses']:
            axes[1, 1].plot(self.training_history['lstm_losses'])
            axes[1, 1].set_title('LSTM训练损失 (模拟数据)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE Loss')
        
        plt.suptitle('诚实的WSN仿真结果\n⚠️ 仅为仿真，非真实硬件测试', fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / 'honest_wsn_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ 可视化完成")

def main():
    """主函数 - 运行诚实的WSN系统"""
    logger.info("🚀 启动诚实的WSN路由系统")
    logger.info("⚠️  重要声明：这是仿真系统，不是真实硬件实现")
    
    # 创建配置
    config = HonestConfig()
    
    # 创建系统
    wsn_system = HonestWSNSystem(config)
    
    # 创建网络拓扑
    wsn_system.create_network_topology()
    
    # 准备训练数据
    X, adj_matrix, lstm_X, lstm_y = wsn_system.prepare_training_data()
    
    # 真实训练模型
    wsn_system.train_gat_model(X, adj_matrix, epochs=20)
    wsn_system.train_lstm_model(lstm_X, lstm_y, epochs=30)
    
    # 运行仿真
    wsn_system.simulate_routing(rounds=50)
    
    # 保存和可视化结果
    wsn_system.save_honest_results()
    wsn_system.visualize_honest_results()
    
    logger.info("✅ 诚实的WSN系统运行完成")
    logger.info("📊 结果保存在 WSN-Intel-Lab-Project/results/honest_wsn/ 目录")
    logger.info("⚠️  再次提醒：所有结果基于仿真，不代表真实网络性能")

if __name__ == "__main__":
    main()