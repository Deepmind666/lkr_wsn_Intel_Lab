#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模糊逻辑可靠性评估模型

该模块实现了基于模糊逻辑的WSN数据可靠性评估模型。
模糊逻辑是一种处理不确定性和模糊性的方法，适合评估WSN中的数据可靠性。
在WSN中，模糊逻辑可用于评估节点和链路的可靠性，考虑能量水平、链路质量、数据一致性等因素。
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FuzzyReliabilityModel:
    """
    模糊逻辑可靠性评估模型
    
    该类使用模糊逻辑评估WSN中节点和链路的可靠性，考虑多种因素。
    """
    
    def __init__(self):
        """
        初始化模糊逻辑可靠性评估模型
        """
        # 创建模糊控制系统
        self._create_node_reliability_system()
        self._create_link_reliability_system()
        
        logger.info("模糊逻辑可靠性评估模型初始化完成")
    
    def _create_node_reliability_system(self):
        """
        创建节点可靠性评估的模糊控制系统
        """
        # 创建模糊变量
        energy_level = ctrl.Antecedent(np.arange(0, 101, 1), 'energy_level')
        data_consistency = ctrl.Antecedent(np.arange(0, 101, 1), 'data_consistency')
        prediction_error = ctrl.Antecedent(np.arange(0, 101, 1), 'prediction_error')
        node_reliability = ctrl.Consequent(np.arange(0, 101, 1), 'node_reliability')
        
        # 定义模糊集
        # 能量水平
        energy_level['very_low'] = fuzz.trimf(energy_level.universe, [0, 0, 20])
        energy_level['low'] = fuzz.trimf(energy_level.universe, [0, 20, 40])
        energy_level['medium'] = fuzz.trimf(energy_level.universe, [20, 50, 80])
        energy_level['high'] = fuzz.trimf(energy_level.universe, [60, 80, 100])
        energy_level['very_high'] = fuzz.trimf(energy_level.universe, [80, 100, 100])
        
        # 数据一致性
        data_consistency['very_low'] = fuzz.trimf(data_consistency.universe, [0, 0, 20])
        data_consistency['low'] = fuzz.trimf(data_consistency.universe, [0, 20, 40])
        data_consistency['medium'] = fuzz.trimf(data_consistency.universe, [20, 50, 80])
        data_consistency['high'] = fuzz.trimf(data_consistency.universe, [60, 80, 100])
        data_consistency['very_high'] = fuzz.trimf(data_consistency.universe, [80, 100, 100])
        
        # 预测误差
        prediction_error['very_low'] = fuzz.trimf(prediction_error.universe, [0, 0, 20])
        prediction_error['low'] = fuzz.trimf(prediction_error.universe, [0, 20, 40])
        prediction_error['medium'] = fuzz.trimf(prediction_error.universe, [20, 50, 80])
        prediction_error['high'] = fuzz.trimf(prediction_error.universe, [60, 80, 100])
        prediction_error['very_high'] = fuzz.trimf(prediction_error.universe, [80, 100, 100])
        
        # 节点可靠性
        node_reliability['very_low'] = fuzz.trimf(node_reliability.universe, [0, 0, 20])
        node_reliability['low'] = fuzz.trimf(node_reliability.universe, [0, 20, 40])
        node_reliability['medium'] = fuzz.trimf(node_reliability.universe, [20, 50, 80])
        node_reliability['high'] = fuzz.trimf(node_reliability.universe, [60, 80, 100])
        node_reliability['very_high'] = fuzz.trimf(node_reliability.universe, [80, 100, 100])
        
        # 定义模糊规则
        rule1 = ctrl.Rule(energy_level['very_low'], node_reliability['very_low'])
        rule2 = ctrl.Rule(energy_level['low'] & data_consistency['low'], node_reliability['low'])
        rule3 = ctrl.Rule(energy_level['medium'] & data_consistency['medium'] & prediction_error['medium'], node_reliability['medium'])
        rule4 = ctrl.Rule(energy_level['high'] & data_consistency['high'] & prediction_error['low'], node_reliability['high'])
        rule5 = ctrl.Rule(energy_level['very_high'] & data_consistency['very_high'] & prediction_error['very_low'], node_reliability['very_high'])
        rule6 = ctrl.Rule(prediction_error['very_high'], node_reliability['very_low'])
        rule7 = ctrl.Rule(data_consistency['very_low'], node_reliability['very_low'])
        rule8 = ctrl.Rule(energy_level['medium'] & data_consistency['high'] & prediction_error['low'], node_reliability['high'])
        rule9 = ctrl.Rule(energy_level['high'] & data_consistency['medium'] & prediction_error['medium'], node_reliability['medium'])
        
        # 创建控制系统
        node_reliability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.node_reliability_sim = ctrl.ControlSystemSimulation(node_reliability_ctrl)
        
        logger.info("节点可靠性评估模糊控制系统创建完成")
    
    def _create_link_reliability_system(self):
        """
        创建链路可靠性评估的模糊控制系统
        """
        # 创建模糊变量
        link_quality = ctrl.Antecedent(np.arange(0, 101, 1), 'link_quality')
        distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
        interference = ctrl.Antecedent(np.arange(0, 101, 1), 'interference')
        link_reliability = ctrl.Consequent(np.arange(0, 101, 1), 'link_reliability')
        
        # 定义模糊集
        # 链路质量
        link_quality['very_low'] = fuzz.trimf(link_quality.universe, [0, 0, 20])
        link_quality['low'] = fuzz.trimf(link_quality.universe, [0, 20, 40])
        link_quality['medium'] = fuzz.trimf(link_quality.universe, [20, 50, 80])
        link_quality['high'] = fuzz.trimf(link_quality.universe, [60, 80, 100])
        link_quality['very_high'] = fuzz.trimf(link_quality.universe, [80, 100, 100])
        
        # 距离（归一化，越小越好）
        distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 20])
        distance['close'] = fuzz.trimf(distance.universe, [0, 20, 40])
        distance['medium'] = fuzz.trimf(distance.universe, [20, 50, 80])
        distance['far'] = fuzz.trimf(distance.universe, [60, 80, 100])
        distance['very_far'] = fuzz.trimf(distance.universe, [80, 100, 100])
        
        # 干扰（越小越好）
        interference['very_low'] = fuzz.trimf(interference.universe, [0, 0, 20])
        interference['low'] = fuzz.trimf(interference.universe, [0, 20, 40])
        interference['medium'] = fuzz.trimf(interference.universe, [20, 50, 80])
        interference['high'] = fuzz.trimf(interference.universe, [60, 80, 100])
        interference['very_high'] = fuzz.trimf(interference.universe, [80, 100, 100])
        
        # 链路可靠性
        link_reliability['very_low'] = fuzz.trimf(link_reliability.universe, [0, 0, 20])
        link_reliability['low'] = fuzz.trimf(link_reliability.universe, [0, 20, 40])
        link_reliability['medium'] = fuzz.trimf(link_reliability.universe, [20, 50, 80])
        link_reliability['high'] = fuzz.trimf(link_reliability.universe, [60, 80, 100])
        link_reliability['very_high'] = fuzz.trimf(link_reliability.universe, [80, 100, 100])
        
        # 定义模糊规则
        rule1 = ctrl.Rule(link_quality['very_low'], link_reliability['very_low'])
        rule2 = ctrl.Rule(distance['very_far'], link_reliability['very_low'])
        rule3 = ctrl.Rule(interference['very_high'], link_reliability['very_low'])
        rule4 = ctrl.Rule(link_quality['low'] & distance['far'] & interference['high'], link_reliability['low'])
        rule5 = ctrl.Rule(link_quality['medium'] & distance['medium'] & interference['medium'], link_reliability['medium'])
        rule6 = ctrl.Rule(link_quality['high'] & distance['close'] & interference['low'], link_reliability['high'])
        rule7 = ctrl.Rule(link_quality['very_high'] & distance['very_close'] & interference['very_low'], link_reliability['very_high'])
        rule8 = ctrl.Rule(link_quality['high'] & distance['medium'] & interference['low'], link_reliability['high'])
        rule9 = ctrl.Rule(link_quality['medium'] & distance['close'] & interference['low'], link_reliability['high'])
        
        # 创建控制系统
        link_reliability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.link_reliability_sim = ctrl.ControlSystemSimulation(link_reliability_ctrl)
        
        logger.info("链路可靠性评估模糊控制系统创建完成")
    
    def evaluate_node_reliability(self, energy_level, data_consistency, prediction_error):
        """
        评估节点可靠性
        
        Args:
            energy_level: 能量水平（0-100）
            data_consistency: 数据一致性（0-100）
            prediction_error: 预测误差（0-100，越小越好）
            
        Returns:
            float: 节点可靠性（0-100）
        """
        # 输入值范围检查
        energy_level = np.clip(energy_level, 0, 100)
        data_consistency = np.clip(data_consistency, 0, 100)
        prediction_error = np.clip(prediction_error, 0, 100)
        
        # 设置输入
        self.node_reliability_sim.input['energy_level'] = energy_level
        self.node_reliability_sim.input['data_consistency'] = data_consistency
        self.node_reliability_sim.input['prediction_error'] = prediction_error
        
        # 计算
        try:
            self.node_reliability_sim.compute()
            reliability = self.node_reliability_sim.output['node_reliability']
            return reliability
        except Exception as e:
            logger.error(f"计算节点可靠性时出错: {e}")
            return 0.0
    
    def evaluate_link_reliability(self, link_quality, distance, interference):
        """
        评估链路可靠性
        
        Args:
            link_quality: 链路质量（0-100）
            distance: 距离（0-100，归一化，越小越好）
            interference: 干扰（0-100，越小越好）
            
        Returns:
            float: 链路可靠性（0-100）
        """
        # 输入值范围检查
        link_quality = np.clip(link_quality, 0, 100)
        distance = np.clip(distance, 0, 100)
        interference = np.clip(interference, 0, 100)
        
        # 设置输入
        self.link_reliability_sim.input['link_quality'] = link_quality
        self.link_reliability_sim.input['distance'] = distance
        self.link_reliability_sim.input['interference'] = interference
        
        # 计算
        try:
            self.link_reliability_sim.compute()
            reliability = self.link_reliability_sim.output['link_reliability']
            return reliability
        except Exception as e:
            logger.error(f"计算链路可靠性时出错: {e}")
            return 0.0
    
    def evaluate_path_reliability(self, path, network):
        """
        评估路径可靠性
        
        Args:
            path: 路径（节点ID列表）
            network: 网络对象，提供节点和链路信息
            
        Returns:
            float: 路径可靠性（0-100）
        """
        if len(path) < 2:
            logger.warning("路径至少需要两个节点")
            return 0.0
        
        # 计算节点可靠性
        node_reliabilities = []
        for node_id in path:
            # 从网络获取节点信息
            energy_level = network.get_node_energy(node_id)
            data_consistency = network.get_node_data_consistency(node_id)
            prediction_error = network.get_node_prediction_error(node_id)
            
            # 评估节点可靠性
            node_reliability = self.evaluate_node_reliability(energy_level, data_consistency, prediction_error)
            node_reliabilities.append(node_reliability)
        
        # 计算链路可靠性
        link_reliabilities = []
        for i in range(len(path) - 1):
            # 从网络获取链路信息
            link_quality = network.get_link_quality(path[i], path[i+1])
            distance = network.get_link_distance(path[i], path[i+1])
            interference = network.get_link_interference(path[i], path[i+1])
            
            # 评估链路可靠性
            link_reliability = self.evaluate_link_reliability(link_quality, distance, interference)
            link_reliabilities.append(link_reliability)
        
        # 计算路径可靠性（最弱环节原则）
        min_node_reliability = min(node_reliabilities)
        min_link_reliability = min(link_reliabilities)
        path_reliability = min(min_node_reliability, min_link_reliability)
        
        return path_reliability
    
    def plot_node_reliability_surface(self, save_path=None):
        """
        绘制节点可靠性三维曲面图
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        # 创建输入网格
        energy_level = np.arange(0, 101, 10)
        data_consistency = np.arange(0, 101, 10)
        energy_level_grid, data_consistency_grid = np.meshgrid(energy_level, data_consistency)
        
        # 计算输出
        reliability = np.zeros_like(energy_level_grid)
        prediction_error = 20  # 固定预测误差为中低水平
        
        for i in range(len(energy_level)):
            for j in range(len(data_consistency)):
                reliability[j, i] = self.evaluate_node_reliability(
                    energy_level[i], data_consistency[j], prediction_error
                )
        
        # 绘制三维曲面
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(energy_level_grid, data_consistency_grid, reliability, 
                              cmap='viridis', edgecolor='none', alpha=0.8)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # 设置标签
        ax.set_xlabel('能量水平')
        ax.set_ylabel('数据一致性')
        ax.set_zlabel('节点可靠性')
        ax.set_title(f'节点可靠性曲面 (预测误差 = {prediction_error})')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"节点可靠性曲面图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_link_reliability_surface(self, save_path=None):
        """
        绘制链路可靠性三维曲面图
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        # 创建输入网格
        link_quality = np.arange(0, 101, 10)
        distance = np.arange(0, 101, 10)
        link_quality_grid, distance_grid = np.meshgrid(link_quality, distance)
        
        # 计算输出
        reliability = np.zeros_like(link_quality_grid)
        interference = 20  # 固定干扰为中低水平
        
        for i in range(len(link_quality)):
            for j in range(len(distance)):
                reliability[j, i] = self.evaluate_link_reliability(
                    link_quality[i], distance[j], interference
                )
        
        # 绘制三维曲面
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(link_quality_grid, distance_grid, reliability, 
                              cmap='viridis', edgecolor='none', alpha=0.8)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # 设置标签
        ax.set_xlabel('链路质量')
        ax.set_ylabel('距离')
        ax.set_zlabel('链路可靠性')
        ax.set_title(f'链路可靠性曲面 (干扰 = {interference})')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"链路可靠性曲面图已保存到: {save_path}")
        else:
            plt.show()

# 模拟网络类（用于测试）
class MockNetwork:
    """
    模拟网络类，用于测试模糊逻辑可靠性模型
    """
    
    def __init__(self, node_count=10):
        """
        初始化模拟网络
        
        Args:
            node_count: 节点数量
        """
        self.node_count = node_count
        self.nodes = {}
        self.links = {}
        
        # 初始化节点
        for i in range(node_count):
            self.nodes[i] = {
                'energy_level': np.random.uniform(30, 100),
                'data_consistency': np.random.uniform(50, 100),
                'prediction_error': np.random.uniform(0, 50)
            }
        
        # 初始化链路
        for i in range(node_count):
            for j in range(i+1, node_count):
                self.links[(i, j)] = {
                    'link_quality': np.random.uniform(50, 100),
                    'distance': np.random.uniform(0, 100),
                    'interference': np.random.uniform(0, 50)
                }
    
    def get_node_energy(self, node_id):
        """
        获取节点能量水平
        """
        return self.nodes[node_id]['energy_level']
    
    def get_node_data_consistency(self, node_id):
        """
        获取节点数据一致性
        """
        return self.nodes[node_id]['data_consistency']
    
    def get_node_prediction_error(self, node_id):
        """
        获取节点预测误差
        """
        return self.nodes[node_id]['prediction_error']
    
    def get_link_quality(self, node1, node2):
        """
        获取链路质量
        """
        key = (min(node1, node2), max(node1, node2))
        return self.links[key]['link_quality']
    
    def get_link_distance(self, node1, node2):
        """
        获取链路距离
        """
        key = (min(node1, node2), max(node1, node2))
        return self.links[key]['distance']
    
    def get_link_interference(self, node1, node2):
        """
        获取链路干扰
        """
        key = (min(node1, node2), max(node1, node2))
        return self.links[key]['interference']

# 测试函数
def test_fuzzy_reliability_model():
    """
    测试模糊逻辑可靠性模型
    """
    # 创建模糊逻辑可靠性模型
    model = FuzzyReliabilityModel()
    
    # 测试节点可靠性评估
    logger.info("测试节点可靠性评估:")
    test_cases = [
        (90, 90, 10),  # 高能量，高一致性，低误差
        (50, 50, 50),  # 中等能量，中等一致性，中等误差
        (20, 20, 80),  # 低能量，低一致性，高误差
        (10, 90, 10),  # 低能量，高一致性，低误差
        (90, 10, 90)   # 高能量，低一致性，高误差
    ]
    
    for energy, consistency, error in test_cases:
        reliability = model.evaluate_node_reliability(energy, consistency, error)
        logger.info(f"能量={energy}, 一致性={consistency}, 误差={error} => 可靠性={reliability:.2f}")
    
    # 测试链路可靠性评估
    logger.info("\n测试链路可靠性评估:")
    test_cases = [
        (90, 10, 10),  # 高质量，近距离，低干扰
        (50, 50, 50),  # 中等质量，中等距离，中等干扰
        (20, 80, 80),  # 低质量，远距离，高干扰
        (90, 80, 10),  # 高质量，远距离，低干扰
        (20, 10, 90)   # 低质量，近距离，高干扰
    ]
    
    for quality, distance, interference in test_cases:
        reliability = model.evaluate_link_reliability(quality, distance, interference)
        logger.info(f"质量={quality}, 距离={distance}, 干扰={interference} => 可靠性={reliability:.2f}")
    
    # 测试路径可靠性评估
    logger.info("\n测试路径可靠性评估:")
    network = MockNetwork(node_count=5)
    path = [0, 2, 4]  # 路径：节点0 -> 节点2 -> 节点4
    path_reliability = model.evaluate_path_reliability(path, network)
    logger.info(f"路径 {path} 的可靠性: {path_reliability:.2f}")
    
    # 绘制节点可靠性曲面
    model.plot_node_reliability_surface()
    
    # 绘制链路可靠性曲面
    model.plot_link_reliability_surface()

if __name__ == '__main__':
    test_fuzzy_reliability_model()