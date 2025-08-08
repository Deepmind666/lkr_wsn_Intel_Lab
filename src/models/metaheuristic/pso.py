#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
粒子群优化算法 (PSO) 实现

该模块实现了用于WSN节能路由优化的粒子群优化算法。
粒子群优化是一种基于群体智能的元启发式算法，模拟鸟群或鱼群的社会行为。
在WSN路由优化中，PSO可用于寻找最优的路由路径，以最小化能量消耗。
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PSO:
    """
    粒子群优化算法类
    
    属性:
        n_particles: 粒子数量
        dimensions: 问题维度
        bounds: 搜索空间边界，形式为 [(min_1, max_1), (min_2, max_2), ...]
        w: 惯性权重
        c1: 认知系数
        c2: 社会系数
        max_iter: 最大迭代次数
        fitness_func: 适应度函数
        minimize: 是否为最小化问题
    """
    
    def __init__(self, n_particles, dimensions, bounds, w=0.7, c1=1.5, c2=1.5, max_iter=100, fitness_func=None, minimize=True):
        """
        初始化PSO算法
        
        Args:
            n_particles: 粒子数量
            dimensions: 问题维度
            bounds: 搜索空间边界，形式为 [(min_1, max_1), (min_2, max_2), ...]
            w: 惯性权重
            c1: 认知系数
            c2: 社会系数
            max_iter: 最大迭代次数
            fitness_func: 适应度函数
            minimize: 是否为最小化问题
        """
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.fitness_func = fitness_func
        self.minimize = minimize
        
        # 初始化粒子位置和速度
        self.positions = np.zeros((n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        
        # 初始化个体最优位置和适应度
        self.pbest_positions = np.zeros((n_particles, dimensions))
        self.pbest_fitness = np.zeros(n_particles)
        
        # 初始化全局最优位置和适应度
        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = float('inf') if minimize else float('-inf')
        
        # 初始化适应度历史记录
        self.fitness_history = []
        
        # 初始化粒子
        self._initialize_particles()
    
    def _initialize_particles(self):
        """
        初始化粒子位置和速度
        """
        # 初始化粒子位置
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            self.positions[:, i] = np.random.uniform(min_val, max_val, self.n_particles)
        
        # 初始化粒子速度
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            velocity_range = (max_val - min_val) * 0.1  # 速度范围为边界范围的10%
            self.velocities[:, i] = np.random.uniform(-velocity_range, velocity_range, self.n_particles)
        
        # 初始化个体最优位置和适应度
        self.pbest_positions = self.positions.copy()
        
        # 计算初始适应度
        for i in range(self.n_particles):
            fitness = self.fitness_func(self.positions[i])
            self.pbest_fitness[i] = fitness
            
            # 更新全局最优
            if (self.minimize and fitness < self.gbest_fitness) or (not self.minimize and fitness > self.gbest_fitness):
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()
    
    def _update_velocities(self):
        """
        更新粒子速度
        """
        r1 = np.random.random((self.n_particles, self.dimensions))
        r2 = np.random.random((self.n_particles, self.dimensions))
        
        cognitive_component = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_component = self.c2 * r2 * (self.gbest_position - self.positions)
        
        self.velocities = self.w * self.velocities + cognitive_component + social_component
    
    def _update_positions(self):
        """
        更新粒子位置并处理边界
        """
        # 更新位置
        self.positions = self.positions + self.velocities
        
        # 处理边界
        for i in range(self.dimensions):
            min_val, max_val = self.bounds[i]
            self.positions[:, i] = np.clip(self.positions[:, i], min_val, max_val)
    
    def _update_best_positions(self):
        """
        更新个体最优和全局最优位置
        """
        for i in range(self.n_particles):
            fitness = self.fitness_func(self.positions[i])
            
            # 更新个体最优
            if (self.minimize and fitness < self.pbest_fitness[i]) or (not self.minimize and fitness > self.pbest_fitness[i]):
                self.pbest_fitness[i] = fitness
                self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if (self.minimize and fitness < self.gbest_fitness) or (not self.minimize and fitness > self.gbest_fitness):
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
    
    def optimize(self, verbose=True):
        """
        运行PSO优化算法
        
        Args:
            verbose: 是否打印优化过程
            
        Returns:
            tuple: (最优位置, 最优适应度, 适应度历史)
        """
        if verbose:
            logger.info("开始PSO优化")
            logger.info(f"粒子数量: {self.n_particles}, 维度: {self.dimensions}, 最大迭代次数: {self.max_iter}")
            logger.info(f"惯性权重: {self.w}, 认知系数: {self.c1}, 社会系数: {self.c2}")
            logger.info(f"优化类型: {'最小化' if self.minimize else '最大化'}")
        
        # 迭代优化
        iterator = tqdm(range(self.max_iter)) if verbose else range(self.max_iter)
        for t in iterator:
            # 更新速度和位置
            self._update_velocities()
            self._update_positions()
            
            # 更新最优位置
            self._update_best_positions()
            
            # 记录当前最优适应度
            self.fitness_history.append(self.gbest_fitness)
            
            if verbose:
                if t % 10 == 0 or t == self.max_iter - 1:
                    logger.info(f"迭代 {t+1}/{self.max_iter}, 当前最优适应度: {self.gbest_fitness:.6f}")
        
        if verbose:
            logger.info(f"PSO优化完成，最优适应度: {self.gbest_fitness:.6f}")
        
        return self.gbest_position, self.gbest_fitness, self.fitness_history
    
    def plot_convergence(self, save_path=None):
        """
        绘制收敛曲线
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('最优适应度')
        plt.title('PSO收敛曲线')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"收敛曲线已保存到: {save_path}")
        else:
            plt.show()

# WSN路由优化的适应度函数示例
def wsn_routing_fitness(solution, network=None, source_node=None, sink_node=None, alpha=0.6, beta=0.3, gamma=0.1):
    """
    WSN路由优化的适应度函数
    
    该函数评估路由方案的适应度，考虑能耗、预测误差和数据可靠性
    
    Args:
        solution: 路由解决方案（编码为粒子位置）
        network: WSN网络对象
        source_node: 源节点ID
        sink_node: 汇聚节点ID
        alpha: 能耗权重
        beta: 预测误差权重
        gamma: 数据可靠性权重
        
    Returns:
        float: 适应度值（越小越好）
    """
    if network is None:
        # 如果没有提供网络，返回一个模拟的适应度值
        # 这里仅用于测试，实际应用中应该提供网络对象
        return np.sum(solution ** 2)  # 使用简单的球函数作为示例
    
    # 解码路由路径
    path = decode_routing_path(solution, network, source_node, sink_node)
    
    # 计算能耗
    energy_consumption = calculate_energy_consumption(path, network)
    
    # 计算预测误差
    prediction_error = calculate_prediction_error(path, network)
    
    # 计算数据可靠性
    data_reliability = calculate_data_reliability(path, network)
    
    # 计算加权适应度
    fitness = alpha * energy_consumption + beta * prediction_error + gamma * (1 - data_reliability)
    
    return fitness

# 辅助函数（实际应用中需要实现这些函数）
def decode_routing_path(solution, network, source_node, sink_node):
    """
    将PSO解决方案解码为路由路径
    
    Args:
        solution: PSO解决方案
        network: WSN网络对象
        source_node: 源节点ID
        sink_node: 汇聚节点ID
        
    Returns:
        list: 路由路径（节点ID列表）
    """
    # 这里是一个简化的实现，实际应用中需要根据具体编码方式解码
    # 例如，solution可以编码为节点选择概率或路径选择
    path = [source_node]
    current_node = source_node
    
    # 简单示例：根据solution值选择下一个节点
    while current_node != sink_node:
        neighbors = network.get_neighbors(current_node)
        if not neighbors:
            break
        
        # 使用solution值作为选择邻居的权重
        weights = solution[:len(neighbors)]
        weights = np.abs(weights) / np.sum(np.abs(weights))  # 归一化为概率
        next_node = np.random.choice(neighbors, p=weights)
        
        path.append(next_node)
        current_node = next_node
        
        # 避免循环
        if len(path) > network.node_count:
            break
    
    return path

def calculate_energy_consumption(path, network):
    """
    计算路由路径的能量消耗
    
    Args:
        path: 路由路径
        network: WSN网络对象
        
    Returns:
        float: 能量消耗
    """
    # 实际应用中需要根据网络模型计算能耗
    energy = 0.0
    for i in range(len(path) - 1):
        energy += network.get_energy_cost(path[i], path[i+1])
    return energy

def calculate_prediction_error(path, network):
    """
    计算路由路径的预测误差
    
    Args:
        path: 路由路径
        network: WSN网络对象
        
    Returns:
        float: 预测误差
    """
    # 实际应用中需要根据预测模型计算误差
    error = 0.0
    for node in path:
        error += network.get_prediction_error(node)
    return error / len(path)

def calculate_data_reliability(path, network):
    """
    计算路由路径的数据可靠性
    
    Args:
        path: 路由路径
        network: WSN网络对象
        
    Returns:
        float: 数据可靠性（0-1之间，越大越好）
    """
    # 实际应用中需要根据可靠性模型计算可靠性
    reliability = 1.0
    for i in range(len(path) - 1):
        link_reliability = network.get_link_reliability(path[i], path[i+1])
        reliability *= link_reliability  # 假设可靠性是独立的
    return reliability

# 测试函数
def test_pso():
    """
    测试PSO算法
    """
    # 定义测试函数（球函数）
    def sphere(x):
        return np.sum(x ** 2)
    
    # 设置PSO参数
    n_particles = 30
    dimensions = 10
    bounds = [(-5, 5)] * dimensions
    max_iter = 100
    
    # 创建PSO实例
    pso = PSO(
        n_particles=n_particles,
        dimensions=dimensions,
        bounds=bounds,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iter=max_iter,
        fitness_func=sphere,
        minimize=True
    )
    
    # 运行优化
    best_position, best_fitness, fitness_history = pso.optimize(verbose=True)
    
    # 打印结果
    logger.info(f"最优位置: {best_position}")
    logger.info(f"最优适应度: {best_fitness:.6f}")
    
    # 绘制收敛曲线
    pso.plot_convergence()

if __name__ == '__main__':
    test_pso()