"""
Enhanced EEHFR WSN系统 - ACO蚁群路由优化模块
基于用户调研文件中的混合元启发式优化设计
实现智能路由发现和多目标优化
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
import json
import time

@dataclass
class Route:
    """路由类"""
    path: List[int]
    cost: float
    energy_consumption: float
    reliability: float
    latency: float

class ACORouter:
    """
    蚁群优化路由算法
    专门用于WSN多目标路由优化
    """
    
    def __init__(self, n_ants: int = 20, n_iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1,
                 q0: float = 0.9, tau0: float = 0.1):
        """
        初始化ACO路由器
        
        Args:
            n_ants: 蚂蚁数量
            n_iterations: 迭代次数
            alpha: 信息素重要程度
            beta: 启发信息重要程度
            rho: 信息素挥发系数
            q0: 开发vs探索参数
            tau0: 初始信息素浓度
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.tau0 = tau0
        
        # 路由历史和性能统计
        self.routing_history = []
        self.best_routes = []
        self.convergence_data = []
        self.pheromone_matrix = None
        
        # 多目标权重
        self.weights = {
            'energy': 0.4,
            'distance': 0.3,
            'reliability': 0.2,
            'latency': 0.1
        }
        
    def initialize_pheromone_matrix(self, n_nodes: int):
        """初始化信息素矩阵"""
        self.pheromone_matrix = np.full((n_nodes, n_nodes), self.tau0)
        np.fill_diagonal(self.pheromone_matrix, 0)  # 自己到自己的信息素为0
        
    def calculate_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """计算节点间距离矩阵"""
        n_nodes = len(positions)
        distance_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    distance_matrix[i, j] = np.sqrt(
                        np.sum((positions[i] - positions[j])**2))
        
        return distance_matrix
    
    def calculate_energy_matrix(self, nodes_energy: np.ndarray) -> np.ndarray:
        """计算能量消耗矩阵"""
        n_nodes = len(nodes_energy)
        energy_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # 能量消耗与距离和节点剩余能量相关
                    base_energy = 0.1  # 基础传输能量
                    distance_factor = 0.01  # 距离因子
                    energy_factor = 1.0 / (nodes_energy[i] + 0.1)  # 能量因子
                    
                    energy_matrix[i, j] = base_energy + distance_factor + energy_factor
        
        return energy_matrix
    
    def calculate_reliability_matrix(self, nodes_trust: np.ndarray, 
                                   distance_matrix: np.ndarray) -> np.ndarray:
        """计算链路可靠性矩阵"""
        n_nodes = len(nodes_trust)
        reliability_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # 可靠性与节点信任度和距离相关
                    trust_factor = (nodes_trust[i] + nodes_trust[j]) / 2
                    distance_factor = 1.0 / (1.0 + distance_matrix[i, j] / 100.0)
                    
                    reliability_matrix[i, j] = trust_factor * distance_factor
        
        return reliability_matrix
    
    def calculate_heuristic_matrix(self, distance_matrix: np.ndarray,
                                 energy_matrix: np.ndarray,
                                 reliability_matrix: np.ndarray) -> np.ndarray:
        """计算启发信息矩阵"""
        # 归一化各个矩阵
        norm_distance = 1.0 / (distance_matrix + 1e-10)
        norm_energy = 1.0 / (energy_matrix + 1e-10)
        norm_reliability = reliability_matrix
        
        # 多目标启发信息
        heuristic = (self.weights['distance'] * norm_distance +
                    self.weights['energy'] * norm_energy +
                    self.weights['reliability'] * norm_reliability)
        
        return heuristic
    
    def construct_ant_solution(self, start_node: int, end_node: int,
                             heuristic_matrix: np.ndarray) -> Route:
        """构造单只蚂蚁的解"""
        current_node = start_node
        path = [current_node]
        total_cost = 0.0
        total_energy = 0.0
        total_reliability = 1.0
        total_latency = 0.0
        
        visited = {current_node}
        
        while current_node != end_node:
            # 获取可访问的邻居节点
            neighbors = []
            probabilities = []
            
            for next_node in range(len(self.pheromone_matrix)):
                if next_node not in visited and next_node != current_node:
                    neighbors.append(next_node)
                    
                    # 计算转移概率
                    pheromone = self.pheromone_matrix[current_node, next_node]
                    heuristic = heuristic_matrix[current_node, next_node]
                    
                    prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                    probabilities.append(prob)
            
            if not neighbors:
                # 无法到达目标，返回无效路由
                return Route(path, float('inf'), float('inf'), 0.0, float('inf'))
            
            # 选择下一个节点
            probabilities = np.array(probabilities)
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
                
                # ε-贪心策略
                if np.random.random() < self.q0:
                    # 开发：选择最优
                    next_idx = np.argmax(probabilities)
                else:
                    # 探索：轮盘赌选择
                    next_idx = np.random.choice(len(neighbors), p=probabilities)
                
                next_node = neighbors[next_idx]
            else:
                next_node = np.random.choice(neighbors)
            
            # 更新路径和成本
            path.append(next_node)
            visited.add(next_node)
            
            # 计算各项成本
            distance = np.sqrt(np.sum((np.random.rand(2) - np.random.rand(2))**2))  # 模拟距离
            energy = 0.1 + 0.01 * distance  # 模拟能量消耗
            reliability = 0.9 + np.random.normal(0, 0.05)  # 模拟可靠性
            latency = distance * 0.1  # 模拟延迟
            
            total_cost += distance
            total_energy += energy
            total_reliability *= max(0.1, reliability)
            total_latency += latency
            
            current_node = next_node
            
            # 防止无限循环
            if len(path) > len(self.pheromone_matrix) * 2:
                break
        
        return Route(path, total_cost, total_energy, total_reliability, total_latency)
    
    def update_pheromone(self, routes: List[Route]):
        """更新信息素"""
        # 信息素挥发
        self.pheromone_matrix *= (1 - self.rho)
        
        # 信息素增强
        for route in routes:
            if route.cost < float('inf') and len(route.path) > 1:
                # 计算信息素增量
                delta_tau = 1.0 / route.cost
                
                # 更新路径上的信息素
                for i in range(len(route.path) - 1):
                    from_node = route.path[i]
                    to_node = route.path[i + 1]
                    self.pheromone_matrix[from_node, to_node] += delta_tau
        
        # 限制信息素范围
        self.pheromone_matrix = np.clip(self.pheromone_matrix, 
                                      self.tau0 * 0.01, self.tau0 * 100)
    
    def find_optimal_routes(self, cluster_heads: List[int], 
                          base_station_id: int,
                          nodes_positions: np.ndarray,
                          nodes_energy: np.ndarray,
                          nodes_trust: np.ndarray) -> Tuple[List[Route], Dict]:
        """
        寻找最优路由
        
        Args:
            cluster_heads: 簇头节点列表
            base_station_id: 基站节点ID
            nodes_positions: 节点位置数组
            nodes_energy: 节点能量数组
            nodes_trust: 节点信任度数组
            
        Returns:
            (最优路由列表, 优化统计信息)
        """
        
        print(f"启动ACO路由优化 - 蚂蚁数: {self.n_ants}, 迭代数: {self.n_iterations}")
        
        # 构建完整的节点列表
        all_nodes = cluster_heads + [base_station_id]
        n_nodes = len(all_nodes)
        
        # 初始化信息素矩阵
        self.initialize_pheromone_matrix(n_nodes)
        
        # 计算各种矩阵
        distance_matrix = self.calculate_distance_matrix(nodes_positions)
        energy_matrix = self.calculate_energy_matrix(nodes_energy)
        reliability_matrix = self.calculate_reliability_matrix(nodes_trust, distance_matrix)
        heuristic_matrix = self.calculate_heuristic_matrix(
            distance_matrix, energy_matrix, reliability_matrix)
        
        # 优化统计
        best_routes = []
        iteration_best_costs = []
        
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            iteration_routes = []
            
            # 为每个簇头寻找到基站的路由
            for ch_idx, cluster_head in enumerate(cluster_heads):
                ant_routes = []
                
                # 多只蚂蚁并行搜索
                for ant in range(self.n_ants):
                    route = self.construct_ant_solution(
                        ch_idx, n_nodes - 1, heuristic_matrix)  # 基站是最后一个节点
                    ant_routes.append(route)
                
                # 选择最优路由
                valid_routes = [r for r in ant_routes if r.cost < float('inf')]
                if valid_routes:
                    best_route = min(valid_routes, key=lambda r: r.cost)
                    iteration_routes.append(best_route)
            
            # 更新信息素
            if iteration_routes:
                self.update_pheromone(iteration_routes)
                
                # 记录最优路由
                best_cost = min(route.cost for route in iteration_routes)
                iteration_best_costs.append(best_cost)
                
                if not best_routes or best_cost < min(r.cost for r in best_routes):
                    best_routes = iteration_routes.copy()
            
            # 记录收敛数据
            self.convergence_data.append({
                'iteration': iteration,
                'best_cost': min(iteration_best_costs) if iteration_best_costs else float('inf'),
                'avg_cost': np.mean(iteration_best_costs[-10:]) if iteration_best_costs else float('inf'),
                'n_valid_routes': len(iteration_routes)
            })
            
            if iteration % 20 == 0:
                avg_cost = np.mean(iteration_best_costs[-10:]) if iteration_best_costs else float('inf')
                print(f"   迭代 {iteration}: 平均成本 = {avg_cost:.4f}")
        
        optimization_time = time.time() - start_time
        
        # 优化统计信息
        stats = {
            'optimization_time': optimization_time,
            'total_iterations': self.n_iterations,
            'best_cost': min(iteration_best_costs) if iteration_best_costs else float('inf'),
            'convergence_data': self.convergence_data,
            'final_pheromone_stats': {
                'mean': np.mean(self.pheromone_matrix),
                'std': np.std(self.pheromone_matrix),
                'max': np.max(self.pheromone_matrix),
                'min': np.min(self.pheromone_matrix)
            }
        }
        
        print(f"ACO路由优化完成 - 最优成本: {stats['best_cost']:.4f}")
        print(f"   优化时间: {optimization_time:.2f}秒")
        
        return best_routes, stats
    
    def visualize_routes(self, routes: List[Route], nodes_positions: np.ndarray,
                        cluster_heads: List[int], save_path: str = None) -> plt.Figure:
        """可视化路由结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 路由拓扑图
        ax1 = axes[0, 0]
        
        # 绘制节点
        ax1.scatter(nodes_positions[:-1, 0], nodes_positions[:-1, 1], 
                   c='lightblue', s=100, label='簇头节点', alpha=0.7)
        ax1.scatter(nodes_positions[-1, 0], nodes_positions[-1, 1], 
                   c='red', s=200, marker='s', label='基站', alpha=0.8)
        
        # 绘制路由路径
        colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))
        for i, (route, color) in enumerate(zip(routes, colors)):
            if len(route.path) > 1:
                path_positions = nodes_positions[route.path]
                ax1.plot(path_positions[:, 0], path_positions[:, 1], 
                        color=color, linewidth=2, alpha=0.7, 
                        label=f'路由 {i+1}')
        
        ax1.set_title('ACO优化路由拓扑')
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收敛曲线
        ax2 = axes[0, 1]
        if self.convergence_data:
            iterations = [d['iteration'] for d in self.convergence_data]
            best_costs = [d['best_cost'] for d in self.convergence_data]
            avg_costs = [d['avg_cost'] for d in self.convergence_data]
            
            ax2.plot(iterations, best_costs, 'b-', label='最优成本', linewidth=2)
            ax2.plot(iterations, avg_costs, 'r--', label='平均成本', linewidth=2)
            ax2.set_title('ACO收敛曲线')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('路由成本')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 路由性能对比
        ax3 = axes[1, 0]
        if routes:
            route_metrics = ['成本', '能耗', '可靠性', '延迟']
            route_values = [
                [r.cost for r in routes],
                [r.energy_consumption for r in routes],
                [r.reliability for r in routes],
                [r.latency for r in routes]
            ]
            
            x = np.arange(len(routes))
            width = 0.2
            
            for i, (metric, values) in enumerate(zip(route_metrics, route_values)):
                # 归一化值
                norm_values = np.array(values) / (max(values) + 1e-10)
                ax3.bar(x + i * width, norm_values, width, label=metric, alpha=0.7)
            
            ax3.set_title('路由性能对比')
            ax3.set_xlabel('路由编号')
            ax3.set_ylabel('归一化性能值')
            ax3.set_xticks(x + width * 1.5)
            ax3.set_xticklabels([f'路由{i+1}' for i in range(len(routes))])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 信息素热力图
        ax4 = axes[1, 1]
        if self.pheromone_matrix is not None:
            im = ax4.imshow(self.pheromone_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_title('信息素分布热力图')
            ax4.set_xlabel('目标节点')
            ax4.set_ylabel('源节点')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ACO路由图表已保存到: {save_path}")
        
        return fig
    
    def get_routing_summary(self) -> Dict:
        """获取路由总结"""
        summary = {
            "algorithm": "Ant Colony Optimization for Routing",
            "parameters": {
                "n_ants": self.n_ants,
                "n_iterations": self.n_iterations,
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "q0": self.q0,
                "tau0": self.tau0
            },
            "weights": self.weights,
            "convergence_data": self.convergence_data[-10:] if self.convergence_data else [],
            "pheromone_stats": {
                "mean": float(np.mean(self.pheromone_matrix)) if self.pheromone_matrix is not None else 0,
                "std": float(np.std(self.pheromone_matrix)) if self.pheromone_matrix is not None else 0
            }
        }
        
        return summary
    
    def save_results(self, routes: List[Route], save_path: str):
        """保存路由结果"""
        results = {
            "routes": [
                {
                    "path": route.path,
                    "cost": route.cost,
                    "energy_consumption": route.energy_consumption,
                    "reliability": route.reliability,
                    "latency": route.latency
                }
                for route in routes
            ],
            "summary": self.get_routing_summary()
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"ACO路由结果已保存到: {save_path}")

if __name__ == "__main__":
    # 测试ACO路由器
    aco = ACORouter(n_ants=15, n_iterations=50)
    
    # 创建测试数据
    np.random.seed(42)
    cluster_heads = [0, 1, 2, 3, 4, 5]
    base_station = 6
    
    # 节点位置（包括基站）
    positions = np.random.uniform(0, 100, (7, 2))
    positions[-1] = [50, 50]  # 基站在中心
    
    # 节点能量和信任度
    energy = np.random.uniform(0.3, 1.0, 7)
    trust = np.random.uniform(0.6, 1.0, 7)
    
    # 运行路由优化
    routes, stats = aco.find_optimal_routes(
        cluster_heads, base_station, positions, energy, trust)
    
    print(f"找到 {len(routes)} 条路由")
    for i, route in enumerate(routes):
        print(f"路由 {i+1}: {route.path}, 成本: {route.cost:.4f}")
    
    # 可视化结果
    aco.visualize_routes(routes, positions, cluster_heads, "aco_routes.png")
    
    # 保存结果
    aco.save_results(routes, "aco_results.json")