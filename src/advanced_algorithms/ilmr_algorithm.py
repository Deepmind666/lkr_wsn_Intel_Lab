"""
ILMR: Interpretable Lightweight Meta-heuristic Routing Algorithm
可解释的轻量级元启发式路由算法

核心创新：
1. 多元启发式算法融合（PSO+ACO+GA）
2. 可解释性机制与决策透明度
3. 轻量级计算优化
4. 自适应参数调整

作者: WSN研究团队
日期: 2025年1月
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RouteDecision:
    """路由决策记录"""
    decision_id: str
    timestamp: float
    source_node: int
    destination_node: int
    selected_path: List[int]
    alternative_paths: List[List[int]]
    decision_factors: Dict[str, float]
    algorithm_weights: Dict[str, float]
    confidence_score: float
    explanation: str

@dataclass
class PerformanceMetrics:
    """性能指标"""
    energy_consumption: float
    path_length: float
    reliability: float
    latency: float
    throughput: float
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """计算综合评分"""
        if weights is None:
            weights = {
                'energy': 0.3,
                'path_length': 0.2,
                'reliability': 0.25,
                'latency': 0.15,
                'throughput': 0.1
            }
        
        # 归一化指标（越小越好的指标需要取倒数）
        normalized_energy = 1.0 / (1.0 + self.energy_consumption)
        normalized_path_length = 1.0 / (1.0 + self.path_length)
        normalized_latency = 1.0 / (1.0 + self.latency)
        
        score = (weights['energy'] * normalized_energy +
                weights['path_length'] * normalized_path_length +
                weights['reliability'] * self.reliability +
                weights['latency'] * normalized_latency +
                weights['throughput'] * self.throughput)
        
        return score

class MetaHeuristicAlgorithm(ABC):
    """元启发式算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
        self.computation_time = 0
    
    @abstractmethod
    def find_path(self, source: int, destination: int, 
                  network_graph: np.ndarray, node_features: np.ndarray) -> Tuple[List[int], Dict]:
        """寻找路径的抽象方法"""
        pass
    
    @abstractmethod
    def update_parameters(self, performance_feedback: float):
        """根据性能反馈更新参数"""
        pass

class PSORouting(MetaHeuristicAlgorithm):
    """粒子群优化路由算法"""
    
    def __init__(self, num_particles: int = 20, max_iterations: int = 50,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        super().__init__("PSO")
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        
        # 自适应参数
        self.adaptive_w = True
        self.w_min = 0.4
        self.w_max = 0.9
    
    def find_path(self, source: int, destination: int, 
                  network_graph: np.ndarray, node_features: np.ndarray) -> Tuple[List[int], Dict]:
        """使用PSO寻找最优路径"""
        start_time = time.time()
        
        num_nodes = len(network_graph)
        
        # 初始化粒子群
        particles = self._initialize_particles(source, destination, network_graph, num_nodes)
        
        # 个体最优和全局最优
        personal_best = [particle.copy() for particle in particles]
        personal_best_fitness = [self._evaluate_fitness(path, network_graph, node_features) 
                               for path in personal_best]
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # 速度初始化
        velocities = [np.random.uniform(-1, 1, len(particle)) for particle in particles]
        
        fitness_history = []
        
        # PSO主循环
        for iteration in range(self.max_iterations):
            if self.adaptive_w:
                self.w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations

            for i in range(self.num_particles):
                # 验证并修复无效粒子
                if not particles[i] or len(particles[i]) < 2:
                    particles[i] = self._generate_random_path(source, destination, network_graph, num_nodes)
                    if not particles[i]: continue # 如果无法生成有效路径，则跳过

                if not personal_best[i] or len(personal_best[i]) < 2:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = self._evaluate_fitness(personal_best[i], network_graph, node_features)

                if not global_best or len(global_best) < 2:
                    global_best = particles[i].copy()
                    global_best_fitness = personal_best_fitness[i]

                # 动态调整速度向量的长度
                max_len = max(len(particles[i]), len(personal_best[i]), len(global_best))
                particle_pos = np.pad(particles[i], (0, max_len - len(particles[i])), 'constant', constant_values=-1)
                pb_pos = np.pad(personal_best[i], (0, max_len - len(personal_best[i])), 'constant', constant_values=-1)
                gb_pos = np.pad(global_best, (0, max_len - len(global_best)), 'constant', constant_values=-1)
                
                if len(velocities[i]) != max_len:
                    velocities[i] = np.resize(velocities[i], max_len)

                # 更新速度
                r1, r2 = np.random.random(), np.random.random()
                cognitive_component = self.c1 * r1 * (pb_pos - particle_pos)
                social_component = self.c2 * r2 * (gb_pos - particle_pos)
                velocities[i] = self.w * velocities[i] + cognitive_component + social_component

                # 更新位置
                particles[i] = self._update_particle_position(
                    particles[i], velocities[i], source, destination, network_graph
                )

                # 评估适应度
                fitness = self._evaluate_fitness(particles[i], network_graph, node_features)

                # 更新个体和全局最优
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i][:]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = particles[i][:]
                        global_best_fitness = fitness
            
            fitness_history.append(global_best_fitness)
            
            # 早停条件
            if len(fitness_history) > 10:
                recent_improvement = abs(fitness_history[-1] - fitness_history[-10])
                if recent_improvement < 1e-6:
                    break
        
        self.computation_time = time.time() - start_time
        
        # 返回结果和元信息
        meta_info = {
            'algorithm': 'PSO',
            'iterations': iteration + 1,
            'final_fitness': global_best_fitness,
            'computation_time': self.computation_time,
            'convergence_history': fitness_history,
            'parameters': {
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'num_particles': self.num_particles
            }
        }
        
        return global_best, meta_info
    
    def _initialize_particles(self, source: int, destination: int, 
                            network_graph: np.ndarray, num_nodes: int) -> List[List[int]]:
        """初始化粒子群"""
        particles = []
        
        for _ in range(self.num_particles):
            # 生成随机路径
            path = self._generate_random_path(source, destination, network_graph, num_nodes)
            particles.append(path)
        
        return particles
    
    def _generate_random_path(self, source: int, destination: int, 
                            network_graph: np.ndarray, num_nodes: int) -> List[int]:
        """生成一条从源到目标的随机路径"""
        path = [source]
        current_node = source
        available_nodes = list(range(num_nodes))
        available_nodes.remove(source)

        while current_node != destination and available_nodes:
            neighbors = [n for n in available_nodes if network_graph[current_node, n] > 0]
            if not neighbors:
                # 如果没有可达的邻居，则回溯或结束
                break

            # 优先选择目标节点
            if destination in neighbors:
                next_node = destination
            else:
                next_node = np.random.choice(neighbors)
            
            path.append(next_node)
            available_nodes.remove(next_node)
            current_node = next_node

        # 如果路径未到达终点，尝试连接到终点
        if path[-1] != destination:
            if network_graph[path[-1], destination] > 0:
                path.append(destination)
            else:
                # 如果无法直接连接，返回一个空路径或进行更复杂的修复
                return [] 

        return path
    
    def _update_particle_position(self, particle: List[int], velocity: np.ndarray,
                                source: int, destination: int, 
                                network_graph: np.ndarray) -> List[int]:
        """更新粒子位置（路径）"""
        # 基于速度向量，对路径进行扰动
        new_path = list(particle)
        num_nodes = len(network_graph)

        # 速度越大，改变路径的可能性越大
        if np.random.rand() < np.linalg.norm(velocity) / num_nodes:
            if len(new_path) > 2:
                # 选择一个点进行修改（不包括起点和终点）
                idx_to_modify = np.random.randint(1, len(new_path) - 1)
                prev_node = new_path[idx_to_modify - 1]

                # 寻找可替换的邻居节点
                neighbors = [n for n in range(num_nodes) 
                             if network_graph[prev_node, n] > 0 and n != new_path[idx_to_modify]]
                
                if neighbors:
                    # 选择一个新节点替换
                    new_node = np.random.choice(neighbors)
                    
                    # 构建新路径片段
                    try:
                        # 尝试从新节点找到一条回到旧路径的路径
                        remaining_path_start = new_path[idx_to_modify+1:]
                        # 这里需要一个辅助函数来找到从 new_node 到 remaining_path_start 中某个点的路径
                        # 为简化，我们直接替换并重新生成到终点的路径
                        new_path = new_path[:idx_to_modify]
                        new_path.append(new_node)
                        
                        # 从新节点继续生成到终点的路径
                        current = new_node
                        visited = set(new_path)
                        while current != destination and len(new_path) < num_nodes:
                            next_neighbors = [n for n in range(num_nodes) if network_graph[current, n] > 0 and n not in visited]
                            if not next_neighbors:
                                break
                            if destination in next_neighbors:
                                next_node = destination
                            else:
                                next_node = np.random.choice(next_neighbors)
                            new_path.append(next_node)
                            visited.add(next_node)
                            current = next_node

                    except Exception as e:
                        # 如果路径构建失败，返回原路径
                        return particle

        # 确保路径合法
        if not new_path or new_path[0] != source or new_path[-1] != destination:
            return particle

        return new_path
    
    def _evaluate_fitness(self, path: List[int], network_graph: np.ndarray, 
                         node_features: np.ndarray) -> float:
        """评估路径适应度"""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # 检查连接是否存在
            if network_graph[current][next_node] == 0:
                return float('inf')
            
            # 计算跳跃成本
            distance_cost = network_graph[current][next_node]
            energy_cost = 1.0 / (node_features[current][2] + 0.01)  # 能量越低成本越高
            
            total_cost += distance_cost + energy_cost
        
        # 路径长度惩罚
        length_penalty = len(path) * 0.1
        
        return total_cost + length_penalty
    
    def update_parameters(self, performance_feedback: float):
        """根据性能反馈更新参数"""
        self.performance_history.append(performance_feedback)
        
        # 自适应调整参数
        if len(self.performance_history) > 5:
            recent_performance = np.mean(self.performance_history[-5:])
            
            if recent_performance < 0.5:  # 性能较差
                self.c1 = min(self.c1 + 0.1, 2.0)  # 增加个体学习
                self.c2 = max(self.c2 - 0.1, 1.0)  # 减少社会学习
            else:  # 性能较好
                self.c1 = max(self.c1 - 0.05, 1.0)
                self.c2 = min(self.c2 + 0.05, 2.0)

class ACORouting(MetaHeuristicAlgorithm):
    """蚁群优化路由算法"""
    
    def __init__(self, num_ants: int = 20, max_iterations: int = 50,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, Q: float = 100):
        super().__init__("ACO")
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta    # 启发式信息重要程度
        self.rho = rho      # 信息素蒸发率
        self.Q = Q          # 信息素强度
        
        self.pheromone_matrix = None
    
    def find_path(self, source: int, destination: int, 
                  network_graph: np.ndarray, node_features: np.ndarray) -> Tuple[List[int], Dict]:
        """使用ACO寻找最优路径"""
        start_time = time.time()
        
        num_nodes = len(network_graph)
        
        # 初始化信息素矩阵
        if self.pheromone_matrix is None or self.pheromone_matrix.shape[0] != num_nodes:
            self.pheromone_matrix = np.ones((num_nodes, num_nodes)) * 0.1
        
        # 计算启发式信息矩阵
        heuristic_matrix = self._calculate_heuristic_matrix(network_graph, node_features)
        
        best_path = None
        best_cost = float('inf')
        cost_history = []
        
        # ACO主循环
        for iteration in range(self.max_iterations):
            paths = []
            costs = []
            
            # 每只蚂蚁构建路径
            for ant in range(self.num_ants):
                path = self._construct_path(source, destination, network_graph, 
                                          heuristic_matrix, num_nodes)
                cost = self._calculate_path_cost(path, network_graph, node_features)
                
                paths.append(path)
                costs.append(cost)
                
                # 更新最优解
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()
            
            # 更新信息素
            self._update_pheromone(paths, costs)
            
            cost_history.append(best_cost)
            
            # 早停条件
            if len(cost_history) > 10:
                recent_improvement = abs(cost_history[-1] - cost_history[-10])
                if recent_improvement < 1e-6:
                    break
        
        self.computation_time = time.time() - start_time
        
        # 返回结果和元信息
        meta_info = {
            'algorithm': 'ACO',
            'iterations': iteration + 1,
            'final_cost': best_cost,
            'computation_time': self.computation_time,
            'convergence_history': cost_history,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'Q': self.Q,
                'num_ants': self.num_ants
            }
        }
        
        return best_path if best_path else [source, destination], meta_info
    
    def _calculate_heuristic_matrix(self, network_graph: np.ndarray, 
                                  node_features: np.ndarray) -> np.ndarray:
        """计算启发式信息矩阵"""
        num_nodes = len(network_graph)
        heuristic_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if network_graph[i][j] > 0:
                    # 距离倒数
                    distance_heuristic = 1.0 / (network_graph[i][j] + 0.01)
                    
                    # 能量启发式
                    energy_heuristic = node_features[j][2]  # 目标节点能量
                    
                    # 综合启发式信息
                    heuristic_matrix[i][j] = distance_heuristic * energy_heuristic
        
        return heuristic_matrix
    
    def _construct_path(self, source: int, destination: int, network_graph: np.ndarray,
                       heuristic_matrix: np.ndarray, num_nodes: int) -> List[int]:
        """构建蚂蚁路径"""
        path = [source]
        current = source
        visited = {source}
        
        while current != destination and len(path) < num_nodes:
            # 计算转移概率
            probabilities = self._calculate_transition_probabilities(
                current, visited, network_graph, heuristic_matrix
            )
            
            if not probabilities:
                break
            
            # 轮盘赌选择下一个节点
            next_node = self._roulette_wheel_selection(probabilities)
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path
    
    def _calculate_transition_probabilities(self, current: int, visited: set,
                                          network_graph: np.ndarray,
                                          heuristic_matrix: np.ndarray) -> Dict[int, float]:
        """计算转移概率"""
        probabilities = {}
        total = 0
        
        for next_node in range(len(network_graph)):
            if (network_graph[current][next_node] > 0 and 
                next_node not in visited):
                
                pheromone = self.pheromone_matrix[current][next_node]
                heuristic = heuristic_matrix[current][next_node]
                
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities[next_node] = probability
                total += probability
        
        # 归一化概率
        if total > 0:
            for node in probabilities:
                probabilities[node] /= total
        
        return probabilities
    
    def _roulette_wheel_selection(self, probabilities: Dict[int, float]) -> int:
        """轮盘赌选择"""
        if not probabilities:
            return -1
        
        rand = np.random.random()
        cumulative = 0
        
        for node, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return node
        
        # 如果没有选中，返回最后一个
        return list(probabilities.keys())[-1]
    
    def _calculate_path_cost(self, path: List[int], network_graph: np.ndarray,
                           node_features: np.ndarray) -> float:
        """计算路径成本"""
        if len(path) < 2:
            return float('inf')
        
        total_cost = 0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if network_graph[current][next_node] == 0:
                return float('inf')
            
            # 距离成本
            distance_cost = network_graph[current][next_node]
            
            # 能量成本
            energy_cost = 1.0 / (node_features[current][2] + 0.01)
            
            total_cost += distance_cost + energy_cost
        
        return total_cost
    
    def _update_pheromone(self, paths: List[List[int]], costs: List[float]):
        """更新信息素"""
        # 信息素蒸发
        self.pheromone_matrix *= (1 - self.rho)
        
        # 信息素增强
        for path, cost in zip(paths, costs):
            if cost < float('inf') and len(path) > 1:
                pheromone_deposit = self.Q / cost
                
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    self.pheromone_matrix[current][next_node] += pheromone_deposit
    
    def update_parameters(self, performance_feedback: float):
        """根据性能反馈更新参数"""
        self.performance_history.append(performance_feedback)
        
        # 自适应调整参数
        if len(self.performance_history) > 5:
            recent_performance = np.mean(self.performance_history[-5:])
            
            if recent_performance < 0.5:  # 性能较差
                self.alpha = min(self.alpha + 0.1, 2.0)  # 增加信息素重要性
                self.rho = min(self.rho + 0.02, 0.3)     # 增加蒸发率
            else:  # 性能较好
                self.alpha = max(self.alpha - 0.05, 0.5)
                self.rho = max(self.rho - 0.01, 0.05)

class GARouting(MetaHeuristicAlgorithm):
    """遗传算法路由"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 30,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        super().__init__("GA")
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def find_path(self, source: int, destination: int, 
                  network_graph: np.ndarray, node_features: np.ndarray) -> Tuple[List[int], Dict]:
        """使用GA寻找最优路径"""
        start_time = time.time()
        
        num_nodes = len(network_graph)
        
        # 初始化种群
        population = self._initialize_population(source, destination, network_graph, num_nodes)
        
        best_path = None
        best_fitness = float('inf')
        fitness_history = []
        
        # GA主循环
        for generation in range(self.max_generations):
            # 评估适应度
            fitness_scores = [self._evaluate_fitness(individual, network_graph, node_features) 
                            for individual in population]
            
            # 更新最优解
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_path = population[min_fitness_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # 选择
            selected_population = self._selection(population, fitness_scores)
            
            # 交叉
            offspring = self._crossover(selected_population, source, destination, network_graph)
            
            # 变异
            offspring = self._mutation(offspring, network_graph)
            
            # 更新种群
            population = offspring
            
            # 早停条件
            if len(fitness_history) > 10:
                recent_improvement = abs(fitness_history[-1] - fitness_history[-10])
                if recent_improvement < 1e-6:
                    break
        
        self.computation_time = time.time() - start_time
        
        # 返回结果和元信息
        meta_info = {
            'algorithm': 'GA',
            'generations': generation + 1,
            'final_fitness': best_fitness,
            'computation_time': self.computation_time,
            'convergence_history': fitness_history,
            'parameters': {
                'population_size': self.population_size,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate
            }
        }
        
        return best_path if best_path else [source, destination], meta_info
    
    def _initialize_population(self, source: int, destination: int, 
                             network_graph: np.ndarray, num_nodes: int) -> List[List[int]]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            path = self._generate_random_path(source, destination, network_graph, num_nodes)
            population.append(path)
        
        return population
    
    def _generate_random_path(self, source: int, destination: int, 
                            network_graph: np.ndarray, num_nodes: int) -> List[int]:
        """生成随机路径"""
        path = [source]
        current = source
        visited = {source}
        max_hops = min(num_nodes, 8)
        
        for _ in range(max_hops):
            if current == destination:
                break
            
            neighbors = [i for i in range(num_nodes) 
                        if network_graph[current][i] > 0 and i not in visited]
            
            if not neighbors:
                break
            
            if destination in neighbors:
                next_node = destination
            else:
                next_node = np.random.choice(neighbors)
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        if path[-1] != destination and network_graph[path[-1]][destination] > 0:
            path.append(destination)
        
        return path
    
    def _evaluate_fitness(self, individual: List[int], network_graph: np.ndarray, 
                         node_features: np.ndarray) -> float:
        """评估个体适应度"""
        if len(individual) < 2:
            return float('inf')
        
        total_cost = 0
        
        for i in range(len(individual) - 1):
            current = individual[i]
            next_node = individual[i + 1]
            
            if network_graph[current][next_node] == 0:
                return float('inf')
            
            distance_cost = network_graph[current][next_node]
            energy_cost = 1.0 / (node_features[current][2] + 0.01)
            
            total_cost += distance_cost + energy_cost
        
        return total_cost + len(individual) * 0.1
    
    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """锦标赛选择"""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # 锦标赛选择
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, population: List[List[int]], source: int, destination: int,
                  network_graph: np.ndarray) -> List[List[int]]:
        """交叉操作"""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2, source, destination, network_graph)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring[:self.population_size]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int], 
                        source: int, destination: int, network_graph: np.ndarray) -> Tuple[List[int], List[int]]:
        """顺序交叉"""
        # 简化的交叉：交换中间部分
        min_len = min(len(parent1), len(parent2))
        if min_len <= 2:
            return parent1.copy(), parent2.copy()
        
        # 选择交叉点
        start = 1
        end = min_len - 1
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 交换中间部分
        if start < end:
            child1[start:end] = parent2[start:end]
            child2[start:end] = parent1[start:end]
        
        # 修复路径连通性
        child1 = self._repair_path(child1, source, destination, network_graph)
        child2 = self._repair_path(child2, source, destination, network_graph)
        
        return child1, child2
    
    def _mutation(self, population: List[List[int]], network_graph: np.ndarray) -> List[List[int]]:
        """变异操作"""
        mutated_population = []
        
        for individual in population:
            if np.random.random() < self.mutation_rate:
                mutated_individual = self._mutate_individual(individual, network_graph)
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual.copy())
        
        return mutated_population
    
    def _mutate_individual(self, individual: List[int], network_graph: np.ndarray) -> List[int]:
        """个体变异"""
        if len(individual) <= 2:
            return individual.copy()
        
        mutated = individual.copy()
        
        # 随机选择一个中间节点进行变异
        mutation_idx = np.random.randint(1, len(mutated) - 1)
        current_node = mutated[mutation_idx - 1]
        next_node = mutated[mutation_idx + 1]
        
        # 寻找替代节点
        alternatives = []
        for node in range(len(network_graph)):
            if (network_graph[current_node][node] > 0 and 
                network_graph[node][next_node] > 0 and
                node not in mutated):
                alternatives.append(node)
        
        if alternatives:
            mutated[mutation_idx] = np.random.choice(alternatives)
        
        return mutated
    
    def _repair_path(self, path: List[int], source: int, destination: int,
                    network_graph: np.ndarray) -> List[int]:
        """修复路径连通性"""
        if len(path) < 2:
            return [source, destination]
        
        repaired = [source]
        
        for i in range(1, len(path)):
            current = repaired[-1]
            target = path[i]
            
            if network_graph[current][target] > 0:
                repaired.append(target)
            else:
                # 寻找中间节点
                for intermediate in range(len(network_graph)):
                    if (network_graph[current][intermediate] > 0 and 
                        network_graph[intermediate][target] > 0):
                        repaired.append(intermediate)
                        repaired.append(target)
                        break
        
        # 确保以目标节点结束
        if repaired[-1] != destination:
            if network_graph[repaired[-1]][destination] > 0:
                repaired.append(destination)
        
        return repaired
    
    def update_parameters(self, performance_feedback: float):
        """根据性能反馈更新参数"""
        self.performance_history.append(performance_feedback)
        
        # 自适应调整参数
        if len(self.performance_history) > 5:
            recent_performance = np.mean(self.performance_history[-5:])
            
            if recent_performance < 0.5:  # 性能较差
                self.mutation_rate = min(self.mutation_rate + 0.02, 0.3)
                self.crossover_rate = max(self.crossover_rate - 0.05, 0.5)
            else:  # 性能较好
                self.mutation_rate = max(self.mutation_rate - 0.01, 0.05)
                self.crossover_rate = min(self.crossover_rate + 0.02, 0.95)

class ExplainabilityEngine:
    """可解释性引擎"""
    
    def __init__(self):
        self.decision_history = []
        self.feature_importance = defaultdict(float)
        self.algorithm_performance = defaultdict(list)
    
    def explain_decision(self, route_decision: RouteDecision) -> Dict[str, Union[str, float, Dict]]:
        """解释路由决策"""
        explanation = {
            'decision_summary': self._generate_decision_summary(route_decision),
            'factor_analysis': self._analyze_decision_factors(route_decision),
            'algorithm_contribution': self._analyze_algorithm_contribution(route_decision),
            'confidence_analysis': self._analyze_confidence(route_decision),
            'alternative_analysis': self._analyze_alternatives(route_decision),
            'risk_assessment': self._assess_risks(route_decision)
        }
        
        return explanation
    
    def _generate_decision_summary(self, decision: RouteDecision) -> str:
        """生成决策摘要"""
        path_length = len(decision.selected_path)
        dominant_algorithm = max(decision.algorithm_weights.items(), key=lambda x: x[1])
        
        summary = (f"选择了长度为{path_length}跳的路径 {' -> '.join(map(str, decision.selected_path))}。"
                  f"决策主要基于{dominant_algorithm[0]}算法（权重{dominant_algorithm[1]:.2f}），"
                  f"置信度为{decision.confidence_score:.2f}。")
        
        return summary
    
    def _analyze_decision_factors(self, decision: RouteDecision) -> Dict[str, float]:
        """分析决策因子"""
        factors = decision.decision_factors.copy()
        
        # 归一化因子重要性
        total_importance = sum(abs(v) for v in factors.values())
        if total_importance > 0:
            normalized_factors = {k: abs(v) / total_importance for k, v in factors.items()}
        else:
            normalized_factors = factors
        
        # 排序因子
        sorted_factors = dict(sorted(normalized_factors.items(), 
                                   key=lambda x: x[1], reverse=True))
        
        return sorted_factors
    
    def _analyze_algorithm_contribution(self, decision: RouteDecision) -> Dict[str, Dict]:
        """分析算法贡献"""
        contributions = {}
        
        for algorithm, weight in decision.algorithm_weights.items():
            contributions[algorithm] = {
                'weight': weight,
                'contribution_level': self._categorize_contribution(weight),
                'historical_performance': np.mean(self.algorithm_performance.get(algorithm, [0.5]))
            }
        
        return contributions
    
    def _categorize_contribution(self, weight: float) -> str:
        """分类贡献水平"""
        if weight >= 0.6:
            return "主导"
        elif weight >= 0.3:
            return "重要"
        elif weight >= 0.1:
            return "辅助"
        else:
            return "微弱"
    
    def _analyze_confidence(self, decision: RouteDecision) -> Dict[str, Union[str, float]]:
        """分析置信度"""
        confidence = decision.confidence_score
        
        if confidence >= 0.8:
            level = "高"
            interpretation = "决策非常可靠，建议采用"
        elif confidence >= 0.6:
            level = "中等"
            interpretation = "决策较为可靠，可以采用但需监控"
        elif confidence >= 0.4:
            level = "低"
            interpretation = "决策可靠性一般，建议谨慎采用"
        else:
            level = "很低"
            interpretation = "决策可靠性差，建议重新评估"
        
        return {
            'confidence_score': confidence,
            'confidence_level': level,
            'interpretation': interpretation
        }
    
    def _analyze_alternatives(self, decision: RouteDecision) -> Dict[str, Union[int, str]]:
        """分析备选方案"""
        num_alternatives = len(decision.alternative_paths)
        
        if num_alternatives == 0:
            diversity = "无备选方案"
        elif num_alternatives <= 2:
            diversity = "备选方案较少"
        elif num_alternatives <= 5:
            diversity = "备选方案适中"
        else:
            diversity = "备选方案丰富"
        
        return {
            'num_alternatives': num_alternatives,
            'diversity_assessment': diversity
        }
    
    def _assess_risks(self, decision: RouteDecision) -> Dict[str, Union[str, float]]:
        """评估风险"""
        risks = []
        risk_score = 0
        
        # 路径长度风险
        path_length = len(decision.selected_path)
        if path_length > 6:
            risks.append("路径较长，可能影响延迟")
            risk_score += 0.2
        
        # 置信度风险
        if decision.confidence_score < 0.5:
            risks.append("决策置信度较低")
            risk_score += 0.3
        
        # 算法多样性风险
        max_weight = max(decision.algorithm_weights.values())
        if max_weight > 0.8:
            risks.append("过度依赖单一算法")
            risk_score += 0.2
        
        # 备选方案风险
        if len(decision.alternative_paths) < 2:
            risks.append("备选方案不足")
            risk_score += 0.1
        
        risk_level = "低" if risk_score < 0.3 else "中等" if risk_score < 0.6 else "高"
        
        return {
            'risk_factors': risks,
            'risk_score': min(risk_score, 1.0),
            'risk_level': risk_level
        }
    
    def update_feature_importance(self, features: Dict[str, float], performance: float):
        """更新特征重要性"""
        for feature, value in features.items():
            # 使用指数移动平均更新重要性
            alpha = 0.1
            self.feature_importance[feature] = (
                alpha * performance * abs(value) + 
                (1 - alpha) * self.feature_importance[feature]
            )
    
    def get_global_insights(self) -> Dict[str, Union[Dict, List]]:
        """获取全局洞察"""
        insights = {
            'most_important_features': dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], reverse=True
            )[:5]),
            'algorithm_performance_ranking': self._rank_algorithms(),
            'decision_patterns': self._analyze_decision_patterns(),
            'recommendations': self._generate_recommendations()
        }
        
        return insights
    
    def _rank_algorithms(self) -> Dict[str, float]:
        """算法性能排名"""
        rankings = {}
        for algorithm, performances in self.algorithm_performance.items():
            rankings[algorithm] = np.mean(performances) if performances else 0
        
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_decision_patterns(self) -> List[str]:
        """分析决策模式"""
        patterns = []
        
        if len(self.decision_history) > 10:
            # 分析路径长度趋势
            path_lengths = [len(d.selected_path) for d in self.decision_history[-10:]]
            avg_length = np.mean(path_lengths)
            
            if avg_length > 5:
                patterns.append("倾向于选择较长路径")
            elif avg_length < 3:
                patterns.append("倾向于选择较短路径")
            
            # 分析算法偏好
            algorithm_usage = defaultdict(int)
            for decision in self.decision_history[-10:]:
                dominant_alg = max(decision.algorithm_weights.items(), key=lambda x: x[1])[0]
                algorithm_usage[dominant_alg] += 1
            
            most_used = max(algorithm_usage.items(), key=lambda x: x[1])
            if most_used[1] > 5:
                patterns.append(f"频繁使用{most_used[0]}算法")
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于特征重要性的建议
        if 'energy' in self.feature_importance and self.feature_importance['energy'] > 0.5:
            recommendations.append("能量因子对决策影响较大，建议优化能量管理策略")
        
        # 基于算法性能的建议
        algorithm_rankings = self._rank_algorithms()
        if algorithm_rankings:
            best_algorithm = list(algorithm_rankings.keys())[0]
            recommendations.append(f"建议增加{best_algorithm}算法的权重")
        
        return recommendations

class ILMRAlgorithm:
    """ILMR主算法类"""
    
    def __init__(self, algorithms: List[MetaHeuristicAlgorithm] = None):
        # 初始化元启发式算法
        if algorithms is None:
            self.algorithms = [
                PSORouting(num_particles=15, max_iterations=30),
                ACORouting(num_ants=15, max_iterations=30),
                GARouting(population_size=30, max_generations=20)
            ]
        else:
            self.algorithms = algorithms
        
        # 算法权重（动态调整）
        self.algorithm_weights = {alg.name: 1.0 / len(self.algorithms) for alg in self.algorithms}
        
        # 可解释性引擎
        self.explainability_engine = ExplainabilityEngine()
        
        # 性能历史
        self.performance_history = {
            'network_lifetime': [],
            'energy_efficiency': [],
            'routing_success_rate': [],
            'average_path_length': [],
            'computation_time': []
        }
        
        # 决策计数器
        self.decision_counter = 0
    
    def find_optimal_route(self, source: int, destination: int, 
                          network_graph: np.ndarray, node_features: np.ndarray) -> RouteDecision:
        """寻找最优路由"""
        start_time = time.time()
        
        # 并行运行所有算法
        algorithm_results = {}
        for algorithm in self.algorithms:
            path, meta_info = algorithm.find_path(source, destination, network_graph, node_features)
            
            # 计算路径性能指标
            performance = self._evaluate_path_performance(path, network_graph, node_features)
            
            algorithm_results[algorithm.name] = {
                'path': path,
                'meta_info': meta_info,
                'performance': performance
            }
        
        # 融合算法结果
        selected_path, confidence_score = self._fuse_algorithm_results(algorithm_results)
        
        # 生成备选路径
        alternative_paths = self._generate_alternative_paths(
            algorithm_results, selected_path
        )
        
        # 计算决策因子
        decision_factors = self._calculate_decision_factors(
            selected_path, network_graph, node_features
        )
        
        # 创建路由决策对象
        decision_id = f"ILMR_{self.decision_counter:06d}"
        self.decision_counter += 1
        
        route_decision = RouteDecision(
            decision_id=decision_id,
            timestamp=time.time(),
            source_node=source,
            destination_node=destination,
            selected_path=selected_path,
            alternative_paths=alternative_paths,
            decision_factors=decision_factors,
            algorithm_weights=self.algorithm_weights.copy(),
            confidence_score=confidence_score,
            explanation=""
        )
        
        # 生成解释
        explanation = self.explainability_engine.explain_decision(route_decision)
        route_decision.explanation = explanation['decision_summary']
        
        # 更新算法权重
        self._update_algorithm_weights(algorithm_results)
        
        # 记录决策历史
        self.explainability_engine.decision_history.append(route_decision)
        
        computation_time = time.time() - start_time
        self.performance_history['computation_time'].append(computation_time)
        
        return route_decision
    
    def _evaluate_path_performance(self, path: List[int], network_graph: np.ndarray,
                                 node_features: np.ndarray) -> PerformanceMetrics:
        """评估路径性能"""
        if len(path) < 2:
            return PerformanceMetrics(
                energy_consumption=float('inf'),
                path_length=float('inf'),
                reliability=0,
                latency=float('inf'),
                throughput=0
            )
        
        # 计算能量消耗
        energy_consumption = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if network_graph[current][next_node] == 0:
                return PerformanceMetrics(
                    energy_consumption=float('inf'),
                    path_length=float('inf'),
                    reliability=0,
                    latency=float('inf'),
                    throughput=0
                )
            
            # 传输能耗
            distance = network_graph[current][next_node]
            energy_consumption += 0.001 * (distance ** 2)
            
            # 节点处理能耗
            energy_consumption += 0.01 / (node_features[current][2] + 0.01)
        
        # 计算路径长度
        path_length = len(path) - 1
        
        # 计算可靠性（基于节点能量）
        min_energy = min(node_features[node][2] for node in path)
        avg_energy = np.mean([node_features[node][2] for node in path])
        reliability = min_energy * 0.6 + avg_energy * 0.4
        
        # 计算延迟（基于跳数和距离）
        total_distance = sum(network_graph[path[i]][path[i+1]] for i in range(len(path)-1))
        latency = path_length * 0.1 + total_distance * 0.01
        
        # 计算吞吐量（基于最小节点容量）
        min_capacity = min(node_features[node][2] for node in path)
        throughput = min_capacity * 10  # 简化计算
        
        return PerformanceMetrics(
            energy_consumption=energy_consumption,
            path_length=path_length,
            reliability=reliability,
            latency=latency,
            throughput=throughput
        )
    
    def _fuse_algorithm_results(self, algorithm_results: Dict) -> Tuple[List[int], float]:
        """融合算法结果"""
        # 计算加权性能分数
        weighted_scores = {}
        
        for alg_name, result in algorithm_results.items():
            performance = result['performance']
            weight = self.algorithm_weights[alg_name]
            
            # 计算综合性能分数
            composite_score = performance.get_composite_score()
            weighted_scores[alg_name] = composite_score * weight
        
        # 选择最佳算法的路径
        best_algorithm = max(weighted_scores.items(), key=lambda x: x[1])[0]
        selected_path = algorithm_results[best_algorithm]['path']
        
        # 计算置信度
        scores = list(weighted_scores.values())
        if len(scores) > 1:
            max_score = max(scores)
            second_max = sorted(scores, reverse=True)[1]
            confidence_score = (max_score - second_max) / max_score if max_score > 0 else 0
        else:
            confidence_score = 0.5
        
        confidence_score = min(max(confidence_score, 0), 1)
        
        return selected_path, confidence_score
    
    def _generate_alternative_paths(self, algorithm_results: Dict, 
                                  selected_path: List[int]) -> List[List[int]]:
        """生成备选路径"""
        alternatives = []
        
        for alg_name, result in algorithm_results.items():
            path = result['path']
            if path != selected_path and path not in alternatives:
                alternatives.append(path)
        
        return alternatives[:3]  # 最多保留3个备选方案
    
    def _calculate_decision_factors(self, path: List[int], network_graph: np.ndarray,
                                  node_features: np.ndarray) -> Dict[str, float]:
        """计算决策因子"""
        if len(path) < 2:
            return {}
        
        factors = {}
        
        # 能量因子
        path_energies = [node_features[node][2] for node in path]
        factors['energy_min'] = min(path_energies)
        factors['energy_avg'] = np.mean(path_energies)
        factors['energy_variance'] = np.var(path_energies)
        
        # 距离因子
        total_distance = sum(network_graph[path[i]][path[i+1]] for i in range(len(path)-1))
        factors['total_distance'] = total_distance
        factors['avg_hop_distance'] = total_distance / (len(path) - 1)
        
        # 拓扑因子
        factors['path_length'] = len(path) - 1
        factors['node_degree_avg'] = np.mean([
            np.sum(network_graph[node] > 0) for node in path
        ])
        
        return factors
    
    def _update_algorithm_weights(self, algorithm_results: Dict):
        """更新算法权重"""
        # 计算每个算法的性能分数
        performance_scores = {}
        for alg_name, result in algorithm_results.items():
            performance = result['performance']
            performance_scores[alg_name] = performance.get_composite_score()
        
        # 使用softmax更新权重
        scores = np.array(list(performance_scores.values()))
        if np.max(scores) > 0:
            # 避免数值溢出
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            softmax_weights = exp_scores / np.sum(exp_scores)
            
            # 更新权重（使用指数移动平均）
            alpha = 0.1
            for i, alg_name in enumerate(performance_scores.keys()):
                self.algorithm_weights[alg_name] = (
                    alpha * softmax_weights[i] + 
                    (1 - alpha) * self.algorithm_weights[alg_name]
                )
        
        # 记录算法性能
        for alg_name, score in performance_scores.items():
            self.explainability_engine.algorithm_performance[alg_name].append(score)
    
    def simulate_network_routing(self, network_graph: np.ndarray, node_features: np.ndarray,
                               routing_requests: List[Tuple[int, int]], 
                               max_rounds: int = 100) -> Dict:
        """模拟网络路由"""
        print("🚀 开始ILMR网络路由模拟...")
        
        simulation_results = {
            'total_requests': len(routing_requests),
            'successful_routes': 0,
            'failed_routes': 0,
            'average_path_length': 0,
            'total_energy_consumption': 0,
            'average_confidence': 0,
            'routing_decisions': [],
            'performance_evolution': []
        }
        
        nodes_data = node_features.copy()
        successful_paths = []
        confidence_scores = []
        total_energy = 0
        
        for round_num in range(max_rounds):
            round_requests = routing_requests[round_num::max_rounds]
            
            if not round_requests:
                continue
            
            round_successful = 0
            round_energy = 0
            
            for source, destination in round_requests:
                # 检查节点是否存活
                if nodes_data[source][3] <= 0 or nodes_data[destination][3] <= 0:
                    simulation_results['failed_routes'] += 1
                    continue
                
                # 寻找路由
                try:
                    route_decision = self.find_optimal_route(
                        source, destination, network_graph, nodes_data
                    )
                    
                    if len(route_decision.selected_path) >= 2:
                        # 成功路由
                        simulation_results['successful_routes'] += 1
                        round_successful += 1
                        
                        successful_paths.append(route_decision.selected_path)
                        confidence_scores.append(route_decision.confidence_score)
                        
                        # 计算能量消耗
                        path_energy = self._calculate_path_energy_consumption(
                            route_decision.selected_path, network_graph, nodes_data
                        )
                        round_energy += path_energy
                        total_energy += path_energy
                        
                        # 更新节点能量
                        for i, node in enumerate(route_decision.selected_path):
                            if i < len(route_decision.selected_path) - 1:
                                nodes_data[node][2] -= path_energy / len(route_decision.selected_path)
                        
                        # 记录决策
                        simulation_results['routing_decisions'].append({
                            'round': round_num,
                            'source': source,
                            'destination': destination,
                            'path': route_decision.selected_path,
                            'confidence': route_decision.confidence_score,
                            'energy_cost': path_energy
                        })
                    else:
                        simulation_results['failed_routes'] += 1
                
                except Exception as e:
                    simulation_results['failed_routes'] += 1
                    print(f"路由失败: {source} -> {destination}, 错误: {e}")
            
            # 更新节点存活状态
            nodes_data[nodes_data[:, 2] <= 0, 3] = 0
            
            # 记录轮次性能
            alive_nodes = np.sum(nodes_data[:, 3] > 0)
            simulation_results['performance_evolution'].append({
                'round': round_num,
                'alive_nodes': alive_nodes,
                'successful_routes': round_successful,
                'round_energy_consumption': round_energy,
                'algorithm_weights': self.algorithm_weights.copy()
            })
            
            # 检查网络连通性
            if alive_nodes < len(nodes_data) * 0.2:  # 80%节点死亡
                print(f"网络在第{round_num}轮失去连通性")
                break
        
        # 计算最终统计
        if successful_paths:
            simulation_results['average_path_length'] = np.mean([
                len(path) for path in successful_paths
            ])
        
        if confidence_scores:
            simulation_results['average_confidence'] = np.mean(confidence_scores)
        
        simulation_results['total_energy_consumption'] = total_energy
        simulation_results['success_rate'] = (
            simulation_results['successful_routes'] / 
            max(simulation_results['total_requests'], 1)
        )
        
        print(f"✅ 路由模拟完成！成功率: {simulation_results['success_rate']:.2%}")
        return simulation_results
    
    def _calculate_path_energy_consumption(self, path: List[int], 
                                         network_graph: np.ndarray,
                                         node_features: np.ndarray) -> float:
        """计算路径能量消耗"""
        if len(path) < 2:
            return 0
        
        total_energy = 0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            # 传输能耗
            distance = network_graph[current][next_node]
            transmission_energy = 0.001 * (distance ** 2)
            
            # 处理能耗
            processing_energy = 0.01
            
            total_energy += transmission_energy + processing_energy
        
        return total_energy
    
    def get_explainability_report(self) -> Dict:
        """获取可解释性报告"""
        return self.explainability_engine.get_global_insights()
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'algorithm_weights': self.algorithm_weights,
            'performance_history': self.performance_history,
            'decision_counter': self.decision_counter,
            'feature_importance': dict(self.explainability_engine.feature_importance),
            'algorithm_performance': dict(self.explainability_engine.algorithm_performance)
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ ILMR模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.algorithm_weights = model_data['algorithm_weights']
        self.performance_history = model_data['performance_history']
        self.decision_counter = model_data['decision_counter']
        
        # 恢复可解释性引擎状态
        self.explainability_engine.feature_importance = defaultdict(
            float, model_data['feature_importance']
        )
        self.explainability_engine.algorithm_performance = defaultdict(
            list, model_data['algorithm_performance']
        )
        
        print(f"✅ ILMR模型已从 {filepath} 加载")

def demonstrate_ilmr():
    """演示ILMR算法"""
    print("🎯 ILMR算法演示")
    print("=" * 60)