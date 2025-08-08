"""
Enhanced EEHFR WSN系统 - PSO粒子群优化模块
基于用户调研文件中的混合元启发式优化设计
实现高效的簇头位置和能耗优化
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

@dataclass
class Particle:
    """粒子类"""
    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    best_position: np.ndarray
    best_fitness: float

class PSOOptimizer:
    """
    粒子群优化算法
    专门用于WSN簇头选择和能耗优化
    """
    
    def __init__(self, n_particles: int = 30, n_iterations: int = 100, 
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0,
                 w_decay: float = 0.95):
        """
        初始化PSO优化器
        
        Args:
            n_particles: 粒子数量
            n_iterations: 迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            w_decay: 惯性权重衰减因子
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.w_initial = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
        # 优化历史
        self.optimization_history = []
        self.best_fitness_history = []
        self.convergence_data = []
        
        # 性能统计
        self.total_evaluations = 0
        self.convergence_iteration = -1
        
    def _initialize_particles(self, n_nodes: int, n_clusters: int) -> List[Particle]:
        """初始化粒子群"""
        particles = []
        
        for _ in range(self.n_particles):
            # 随机初始化簇头位置（节点索引）
            position = np.random.choice(n_nodes, n_clusters, replace=False)
            velocity = np.random.uniform(-1, 1, n_clusters)
            
            particle = Particle(
                position=position.astype(float),
                velocity=velocity,
                fitness=-np.inf,
                best_position=position.astype(float).copy(),
                best_fitness=-np.inf
            )
            particles.append(particle)
        
        return particles
    
    def _evaluate_fitness(self, particle_position: np.ndarray, 
                         nodes_data: np.ndarray) -> float:
        """
        评估粒子适应度
        
        Args:
            particle_position: 粒子位置（簇头节点索引）
            nodes_data: 节点数据 [x, y, energy, ...]
            
        Returns:
            适应度值（越大越好）
        """
        self.total_evaluations += 1
        
        # 确保簇头索引有效
        cluster_heads = np.clip(particle_position, 0, len(nodes_data) - 1).astype(int)
        cluster_heads = np.unique(cluster_heads)  # 去除重复
        
        if len(cluster_heads) < 2:
            return -1000  # 惩罚无效解
        
        # 1. 能量均衡性评估
        ch_energies = nodes_data[cluster_heads, 2]  # 假设第3列是能量
        energy_balance = 1.0 / (1.0 + np.std(ch_energies))
        
        # 2. 簇内距离评估（越小越好）
        total_intra_distance = 0
        for ch_idx in cluster_heads:
            ch_pos = nodes_data[ch_idx, :2]  # x, y坐标
            distances = np.sqrt(np.sum((nodes_data[:, :2] - ch_pos)**2, axis=1))
            total_intra_distance += np.mean(distances)
        
        avg_intra_distance = total_intra_distance / len(cluster_heads)
        distance_score = 1.0 / (1.0 + avg_intra_distance)
        
        # 3. 簇头间距离评估（避免过于集中）
        ch_positions = nodes_data[cluster_heads, :2]
        inter_distances = []
        for i in range(len(cluster_heads)):
            for j in range(i + 1, len(cluster_heads)):
                dist = np.sqrt(np.sum((ch_positions[i] - ch_positions[j])**2))
                inter_distances.append(dist)
        
        if inter_distances:
            avg_inter_distance = np.mean(inter_distances)
            separation_score = min(1.0, avg_inter_distance / 50.0)  # 归一化
        else:
            separation_score = 0.0
        
        # 4. 能量效率评估
        total_energy = np.sum(ch_energies)
        energy_efficiency = total_energy / len(cluster_heads)
        
        # 综合适应度计算
        fitness = (0.3 * energy_balance + 
                  0.3 * distance_score + 
                  0.2 * separation_score + 
                  0.2 * energy_efficiency)
        
        return fitness
    
    def _update_particle(self, particle: Particle, global_best_position: np.ndarray):
        """更新粒子位置和速度"""
        r1, r2 = np.random.rand(2)
        
        # 更新速度
        particle.velocity = (self.w * particle.velocity + 
                           self.c1 * r1 * (particle.best_position - particle.position) + 
                           self.c2 * r2 * (global_best_position - particle.position))
        
        # 限制速度
        max_velocity = 5.0
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
        
        # 更新位置
        particle.position = particle.position + particle.velocity
        
        # 确保位置在有效范围内
        particle.position = np.clip(particle.position, 0, len(particle.position) - 1)
    
    def optimize_cluster_heads(self, nodes_data: np.ndarray, 
                             n_clusters: int = 6) -> Tuple[np.ndarray, float]:
        """
        优化簇头选择
        
        Args:
            nodes_data: 节点数据数组 [x, y, energy, ...]
            n_clusters: 簇头数量
            
        Returns:
            (最优簇头索引, 最优适应度)
        """
        
        print(f"🔄 启动PSO优化 - 粒子数: {self.n_particles}, 迭代数: {self.n_iterations}")
        
        # 初始化粒子群
        particles = self._initialize_particles(len(nodes_data), n_clusters)
        
        # 全局最优
        global_best_position = None
        global_best_fitness = -np.inf
        
        # 收敛检测
        stagnation_count = 0
        convergence_threshold = 1e-6
        
        for iteration in range(self.n_iterations):
            iteration_best_fitness = -np.inf
            
            # 评估所有粒子
            for particle in particles:
                fitness = self._evaluate_fitness(particle.position, nodes_data)
                particle.fitness = fitness
                
                # 更新个体最优
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                # 更新全局最优
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
                
                iteration_best_fitness = max(iteration_best_fitness, fitness)
            
            # 更新所有粒子
            for particle in particles:
                self._update_particle(particle, global_best_position)
            
            # 更新惯性权重
            self.w = max(0.1, self.w * self.w_decay)
            
            # 记录收敛数据
            self.best_fitness_history.append(global_best_fitness)
            self.convergence_data.append({
                'iteration': iteration,
                'best_fitness': global_best_fitness,
                'avg_fitness': np.mean([p.fitness for p in particles]),
                'diversity': self._calculate_diversity(particles)
            })
            
            # 收敛检测
            if iteration > 10:
                improvement = (self.best_fitness_history[-1] - 
                             self.best_fitness_history[-10])
                if improvement < convergence_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                if stagnation_count >= 20:
                    self.convergence_iteration = iteration
                    print(f"✅ PSO在第{iteration}轮收敛")
                    break
            
            if iteration % 20 == 0:
                print(f"   迭代 {iteration}: 最优适应度 = {global_best_fitness:.6f}")
        
        # 转换为整数索引
        optimal_cluster_heads = np.clip(global_best_position, 0, 
                                      len(nodes_data) - 1).astype(int)
        optimal_cluster_heads = np.unique(optimal_cluster_heads)
        
        print(f"✅ PSO优化完成 - 最优适应度: {global_best_fitness:.6f}")
        print(f"   选择的簇头: {optimal_cluster_heads}")
        
        return optimal_cluster_heads, global_best_fitness
    
    def _calculate_diversity(self, particles: List[Particle]) -> float:
        """计算粒子群多样性"""
        positions = np.array([p.position for p in particles])
        center = np.mean(positions, axis=0)
        diversity = np.mean([np.linalg.norm(pos - center) for pos in positions])
        return diversity
    
    def visualize_convergence(self, save_path: str = None) -> plt.Figure:
        """可视化收敛过程"""
        if not self.convergence_data:
            print("❌ 没有收敛数据可视化")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iterations = [d['iteration'] for d in self.convergence_data]
        best_fitness = [d['best_fitness'] for d in self.convergence_data]
        avg_fitness = [d['avg_fitness'] for d in self.convergence_data]
        diversity = [d['diversity'] for d in self.convergence_data]
        
        # 适应度收敛曲线
        axes[0, 0].plot(iterations, best_fitness, 'b-', label='最优适应度', linewidth=2)
        axes[0, 0].plot(iterations, avg_fitness, 'r--', label='平均适应度', linewidth=2)
        axes[0, 0].set_title('PSO适应度收敛曲线')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('适应度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 多样性变化
        axes[0, 1].plot(iterations, diversity, 'g-', linewidth=2)
        axes[0, 1].set_title('粒子群多样性变化')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('多样性')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 适应度分布直方图
        final_fitness = [d['best_fitness'] for d in self.convergence_data[-10:]]
        axes[1, 0].hist(final_fitness, bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_title('最终适应度分布')
        axes[1, 0].set_xlabel('适应度')
        axes[1, 0].set_ylabel('频次')
        
        # 收敛统计
        convergence_info = f"""
        总迭代次数: {len(iterations)}
        收敛轮次: {self.convergence_iteration if self.convergence_iteration > 0 else '未收敛'}
        最优适应度: {max(best_fitness):.6f}
        总评估次数: {self.total_evaluations}
        """
        
        axes[1, 1].text(0.1, 0.5, convergence_info, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('优化统计信息')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ PSO收敛图表已保存到: {save_path}")
        
        return fig
    
    def get_optimization_summary(self) -> Dict:
        """获取优化总结"""
        if not self.convergence_data:
            return {"message": "暂无优化数据"}
        
        summary = {
            "algorithm": "Particle Swarm Optimization",
            "parameters": {
                "n_particles": self.n_particles,
                "n_iterations": self.n_iterations,
                "w_initial": self.w_initial,
                "c1": self.c1,
                "c2": self.c2,
                "w_decay": self.w_decay
            },
            "performance": {
                "total_evaluations": self.total_evaluations,
                "convergence_iteration": self.convergence_iteration,
                "best_fitness": max(self.best_fitness_history) if self.best_fitness_history else 0,
                "final_diversity": self.convergence_data[-1]['diversity'] if self.convergence_data else 0
            },
            "convergence_data": self.convergence_data[-10:]  # 最后10轮数据
        }
        
        return summary
    
    def save_results(self, save_path: str):
        """保存优化结果"""
        summary = self.get_optimization_summary()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ PSO优化结果已保存到: {save_path}")

if __name__ == "__main__":
    # 测试PSO优化器
    pso = PSOOptimizer(n_particles=20, n_iterations=50)
    
    # 创建测试节点数据
    np.random.seed(42)
    n_nodes = 30
    nodes_data = np.column_stack([
        np.random.uniform(0, 100, n_nodes),  # x坐标
        np.random.uniform(0, 100, n_nodes),  # y坐标
        np.random.uniform(0.3, 1.0, n_nodes)  # 能量
    ])
    
    # 运行优化
    optimal_heads, best_fitness = pso.optimize_cluster_heads(nodes_data, n_clusters=6)
    
    print(f"最优簇头: {optimal_heads}")
    print(f"最优适应度: {best_fitness}")
    
    # 可视化结果
    pso.visualize_convergence("pso_convergence.png")
    
    # 保存结果
    pso.save_results("pso_results.json")