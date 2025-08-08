"""
Enhanced EEHFR WSN系统 - 终极修复版模块化系统测试
解决所有接口不匹配和数据类型问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入所有核心模块
try:
    from fuzzy_logic_cluster import FuzzyLogicClusterHead
    from pso_optimizer import PSOOptimizer
    from aco_router import ACORouter
    from lstm_predictor import WSNLSTMSystem
    from trust_evaluator import TrustEvaluator, DataReliabilityAnalyzer, TrustMetrics
    print("✓ 所有核心模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    exit(1)

def create_test_nodes_data(n_nodes: int = 25, network_size: tuple = (80, 80)) -> pd.DataFrame:
    """创建测试节点数据，包含所有必需字段"""
    np.random.seed(42)
    
    # 生成节点位置
    positions = np.random.uniform(0, network_size[0], (n_nodes, 2))
    
    # 计算到基站的距离（基站在中心）
    base_station = np.array([network_size[0]/2, network_size[1]/2])
    distances_to_bs = np.sqrt(np.sum((positions - base_station)**2, axis=1))
    max_distance = np.max(distances_to_bs)
    
    # 生成节点数据
    nodes_data = pd.DataFrame({
        'node_id': range(n_nodes),
        'x': positions[:, 0],
        'y': positions[:, 1],
        'energy': np.random.uniform(0.3, 1.0, n_nodes),  # 剩余能量
        'initial_energy': np.ones(n_nodes),  # 初始能量
        'distance_to_bs': distances_to_bs,
        'neighbor_count': np.random.randint(3, 8, n_nodes),
        'trust_value': np.random.uniform(0.6, 1.0, n_nodes),
        'is_alive': np.ones(n_nodes, dtype=bool),
        'cluster_head': np.zeros(n_nodes, dtype=bool)
    })
    
    # 计算比例字段
    nodes_data['energy_ratio'] = nodes_data['energy'] / nodes_data['initial_energy']
    nodes_data['distance_ratio'] = nodes_data['distance_to_bs'] / max_distance
    
    return nodes_data

def test_fuzzy_logic_module():
    """测试模糊逻辑簇头选择模块"""
    print("1. 测试模糊逻辑簇头选择模块...")
    
    try:
        # 创建模糊逻辑选择器
        fuzzy_selector = FuzzyLogicClusterHead()
        
        # 创建测试数据
        nodes_data = create_test_nodes_data(20)
        
        # 选择簇头
        cluster_heads = fuzzy_selector.select_cluster_heads(nodes_data, n_clusters=5)
        
        # 获取性能总结
        summary = fuzzy_selector.get_performance_summary()
        
        print(f"   ✓ 模糊逻辑模块测试成功，选择簇头: {cluster_heads}")
        print(f"   平均模糊评分: {summary['average_score']:.4f}")
        
        return True, cluster_heads
        
    except Exception as e:
        print(f"   ✗ 模糊逻辑模块测试失败: {e}")
        return False, []

def test_pso_module():
    """测试PSO粒子群优化模块"""
    print("2. 测试PSO粒子群优化模块...")
    
    try:
        # 创建PSO优化器
        pso_optimizer = PSOOptimizer(
            n_particles=15,
            n_iterations=20,
            w=0.9,
            c1=2.0,
            c2=2.0
        )
        
        # 创建测试数据
        nodes_data = create_test_nodes_data(20)
        
        # 转换为numpy数组格式 [x, y, energy, ...]
        nodes_array = nodes_data[['x', 'y', 'energy', 'trust_value']].values
        
        # 执行PSO优化
        best_solution, best_fitness = pso_optimizer.optimize_cluster_heads(
            nodes_array, n_clusters=4)
        
        # 获取优化总结
        summary = pso_optimizer.get_optimization_summary()
        
        print(f"   ✓ PSO优化模块测试成功，最佳适应度: {best_fitness:.4f}")
        print(f"   选择的簇头: {best_solution}")
        
        return True, best_solution
        
    except Exception as e:
        print(f"   ✗ PSO优化模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_aco_module():
    """测试ACO蚁群路由优化模块"""
    print("3. 测试ACO蚁群路由优化模块...")
    
    try:
        # 创建ACO路由器
        aco_router = ACORouter(n_ants=10, n_iterations=20)
        
        # 创建测试数据
        nodes_data = create_test_nodes_data(15)
        cluster_heads = [2, 5, 8, 12]
        base_station = 0
        
        # 准备ACO所需的数据
        positions = nodes_data[['x', 'y']].values
        energies = nodes_data['energy'].values
        trusts = nodes_data['trust_value'].values
        
        # 执行路由优化
        routes, stats = aco_router.find_optimal_routes(
            cluster_heads, base_station, positions, energies, trusts)
        
        # 获取路由总结
        summary = aco_router.get_routing_summary()
        
        print(f"   ✓ ACO路由模块测试成功，找到 {len(routes)} 条路由")
        print(f"   最优成本: {stats.get('best_cost', 0):.4f}")
        
        return True, routes
        
    except Exception as e:
        print(f"   ✗ ACO路由模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_lstm_module():
    """测试LSTM时序预测模块"""
    print("4. 测试LSTM时序预测模块...")
    
    try:
        # 创建LSTM系统
        lstm_system = WSNLSTMSystem(sequence_length=10, prediction_horizon=1)
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 200
        test_data = pd.DataFrame({
            'moteid': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'temperature': 20 + 5 * np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 1, n_samples),
            'humidity': 50 + 10 * np.cos(np.arange(n_samples) * 0.08) + np.random.normal(0, 2, n_samples),
            'light': 300 + 100 * np.sin(np.arange(n_samples) * 0.05) + np.random.normal(0, 20, n_samples)
        })
        
        feature_columns = ['temperature', 'humidity', 'light']
        
        # 准备数据
        train_loader, val_loader, test_loader = lstm_system.prepare_data(
            test_data, feature_columns, batch_size=16)
        
        # 构建模型
        lstm_system.build_model(input_size=len(feature_columns), hidden_size=32, num_layers=2)
        
        # 训练模型
        training_stats = lstm_system.train_model(
            train_loader, val_loader, epochs=10, learning_rate=0.001)
        
        # 评估模型
        evaluation_metrics = lstm_system.evaluate_model(test_loader)
        
        print(f"   ✓ LSTM预测模块测试成功，RMSE: {evaluation_metrics['rmse']:.4f}")
        print(f"   R²评分: {evaluation_metrics['r2']:.4f}")
        
        return True, evaluation_metrics
        
    except Exception as e:
        print(f"   ✗ LSTM预测模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_trust_module():
    """测试信任评估模块"""
    print("5. 测试信任评估模块...")
    
    try:
        # 创建信任评估器
        trust_evaluator = TrustEvaluator()
        
        # 创建测试数据
        nodes_data = create_test_nodes_data(10)
        
        # 初始化信任值 - 使用正确的方法名
        node_ids = list(range(10))
        trust_evaluator.initialize_trust(node_ids, initial_trust=0.5)
        
        # 模拟一些交互和信任更新
        for i in range(5):
            # 创建信任度量指标
            metrics = TrustMetrics(
                data_consistency=np.random.uniform(0.7, 0.95),
                packet_delivery_ratio=np.random.uniform(0.85, 0.98),
                energy_efficiency=np.random.uniform(0.6, 0.9),
                communication_reliability=np.random.uniform(0.8, 0.95),
                neighbor_recommendations=np.random.uniform(0.7, 0.9),
                response_time=np.random.uniform(10, 100)
            )
            
            # 模拟邻居数据
            neighbor_data = {
                j: [np.random.normal(25, 2) for _ in range(10)]
                for j in range(3)
            }
            
            # 更新信任值
            trust_evaluator.update_trust(i, metrics, neighbor_data, timestamp=float(i))
        
        # 计算综合信任值
        trust_values = []
        for node_id in range(5):
            trust = trust_evaluator.calculate_composite_trust(node_id)
            trust_values.append(trust)
        
        avg_trust = np.mean(trust_values)
        
        print(f"   ✓ 信任评估模块测试成功，平均信任值: {avg_trust:.4f}")
        
        return True, trust_values
        
    except Exception as e:
        print(f"   ✗ 信任评估模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_data_reliability():
    """测试数据可靠性分析"""
    print("\n=== 数据可靠性分析 ===")
    
    try:
        # 创建数据可靠性分析器
        reliability_analyzer = DataReliabilityAnalyzer()
        
        # 创建测试数据 - 修复数据格式
        sensor_data = {}
        for node_id in range(10):
            sensor_data[node_id] = np.random.normal(25, 2, 50).tolist()
        
        # 在节点9中注入异常数据
        sensor_data[9][-10:] = np.random.normal(50, 1, 10).tolist()  # 异常高温
        
        print("数据可靠性分析结果:")
        
        # 分析每个节点
        anomaly_results = reliability_analyzer.detect_anomalies(sensor_data)
        
        for node_id in range(10):
            status = "异常" if anomaly_results.get(node_id, False) else "正常"
            print(f"  节点 {node_id}: {status}")
            
            # 数据一致性检查（前3个节点）
            if node_id < 3:
                consistency = reliability_analyzer.check_data_consistency(
                    sensor_data[node_id],
                    sensor_data[(node_id + 1) % 3]
                )
                print(f"  节点 {node_id} 数据一致性: {consistency:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据可靠性分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simplified_system_test():
    """运行简化的系统集成测试"""
    print("\n=== 运行简化Enhanced EEHFR系统测试 ===")
    
    try:
        # 系统配置
        n_nodes = 25
        network_size = (80, 80)
        n_clusters = 5
        
        print(f"系统配置: {n_nodes}节点, 网络大小: {network_size}")
        
        # 创建节点数据
        nodes_data = create_test_nodes_data(n_nodes, network_size)
        
        # 1. 模糊逻辑簇头选择
        print("执行模糊逻辑簇头选择...")
        fuzzy_selector = FuzzyLogicClusterHead()
        cluster_heads = fuzzy_selector.select_cluster_heads(nodes_data, n_clusters=n_clusters)
        
        # 2. PSO优化簇头选择
        print("执行PSO优化...")
        pso_optimizer = PSOOptimizer(n_particles=20, n_iterations=30)
        nodes_array = nodes_data[['x', 'y', 'energy', 'trust_value']].values
        pso_solution, pso_fitness = pso_optimizer.optimize_cluster_heads(
            nodes_array, n_clusters=n_clusters)
        
        # 3. ACO路由优化
        print("执行ACO路由优化...")
        aco_router = ACORouter(n_ants=15, n_iterations=25)
        positions = nodes_data[['x', 'y']].values
        energies = nodes_data['energy'].values
        trusts = nodes_data['trust_value'].values
        routes, stats = aco_router.find_optimal_routes(
            cluster_heads, 0, positions, energies, trusts)
        
        # 4. 信任评估
        print("执行信任评估...")
        trust_evaluator = TrustEvaluator()
        
        # 初始化信任值
        node_ids = list(range(min(10, n_nodes)))
        trust_evaluator.initialize_trust(node_ids, initial_trust=0.5)
        
        # 计算综合信任值
        trust_values = []
        for node_id in node_ids:
            trust = trust_evaluator.calculate_composite_trust(node_id)
            trust_values.append(trust)
        
        # 输出结果
        print(f"\n✅ 简化系统测试成功!")
        print(f"模糊逻辑选择的簇头: {cluster_heads}")
        print(f"PSO优化的簇头: {pso_solution} (适应度: {pso_fitness:.4f})")
        print(f"ACO找到路由数量: {len(routes)}")
        print(f"ACO最优成本: {stats.get('best_cost', 0):.4f}")
        print(f"平均信任值: {np.mean(trust_values):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 简化系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("Enhanced EEHFR WSN System - 终极修复版模块化系统测试")
    print("=" * 80)
    
    # 测试各个核心模块
    print("\n=== 测试各个核心模块 ===")
    
    test_results = {}
    
    # 1. 测试模糊逻辑模块
    success, result = test_fuzzy_logic_module()
    test_results['fuzzy_logic'] = success
    
    # 2. 测试PSO模块
    success, result = test_pso_module()
    test_results['pso'] = success
    
    # 3. 测试ACO模块
    success, result = test_aco_module()
    test_results['aco'] = success
    
    # 4. 测试LSTM模块
    success, result = test_lstm_module()
    test_results['lstm'] = success
    
    # 5. 测试信任评估模块
    success, result = test_trust_module()
    test_results['trust'] = success
    
    # 数据可靠性分析
    reliability_success = test_data_reliability()
    test_results['reliability'] = reliability_success
    
    # 运行简化系统测试
    system_success = run_simplified_system_test()
    test_results['system'] = system_success
    
    # 总结测试结果
    print("\n" + "=" * 80)
    successful_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    if successful_tests == total_tests:
        print("🎉 所有测试均成功通过!")
        print("Enhanced EEHFR系统模块化架构运行完美")
        print("\n📊 系统功能验证:")
        print("✅ 模糊逻辑簇头选择 - 智能决策")
        print("✅ PSO粒子群优化 - 全局优化")
        print("✅ ACO蚁群路由优化 - 自适应路由")
        print("✅ LSTM时序预测 - 智能预测")
        print("✅ 信任评估系统 - 安全保障")
        print("✅ 数据可靠性分析 - 质量控制")
        print("✅ 系统集成测试 - 协同工作")
    else:
        print(f"⚠️  测试结果: {successful_tests}/{total_tests} 个模块测试成功")
        failed_modules = [name for name, success in test_results.items() if not success]
        print(f"需要调试的模块: {', '.join(failed_modules)}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()