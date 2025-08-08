"""
真正使用Intel Berkeley真实数据集的WSN路由系统
不再有任何虚假宣传
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
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealDataConfig:
    """真实数据配置"""
    data_dir: str = "data"
    min_samples_per_node: int = 100  # 每个节点最少样本数
    sequence_length: int = 10        # LSTM序列长度
    test_split: float = 0.2          # 测试集比例

class RealIntelDataLoader:
    """真正的Intel Berkeley数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sensor_data = None
        self.topology_data = None
        self.connectivity_data = None
        
    def load_intel_berkeley_data(self) -> Optional[pd.DataFrame]:
        """加载真实的Intel Berkeley数据"""
        logger.info("尝试加载真实Intel Berkeley数据...")
        
        # 尝试多个可能的数据文件位置
        possible_paths = [
            self.data_dir / "data.txt",
            self.data_dir / "intel_berkeley" / "data.txt",
            self.data_dir / "real_datasets" / "intel_berkeley" / "data.txt",
            self.data_dir / "processed" / "cleaned_data.csv"
        ]
        
        for data_path in possible_paths:
            if data_path.exists():
                logger.info(f"找到数据文件: {data_path}")
                try:
                    if data_path.suffix == '.csv':
                        data = pd.read_csv(data_path)
                    else:
                        # Intel Berkeley原始格式
                        # 格式: date time epoch moteid temperature humidity light voltage
                        data = pd.read_csv(data_path, sep=r'\s+', header=None,
                                         names=['date', 'time', 'epoch', 'moteid', 
                                               'temperature', 'humidity', 'light', 'voltage'])
                    
                    logger.info(f"✅ 成功加载数据: {len(data)} 条记录")
                    logger.info(f"   节点数: {data['moteid'].nunique()}")
                    logger.info(f"   时间范围: {data['epoch'].min()} - {data['epoch'].max()}")
                    
                    # 基本数据清洗
                    data = self.clean_data(data)
                    self.sensor_data = data
                    return data
                    
                except Exception as e:
                    logger.warning(f"加载 {data_path} 失败: {e}")
                    continue
        
        logger.error("❌ 未找到任何可用的Intel Berkeley数据文件")
        logger.error("请确保数据文件存在于以下位置之一:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        
        return None
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗真实数据"""
        logger.info("清洗数据...")
        
        original_len = len(data)
        
        # 移除缺失值
        data = data.dropna()
        
        # 移除明显异常值
        for col in ['temperature', 'humidity', 'light', 'voltage']:
            if col in data.columns:
                Q1 = data[col].quantile(0.01)
                Q3 = data[col].quantile(0.99)
                data = data[(data[col] >= Q1) & (data[col] <= Q3)]
        
        # 确保节点ID是整数
        if 'moteid' in data.columns:
            data['moteid'] = data['moteid'].astype(int)
        
        logger.info(f"数据清洗完成: {original_len} -> {len(data)} 条记录")
        return data
    
    def load_topology_data(self) -> Optional[pd.DataFrame]:
        """加载拓扑数据"""
        topology_paths = [
            self.data_dir / "topology.txt",
            self.data_dir / "mote_locs.txt",
            self.data_dir / "processed" / "topology.csv"
        ]
        
        for path in topology_paths:
            if path.exists():
                try:
                    if path.suffix == '.csv':
                        data = pd.read_csv(path)
                    else:
                        data = pd.read_csv(path, sep=r'\s+', header=None,
                                         names=['moteid', 'x', 'y'])
                    
                    logger.info(f"✅ 加载拓扑数据: {len(data)} 个节点")
                    self.topology_data = data
                    return data
                except Exception as e:
                    logger.warning(f"加载拓扑文件 {path} 失败: {e}")
        
        logger.warning("⚠️ 未找到拓扑数据，将使用传感器数据推断")
        return None

class RealLSTMPredictor(nn.Module):
    """基于真实数据的LSTM预测器"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        return output

class RealWSNSystem:
    """基于真实数据的WSN系统"""
    
    def __init__(self, config: RealDataConfig):
        self.config = config
        self.data_loader = RealIntelDataLoader(config.data_dir)
        self.sensor_data = None
        self.topology_data = None
        self.network_graph = None
        
        # 模型组件
        self.lstm_predictor = None
        
        # 训练历史
        self.training_history = {
            'lstm_losses': [],
            'data_source': 'unknown',
            'training_samples': 0,
            'nodes_used': 0
        }
        
        # 性能指标
        self.metrics = {
            'prediction_mae': [],
            'prediction_rmse': [],
            'data_coverage': {},
            'note': '基于真实Intel Berkeley数据集'
        }
    
    def load_real_data(self) -> bool:
        """加载真实数据"""
        logger.info("🔍 开始加载真实数据...")
        
        # 加载传感器数据
        self.sensor_data = self.data_loader.load_intel_berkeley_data()
        if self.sensor_data is None:
            logger.error("❌ 无法加载传感器数据")
            return False
        
        # 加载拓扑数据
        self.topology_data = self.data_loader.load_topology_data()
        
        # 数据统计
        self.analyze_data_quality()
        
        return True
    
    def analyze_data_quality(self):
        """分析数据质量"""
        if self.sensor_data is None:
            return
        
        logger.info("📊 分析数据质量...")
        
        # 基本统计
        total_records = len(self.sensor_data)
        unique_nodes = self.sensor_data['moteid'].nunique()
        
        # 每个节点的数据量
        node_counts = self.sensor_data['moteid'].value_counts()
        
        logger.info(f"总记录数: {total_records}")
        logger.info(f"节点数: {unique_nodes}")
        logger.info(f"平均每节点记录数: {total_records / unique_nodes:.1f}")
        logger.info(f"数据量最少的节点: {node_counts.min()} 条记录")
        logger.info(f"数据量最多的节点: {node_counts.max()} 条记录")
        
        # 检查数据完整性
        sufficient_nodes = (node_counts >= self.config.min_samples_per_node).sum()
        logger.info(f"数据充足的节点数 (>={self.config.min_samples_per_node}条): {sufficient_nodes}")
        
        # 保存统计信息
        self.metrics['data_coverage'] = {
            'total_records': int(total_records),
            'unique_nodes': int(unique_nodes),
            'sufficient_nodes': int(sufficient_nodes),
            'min_records_per_node': int(node_counts.min()),
            'max_records_per_node': int(node_counts.max()),
            'avg_records_per_node': float(total_records / unique_nodes)
        }
        
        self.training_history['data_source'] = 'Intel Berkeley Lab (Real)'
        self.training_history['nodes_used'] = int(sufficient_nodes)
    
    def prepare_lstm_training_data(self):
        """准备LSTM训练数据"""
        if self.sensor_data is None:
            logger.error("❌ 没有可用的传感器数据")
            return None, None
        
        logger.info("准备LSTM训练数据...")
        
        # 选择数据充足的节点
        node_counts = self.sensor_data['moteid'].value_counts()
        valid_nodes = node_counts[node_counts >= self.config.min_samples_per_node].index
        
        if len(valid_nodes) == 0:
            logger.error("❌ 没有足够数据的节点")
            return None, None
        
        logger.info(f"使用 {len(valid_nodes)} 个节点的数据")
        
        sequences = []
        targets = []
        
        for node_id in valid_nodes:
            node_data = self.sensor_data[self.sensor_data['moteid'] == node_id].copy()
            node_data = node_data.sort_values('epoch')
            
            # 提取特征
            features = ['temperature', 'humidity', 'light', 'voltage']
            available_features = [f for f in features if f in node_data.columns]
            
            if len(available_features) < 2:
                continue
            
            feature_data = node_data[available_features].values
            
            # 创建序列
            for i in range(len(feature_data) - self.config.sequence_length):
                seq = feature_data[i:i + self.config.sequence_length]
                target = feature_data[i + self.config.sequence_length, 0]  # 预测温度
                
                sequences.append(seq)
                targets.append(target)
        
        if len(sequences) == 0:
            logger.error("❌ 无法创建训练序列")
            return None, None
        
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(targets)
        
        logger.info(f"✅ 创建了 {len(sequences)} 个训练序列")
        logger.info(f"   特征维度: {X.shape}")
        logger.info(f"   目标维度: {y.shape}")
        
        self.training_history['training_samples'] = len(sequences)
        
        return X, y
    
    def train_lstm_on_real_data(self, epochs=50):
        """使用真实数据训练LSTM"""
        logger.info("🚀 开始使用真实数据训练LSTM...")
        
        X, y = self.prepare_lstm_training_data()
        if X is None or y is None:
            logger.error("❌ 无法准备训练数据")
            return False
        
        # 数据分割
        split_idx = int(len(X) * (1 - self.config.test_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 初始化模型
        input_size = X.shape[2]
        self.lstm_predictor = RealLSTMPredictor(
            input_size=input_size,
            hidden_size=32,
            output_size=1
        )
        
        optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"测试集大小: {len(X_test)}")
        
        # 训练循环
        self.lstm_predictor.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            predictions = self.lstm_predictor(X_train).squeeze()
            loss = criterion(predictions, y_train)
            
            loss.backward()
            optimizer.step()
            
            self.training_history['lstm_losses'].append(loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        # 测试评估
        self.lstm_predictor.eval()
        with torch.no_grad():
            test_predictions = self.lstm_predictor(X_test).squeeze()
            test_mae = torch.mean(torch.abs(test_predictions - y_test)).item()
            test_rmse = torch.sqrt(torch.mean((test_predictions - y_test) ** 2)).item()
            
            self.metrics['prediction_mae'].append(test_mae)
            self.metrics['prediction_rmse'].append(test_rmse)
            
            logger.info(f"✅ 测试结果 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        return True
    
    def save_real_results(self):
        """保存真实结果"""
        results_dir = Path(__file__).parent.parent.parent / "results" / "real_data_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        with open(results_dir / "real_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        with open(results_dir / "real_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存模型
        if self.lstm_predictor:
            torch.save(self.lstm_predictor.state_dict(), results_dir / "real_lstm_model.pth")
        
        logger.info(f"✅ 结果保存到: {results_dir}")
    
    def visualize_real_results(self):
        """可视化真实结果"""
        results_dir = Path(__file__).parent.parent.parent / "results" / "real_data_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # LSTM训练损失
        if self.training_history['lstm_losses']:
            axes[0, 0].plot(self.training_history['lstm_losses'])
            axes[0, 0].set_title('LSTM训练损失 (真实数据)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
        
        # 数据覆盖情况
        if self.metrics['data_coverage']:
            coverage = self.metrics['data_coverage']
            labels = ['总节点', '数据充足节点']
            values = [coverage['unique_nodes'], coverage['sufficient_nodes']]
            axes[0, 1].bar(labels, values)
            axes[0, 1].set_title('数据覆盖情况')
            axes[0, 1].set_ylabel('节点数')
        
        # 预测性能
        if self.metrics['prediction_mae']:
            axes[1, 0].bar(['MAE', 'RMSE'], 
                          [self.metrics['prediction_mae'][0], self.metrics['prediction_rmse'][0]])
            axes[1, 0].set_title('预测性能 (真实数据测试)')
            axes[1, 0].set_ylabel('误差')
        
        # 数据统计
        if self.sensor_data is not None:
            temp_data = self.sensor_data['temperature'].dropna()
            axes[1, 1].hist(temp_data, bins=30, alpha=0.7)
            axes[1, 1].set_title('温度数据分布 (真实数据)')
            axes[1, 1].set_xlabel('温度 (°C)')
            axes[1, 1].set_ylabel('频次')
        
        plt.suptitle('基于真实Intel Berkeley数据的WSN分析', fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / 'real_data_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ 可视化完成")

def main():
    """主函数"""
    logger.info("🚀 启动基于真实数据的WSN系统")
    
    # 配置
    config = RealDataConfig(
        data_dir=str(Path(__file__).parent.parent.parent / "data")
    )
    
    # 创建系统
    wsn_system = RealWSNSystem(config)
    
    # 加载真实数据
    if not wsn_system.load_real_data():
        logger.error("❌ 无法加载真实数据，程序退出")
        return
    
    # 训练模型
    if wsn_system.train_lstm_on_real_data(epochs=50):
        logger.info("✅ 模型训练成功")
    else:
        logger.error("❌ 模型训练失败")
        return
    
    # 保存和可视化结果
    wsn_system.save_real_results()
    wsn_system.visualize_real_results()
    
    logger.info("✅ 基于真实数据的WSN系统运行完成")
    logger.info("📊 这次使用的是真实的Intel Berkeley数据集")

if __name__ == "__main__":
    main()