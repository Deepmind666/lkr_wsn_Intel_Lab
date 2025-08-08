"""
内存优化的真实数据WSN路由系统
解决大数据集内存不足问题
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEfficientConfig:
    """内存优化配置"""
    data_dir: str
    sequence_length: int = 10
    test_split: float = 0.2
    min_samples_per_node: int = 100
    max_samples_per_node: int = 1000  # 限制每个节点的最大样本数
    batch_size: int = 512  # 批处理大小
    sample_ratio: float = 0.1  # 数据采样比例
    max_nodes: int = 20  # 最大使用节点数

class WSNDataset(Dataset):
    """WSN数据集类，支持批处理"""
    
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

class MemoryEfficientLSTM(nn.Module):
    """内存优化的LSTM模型"""
    
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output

class RealDataLoader:
    """真实数据加载器"""
    
    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        self.sensor_data = None
        self.topology_data = None
    
    def load_intel_data(self) -> bool:
        """加载Intel Berkeley数据"""
        try:
            data_file = Path(self.config.data_dir) / "data.txt"
            
            if not data_file.exists():
                logger.error(f"数据文件不存在: {data_file}")
                return False
            
            logger.info(f"找到数据文件: {data_file}")
            
            # 读取数据，使用chunksize避免内存问题
            chunks = []
            chunk_size = 100000
            
            for chunk in pd.read_csv(data_file, sep=' ', header=None, chunksize=chunk_size):
                chunk.columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
                chunks.append(chunk)
                
                # 限制读取的数据量
                if len(chunks) * chunk_size > 500000:  # 最多50万条记录
                    break
            
            self.sensor_data = pd.concat(chunks, ignore_index=True)
            logger.info(f"✅ 成功加载数据: {len(self.sensor_data)} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载数据失败: {e}")
            return False
    
    def clean_data(self):
        """清洗数据"""
        if self.sensor_data is None:
            return
        
        logger.info("清洗数据...")
        original_count = len(self.sensor_data)
        
        # 移除异常值
        self.sensor_data = self.sensor_data.dropna()
        
        # 温度范围过滤 (合理范围: -10 to 50°C)
        self.sensor_data = self.sensor_data[
            (self.sensor_data['temperature'] >= -10) & 
            (self.sensor_data['temperature'] <= 50)
        ]
        
        # 湿度范围过滤 (0-100%)
        if 'humidity' in self.sensor_data.columns:
            self.sensor_data = self.sensor_data[
                (self.sensor_data['humidity'] >= 0) & 
                (self.sensor_data['humidity'] <= 100)
            ]
        
        # 电压范围过滤 (1.8-3.3V)
        if 'voltage' in self.sensor_data.columns:
            self.sensor_data = self.sensor_data[
                (self.sensor_data['voltage'] >= 1.8) & 
                (self.sensor_data['voltage'] <= 3.3)
            ]
        
        logger.info(f"数据清洗完成: {original_count} -> {len(self.sensor_data)} 条记录")

class MemoryEfficientWSNSystem:
    """内存优化的WSN系统"""
    
    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        self.data_loader = RealDataLoader(config)
        self.model = None
        self.metrics = {
            'data_coverage': {},
            'prediction_mae': [],
            'prediction_rmse': [],
            'memory_usage': []
        }
        self.training_history = {
            'lstm_losses': [],
            'data_source': 'Intel Berkeley Lab (Memory Optimized)',
            'training_samples': 0,
            'nodes_used': 0
        }
    
    def load_and_prepare_data(self) -> bool:
        """加载并准备数据"""
        logger.info("🔍 开始加载真实数据...")
        
        if not self.data_loader.load_intel_data():
            return False
        
        self.data_loader.clean_data()
        self.analyze_data_quality()
        
        return True
    
    def analyze_data_quality(self):
        """分析数据质量"""
        data = self.data_loader.sensor_data
        if data is None:
            return
        
        logger.info("📊 分析数据质量...")
        
        total_records = len(data)
        unique_nodes = data['moteid'].nunique()
        node_counts = data['moteid'].value_counts()
        
        logger.info(f"总记录数: {total_records}")
        logger.info(f"节点数: {unique_nodes}")
        logger.info(f"平均每节点记录数: {total_records / unique_nodes:.1f}")
        logger.info(f"数据量最少的节点: {node_counts.min()} 条记录")
        logger.info(f"数据量最多的节点: {node_counts.max()} 条记录")
        
        sufficient_nodes = (node_counts >= self.config.min_samples_per_node).sum()
        logger.info(f"数据充足的节点数 (>={self.config.min_samples_per_node}条): {sufficient_nodes}")
        
        self.metrics['data_coverage'] = {
            'total_records': int(total_records),
            'unique_nodes': int(unique_nodes),
            'sufficient_nodes': int(sufficient_nodes),
            'memory_optimized': True
        }
    
    def prepare_training_data(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """准备训练数据（内存优化版本）"""
        data = self.data_loader.sensor_data
        if data is None:
            logger.error("❌ 没有可用的传感器数据")
            return None, None
        
        logger.info("准备LSTM训练数据（内存优化）...")
        
        # 选择数据充足的节点
        node_counts = data['moteid'].value_counts()
        valid_nodes = node_counts[node_counts >= self.config.min_samples_per_node].index
        
        # 限制节点数量
        if len(valid_nodes) > self.config.max_nodes:
            valid_nodes = valid_nodes[:self.config.max_nodes]
        
        logger.info(f"使用 {len(valid_nodes)} 个节点的数据")
        
        sequences = []
        targets = []
        
        for node_id in valid_nodes:
            node_data = data[data['moteid'] == node_id].copy()
            node_data = node_data.sort_values('epoch')
            
            # 限制每个节点的样本数
            if len(node_data) > self.config.max_samples_per_node:
                # 均匀采样
                indices = np.linspace(0, len(node_data)-1, self.config.max_samples_per_node, dtype=int)
                node_data = node_data.iloc[indices]
            
            # 提取特征
            features = ['temperature', 'humidity', 'light', 'voltage']
            available_features = [f for f in features if f in node_data.columns]
            
            if len(available_features) < 2:
                continue
            
            feature_data = node_data[available_features].values
            
            # 数据标准化
            feature_data = (feature_data - np.mean(feature_data, axis=0)) / (np.std(feature_data, axis=0) + 1e-8)
            
            # 创建序列（采样）
            max_sequences = int(len(feature_data) * self.config.sample_ratio)
            step_size = max(1, (len(feature_data) - self.config.sequence_length) // max_sequences)
            
            for i in range(0, len(feature_data) - self.config.sequence_length, step_size):
                seq = feature_data[i:i + self.config.sequence_length]
                target = feature_data[i + self.config.sequence_length, 0]  # 预测温度
                
                sequences.append(seq)
                targets.append(target)
        
        if len(sequences) == 0:
            logger.error("❌ 无法创建训练序列")
            return None, None
        
        logger.info(f"✅ 创建了 {len(sequences)} 个训练序列")
        logger.info(f"   序列长度: {self.config.sequence_length}")
        logger.info(f"   特征数: {len(available_features)}")
        
        # 数据分割
        split_idx = int(len(sequences) * (1 - self.config.test_split))
        
        train_sequences = sequences[:split_idx]
        train_targets = targets[:split_idx]
        test_sequences = sequences[split_idx:]
        test_targets = targets[split_idx:]
        
        # 创建数据集和数据加载器
        train_dataset = WSNDataset(train_sequences, train_targets)
        test_dataset = WSNDataset(test_sequences, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        self.training_history['training_samples'] = len(sequences)
        self.training_history['nodes_used'] = len(valid_nodes)
        
        return train_loader, test_loader
    
    def train_model(self, epochs=50) -> bool:
        """训练模型"""
        logger.info("🚀 开始训练LSTM模型...")
        
        train_loader, test_loader = self.prepare_training_data()
        if train_loader is None or test_loader is None:
            logger.error("❌ 无法准备训练数据")
            return False
        
        # 获取输入维度
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        
        # 初始化模型
        self.model = MemoryEfficientLSTM(
            input_size=input_size,
            hidden_size=32,
            output_size=1
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"测试批次数: {len(test_loader)}")
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_x).squeeze()
                loss = criterion(predictions, batch_y.squeeze())
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            self.training_history['lstm_losses'].append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 测试评估
        self.model.eval()
        test_mae = 0.0
        test_rmse = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predictions = self.model(batch_x).squeeze()
                
                mae = torch.mean(torch.abs(predictions - batch_y.squeeze())).item()
                rmse = torch.sqrt(torch.mean((predictions - batch_y.squeeze()) ** 2)).item()
                
                test_mae += mae
                test_rmse += rmse
                test_batches += 1
        
        test_mae /= test_batches
        test_rmse /= test_batches
        
        self.metrics['prediction_mae'].append(test_mae)
        self.metrics['prediction_rmse'].append(test_rmse)
        
        logger.info(f"✅ 测试结果 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        return True
    
    def save_results(self):
        """保存结果"""
        results_dir = Path(__file__).parent.parent.parent / "results" / "memory_efficient_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        with open(results_dir / "memory_efficient_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # 保存训练历史
        with open(results_dir / "memory_efficient_training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存模型
        if self.model:
            torch.save(self.model.state_dict(), results_dir / "memory_efficient_lstm_model.pth")
        
        logger.info(f"✅ 结果保存到: {results_dir}")
    
    def visualize_results(self):
        """可视化结果"""
        results_dir = Path(__file__).parent.parent.parent / "results" / "memory_efficient_wsn"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # LSTM训练损失
        if self.training_history['lstm_losses']:
            axes[0, 0].plot(self.training_history['lstm_losses'])
            axes[0, 0].set_title('LSTM训练损失 (内存优化)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MSE Loss')
        
        # 数据使用情况
        if self.metrics['data_coverage']:
            coverage = self.metrics['data_coverage']
            labels = ['总节点', '使用节点']
            values = [coverage['unique_nodes'], self.training_history['nodes_used']]
            axes[0, 1].bar(labels, values)
            axes[0, 1].set_title('节点使用情况')
            axes[0, 1].set_ylabel('节点数')
        
        # 预测性能
        if self.metrics['prediction_mae']:
            axes[1, 0].bar(['MAE', 'RMSE'], 
                          [self.metrics['prediction_mae'][0], self.metrics['prediction_rmse'][0]])
            axes[1, 0].set_title('预测性能 (内存优化)')
            axes[1, 0].set_ylabel('误差')
        
        # 训练样本统计
        axes[1, 1].bar(['训练样本'], [self.training_history['training_samples']])
        axes[1, 1].set_title('训练样本数量')
        axes[1, 1].set_ylabel('样本数')
        
        plt.suptitle('内存优化的真实数据WSN分析', fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / 'memory_efficient_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ 可视化完成")

def main():
    """主函数"""
    logger.info("🚀 启动内存优化的真实数据WSN系统")
    
    # 配置
    config = MemoryEfficientConfig(
        data_dir=str(Path(__file__).parent.parent.parent / "data")
    )
    
    # 创建系统
    wsn_system = MemoryEfficientWSNSystem(config)
    
    # 加载数据
    if not wsn_system.load_and_prepare_data():
        logger.error("❌ 无法加载数据，程序退出")
        return
    
    # 训练模型
    if wsn_system.train_model(epochs=30):
        logger.info("✅ 模型训练成功")
    else:
        logger.error("❌ 模型训练失败")
        return
    
    # 保存和可视化结果
    wsn_system.save_results()
    wsn_system.visualize_results()
    
    logger.info("✅ 内存优化WSN系统运行完成")
    logger.info("📊 使用真实Intel Berkeley数据集（内存优化版本）")
    logger.info("🔧 优化措施：数据采样、批处理、节点限制")

if __name__ == "__main__":
    main()