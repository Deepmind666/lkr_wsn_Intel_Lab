"""
Enhanced EEHFR WSN系统 - LSTM时序预测模块
基于用户调研文件中的轻量级时序预测模型设计
实现传感器数据预测和异常检测
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor(nn.Module):
    """
    轻量级LSTM预测模型
    专门用于WSN传感器数据时序预测
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 4, 
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        初始化LSTM预测器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 全连接层
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        )
        
        # 残差连接（如果输入输出维度相同）
        self.use_residual = (input_size == output_size)
        if self.use_residual:
            self.residual_projection = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        final_output = attn_out[:, -1, :]
        
        # 全连接层
        prediction = self.fc_layers(final_output)
        
        # 残差连接
        if self.use_residual:
            residual = self.residual_projection(x[:, -1, :])
            prediction = prediction + residual
        
        return prediction

class WSNDataPreprocessor:
    """WSN数据预处理器"""
    
    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.feature_names = []
        
    def fit_transform(self, data: pd.DataFrame, 
                     feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并转换数据"""
        self.feature_names = feature_columns
        
        # 数据清洗
        cleaned_data = self._clean_data(data, feature_columns)
        
        # 特征缩放
        scaled_data = self._scale_features(cleaned_data, feature_columns)
        
        # 创建序列
        sequences, targets = self._create_sequences(scaled_data, feature_columns)
        
        return sequences, targets
    
    def transform(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """转换新数据"""
        # 数据清洗
        cleaned_data = self._clean_data(data, self.feature_names)
        
        # 特征缩放
        scaled_data = self._transform_features(cleaned_data, self.feature_names)
        
        # 创建序列
        sequences, targets = self._create_sequences(scaled_data, self.feature_names)
        
        return sequences, targets
    
    def _clean_data(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """数据清洗"""
        cleaned = data.copy()
        
        # 移除异常值
        for col in feature_columns:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            cleaned = cleaned[(cleaned[col] >= lower_bound) & 
                            (cleaned[col] <= upper_bound)]
        
        # 填充缺失值
        cleaned[feature_columns] = cleaned[feature_columns].fillna(
            cleaned[feature_columns].mean())
        
        return cleaned
    
    def _scale_features(self, data: pd.DataFrame, 
                       feature_columns: List[str]) -> pd.DataFrame:
        """特征缩放"""
        scaled_data = data.copy()
        
        for col in feature_columns:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(data[[col]])
            self.scalers[col] = scaler
        
        return scaled_data
    
    def _transform_features(self, data: pd.DataFrame, 
                          feature_columns: List[str]) -> pd.DataFrame:
        """转换特征"""
        scaled_data = data.copy()
        
        for col in feature_columns:
            if col in self.scalers:
                scaled_data[col] = self.scalers[col].transform(data[[col]])
        
        return scaled_data
    
    def _create_sequences(self, data: pd.DataFrame, 
                         feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """创建时序序列"""
        sequences = []
        targets = []
        
        # 按节点分组
        if 'moteid' in data.columns:
            for node_id in data['moteid'].unique():
                node_data = data[data['moteid'] == node_id][feature_columns].values
                
                if len(node_data) >= self.sequence_length + self.prediction_horizon:
                    for i in range(len(node_data) - self.sequence_length - self.prediction_horizon + 1):
                        seq = node_data[i:i + self.sequence_length]
                        target = node_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                        
                        sequences.append(seq)
                        targets.append(target.flatten() if self.prediction_horizon > 1 else target[0])
        else:
            # 单节点数据
            node_data = data[feature_columns].values
            for i in range(len(node_data) - self.sequence_length - self.prediction_horizon + 1):
                seq = node_data[i:i + self.sequence_length]
                target = node_data[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                
                sequences.append(seq)
                targets.append(target.flatten() if self.prediction_horizon > 1 else target[0])
        
        return np.array(sequences), np.array(targets)
    
    def inverse_transform(self, scaled_data: np.ndarray, 
                         feature_name: str) -> np.ndarray:
        """反向转换数据"""
        if feature_name in self.scalers:
            return self.scalers[feature_name].inverse_transform(
                scaled_data.reshape(-1, 1)).flatten()
        return scaled_data

class LSTMTrainer:
    """LSTM训练器"""
    
    def __init__(self, model: LSTMPredictor, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = []
        self.best_model_state = None
        self.best_loss = float('inf')
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              patience: int = 10) -> Dict:
        """训练模型"""
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        train_losses = []
        val_losses = []
        early_stop_counter = 0
        
        print(f"🔄 开始LSTM训练 - 设备: {self.device}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_model_state = self.model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # 记录训练历史
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                      f"Val Loss = {avg_val_loss:.6f}")
            
            # 早停
            if early_stop_counter >= patience:
                print(f"✅ 早停于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        training_stats = {
            'total_epochs': len(train_losses),
            'best_val_loss': self.best_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'training_history': self.training_history
        }
        
        print(f"✅ LSTM训练完成 - 最佳验证损失: {self.best_loss:.6f}")
        
        return training_stats
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # 计算评估指标
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 计算MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-10))) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'targets': targets
        }
        
        print(f"✅ 模型评估完成:")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²: {r2:.6f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return metrics

class WSNLSTMSystem:
    """WSN LSTM预测系统"""
    
    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.preprocessor = WSNDataPreprocessor(sequence_length, prediction_horizon)
        self.model = None
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data: pd.DataFrame, 
                    feature_columns: List[str],
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练数据"""
        
        print("🔄 准备LSTM训练数据...")
        
        # 预处理数据
        sequences, targets = self.preprocessor.fit_transform(data, feature_columns)
        
        print(f"   创建序列: {len(sequences)} 个样本")
        print(f"   序列形状: {sequences.shape}")
        print(f"   目标形状: {targets.shape}")
        
        # 数据分割
        n_samples = len(sequences)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # 转换为PyTorch张量
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(targets)
        
        # 分割数据
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(val_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, input_size: int, hidden_size: int = 64,
                   num_layers: int = 2, dropout: float = 0.2) -> LSTMPredictor:
        """构建LSTM模型"""
        
        output_size = input_size * self.prediction_horizon
        
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=False
        )
        
        self.trainer = LSTMTrainer(self.model, self.device)
        
        print(f"✅ LSTM模型构建完成:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏层大小: {hidden_size}")
        print(f"   层数: {num_layers}")
        print(f"   输出维度: {output_size}")
        print(f"   参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   epochs: int = 100, learning_rate: float = 0.001) -> Dict:
        """训练模型"""
        
        if self.trainer is None:
            raise ValueError("请先构建模型")
        
        return self.trainer.train(train_loader, val_loader, epochs, learning_rate)
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """评估模型"""
        
        if self.trainer is None:
            raise ValueError("请先训练模型")
        
        return self.trainer.evaluate(test_loader)
    
    def visualize_results(self, training_stats: Dict, evaluation_metrics: Dict,
                         save_path: str = None) -> plt.Figure:
        """可视化结果"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 训练损失曲线
        history = training_stats['training_history']
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        
        axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_title('LSTM训练损失曲线')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测vs真实值散点图
        predictions = evaluation_metrics['predictions']
        targets = evaluation_metrics['targets']
        
        # 只显示第一个特征的结果
        axes[0, 1].scatter(targets[:, 0], predictions[:, 0], alpha=0.6)
        axes[0, 1].plot([targets[:, 0].min(), targets[:, 0].max()], 
                       [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
        axes[0, 1].set_title('预测值 vs 真实值')
        axes[0, 1].set_xlabel('真实值')
        axes[0, 1].set_ylabel('预测值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差分布
        residuals = targets[:, 0] - predictions[:, 0]
        axes[0, 2].hist(residuals, bins=50, alpha=0.7, color='green')
        axes[0, 2].set_title('残差分布')
        axes[0, 2].set_xlabel('残差')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 时序预测示例
        sample_size = min(100, len(targets))
        axes[1, 0].plot(targets[:sample_size, 0], 'b-', label='真实值', linewidth=2)
        axes[1, 0].plot(predictions[:sample_size, 0], 'r--', label='预测值', linewidth=2)
        axes[1, 0].set_title('时序预测示例')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('数值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 评估指标
        metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE(%)']
        metrics_values = [
            evaluation_metrics['mae'],
            evaluation_metrics['rmse'],
            evaluation_metrics['r2'],
            evaluation_metrics['mape']
        ]
        
        bars = axes[1, 1].bar(metrics_names, metrics_values, color='purple', alpha=0.7)
        axes[1, 1].set_title('评估指标')
        axes[1, 1].set_ylabel('数值')
        
        for bar, value in zip(bars, metrics_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # 6. 学习率变化
        learning_rates = [h['lr'] for h in history]
        axes[1, 2].plot(epochs, learning_rates, 'g-', linewidth=2)
        axes[1, 2].set_title('学习率变化')
        axes[1, 2].set_xlabel('轮次')
        axes[1, 2].set_ylabel('学习率')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ LSTM结果图表已保存到: {save_path}")
        
        return fig
    
    def save_model(self, save_path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.output_size,
                'bidirectional': self.model.bidirectional
            },
            'preprocessor': self.preprocessor
        }, save_path)
        
        print(f"✅ LSTM模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        config = checkpoint['model_config']
        self.model = LSTMPredictor(**config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.preprocessor = checkpoint['preprocessor']
        self.trainer = LSTMTrainer(self.model, self.device)
        
        print(f"✅ LSTM模型已从 {load_path} 加载")

if __name__ == "__main__":
    # 测试LSTM预测系统
    lstm_system = WSNLSTMSystem(sequence_length=10, prediction_horizon=1)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    test_data = pd.DataFrame({
        'moteid': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'temperature': 20 + 5 * np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 1, n_samples),
        'humidity': 50 + 10 * np.cos(np.arange(n_samples) * 0.08) + np.random.normal(0, 2, n_samples),
        'light': 300 + 100 * np.sin(np.arange(n_samples) * 0.05) + np.random.normal(0, 20, n_samples),
        'voltage': 2.8 + 0.2 * np.random.random(n_samples)
    })
    
    feature_columns = ['temperature', 'humidity', 'light', 'voltage']
    
    # 准备数据
    train_loader, val_loader, test_loader = lstm_system.prepare_data(
        test_data, feature_columns, batch_size=32)
    
    # 构建模型
    lstm_system.build_model(input_size=len(feature_columns), hidden_size=32, num_layers=2)
    
    # 训练模型
    training_stats = lstm_system.train_model(train_loader, val_loader, epochs=50)
    
    # 评估模型
    evaluation_metrics = lstm_system.evaluate_model(test_loader)
    
    # 可视化结果
    lstm_system.visualize_results(training_stats, evaluation_metrics, "lstm_results.png")
    
    # 保存模型
    lstm_system.save_model("lstm_model.pth")