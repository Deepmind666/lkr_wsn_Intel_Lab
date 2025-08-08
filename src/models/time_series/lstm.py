#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM时序预测模型

该模块实现了基于LSTM（长短期记忆网络）的时序预测模型，用于WSN传感器数据的预测。
LSTM是一种特殊的RNN（循环神经网络），能够学习长期依赖关系，适合时序数据预测。
在WSN中，LSTM可用于预测传感器读数，如温度、湿度、光照等，以减少传输频率，从而节省能源。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    LSTM时序预测模型
    
    属性:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        output_size: 输出维度
        dropout: Dropout比例
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比例
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_size)
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out

class LSTMPredictor:
    """
    LSTM预测器类
    
    属性:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        output_size: 输出维度
        seq_length: 序列长度
        dropout: Dropout比例
        learning_rate: 学习率
        device: 训练设备（CPU或GPU）
        model: LSTM模型
        optimizer: 优化器
        criterion: 损失函数
        scaler: 数据归一化器
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, seq_length=24, 
                 dropout=0.2, learning_rate=0.001, device=None):
        """
        初始化LSTM预测器
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度
            seq_length: 序列长度
            dropout: Dropout比例
            learning_rate: 学习率
            device: 训练设备（CPU或GPU）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_length = seq_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 创建损失函数
        self.criterion = nn.MSELoss()
        
        # 创建归一化器
        self.scaler = None
        
        logger.info(f"LSTM预测器初始化完成")
        logger.info(f"模型结构: {self.model}")
    
    def _create_sequences(self, data):
        """
        创建时序序列
        
        Args:
            data: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            tuple: (X, y)，其中X形状为 (n_sequences, seq_length, input_size)，y形状为 (n_sequences, output_size)
        """
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length])
            y.append(data[i+self.seq_length, :self.output_size])
        
        return np.array(X), np.array(y)
    
    def _normalize_data(self, data, fit=True):
        """
        归一化数据
        
        Args:
            data: 输入数据
            fit: 是否拟合归一化器
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        if fit:
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(data)
        else:
            if self.scaler is None:
                raise ValueError("归一化器未初始化，请先使用fit=True进行拟合")
            return self.scaler.transform(data)
    
    def _inverse_normalize(self, data):
        """
        反归一化数据
        
        Args:
            data: 归一化后的数据
            
        Returns:
            np.ndarray: 原始尺度的数据
        """
        if self.scaler is None:
            raise ValueError("归一化器未初始化，无法进行反归一化")
        return self.scaler.inverse_transform(data)
    
    def train(self, train_data, val_data=None, epochs=100, batch_size=32, patience=10, verbose=True):
        """
        训练LSTM模型
        
        Args:
            train_data: 训练数据，形状为 (n_samples, n_features)
            val_data: 验证数据，形状为 (n_samples, n_features)
            epochs: 训练轮数
            batch_size: 批次大小
            patience: 早停耐心值
            verbose: 是否打印训练过程
            
        Returns:
            dict: 训练历史记录
        """
        # 归一化数据
        train_data_norm = self._normalize_data(train_data, fit=True)
        
        # 创建序列
        X_train, y_train = self._create_sequences(train_data_norm)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证数据处理
        if val_data is not None:
            val_data_norm = self._normalize_data(val_data, fit=False)
            X_val, y_val = self._create_sequences(val_data_norm)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 早停设置
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 训练循环
        if verbose:
            logger.info(f"开始训练LSTM模型，共 {epochs} 轮")
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            # 训练批次循环
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
            for X_batch, y_batch in train_iterator:
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # 计算平均训练损失
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_data is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    # 保存最佳模型
                    best_model_state = self.model.state_dict().copy()
                else:
                    early_stop_counter += 1
                
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # 早停
                if early_stop_counter >= patience:
                    if verbose:
                        logger.info(f"早停触发，在 {epoch+1-patience} 轮达到最佳验证损失")
                    # 恢复最佳模型
                    self.model.load_state_dict(best_model_state)
                    break
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        if verbose:
            logger.info("LSTM模型训练完成")
        
        return history
    
    def evaluate(self, data_loader):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            float: 平均损失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(data_loader.dataset)
    
    def predict(self, data):
        """
        预测
        
        Args:
            data: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测结果，形状为 (n_samples - seq_length, output_size)
        """
        # 归一化数据
        data_norm = self._normalize_data(data, fit=False)
        
        # 创建序列
        X, _ = self._create_sequences(data_norm)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测模式
        self.model.eval()
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 准备反归一化
        dummy = np.zeros((predictions.shape[0], data.shape[1]))
        dummy[:, :self.output_size] = predictions
        
        # 反归一化
        dummy_inverse = self._inverse_normalize(dummy)
        
        # 提取预测结果
        predictions_inverse = dummy_inverse[:, :self.output_size]
        
        return predictions_inverse
    
    def predict_next_n_steps(self, initial_sequence, n_steps):
        """
        预测未来n步
        
        Args:
            initial_sequence: 初始序列，形状为 (seq_length, n_features)
            n_steps: 预测步数
            
        Returns:
            np.ndarray: 预测结果，形状为 (n_steps, output_size)
        """
        # 检查初始序列长度
        if len(initial_sequence) < self.seq_length:
            raise ValueError(f"初始序列长度必须至少为 {self.seq_length}")
        
        # 取最后seq_length个样本作为初始序列
        sequence = initial_sequence[-self.seq_length:].copy()
        
        # 归一化初始序列
        sequence_norm = self._normalize_data(sequence, fit=False)
        
        # 预测结果
        predictions = []
        
        # 预测模式
        self.model.eval()
        
        # 逐步预测
        for _ in range(n_steps):
            # 准备输入
            X = sequence_norm.reshape(1, self.seq_length, -1)
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # 预测下一步
            with torch.no_grad():
                next_step = self.model(X_tensor).cpu().numpy()[0]
            
            # 准备反归一化
            dummy = np.zeros((1, sequence.shape[1]))
            dummy[0, :self.output_size] = next_step
            
            # 反归一化
            dummy_inverse = self._inverse_normalize(dummy)
            
            # 提取预测结果
            next_step_inverse = dummy_inverse[0, :self.output_size]
            predictions.append(next_step_inverse)
            
            # 更新序列
            new_row = np.zeros((1, sequence.shape[1]))
            new_row[0, :self.output_size] = next_step
            sequence_norm = np.vstack([sequence_norm[1:], new_row])
        
        return np.array(predictions)
    
    def calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 评估指标
        """
        metrics = {}
        
        # 均方误差
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # 均方根误差
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 平均绝对误差
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 决定系数
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 平均绝对百分比误差
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title='LSTM预测结果', save_path=None):
        """
        绘制预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_path: 保存路径，如果为None则显示图形
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值
        plt.plot(y_true, 'b-', label='真实值')
        
        # 绘制预测值
        plt.plot(y_pred, 'r--', label='预测值')
        
        plt.title(title)
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"预测结果图已保存到: {save_path}")
        else:
            plt.show()
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'seq_length': self.seq_length,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        
        # 保存归一化器
        if self.scaler is not None:
            model_state['scaler'] = self.scaler
        
        # 保存模型
        torch.save(model_state, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        # 加载模型状态
        model_state = torch.load(path, map_location=self.device)
        
        # 更新模型参数
        self.input_size = model_state['input_size']
        self.hidden_size = model_state['hidden_size']
        self.num_layers = model_state['num_layers']
        self.output_size = model_state['output_size']
        self.seq_length = model_state['seq_length']
        self.dropout = model_state['dropout']
        self.learning_rate = model_state['learning_rate']
        
        # 重新创建模型
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(self.device)
        
        # 加载模型状态字典
        self.model.load_state_dict(model_state['model_state_dict'])
        
        # 重新创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        # 加载归一化器
        if 'scaler' in model_state:
            self.scaler = model_state['scaler']
        
        logger.info(f"模型已从 {path} 加载")

# 测试函数
def test_lstm_predictor():
    """
    测试LSTM预测器
    """
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    
    # 创建正弦波 + 噪声
    signal = np.sin(0.1 * time) + 0.1 * np.random.randn(n_samples)
    
    # 添加趋势
    trend = 0.001 * time
    signal = signal + trend
    
    # 创建特征矩阵
    features = np.column_stack([signal, np.cos(0.1 * time), time / n_samples])
    
    # 划分训练集和测试集
    train_size = int(0.8 * n_samples)
    train_data = features[:train_size]
    test_data = features[train_size:]
    
    # 创建LSTM预测器
    predictor = LSTMPredictor(
        input_size=features.shape[1],
        hidden_size=64,
        num_layers=2,
        output_size=1,
        seq_length=24,
        dropout=0.2,
        learning_rate=0.001
    )
    
    # 训练模型
    history = predictor.train(
        train_data=train_data,
        val_data=None,
        epochs=50,
        batch_size=32,
        patience=10,
        verbose=True
    )
    
    # 预测
    predictions = predictor.predict(test_data)
    
    # 计算指标
    y_true = test_data[24:, 0]  # 第一列是目标变量
    y_pred = predictions[:, 0]
    metrics = predictor.calculate_metrics(y_true, y_pred)
    
    # 打印指标
    logger.info("评估指标:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.6f}")
    
    # 绘制预测结果
    predictor.plot_predictions(y_true, y_pred, title='LSTM预测结果')
    
    # 预测未来10步
    future_predictions = predictor.predict_next_n_steps(test_data[:24], 10)
    logger.info(f"未来10步预测: {future_predictions.flatten()}")

if __name__ == '__main__':
    test_lstm_predictor()