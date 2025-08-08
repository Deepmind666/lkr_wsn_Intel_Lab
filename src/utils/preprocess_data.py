#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intel Berkeley Lab数据集预处理脚本

该脚本用于预处理Intel Berkeley Lab数据集，包括数据清洗、归一化、特征工程等。

数据集格式：
- data.txt: 包含日期、时间、纪元、moteid、温度、湿度、光照和电压
- topology.txt: 包含节点ID和位置坐标
- connectivity.txt: 包含节点间的连接关系
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    加载数据文件
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        pd.DataFrame: 数据DataFrame
    """
    try:
        # 定义列名
        columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
        
        # 加载数据
        logger.info(f"加载数据文件: {data_path}")
        df = pd.read_csv(data_path, sep=' ', header=None, names=columns)
        
        # 转换日期和时间
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed')
        
        # 删除原始日期和时间列
        df = df.drop(['date', 'time'], axis=1)
        
        # 将datetime设为索引
        df = df.set_index('datetime')
        
        logger.info(f"数据加载完成，共 {len(df)} 行记录")
        return df
    except Exception as e:
        logger.error(f"加载数据文件时出错: {e}")
        return None

def load_topology(topology_path):
    """
    加载拓扑文件
    
    Args:
        topology_path: 拓扑文件路径
        
    Returns:
        pd.DataFrame: 拓扑DataFrame
    """
    try:
        # 定义列名
        columns = ['moteid', 'x', 'y']
        
        # 加载数据
        logger.info(f"加载拓扑文件: {topology_path}")
        df = pd.read_csv(topology_path, sep=' ', header=None, names=columns)
        
        logger.info(f"拓扑加载完成，共 {len(df)} 个节点")
        return df
    except Exception as e:
        logger.error(f"加载拓扑文件时出错: {e}")
        return None

def load_connectivity(connectivity_path):
    """
    加载连接文件
    
    Args:
        connectivity_path: 连接文件路径
        
    Returns:
        pd.DataFrame: 连接DataFrame
    """
    try:
        # 定义列名
        columns = ['from_id', 'to_id']
        
        # 加载数据
        logger.info(f"加载连接文件: {connectivity_path}")
        df = pd.read_csv(connectivity_path, sep=' ', header=None, names=columns)
        
        logger.info(f"连接加载完成，共 {len(df)} 条连接")
        return df
    except Exception as e:
        logger.error(f"加载连接文件时出错: {e}")
        return None

def clean_data(df):
    """
    清洗数据，处理缺失值和异常值
    
    Args:
        df: 数据DataFrame
        
    Returns:
        pd.DataFrame: 清洗后的DataFrame
    """
    logger.info("开始清洗数据")
    
    # 原始数据统计
    original_rows = len(df)
    logger.info(f"原始数据: {original_rows} 行")
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    logger.info(f"缺失值统计:\n{missing_values}")
    
    # 删除缺失值
    df = df.dropna()
    logger.info(f"删除缺失值后: {len(df)} 行")
    
    # 检查异常值
    # 温度范围：-10°C ~ 50°C
    # 湿度范围：0% ~ 100%
    # 光照范围：0 ~ 1000 lux
    # 电压范围：2.0V ~ 3.0V
    logger.info("检查异常值")
    logger.info(f"温度范围: {df['temperature'].min():.2f} ~ {df['temperature'].max():.2f}°C")
    logger.info(f"湿度范围: {df['humidity'].min():.2f} ~ {df['humidity'].max():.2f}%")
    logger.info(f"光照范围: {df['light'].min():.2f} ~ {df['light'].max():.2f} lux")
    logger.info(f"电压范围: {df['voltage'].min():.2f} ~ {df['voltage'].max():.2f}V")
    
    # 过滤异常值
    df = df[(df['temperature'] >= -10) & (df['temperature'] <= 50)]
    df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
    df = df[(df['light'] >= 0) & (df['light'] <= 1000)]
    df = df[(df['voltage'] >= 2.0) & (df['voltage'] <= 3.0)]
    
    logger.info(f"过滤异常值后: {len(df)} 行")
    
    # 计算清洗比例
    clean_ratio = len(df) / original_rows * 100
    logger.info(f"数据清洗完成，保留了 {clean_ratio:.2f}% 的原始数据")
    
    return df

def normalize_data(df, method='minmax'):
    """
    数据归一化
    
    Args:
        df: 数据DataFrame
        method: 归一化方法，'minmax'或'standard'
        
    Returns:
        pd.DataFrame: 归一化后的DataFrame
        dict: 归一化器字典
    """
    logger.info(f"开始数据归一化，使用 {method} 方法")
    
    # 需要归一化的列
    columns = ['temperature', 'humidity', 'light', 'voltage']
    
    # 创建归一化器字典
    scalers = {}
    
    # 复制DataFrame
    normalized_df = df.copy()
    
    # 对每列进行归一化
    for col in columns:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
            
        # 归一化
        normalized_df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        
        # 保存归一化器
        scalers[col] = scaler
        
        logger.info(f"列 {col} 归一化完成")
        
    logger.info("数据归一化完成")
    
    return normalized_df, scalers

def generate_features(df):
    """
    生成特征
    
    Args:
        df: 数据DataFrame
        
    Returns:
        pd.DataFrame: 添加特征后的DataFrame
    """
    logger.info("开始生成特征")
    
    # 复制DataFrame
    feature_df = df.copy()
    
    # 添加时间特征
    feature_df['hour'] = feature_df.index.hour
    feature_df['day'] = feature_df.index.day
    feature_df['month'] = feature_df.index.month
    feature_df['dayofweek'] = feature_df.index.dayofweek
    
    # 添加滞后特征
    for col in ['temperature', 'humidity', 'light', 'voltage']:
        # 添加1小时、3小时、6小时、12小时、24小时的滞后特征
        for lag in [1, 3, 6, 12, 24]:
            feature_df[f'{col}_lag_{lag}h'] = feature_df[col].shift(lag)
    
    # 添加滚动统计特征
    for col in ['temperature', 'humidity', 'light', 'voltage']:
        # 添加1小时、3小时、6小时、12小时、24小时的滚动平均
        for window in [1, 3, 6, 12, 24]:
            feature_df[f'{col}_rolling_mean_{window}h'] = feature_df[col].rolling(window=window).mean()
            feature_df[f'{col}_rolling_std_{window}h'] = feature_df[col].rolling(window=window).std()
    
    # 删除缺失值
    feature_df = feature_df.dropna()
    
    logger.info(f"特征生成完成，共 {len(feature_df.columns)} 个特征")
    
    return feature_df

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分数据集
    
    Args:
        df: 数据DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger.info("开始划分数据集")
    
    # 检查比例和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例和必须为1"
    
    # 按时间顺序划分
    df = df.sort_index()
    
    # 计算划分点
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # 划分数据集
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"数据集划分完成，训练集: {len(train_df)} 行，验证集: {len(val_df)} 行，测试集: {len(test_df)} 行")
    
    return train_df, val_df, test_df

def save_data(df, save_path):
    """
    保存数据
    
    Args:
        df: 数据DataFrame
        save_path: 保存路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存数据
        df.to_csv(save_path)
        logger.info(f"数据保存成功: {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")
        return False

def visualize_data(df, save_dir):
    """
    可视化数据
    
    Args:
        df: 数据DataFrame
        save_dir: 保存目录
        
    Returns:
        bool: 可视化是否成功
    """
    try:
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 可视化每个节点的温度、湿度、光照和电压
        for moteid in df['moteid'].unique():
            node_df = df[df['moteid'] == moteid]
            
            # 创建图形
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            
            # 绘制温度
            axes[0].plot(node_df.index, node_df['temperature'])
            axes[0].set_title(f'Node {moteid} - Temperature')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].grid(True)
            
            # 绘制湿度
            axes[1].plot(node_df.index, node_df['humidity'])
            axes[1].set_title(f'Node {moteid} - Humidity')
            axes[1].set_ylabel('Humidity (%)')
            axes[1].grid(True)
            
            # 绘制光照
            axes[2].plot(node_df.index, node_df['light'])
            axes[2].set_title(f'Node {moteid} - Light')
            axes[2].set_ylabel('Light (lux)')
            axes[2].grid(True)
            
            # 绘制电压
            axes[3].plot(node_df.index, node_df['voltage'])
            axes[3].set_title(f'Node {moteid} - Voltage')
            axes[3].set_ylabel('Voltage (V)')
            axes[3].set_xlabel('Time')
            axes[3].grid(True)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            save_path = os.path.join(save_dir, f'node_{moteid}.png')
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"节点 {moteid} 的数据可视化保存到: {save_path}")
            
        # 可视化所有节点的平均温度、湿度、光照和电压
        avg_df = df.groupby(df.index).mean()
        
        # 创建图形
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 绘制温度
        axes[0].plot(avg_df.index, avg_df['temperature'])
        axes[0].set_title('Average Temperature')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].grid(True)
        
        # 绘制湿度
        axes[1].plot(avg_df.index, avg_df['humidity'])
        axes[1].set_title('Average Humidity')
        axes[1].set_ylabel('Humidity (%)')
        axes[1].grid(True)
        
        # 绘制光照
        axes[2].plot(avg_df.index, avg_df['light'])
        axes[2].set_title('Average Light')
        axes[2].set_ylabel('Light (lux)')
        axes[2].grid(True)
        
        # 绘制电压
        axes[3].plot(avg_df.index, avg_df['voltage'])
        axes[3].set_title('Average Voltage')
        axes[3].set_ylabel('Voltage (V)')
        axes[3].set_xlabel('Time')
        axes[3].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        save_path = os.path.join(save_dir, 'average.png')
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"平均数据可视化保存到: {save_path}")
        
        return True
    except Exception as e:
        logger.error(f"可视化数据时出错: {e}")
        return False

def preprocess_intel_lab_dataset(raw_dir, processed_dir):
    """
    预处理Intel Berkeley Lab数据集
    
    Args:
        raw_dir: 原始数据目录
        processed_dir: 处理后的数据目录
        
    Returns:
        bool: 预处理是否成功
    """
    # 创建处理后的数据目录
    os.makedirs(processed_dir, exist_ok=True)
    
    # 加载数据
    data_path = os.path.join(raw_dir, 'data.txt')
    topology_path = os.path.join(raw_dir, 'mote_locs.txt')
    connectivity_path = os.path.join(raw_dir, 'connectivity.txt')
    
    df = load_data(data_path)
    topology_df = load_topology(topology_path)
    connectivity_df = load_connectivity(connectivity_path)
    
    if df is None or topology_df is None or connectivity_df is None:
        logger.error("加载数据失败")
        return False
    
    # 清洗数据
    cleaned_df = clean_data(df)
    
    # 保存清洗后的数据
    cleaned_path = os.path.join(processed_dir, 'cleaned_data.csv')
    if not save_data(cleaned_df, cleaned_path):
        logger.error("保存清洗后的数据失败")
        return False
    
    # 归一化数据
    normalized_df, scalers = normalize_data(cleaned_df, method='minmax')
    
    # 保存归一化后的数据
    normalized_path = os.path.join(processed_dir, 'normalized_data.csv')
    if not save_data(normalized_df, normalized_path):
        logger.error("保存归一化后的数据失败")
        return False
    
    # 生成特征
    feature_df = generate_features(normalized_df)
    
    # 保存特征数据
    feature_path = os.path.join(processed_dir, 'feature_data.csv')
    if not save_data(feature_df, feature_path):
        logger.error("保存特征数据失败")
        return False
    
    # 划分数据集
    train_df, val_df, test_df = split_data(feature_df)
    
    # 创建训练集、验证集和测试集目录
    split_dir = os.path.join(processed_dir, 'train_test_split')
    os.makedirs(split_dir, exist_ok=True)
    
    # 保存训练集、验证集和测试集
    train_path = os.path.join(split_dir, 'train.csv')
    val_path = os.path.join(split_dir, 'val.csv')
    test_path = os.path.join(split_dir, 'test.csv')
    
    if not save_data(train_df, train_path) or not save_data(val_df, val_path) or not save_data(test_df, test_path):
        logger.error("保存训练集、验证集和测试集失败")
        return False
    
    # 保存拓扑和连接数据
    topology_path = os.path.join(processed_dir, 'topology.csv')
    connectivity_path = os.path.join(processed_dir, 'connectivity.csv')
    
    if not save_data(topology_df, topology_path) or not save_data(connectivity_df, connectivity_path):
        logger.error("保存拓扑和连接数据失败")
        return False
    
    # 可视化数据
    visualization_dir = os.path.join(processed_dir, 'visualization')
    if not visualize_data(cleaned_df, visualization_dir):
        logger.error("可视化数据失败")
        return False
    
    logger.info("数据预处理完成")
    return True

def main():
    """
    主函数
    """
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='预处理Intel Berkeley Lab数据集')
    parser.add_argument('--raw_dir', type=str, default='../../data/raw',
                        help='原始数据目录')
    parser.add_argument('--processed_dir', type=str, default='../../data/processed',
                        help='处理后的数据目录')
    args = parser.parse_args()
    
    # 获取绝对路径
    raw_dir = os.path.abspath(args.raw_dir)
    processed_dir = os.path.abspath(args.processed_dir)
    
    logger.info(f"开始预处理Intel Berkeley Lab数据集")
    logger.info(f"原始数据目录: {raw_dir}")
    logger.info(f"处理后的数据目录: {processed_dir}")
    
    # 预处理数据集
    if preprocess_intel_lab_dataset(raw_dir, processed_dir):
        logger.info("数据预处理成功")
    else:
        logger.error("数据预处理失败")

if __name__ == '__main__':
    main()