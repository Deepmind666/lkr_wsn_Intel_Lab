#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intel Berkeley Lab数据集下载脚本

该脚本用于下载Intel Berkeley Lab数据集，该数据集包含54个静态节点，
传感温度、湿度、光照、电压，每31秒采样一次。

数据集来源：http://db.csail.mit.edu/labdata/labdata.html
"""

import os
import sys
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据集URL
DATASET_URLS = {
    'data': 'http://db.csail.mit.edu/labdata/data.txt.gz',
    'topology': 'http://db.csail.mit.edu/labdata/mote_locs.txt',
    'connectivity': 'http://db.csail.mit.edu/labdata/connectivity.txt',
}

def download_file(url, save_path):
    """
    下载文件并显示进度条
    
    Args:
        url: 文件URL
        save_path: 保存路径
        
    Returns:
        bool: 下载是否成功
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # 创建进度条
        t = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"下载 {os.path.basename(url)}")
        
        # 下载文件
        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        # 检查文件大小是否正确（仅当服务器提供了content-length且文件不为空时检查）
        if total_size != 0 and os.path.getsize(save_path) == 0:
            logger.error(f"下载的文件为空: {save_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"下载文件时出错: {e}")
        return False

def extract_gz(gz_path, extract_dir):
    """
    解压.gz文件
    
    Args:
        gz_path: .gz文件路径
        extract_dir: 解压目录
        
    Returns:
        str: 解压后的文件路径
    """
    try:
        import gzip
        import shutil
        
        # 创建解压目录
        os.makedirs(extract_dir, exist_ok=True)
        
        # 解压文件名
        extract_path = os.path.join(extract_dir, os.path.basename(gz_path)[:-3])
        
        # 解压文件
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        return extract_path
    except Exception as e:
        logger.error(f"解压文件时出错: {e}")
        return None

def download_intel_lab_dataset(save_dir='../../data/raw'):
    """
    下载Intel Berkeley Lab数据集
    
    Args:
        save_dir: 保存目录
        
    Returns:
        bool: 下载是否成功
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载数据集
    success = True
    for name, url in DATASET_URLS.items():
        # 确定保存路径，对于topology使用固定名称
        if name == 'topology':
            save_path = os.path.join(save_dir, 'mote_locs.txt')
        else:
            save_path = os.path.join(save_dir, os.path.basename(url))
            
        logger.info(f"下载 {name} 数据: {url}")
        
        if not download_file(url, save_path):
            success = False
            continue
            
        # 如果是.gz文件，解压
        if save_path.endswith('.gz'):
            logger.info(f"解压 {save_path}")
            extract_path = extract_gz(save_path, save_dir)
            if extract_path is None:
                success = False
                continue
                
    return success

def verify_dataset(data_dir):
    """
    验证数据集是否完整
    
    Args:
        data_dir: 数据目录
        
    Returns:
        bool: 数据集是否完整
    """
    required_files = ['data.txt', 'mote_locs.txt', 'connectivity.txt']
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"缺少文件: {file_path}")
            return False
            
    return True

def print_dataset_info(data_dir):
    """
    打印数据集信息
    
    Args:
        data_dir: 数据目录
    """
    data_path = os.path.join(data_dir, 'data.txt')
    topology_path = os.path.join(data_dir, 'mote_locs.txt')
    connectivity_path = os.path.join(data_dir, 'connectivity.txt')
    
    # 数据文件信息
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            lines = f.readlines()
            num_lines = len(lines)
            logger.info(f"数据文件包含 {num_lines} 行记录")
            
            if num_lines > 0:
                logger.info(f"数据文件第一行: {lines[0].strip()}")
                
    # 拓扑文件信息
    if os.path.exists(topology_path):
        with open(topology_path, 'r') as f:
            lines = f.readlines()
            num_nodes = len(lines)
            logger.info(f"拓扑文件包含 {num_nodes} 个节点")
            
    # 连接文件信息
    if os.path.exists(connectivity_path):
        with open(connectivity_path, 'r') as f:
            lines = f.readlines()
            num_connections = len(lines)
            logger.info(f"连接文件包含 {num_connections} 条连接记录")

def main():
    """
    主函数
    """
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='下载Intel Berkeley Lab数据集')
    parser.add_argument('--save_dir', type=str, default='../../data/raw',
                        help='保存目录')
    args = parser.parse_args()
    
    # 获取绝对路径
    save_dir = os.path.abspath(args.save_dir)
    
    logger.info(f"开始下载Intel Berkeley Lab数据集到 {save_dir}")
    
    # 下载数据集
    if download_intel_lab_dataset(save_dir):
        logger.info("数据集下载完成")
        
        # 验证数据集
        if verify_dataset(save_dir):
            logger.info("数据集验证通过")
            print_dataset_info(save_dir)
        else:
            logger.error("数据集验证失败，请检查数据文件是否完整")
    else:
        logger.error("数据集下载失败，请检查网络连接或手动下载")
        logger.info("数据集下载地址: http://db.csail.mit.edu/labdata/labdata.html")

if __name__ == '__main__':
    main()