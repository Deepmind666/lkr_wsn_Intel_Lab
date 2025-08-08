import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
import numpy as np
import networkx as nx

# --- 1. GNN模型定义 ---
class ChainGNN(nn.Module):
    """用于预测最优链式拓扑的图神经网络模型"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ChainGNN, self).__init__()
        # 可以选用GCN, GraphSAGE, GAT等不同卷积层进行实验对比
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=2, concat=True)
        self.conv3 = GATConv(hidden_channels * 2, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        # 输出层：预测每个节点与其邻居节点的连接“倾向性”
        # 这里输出的是一个代表连接分数的标量，分数越高代表越应该连接
        x = self.conv3(x, edge_index)
        return x

# --- 2. 训练数据生成器 ---
class WSNDataGenerator:
    """生成用于训练GNN的WSN拓扑数据"""
    def __init__(self, num_nodes_range=(50, 100), area_size=(100, 100)):
        self.num_nodes_range = num_nodes_range
        self.area_size = area_size

    def _generate_random_wsn(self):
        """生成一个随机的WSN网络拓扑"""
        num_nodes = np.random.randint(*self.num_nodes_range)
        nodes = {
            i: (np.random.rand() * self.area_size[0], np.random.rand() * self.area_size[1])
            for i in range(num_nodes)
        }
        G = nx.random_geometric_graph(num_nodes, radius=25, pos=nodes)
        return G

    def _get_greedy_chain(self, G):
        """使用贪心算法生成次优链（作为模型的输入特征之一）"""
        # 此处简化实现，实际应使用您项目中的build_energy_efficient_chains逻辑
        if not G.nodes:
            return []
        path = nx.approximation.traveling_salesman_problem(G, cycle=False)
        return path

    def _get_optimal_chain(self, G):
        """通过全局优化算法生成最优链（作为训练标签）"""
        # 注意：这是一个NP-hard问题，只能对小规模网络求解
        # 实际研究中，可以使用高质量的近似解（如LKH算法）作为“最优”标签
        if not G.nodes or len(G.nodes) > 15: # 限制规模以保证可计算性
            return self._get_greedy_chain(G)
        path = nx.approximation.traveling_salesman_problem(G, cycle=False, method=nx.approximation.christofides)
        return path

    def generate_dataset(self, num_samples=100):
        """生成包含多个图样本的数据集"""
        dataset = []
        for _ in range(num_samples):
            G = self._generate_random_wsn()
            if len(G.nodes) == 0:
                continue

            # 节点特征：[x坐标, y坐标, 节点度, 能量（模拟）]
            features = []
            for node in G.nodes():
                pos = G.nodes[node]['pos']
                degree = G.degree(node)
                energy = np.random.uniform(0.5, 1.0)
                features.append([pos[0], pos[1], degree, energy])
            x = torch.tensor(features, dtype=torch.float)

            # 边索引
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

            # 标签：构建一个邻接矩阵表示最优链
            optimal_chain = self._get_optimal_chain(G)
            y = torch.zeros((len(G.nodes), len(G.nodes)))
            for i in range(len(optimal_chain) - 1):
                u, v = optimal_chain[i], optimal_chain[i+1]
                y[u, v] = y[v, u] = 1.0 # 对称矩阵

            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)
        return dataset

# --- 3. 训练和评估框架 ---
class GNNTrainer:
    """GNN模型的训练和评估器"""
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # 使用BCEWithLogitsLoss，因为它在数值上更稳定，且适合处理图连接预测这类问题
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_loader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                self.optimizer.zero_grad()
                
                # 模型预测的是节点级别的连接分数
                node_scores = self.model(data)
                
                # 将节点分数转换为边级别的预测
                # 通过源节点和目标节点分数的点积来预测边的存在概率
                edge_pred = node_scores[data.edge_index[0]] * node_scores[data.edge_index[1]]
                edge_pred = torch.sum(edge_pred, dim=1)

                # 准备标签
                edge_label = data.y[data.edge_index[0], data.edge_index[1]]

                loss = self.criterion(edge_pred, edge_label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1:03d}, Loss: {avg_loss:.4f}')

    def predict_chain(self, wsn_graph):
        """使用训练好的模型为新的WSN图预测最优链"""
        self.model.eval()
        with torch.no_grad():
            # ... (此处省略将wsn_graph转换为torch_geometric.data.Data的逻辑)
            # ... (使用模型进行预测)
            # ... (根据预测分数，使用贪心或波束搜索等方法解码出最终的链)
            pass
        print("预测功能待实现...")

# --- 4. 主程序入口 ---
if __name__ == '__main__':
    print("🚀 1. 初始化数据生成器...")
    data_generator = WSNDataGenerator()

    print("\n🚀 2. 生成训练和测试数据集...")
    train_dataset = data_generator.generate_dataset(num_samples=200)
    test_dataset = data_generator.generate_dataset(num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"\n   - 生成了 {len(train_dataset)} 个训练样本")
    print(f"   - 生成了 {len(test_dataset)} 个测试样本")

    print("\n🚀 3. 初始化GNN模型...")
    # 输入特征维度为4 (x, y, degree, energy)
    # 输出维度为8，代表一个8维的连接倾向性嵌入向量
    model = ChainGNN(in_channels=4, hidden_channels=32, out_channels=8)
    print(model)

    print("\n🚀 4. 开始训练模型...")
    trainer = GNNTrainer(model)
    trainer.train(train_loader, epochs=50)

    print("\n🚀 5. 模型训练完成！")
    print("   下一步：实现predict_chain方法，并将其集成到您的主仿真循环中。")