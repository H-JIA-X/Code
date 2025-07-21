
# ## 任务三、 图分类
# 实现要求：基于现有模型框架(DGL, Pytorch_geometric)实现图分类任务
#
#
# 使用TUDataset, ZINC数据集
#
#
# 分析不同的池化方法对图分类性能的影响(AvgPooling, MaxPooling, MinPooling)
#
#
# 测试GCN, GAT, GraphSAGE, GIN模型



import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter
from torch.nn import Linear
def global_min_pool(x, batch):
    return scatter(x, batch, dim=0, reduce='min')
# 加载数据集
dataset = TUDataset(root=f'./data', name='ENZYMES')
# dataset = ZINC(root='/dataset/molecules')




class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, pooling):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, num_classes)
        self.pooling = pooling

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pooling(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, pooling):  # 添加pooling参数
        super(GAT, self).__init__()
        self.GAT1 = GATConv(num_features, 8, heads=8, concat=True, dropout=0.6)
        self.GAT2 = GATConv(8 * 8, 128, dropout=0.6)  # 输出特征维度128
        self.fc = torch.nn.Linear(128, num_classes)   # 添加全连接层
        self.pooling = pooling                         # 存储池化方法

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch  # 获取batch信息
        x = F.relu(self.GAT1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.GAT2(x, edge_index)
        x = self.pooling(x, batch)        # 应用池化操作
        x = self.fc(x)                    # 全连接层分类
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, pooling):  # 添加pooling
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(num_features, 64)
        self.sage2 = SAGEConv(64, 128)  # 统一输出维度128
        self.fc = torch.nn.Linear(128, num_classes)   # 添加全连接层
        self.pooling = pooling                        # 存储池化方法

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        x = self.pooling(x, batch)        # 应用池化操作
        x = self.fc(x)                    # 全连接层分类
        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pooling):  # 添加pooling
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(Linear(in_channels, 64),
                                       torch.nn.ReLU(),
                                       Linear(64, 64)))
        self.conv2 = GINConv(torch.nn.Sequential(Linear(64, 64),
                                       torch.nn.ReLU(),
                                       Linear(64, 128)))  # 输出特征维度128
        self.fc = torch.nn.Linear(128, out_channels)      # 添加全连接层
        self.pooling = pooling                            # 存储池化方法
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.pooling(x, batch)        # 应用池化操作
        x = self.fc(x)                    # 全连接层分类
        return F.log_softmax(x, dim=1)


# class GAT(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling):
#         super(GAT ,self).__init__()
#         self.GAT1 = GATConv(num_features, 64, heads = 8, concat = True, dropout = 0.6)
#         self.GAT2 = GATConv( 64, 128, dropout = 0.6)
#         self.fc = torch.nn.Linear(128, num_classes)
#         self.pooling = pooling
#
#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.GAT1(x, edge_index))
#         x = F.relu(self.GAT2(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
#
#
#
#
#
# class GraphSAGE(torch.nn.Module):
#
#     def __init__(self, num_features, num_classes, pooling):
#         super(GraphSAGE, self).__init__()
#         self.sage1 = SAGEConv(num_features, 64)  # 定义两层GraphSAGE层
#         self.sage2 = SAGEConv(64, 128)
#         self.fc = torch.nn.Linear(128, num_classes)
#         self.pooling = pooling
#     def forward(self, x, edge_index, batch):
#         # x, edge_index = data.x, data.edge_index
#         x = F.relu(self.sage1(x, edge_index))
#         x = F.relu(self.sage2(x, edge_index))
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)
#         # x = self.sage1(x, edge_index)
#         # x = F.relu(x)
#         # x = F.dropout(x, training=self.training)
#         # x = self.sage2(x, edge_index)
#         #
#         # return F.log_softmax(x, dim=1)
#
#
#
#
# class GIN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, pooling):
#         super(GIN, self).__init__()
#         self.conv1 = GINConv(torch.nn.Sequential(Linear(num_features, 64), torch.nn.ReLU(), Linear(64, 64)))
#         self.conv2 = GINConv(torch.nn.Sequential(Linear(64, 128), torch.nn.ReLU(), Linear(128, 128)))
#         self.fc = torch.nn.Linear(128, num_classes)
#         self.pooling = pooling
#
#     def forward(self, x, edge_index, batch):
#         # x, edge_index = data.x, data.edge_index
#
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.dropout(x)
#         x = self.pooling(x, batch)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)




# 训练
def train(model, loader, optimizer, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# 测试
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# 主程序
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 池化方法
pooling_methods = {
    'AvgPooling': global_mean_pool,
    'MaxPooling': global_max_pool,
    'MinPooling': global_min_pool,
}

# 模型
model_classes = {
    'GCN': GCN,
    'GAT': GAT,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
}


for pool_name, pool in pooling_methods.items():
    for model_name, model_class in model_classes.items():
        model = model_class(dataset.num_features, dataset.num_classes, pooling=pool).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 训练模型
        for epoch in range(1, 101):
            train(model, train_loader, optimizer, device)

        # 测试模型
        acc = test(model, test_loader, device)
        print(f'Model: {model_name}, Pooling: {pool_name}, Accuracy: {acc:.4f}')


