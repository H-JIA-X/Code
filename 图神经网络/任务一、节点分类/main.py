import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
class GraphModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, model_type='GAT', hidden=16, heads=4):
        super(GraphModel, self).__init__()
        self.model_type = model_type
        if model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden, heads=heads)
            self.conv2 = GATConv(hidden * heads, out_channels)
        elif model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out_channels)
        elif model_type == 'GIN':
            self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)))
            self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden, out_channels), torch.nn.ReLU(), torch.nn.Linear(out_channels, out_channels)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
        pred = out.argmax(dim=1)
        correct = int(pred[mask].eq(data.y[mask]).sum().item())
        acc = correct / int(mask.sum())
        return acc



# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集列表
datasets = ['Cora', 'Citeseer']

# 定义模型、优化器和损失函数
models = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
# model_type = 'GAT'  # 可以选择 'GAT', 'GCN', 'GraphSAGE', 'GIN' 中的任意一个
hidden_channels = 16
heads = 4
num_epochs = 200
lr = 0.01
weight_decay = 5e-4

for name in datasets:
    print(f"Training on {name} dataset")

    # 加载数据集
    dataset = Planetoid(root=f'./data/{name}', name=name)
    data = dataset[0].to(device)

    history = {model_name: {'train_loss': [], 'val_acc': []} for model_name in models}
    for model_type in models:
        # 实例化模型
        model = GraphModel(in_channels=dataset.num_node_features, out_channels=dataset.num_classes,
                           model_type=model_type, hidden=hidden_channels, heads=heads).to(device)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # 实例化Trainer类
        trainer = Trainer(model, optimizer, criterion, device)

        # 训练模型
        best_val_acc = 0
        best_test_acc = 0

        for epoch in range(num_epochs):
            loss = trainer.train(data)
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
            history[model_type]['train_loss'].append(loss)
            # 在每个epoch结束时评估模型
            val_acc = trainer.evaluate(data, data.val_mask)
            print(f'Validation Accuracy: {val_acc:.4f}')
            history[model_type]['val_acc'].append(val_acc)
            # 保存最佳验证集准确率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = trainer.evaluate(data, data.test_mask)
                print(f'Best Test Accuracy: {best_test_acc:.4f}')

        # 输出最佳测试集准确率
        print(f"{name} dataset - Best Test Accuracy: {best_test_acc:.4f}\n")

    # # 训练损失对比
    # plt.subplot(1, 2, 1)
    # for model_name in models:
    #     plt.plot(history[model_name]['train_loss'], label=model_name)
    # plt.title('Training Loss Comparison')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # # 验证准确率对比
    # plt.subplot(1, 2, 2)
    # for model_name in models:
    #     plt.plot(history[model_name]['val_acc'], label=model_name)
    # plt.title('Validation Accuracy Comparison')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # 可视化部分修改
    plt.figure(figsize=(14, 6))
    plt.style.use('seaborn')

    # 训练损失子图
    ax1 = plt.subplot(1, 2, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (model_name, c) in enumerate(zip(models, colors)):
        ax1.plot(history[model_name]['train_loss'],
                 color=c,
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.8,
                 label=model_name)
    ax1.set_title('Training Loss Trajectory', fontsize=14, pad=12)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white')

    # 验证准确率子图
    ax2 = plt.subplot(1, 2, 2)
    for idx, (model_name, c) in enumerate(zip(models, colors)):
        ax2.plot(history[model_name]['val_acc'],
                 color=c,
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.8,
                 label=model_name)
    ax2.set_title('Validation Accuracy Progression', fontsize=14, pad=12)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white')

    # 调整布局并保存
    plt.tight_layout(w_pad=4)
    plt.savefig(f'./performance_comparison_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 添加最佳精度输出
    print("\n=== 模型最佳验证精度 ===")
    for model_name in models:
        best_acc = max(history[model_name]['val_acc']) * 100
        print(f'{model_name}: {best_acc:.2f}%')