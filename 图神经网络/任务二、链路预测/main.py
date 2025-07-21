import argparse
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
import torch.optim as optim
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges, negative_sampling


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
setup_seed(42)
# 命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Base-Graph Neural Network')
    parser.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Pubmed'], default='Cora',
                        help="Dataset selection")
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--model', choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'], default='GCN',
                        help="Model selection")
    parser.add_argument('--lr', default=0.01, help="Learning Rate selection")
    parser.add_argument('--wd', default=5e-4, help="weight_decay selection")
    parser.add_argument('--epochs', default=200, help="train epochs selection")
    return parser.parse_args()


# 加载数据集
def load_dataset(name):
    dataset = Planetoid(root='dataset/' + name, name=name, transform=T.NormalizeFeatures())
    return dataset


# 定义模型
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, model_type, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

        # if model_type == 'GCN':
        #     self.conv1 = GCNConv(in_channels, hidden_channels)
        #     self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # elif model_type == 'GAT':
        #     self.conv1 = GATConv(in_channels, hidden_channels)
        #     self.conv2 = GATConv(hidden_channels, hidden_channels)
        # elif model_type == 'SAGE':
        #     self.conv1 = SAGEConv(in_channels, hidden_channels)
        #     self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # elif model_type == 'ChebNet':
        #     self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
        #     self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2)
        # elif model_type == 'TransformerConv':
        #     self.conv1 = TransformerConv(in_channels, hidden_channels)
        #     self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        # else:
        #     raise NotImplementedError
        if model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        elif model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GIN':
            self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)))
            self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, x, edge_index, edge_label_index, edge_label, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    # 只对标记的边计算损失
    pos_out = out[edge_label_index[0], :] * out[edge_label_index[1], :]
    pos_out = torch.sum(pos_out, dim=1)
    loss = F.binary_cross_entropy_with_logits(pos_out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, edge_index, edge_label_index, edge_label):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pos_out = out[edge_label_index[0], :] * out[edge_label_index[1], :]
        pos_out = torch.sum(pos_out, dim=1)
        predictions = torch.sigmoid(pos_out) > 0.5
        acc = accuracy_score(edge_label.cpu(), predictions.cpu())
        prec = precision_score(edge_label.cpu(), predictions.cpu())
        rec = recall_score(edge_label.cpu(), predictions.cpu())
        f1 = f1_score(edge_label.cpu(), predictions.cpu())

    return acc, prec, rec, f1

if __name__ == "__main__":
    datasets = ['Cora', 'Citeseer']
    models = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in datasets:
        dataset = load_dataset(name)
        data = dataset[0]
        print(data.edge_index)
        # data = dataset[0]
        # 使用RandomLinkSplit转换进行边的训练/验证/测试集划分
        transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.8, neg_sampling_ratio=1.0)
        train_data, val_data, test_data = transform(data)
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
        # 将edge_label转换为float，因为binary_cross_entropy_with_logits期望float类型的labels
        print("Train data:")
        print(train_data)
        print("Validation data:")
        print(val_data)
        print("Test data:")
        print(test_data)
        train_edge_label = train_data['edge_label'].to(torch.float)
        test_edge_label = test_data['edge_label'].to(torch.float)
        history = {model_name: {'train_loss': [], 'val_acc': []} for model_name in models}
        for model_name in models:

            model = GNN(in_channels=data.x.shape[1], hidden_channels=args.hidden_dim,
                        model_type=model_name, dropout_rate=args.dropout_rate).to(device)
            print(model)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            for epoch in range(args.epochs):
                loss = train(model, train_data.x, train_data.edge_index, train_data.edge_label_index, train_edge_label,
                             optimizer)
                history[model_name]['train_loss'].append(loss)
                acc, prec, rec, f1 = test(model, test_data.x, test_data.edge_index, test_data.edge_label_index, test_edge_label)
                history[model_name]['val_acc'].append(acc)
                print(f'Epoch: [{epoch:03d}/200], Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}')
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
