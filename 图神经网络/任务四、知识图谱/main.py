import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
class KnowledgeGraphDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        return (self.entity2id[head], self.relation2id[relation], self.entity2id[tail])

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        score = torch.norm(head_emb + relation_emb - tail_emb, p=1, dim=1)
        return score

    def loss(self, positive_score, negative_score):
        return torch.mean(torch.max(positive_score - negative_score + self.margin, torch.tensor(0.0)))

class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(RotatE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, head, relation, tail):
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        head_emb = head_emb.view(-1, self.embedding_dim, 2)
        tail_emb = tail_emb.view(-1, self.embedding_dim, 2)

        real_head, imag_head = head_emb[:, :, 0], head_emb[:, :, 1]
        real_tail, imag_tail = tail_emb[:, :, 0], tail_emb[:, :, 1]
        phase_relation = relation_emb / (self.embedding_dim / np.pi)

        real_relation = torch.cos(phase_relation)
        imag_relation = torch.sin(phase_relation)

        real_score = real_head * real_relation - imag_head * imag_relation
        imag_score = real_head * imag_relation + imag_head * real_relation
        score = torch.sqrt((real_score - real_tail) ** 2 + (imag_score - imag_tail) ** 2).sum(dim=1)
        return score

    def loss(self, positive_score, negative_score):
        return torch.mean(torch.max(positive_score - negative_score + self.margin, torch.tensor(0.0)))

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(ConvE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0)
        self.fc = nn.Linear(32 * (embedding_dim - 2) * (embedding_dim - 2), embedding_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, head, relation, tail):
        # 调整head_emb和relation_emb形状为(批量大小, 1, embedding_dim, embedding_dim // 2)
        # 将embedding维度调整为合适的二维形状，满足卷积核大小要求
        side_len = int(self.embedding_dim ** 0.5)
        head_emb = self.entity_embeddings(head).view(-1, 1, side_len, side_len)
        relation_emb = self.relation_embeddings(relation).view(-1, 1, side_len, side_len)
        # 由于卷积核大小为(3,3)，输入的高和宽至少为3，调整embedding_dim以满足要求
        # 如果embedding_dim较小，可以考虑调整卷积核大小或padding
        concat = torch.cat([head_emb, relation_emb], 2)
        x = self.conv1(concat)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        tail_emb = self.entity_embeddings(tail)
        score = torch.norm(x - tail_emb, p=2, dim=1)
        return score

    def loss(self, positive_score, negative_score):
        return torch.mean(torch.max(positive_score - negative_score + self.margin, torch.tensor(0.0)))


def train(model, dataloader, optimizer, name, num_epochs=100):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for head, relation, tail in dataloader:
            head = head.to(device)
            relation = relation.to(device)
            tail = tail.to(device)
            # 正样本
            positive_score = model(head, relation, tail)

            # 负样本生成 (head, relation, random tail)
            # 确保负样本的索引在实体索引的有效范围内
            negative_tail = torch.randint(0, num_entities, tail.shape).to(device)
            negative_score = model(head, relation, negative_tail)

            loss = model.loss(positive_score, negative_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history[name]['train_loss'].append(total_loss / len(dataloader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


def load_data(entity_file, relation_file, train_file):
    # 读取实体到ID的映射
    entity2id = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                entity2id[parts[0]] = int(parts[1])

    # 读取关系到ID的映射
    relation2id = {}
    with open(relation_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                relation2id[parts[0]] = int(parts[1])

    # 读取训练三元组
    triples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                triples.append((head, relation, tail))

    return triples, entity2id, relation2id


# 使用示例
triples, entity2id, relation2id = load_data(
    './data/entity2id.txt',
    './data/relation2id.txt',
    './data/train.txt'
)

# 参数设置
embedding_dim = 50
num_entities = len(entity2id)
num_relations = len(relation2id)

# 数据加载
dataset = KnowledgeGraphDataset(triples, entity2id, relation2id)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
models = ['TransE', 'RotatE']
history = {model_name: {'train_loss': []} for model_name in models}
for name in ['TransE', 'RotatE']:
    if name == 'TransE':
        model = TransE(num_entities, num_relations, embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, dataloader, optimizer, name, num_epochs=50)
    if name == 'RotatE':
        model = RotatE(num_entities, num_relations, embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train(model, dataloader, optimizer, name, num_epochs=50)

plt.figure(figsize=(14, 6))


for name in models:
    plt.plot(history[name]['train_loss'], label=name)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(title='Model')
plt.show()
