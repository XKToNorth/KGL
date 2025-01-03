import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 确保设备的一致性，选择使用 GPU（如果可用），否则使用 CPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# -------------------------
# Step 1: 数据加载与预处理
# -------------------------
def load_mapping(file_path):
    """加载 entity2id 或 relation2id 映射文件"""
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            name, idx = line.strip().split()
            mapping[name] = int(idx)
    return mapping


def load_triplets(file_path, entity2id, relation2id):
    """加载三元组数据，将实体和关系映射为ID"""
    triplets = []
    with open(file_path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split()
            triplets.append((entity2id[head], entity2id[tail], relation2id[relation]))
    return triplets


class FB15kDataset(Dataset):
    """自定义数据集"""

    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


# -------------------------
# Step 2: 定义 TransE 模型
# -------------------------
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.embedding_dim = embedding_dim
        self.initialize_embeddings()

    def initialize_embeddings(self):
        """初始化嵌入向量"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail):
        """计算正样本和负样本之间的距离"""
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        return -torch.norm(head_emb + relation_emb - tail_emb, p=2, dim=1)  # 距离变为分数返回


# -------------------------
# Step 3: 改进负样本生成
# -------------------------
def generate_hard_negative_sample(head, relation, tail, model, batch_size=2048):
    """基于模型得分生成困难负样本（分批计算减少内存占用）"""
    num_entities = model.entity_embeddings.weight.size(0)
    neg_head, neg_tail = head.clone(), tail.clone()

    with torch.no_grad():
        if torch.rand(1).item() < 0.5:
            # 替换头实体
            candidate_scores = []
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                candidate_heads = torch.arange(start, end, device=head.device).unsqueeze(0).expand(len(head), -1)
                scores = torch.norm(
                    model.entity_embeddings(candidate_heads) + model.relation_embeddings(relation).unsqueeze(1)
                    - model.entity_embeddings(tail).unsqueeze(1),
                    p=2, dim=2
                )
                candidate_scores.append(scores)
            candidate_scores = torch.cat(candidate_scores, dim=1)
            neg_head = torch.argmin(candidate_scores, dim=1)
        else:
            # 替换尾实体
            candidate_scores = []
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                candidate_tails = torch.arange(start, end, device=tail.device).unsqueeze(0).expand(len(tail), -1)
                scores = torch.norm(
                    model.entity_embeddings(head).unsqueeze(1) + model.relation_embeddings(relation).unsqueeze(1)
                    - model.entity_embeddings(candidate_tails),
                    p=2, dim=2
                )
                candidate_scores.append(scores)
            candidate_scores = torch.cat(candidate_scores, dim=1)
            neg_tail = torch.argmin(candidate_scores, dim=1)

    return neg_head, neg_tail


# -------------------------
# Step 4: 自对抗负采样损失
# -------------------------
class AdversarialLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(AdversarialLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        pos_loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        neg_weights = torch.softmax(neg_scores / self.temperature, dim=1)
        neg_loss = -(neg_weights * torch.log(1 - torch.sigmoid(neg_scores))).sum(dim=1).mean()
        return pos_loss + neg_loss


# -------------------------
# Step 5: 模型训练
# -------------------------
def train_model(train_data, num_entities, num_relations, embedding_dim=500, epochs=100, batch_size=1024, lr=0.002, margin=4.0):
    model = TransE(num_entities, num_relations, embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 改为基于对比学习的损失函数
    loss_fn = nn.MarginRankingLoss(margin=margin)

    train_loader = DataLoader(FB15kDataset(train_data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            head, tail, relation = batch
            head, tail, relation = head.to(device), tail.to(device), relation.to(device)

            # 正样本计算
            pos_scores = model(head, relation, tail)

            # 生成负样本
            neg_head, neg_tail = generate_hard_negative_sample(head, relation, tail, model)
            neg_scores = model(neg_head, relation, neg_tail)

            # 保证 `neg_scores` 是正确的维度
            if neg_scores.dim() == 1:
                neg_scores = neg_scores.unsqueeze(1)  # 转换为 2D 张量

            # 计算损失
            y = torch.ones_like(pos_scores, device=pos_scores.device)
            loss = loss_fn(pos_scores, neg_scores.squeeze(), y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model



# -------------------------
# Step 6: 保存嵌入向量
# -------------------------
def save_embeddings(file_path, embeddings, id2entity):
    """保存嵌入向量到文件"""
    with open(file_path, 'w') as f:
        for idx, embedding in enumerate(embeddings):
            entity = id2entity[idx]
            embedding_str = ', '.join(map(str, embedding.tolist()))
            f.write(f"{entity}\t[{embedding_str}]\n")


# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    # 文件路径
    entity2id_path = "/data_disk/libh/KRL/WN18/entity2id.txt"
    relation2id_path = "/data_disk/libh/KRL/WN18/relation2id.txt"
    train_path = "/data_disk/libh/KRL/WN18/train.txt"
    entity_embeddings_path = "entity_embeddings_WN18.txt"
    relation_embeddings_path = "relation_embeddings_WN18.txt"

    # 加载数据
    entity2id = load_mapping(entity2id_path)
    relation2id = load_mapping(relation2id_path)
    train_data = load_triplets(train_path, entity2id, relation2id)

    # 模型训练
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    embedding_dim = 300
    model = train_model(train_data, num_entities, num_relations, embedding_dim, epochs=50, batch_size=1024)

    # 保存嵌入向量
    save_embeddings(entity_embeddings_path, model.entity_embeddings.weight.cpu().detach(),
                    {v: k for k, v in entity2id.items()})
    save_embeddings(relation_embeddings_path, model.relation_embeddings.weight.cpu().detach(),
                    {v: k for k, v in relation2id.items()})

