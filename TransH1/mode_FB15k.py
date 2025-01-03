import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 确保设备的一致性，选择使用 GPU（如果可用），否则使用 CPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
            head, tail, relation = line.strip().split()
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
# Step 2: 定义 TransH 模型
# -------------------------
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransH, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.relation_norms = nn.Embedding(num_relations, embedding_dim)  # 每个关系的超平面向量
        self.embedding_dim = embedding_dim
        self.initialize_embeddings()

    def initialize_embeddings(self):
        """初始化嵌入向量"""
        nn.init.uniform_(self.entity_embeddings.weight, a=-6/np.sqrt(self.embedding_dim), b=6/np.sqrt(self.embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, a=-6/np.sqrt(self.embedding_dim), b=6/np.sqrt(self.embedding_dim))
        nn.init.uniform_(self.relation_norms.weight, a=-6/np.sqrt(self.embedding_dim), b=6/np.sqrt(self.embedding_dim))

    def project_to_relation_space(self, entity_emb, relation_emb, relation_norm):
        """将实体嵌入投影到关系超平面"""
        # 计算实体与关系超平面的投影
        norm_relation_emb = relation_norm / relation_norm.norm(p=2, dim=1, keepdim=True)  # 归一化关系的超平面
        projection = entity_emb - (entity_emb * norm_relation_emb).sum(dim=1, keepdim=True) * norm_relation_emb
        return projection

    def forward(self, head, relation, tail):
        """计算正样本和负样本之间的距离"""
        head_emb = self.entity_embeddings(head)
        relation_emb = self.relation_embeddings(relation)
        tail_emb = self.entity_embeddings(tail)
        relation_norm = self.relation_norms(relation)

        # 投影到关系的超平面
        head_proj = self.project_to_relation_space(head_emb, relation_emb, relation_norm)
        tail_proj = self.project_to_relation_space(tail_emb, relation_emb, relation_norm)

        # 计算距离
        return torch.norm(head_proj + relation_emb - tail_proj, p=2, dim=1)

# -------------------------
# Step 3: 训练模型
# -------------------------
def train_model(train_data, num_entities, num_relations, embedding_dim=100, epochs=100, batch_size=256, lr=0.001, margin=1.0):
    # 初始化模型、损失函数和优化器
    model = TransH(num_entities, num_relations, embedding_dim).to(device)  # 将模型移到指定设备
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MarginRankingLoss(margin=margin)

    # 数据加载
    train_loader = DataLoader(FB15kDataset(train_data), batch_size=batch_size, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            head, tail, relation = batch
            # 将数据移到指定设备
            head, tail, relation = head.to(device), tail.to(device), relation.to(device)

            # 正样本
            pos_dist = model(head, relation, tail)

            # 负样本
            neg_head = torch.randint(0, num_entities, head.size()).to(device)
            neg_tail = torch.randint(0, num_entities, tail.size()).to(device)
            neg_dist = model(neg_head, relation, neg_tail)

            # 损失计算
            target = torch.tensor([-1], dtype=torch.float, device=head.device)
            loss = loss_fn(pos_dist, neg_dist, target)
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model

# -------------------------
# Step 4: 保存嵌入向量
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
    train_path = "/data_disk/libh/KRL/WN18/train1.txt"
    entity_embeddings_path = "entity_embeddings.txt"
    relation_embeddings_path = "relation_embeddings.txt"

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
    save_embeddings(entity_embeddings_path, model.entity_embeddings.weight.cpu().detach(), {v: k for k, v in entity2id.items()})
    save_embeddings(relation_embeddings_path, model.relation_embeddings.weight.cpu().detach(), {v: k for k, v in relation2id.items()})
