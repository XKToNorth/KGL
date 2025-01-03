import torch
import numpy as np
import json
from torch.nn.functional import cosine_similarity

# 设备选择
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -------------------------
# Step 1: 加载嵌入向量
# -------------------------
def load_embeddings(file_path):
    """加载嵌入向量文件"""
    embeddings = []
    id_to_name = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            entity_or_relation = parts[0]
            embedding = np.array(eval(parts[1]))
            embeddings.append(torch.tensor(embedding, device=device))
            id_to_name[idx] = entity_or_relation
    return torch.stack(embeddings), id_to_name

# -------------------------
# Step 2: 加载测试集
# -------------------------
def load_test_data(file_path, entity_name_to_id, relation_name_to_id):
    """加载测试集文件并过滤"""
    test_data = []
    with open(file_path, 'r') as f:
        for line in f:
            head, tail, relation = line.strip().split()
            if head not in entity_name_to_id or tail not in entity_name_to_id or relation not in relation_name_to_id:
                continue
            test_data.append((
                entity_name_to_id[head],
                entity_name_to_id[tail],
                relation_name_to_id[relation]
            ))
    return test_data

# -------------------------
# Step 3: Link Prediction
# -------------------------
def link_prediction(head_id, tail_id, entity_embeddings, relation_embeddings):
    """给定头实体和尾实体ID，预测最可能的关系"""
    head_emb = entity_embeddings[head_id]
    tail_emb = entity_embeddings[tail_id]

    distances = torch.norm(relation_embeddings + head_emb - tail_emb, p=2, dim=1)
    top_5_ids = torch.topk(-distances, k=5).indices.tolist()
    return top_5_ids

# -------------------------
# Step 4: Entity Prediction
# -------------------------
def entity_prediction(head_id, relation_id, entity_embeddings, relation_embeddings):
    """给定头实体和关系ID，预测最可能的尾实体"""
    head_emb = entity_embeddings[head_id]
    relation_emb = relation_embeddings[relation_id]

    distances = torch.norm(entity_embeddings + head_emb + relation_emb, p=2, dim=1)
    top_5_ids = torch.topk(-distances, k=5).indices.tolist()
    return top_5_ids

# -------------------------
# Step 5: 生成预测结果
# -------------------------
def generate_predictions(test_data, entity_embeddings, relation_embeddings, id_to_entity, id_to_relation):
    link_predictions = []
    entity_predictions = []

    for head_id, tail_id, relation_id in test_data:
        # Link Prediction
        link_pred_ids = link_prediction(head_id, tail_id, entity_embeddings, relation_embeddings)
        link_preds = [id_to_relation[rel_id] for rel_id in link_pred_ids]
        link_predictions.append({
            "head": id_to_entity[head_id],
            "tail": id_to_entity[tail_id],
            "true_relation": id_to_relation[relation_id],
            "predicted_relations": link_preds
        })

        # Entity Prediction
        entity_pred_ids = entity_prediction(head_id, relation_id, entity_embeddings, relation_embeddings)
        entity_preds = [id_to_entity[ent_id] for ent_id in entity_pred_ids]
        entity_predictions.append({
            "head": id_to_entity[head_id],
            "relation": id_to_relation[relation_id],
            "true_tail": id_to_entity[tail_id],
            "predicted_entities": entity_preds
        })

    # 保存预测结果
    with open("link_predictions.json", "w") as f:
        json.dump(link_predictions, f, indent=4, ensure_ascii=False)

    with open("entity_predictions.json", "w") as f:
        json.dump(entity_predictions, f, indent=4, ensure_ascii=False)

# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    # 文件路径
    entity_embeddings_path = "entity_embeddings.txt"
    relation_embeddings_path = "relation_embeddings.txt"
    test_path = "/data_disk/libh/KRL/FB15k/test.txt"

    # 加载嵌入向量
    entity_embeddings, id_to_entity = load_embeddings(entity_embeddings_path)
    relation_embeddings, id_to_relation = load_embeddings(relation_embeddings_path)

    # 创建名称到ID的映射
    entity_name_to_id = {name: idx for idx, name in id_to_entity.items()}
    relation_name_to_id = {name: idx for idx, name in id_to_relation.items()}

    # 加载测试数据并过滤
    test_data = load_test_data(test_path, entity_name_to_id, relation_name_to_id)

    # 生成预测结果
    generate_predictions(test_data, entity_embeddings, relation_embeddings, id_to_entity, id_to_relation)

