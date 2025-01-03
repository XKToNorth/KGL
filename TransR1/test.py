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
    embeddings = []
    id_to_name = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            entity_or_relation = parts[0]
            embedding = np.array(eval(parts[1]), dtype=np.float32)
            embeddings.append(torch.tensor(embedding, dtype=torch.float32, device=device))
            id_to_name[idx] = entity_or_relation
    return torch.stack(embeddings), id_to_name


def load_projection_matrices(file_path):
    matrices = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            _, matrix_str = parts
            matrix = torch.tensor(eval(matrix_str), dtype=torch.float32, device=device)
            matrices.append(matrix)
    return torch.stack(matrices)


# -------------------------
# Step 2: 加载测试集
# -------------------------
def load_test_data(file_path, entity_name_to_id, relation_name_to_id):
    test_data = []
    with open(file_path, 'r') as f:
        for line in f:
            head, tail, relation = line.strip().split()
            if head in entity_name_to_id and tail in entity_name_to_id and relation in relation_name_to_id:
                test_data.append((
                    entity_name_to_id[head],
                    entity_name_to_id[tail],
                    relation_name_to_id[relation]
                ))
    return test_data


# -------------------------
# Step 3: Link Prediction
# -------------------------
def link_prediction_batch(head_ids, tail_ids, entity_embeddings, relation_embeddings, projection_matrices):
    head_embs = entity_embeddings[head_ids]
    tail_embs = entity_embeddings[tail_ids]
    distances = []

    for relation_id, proj_matrix in enumerate(projection_matrices):
        proj_matrix = proj_matrix.to(torch.float32)
        head_proj = torch.matmul(proj_matrix, head_embs.T)
        tail_proj = torch.matmul(proj_matrix, tail_embs.T)
        relation_emb = relation_embeddings[relation_id].to(torch.float32)

        dist = torch.norm(head_proj + relation_emb.unsqueeze(1) - tail_proj, p=2, dim=0)
        distances.append(dist)

    distances = torch.stack(distances)
    top_5_ids = torch.topk(-distances, k=5, dim=0).indices
    return top_5_ids


# -------------------------
# Step 4: Entity Prediction
# -------------------------
def entity_prediction_batch(head_ids, relation_ids, entity_embeddings, relation_embeddings, projection_matrices):
    """批量计算 Entity Prediction"""
    head_embs = entity_embeddings[head_ids]
    relation_embs = relation_embeddings[relation_ids]
    similarities = []

    for idx, proj_matrix in enumerate(projection_matrices[relation_ids]):
        proj_matrix = proj_matrix.to(torch.float32)

        # 确保 head_embs[idx] 是 2D 张量
        head_proj = torch.matmul(proj_matrix, head_embs[idx].unsqueeze(1))

        # 实体嵌入矩阵的转置
        tail_embs = entity_embeddings.T
        tail_proj = torch.matmul(proj_matrix, tail_embs)

        # 计算相似度
        similarity = -torch.norm(head_proj + relation_embs[idx].unsqueeze(1) - tail_proj, p=2, dim=0)
        similarities.append(similarity)

    similarities = torch.stack(similarities)
    top_5_ids = torch.topk(similarities, k=5, dim=1).indices
    return top_5_ids


# -------------------------
# Step 5: 生成预测结果
# -------------------------
def generate_predictions(test_data, entity_embeddings, relation_embeddings, id_to_entity, id_to_relation,
                         projection_matrices):
    link_predictions = []
    entity_predictions = []

    batch_size = 256
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        head_ids = [x[0] for x in batch]
        tail_ids = [x[1] for x in batch]
        relation_ids = [x[2] for x in batch]

        link_pred_ids = link_prediction_batch(head_ids, tail_ids, entity_embeddings, relation_embeddings, projection_matrices)
        for idx, (head_id, tail_id, relation_id) in enumerate(batch):
            try:
                link_preds = [id_to_relation[rel_id] for rel_id in link_pred_ids[:, idx]]
            except KeyError:
                link_preds = []
            link_predictions.append({
                "head": id_to_entity[head_id],
                "tail": id_to_entity[tail_id],
                "true_relation": id_to_relation.get(relation_id, "UNKNOWN"),
                "predicted_relations": link_preds
            })

        entity_pred_ids = entity_prediction_batch(head_ids, relation_ids, entity_embeddings, relation_embeddings, projection_matrices)
        for idx, (head_id, relation_id, tail_id) in enumerate(batch):
            try:
                entity_preds = [id_to_entity[ent_id] for ent_id in entity_pred_ids[idx]]
            except KeyError:
                entity_preds = []
            entity_predictions.append({
                "head": id_to_entity[head_id],
                "relation": id_to_relation.get(relation_id, "UNKNOWN"),
                "true_tail": id_to_entity[tail_id],
                "predicted_entities": entity_preds
            })

    with open("link_predictions.json", "w") as f:
        json.dump(link_predictions, f, indent=4, ensure_ascii=False)

    with open("entity_predictions.json", "w") as f:
        json.dump(entity_predictions, f, indent=4, ensure_ascii=False)


# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    entity_embeddings_path = "entity_embeddings.txt"
    relation_embeddings_path = "relation_embeddings.txt"
    projection_matrices_path = "projection_matrices.txt"
    test_path = "/data_disk/libh/KRL/FB15k/test.txt"

    entity_embeddings, id_to_entity = load_embeddings(entity_embeddings_path)
    relation_embeddings, id_to_relation = load_embeddings(relation_embeddings_path)
    projection_matrices = load_projection_matrices(projection_matrices_path)

    entity_name_to_id = {name: idx for idx, name in id_to_entity.items()}
    relation_name_to_id = {name: idx for idx, name in id_to_relation.items()}

    test_data = load_test_data(test_path, entity_name_to_id, relation_name_to_id)

    generate_predictions(test_data, entity_embeddings, relation_embeddings, id_to_entity, id_to_relation,
                         projection_matrices)
