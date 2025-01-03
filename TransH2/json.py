import json
import torch
import torch.nn.functional as F


# 加载映射文件
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            name, id_ = line.strip().split('\t')
            mapping[name] = int(id_)
    return mapping


# 加载嵌入文件
def load_embeddings(file_path, device):
    embeddings = {}
    with open(file_path, 'r') as f:
        for line in f:
            id_, vector = line.strip().split('\t')
            embeddings[int(id_)] = torch.tensor(json.loads(vector), dtype=torch.float32, device=device)
    return embeddings


# 计算得分
def compute_score_batch(heads, relations, tails, metric='L2'):
    if metric == 'L1':
        return torch.norm(heads + relations - tails, p=1, dim=1)
    return torch.norm(heads + relations - tails, p=2, dim=1)


# Link Prediction
def link_prediction(entity_embeddings, relation_embeddings, test_data, top_k=5, device="cuda"):
    predictions = []

    # 将实体和关系嵌入转换为 GPU 上的张量
    entity_ids = list(entity_embeddings.keys())
    relation_ids = list(relation_embeddings.keys())

    entity_tensor = torch.stack([entity_embeddings[e] for e in entity_ids])
    relation_tensor = torch.stack([relation_embeddings[r] for r in relation_ids])

    for head, tail, true_relation in test_data:
        head_emb = entity_embeddings[head]
        tail_emb = entity_embeddings[tail]

        # 扩展矩阵并计算得分
        head_batch = head_emb.repeat(len(relation_ids), 1)  # 扩展成 [len(relations), embedding_dim]
        tail_batch = tail_emb.repeat(len(relation_ids), 1)  # 扩展成 [len(relations), embedding_dim]
        relation_batch = relation_tensor  # 已经是 [len(relations), embedding_dim]

        scores = compute_score_batch(head_batch, relation_batch, tail_batch)
        sorted_indices = torch.argsort(scores)[:top_k]
        top_relations = [relation_ids[i] for i in sorted_indices]

        predictions.append({
            "head": head,
            "tail": tail,
            "true_relation": true_relation,
            "predicted_relations": top_relations
        })

    return predictions


# Entity Prediction
def entity_prediction(entity_embeddings, relation_embeddings, test_data, top_k=5, device="cuda"):
    predictions = []

    # 将实体和关系嵌入转换为 GPU 上的张量
    entity_ids = list(entity_embeddings.keys())
    relation_ids = list(relation_embeddings.keys())

    entity_tensor = torch.stack([entity_embeddings[e] for e in entity_ids])
    relation_tensor = torch.stack([relation_embeddings[r] for r in relation_ids])

    for head, relation, true_tail in test_data:
        head_emb = entity_embeddings[head]
        rel_emb = relation_embeddings[relation]

        # 扩展矩阵并计算得分
        head_batch = head_emb.repeat(len(entity_ids), 1)  # 扩展成 [len(entities), embedding_dim]
        rel_batch = rel_emb.repeat(len(entity_ids), 1)  # 扩展成 [len(entities), embedding_dim]
        entity_batch = entity_tensor  # 已经是 [len(entities), embedding_dim]

        scores = compute_score_batch(head_batch, rel_batch, entity_batch)
        sorted_indices = torch.argsort(scores)[:top_k]
        top_entities = [entity_ids[i] for i in sorted_indices]

        predictions.append({
            "head": head,
            "relation": relation,
            "true_tail": true_tail,
            "predicted_tails": top_entities
        })

    return predictions


# 保存为 JSON 文件
def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# 主函数
if __name__ == '__main__':
    # 路径配置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    entity2id_file = "/data_disk/libh/KRL/FB15k/entity2id.txt"
    relation2id_file = "/data_disk/libh/KRL/FB15k/relation2id.txt"
    entity_embedding_file = "entity_50dim"
    relation_embedding_file = "relation_50dim"
    test_data_file = "/data_disk/libh/KRL/FB15k/test.txt"

    # 加载映射
    print("Loading mappings...")
    entity2id = load_mapping(entity2id_file)
    relation2id = load_mapping(relation2id_file)

    # 加载嵌入
    print("Loading embeddings...")
    entity_embeddings = load_embeddings(entity_embedding_file, device)
    relation_embeddings = load_embeddings(relation_embedding_file, device)

    # 加载测试数据并转换
    print("Loading and converting test data...")
    with open(test_data_file, 'r') as f:
        test_data_raw = [line.strip().split('\t') for line in f.readlines()]
    test_data_lp = [
        (entity2id[head], entity2id[tail], relation2id[relation])
        for head, tail, relation in test_data_raw
    ]
    test_data_ep = [
        (entity2id[head], relation2id[relation], entity2id[tail])
        for head, tail, relation in test_data_raw
    ]

    # 执行任务
    print("Running link prediction...")
    link_pred_results = link_prediction(entity_embeddings, relation_embeddings, test_data_lp, device=device)
    save_to_json(link_pred_results, "link_prediction_results.json")

    print("Running entity prediction...")
    entity_pred_results = entity_prediction(entity_embeddings, relation_embeddings, test_data_ep, device=device)
    save_to_json(entity_pred_results, "entity_prediction_results.json")

    print("Results saved.")

