import json

# 文件路径
entity_predictions_path = "entity_prediction_results.json"
link_predictions_path = "link_prediction_results.json"

def calculate_hit_score(predictions, task_name):
    """计算Hit Score"""
    total_scores = 0  # 累积分数
    num_samples = len(predictions)  # 样本总数

    for item in predictions:
        if task_name == "link_prediction":
            true_values = set([item["true_relation"]])  # 真实关系
            predicted_values = item["predicted_relations"]  # 预测的关系列表
        elif task_name == "entity_prediction":
            true_values = set([item["true_tail"]])  # 真实尾实体
            predicted_values = item["predicted_tails"]  # 预测的实体列表
        else:
            raise ValueError("任务名称不正确，只支持 link_prediction 和 entity_prediction")

        # 计算分数
        for rank, prediction in enumerate(predicted_values):
            if prediction in true_values:
                total_scores += 1 / (rank + 1)
                break  # 只计入第一个命中的排名

    # 平均分数
    average_score = total_scores / num_samples if num_samples > 0 else 0
    return average_score

def main():
    # 加载预测结果
    with open(link_predictions_path, "r") as f:
        link_predictions = json.load(f)

    with open(entity_predictions_path, "r") as f:
        entity_predictions = json.load(f)

    # 评估任务
    link_score = calculate_hit_score(link_predictions, "link_prediction")
    entity_score = calculate_hit_score(entity_predictions, "entity_prediction")

    # 输出结果
    print(f"Link Prediction Average Hit Score: {link_score:.12f}")
    print(f"Entity Prediction Average Hit Score: {entity_score:.12f}")

if __name__ == "__main__":
    main()
