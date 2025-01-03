import json
eval_path = "推理出来的"
ground_truth_path = "原测试集的"

eval_dict = json.load(open(eval_path))
label_dict = json.load(open(ground_truth_path))

for task in ['link_prediction', 'entity_prediction']:
    evals = eval_dict[task]
    labels = label_dict[task]
    total_scores = 0
    for eval, label in zip(evals, labels):
        eval_output = eval["output"]
        label_truth = set(label["ground_truth"])
        for idx, output in enumerate(eval_output):
            if output in label_truth:
                total_scores += 1.0/ (idx+1)

    print(task, total_scores/len(evals))

