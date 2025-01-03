import codecs
import copy
import math
import random
import time

import numpy as np

entity2id = {}
relation2id = {}
loss_ls = []

def data_loader(file):
    # 加载数据文件，train.txt, entity2id.txt, relation2id.txt
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []

    # 读取训练数据，存储头实体、关系、尾实体
    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]  # 头实体
            r_ = relation2id[triple[1]]  # 关系
            t_ = entity2id[triple[2]]  # 尾实体

            triple_list.append([h_, r_, t_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list

# L2范数计算函数（用于距离计算）
def distanceL2(h, r, t):
    return np.sum(np.square(h + r - t))

# L1范数计算函数（用于距离计算）
def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))

class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.learning_rate = learning_rate  # 学习率
        self.margin = margin  # 边际损失
        self.entity = entity_set  # 实体集合
        self.relation = relation_set  # 关系集合
        self.triple_list = triple_list  # 三元组列表
        self.L1 = L1  # 是否使用L1范数

        self.loss = 0

    def emb_initialize(self):
        # 初始化实体和关系的嵌入
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        nbatches = 400
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            # Sbatch: 从三元组中随机选择一个批次
            Sbatch = random.sample(self.triple_list, batch_size)
            Tbatch = []

            # 为每个三元组生成一个腐败三元组（head或tail替换）
            for triple in Sbatch:
                corrupted_triple = self.Corrupt(triple)
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))
            self.update_embeddings(Tbatch)

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)
            loss_ls.append(self.loss)

            # 每20个epoch保存一次临时结果
            if epoch % 20 == 0:
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        # 保存最终的嵌入结果和损失
        print("写入文件...")
        with codecs.open("entity_50dim", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("relation_50dim", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")

        with codecs.open("loss", "w") as f3:
            f3.write(str(loss_ls))

        print("写入完成")

    def Corrupt(self, triple):
        # 随机替换头实体或尾实体
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        if seed > 0.5:
            # 替换头实体
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0] = rand_head

        else:
            # 替换尾实体
            rand_tail = triple[2]  # 现在尾实体是triple[2]
            while rand_tail == triple[2]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[2] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            # 正确的头实体和尾实体
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[2]]  # 现在尾实体在triple[2]
            relation_update = copy_relation[triple[1]]

            # 错误的头实体和尾实体
            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[2]]  # 错误的尾实体

            # 原始三元组的向量
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[2]]
            relation = self.relation[triple[1]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[2]]

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                if triple[0] == corrupted_triple[0]:
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[2] == corrupted_triple[2]:  # 如果替换的是尾实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                relation_update -= self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg

        # 正规化实体和关系嵌入
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        # 计算hinge loss
        return max(0, dist_correct - dist_corrupt + self.margin)

if __name__ == '__main__':
    file1 = "/data_disk/libh/KRL/WN18/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("加载文件中...")
    print("加载完成。实体数量 : %d , 关系数量 : %d , 三元组数量 : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.001, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=200)
