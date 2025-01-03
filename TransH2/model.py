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

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list


class TransH:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1
        self.loss = 0
    def emb_initialize(self):
        relation_dict = {}
        entity_dict = {}
        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)
        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)
            relation_dict[relation + "_proj"] = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                                                   6 / math.sqrt(self.embedding_dim),
                                                                   self.embedding_dim)
        self.relation = relation_dict
        self.entity = entity_dict
    def project_on_hyperplane(self, entity_vector, relation_vector):
        projection = entity_vector - np.dot(entity_vector, relation_vector) * relation_vector
        return projection
    def distanceL2(self, h, r, t):
        h_proj = self.project_on_hyperplane(h, r)
        t_proj = self.project_on_hyperplane(t, r)
        return np.sum(np.square(h_proj + r - t_proj))
    def distanceL1(self, h, r, t):
        h_proj = self.project_on_hyperplane(h, r)
        t_proj = self.project_on_hyperplane(t, r)
        return np.sum(np.fabs(h_proj + r - t_proj))
    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)
        for triple, corrupted_triple in Tbatch:
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]
            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]
            h_correct_proj = self.project_on_hyperplane(h_correct, self.relation[triple[2] + "_proj"])
            t_correct_proj = self.project_on_hyperplane(t_correct, self.relation[triple[2] + "_proj"])
            h_corrupt_proj = self.project_on_hyperplane(h_corrupt, self.relation[corrupted_triple[2] + "_proj"])
            t_corrupt_proj = self.project_on_hyperplane(t_corrupt, self.relation[corrupted_triple[2] + "_proj"])
            if self.L1:
                dist_correct = self.distanceL1(h_correct_proj, relation, t_correct_proj)
                dist_corrupt = self.distanceL1(h_corrupt_proj, relation, t_corrupt_proj)
            else:
                dist_correct = self.distanceL2(h_correct_proj, relation, t_correct_proj)
                dist_corrupt = self.distanceL2(h_corrupt_proj, relation, t_corrupt_proj)

            err = self.hinge_loss(dist_correct, dist_corrupt)
            if err > 0:
                self.loss += err
                grad_pos = 2 * (h_correct_proj + relation - t_correct_proj)
                grad_neg = 2 * (h_corrupt_proj + relation - t_corrupt_proj)
                if self.L1:
                    for i in range(len(grad_pos)):
                        grad_pos[i] = 1 if grad_pos[i] > 0 else -1
                    for i in range(len(grad_neg)):
                        grad_neg[i] = 1 if grad_neg[i] > 0 else -1
                h_correct_update = copy_entity[triple[0]] - self.learning_rate * grad_pos
                t_correct_update = copy_entity[triple[1]] - (-1) * self.learning_rate * grad_pos
                h_corrupt_update = copy_entity[corrupted_triple[0]] - (-1) * self.learning_rate * grad_neg
                t_corrupt_update = copy_entity[corrupted_triple[1]] - self.learning_rate * grad_neg
                relation_update = copy_relation[triple[2]] - self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg
                projection_update = copy_relation[triple[2] + "_proj"] - self.learning_rate * grad_pos
                projection_update -= (-1) * self.learning_rate * grad_neg
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)
    def train(self, epochs):
        nbatches = 800
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0
            Sbatch = random.sample(self.triple_list, batch_size)
            Tbatch = []
            for triple in Sbatch:
                corrupted_triple = self.Corrupt(triple)
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))
            self.update_embeddings(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)
            loss_ls.append(self.loss)
            if epoch % 20 == 0:
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        if "_proj" not in r:
                            f_r.write(r + "\t")
                            f_r.write(str(list(self.relation[r])))
                            f_r.write("\n")
        print("写入文件...")
        with codecs.open("entity_50dim", "w") as f1:
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")
        with codecs.open("relation_50dim", "w") as f2:
            for r in self.relation.keys():
                if "_proj" not in r:
                    f2.write(r + "\t")
                    f2.write(str(list(self.relation[r])))
                    f2.write("\n")
        with codecs.open("loss", "w") as f3:
            f3.write(str(loss_ls))
        print("写入完成")
    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        if seed > 0.5:
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(list(self.entity.keys()), 1)[0]
            corrupted_triple[0] = rand_head
        else:
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(list(self.entity.keys()), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple
def main():
    file1 = "/data_disk/libh/KRL/FB15k/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=500, learning_rate=0.001, margin=1, L1=True)
    transH.emb_initialize()
    transH.train(epochs=100)

if __name__ == '__main__':
    main()
