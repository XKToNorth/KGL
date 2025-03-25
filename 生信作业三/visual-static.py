import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_blobs
from scipy.spatial import distance_matrix

# 生成一组随机点
num_nodes = 50
X, _ = make_blobs(n_samples=num_nodes, centers=3, cluster_std=1.0, random_state=42)

# 计算距离矩阵
dist_matrix = distance_matrix(X, X)


# 可视化函数，增强可视化效果
def plot_graph(G, title, pos, edge_weights=None, edge_styles=None):
    plt.figure(figsize=(8, 8))

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300, edgecolors='black')

    # 绘制边
    for (i, j), style in edge_styles.items():
        nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], style=style, edge_color='gray', width=1.5)

    # 绘制带权重的边标签
    if edge_weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8, label_pos=0.3)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title, fontsize=14)
    plt.show()


# 计算节点布局
pos = nx.spring_layout(nx.complete_graph(num_nodes), seed=42)

# 1. ε-近邻图 (epsilon-nearest neighbor graph)
epsilon = 1.5
G_epsilon = nx.Graph()
edge_styles = {}
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if dist_matrix[i, j] < epsilon:
            G_epsilon.add_edge(i, j)
            edge_styles[(i, j)] = 'dashed'  # ε-近邻边用虚线表示
plot_graph(G_epsilon, "ε-nearest neighbor graph", pos, edge_styles=edge_styles)

# 2. k-近邻图 (k-nearest neighbor graph)
k = 3
knn_graph = kneighbors_graph(X, k, mode='connectivity', include_self=False)
G_knn = nx.from_scipy_sparse_array(knn_graph)  # 兼容新版 networkx
edge_styles = {edge: 'solid' for edge in G_knn.edges()}  # k-近邻边用实线表示
plot_graph(G_knn, "k-nearest neighbor graph", pos, edge_styles=edge_styles)

# 3. 全连通图 (Fully connected graph)
G_fully = nx.Graph()
edge_weights = {}
edge_styles = {}
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        G_fully.add_edge(i, j, weight=round(dist_matrix[i, j], 2))
        edge_weights[(i, j)] = round(dist_matrix[i, j], 2)
        edge_styles[(i, j)] = 'solid'  # 全连通图边用实线表示
plot_graph(G_fully, "Fully connected graph", pos, edge_weights=edge_weights, edge_styles=edge_styles)
