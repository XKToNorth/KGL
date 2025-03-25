import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import NearestNeighbors
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
n_points = 20
points = np.random.rand(n_points, 2)
pos = {i: points[i] for i in range(n_points)}
epsilon = 0.25
k = 3
epsilon_edges = []
for i in range(n_points):
    for j in range(i+1, n_points):
        if np.linalg.norm(points[i] - points[j]) < epsilon:
            epsilon_edges.append((i, j))
epsilon_edges_sorted = epsilon_edges.copy()
G_epsilon = nx.Graph()
G_epsilon.add_nodes_from(range(n_points))
G_epsilon.add_edges_from(epsilon_edges)
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
distances, indices = nbrs.kneighbors(points)
k_edges = set()
for i in range(n_points):
    for j in indices[i][1:]:
        k_edges.add(tuple(sorted((i, j))))
k_edges = list(k_edges)
k_edges_sorted = k_edges.copy()
G_knn = nx.Graph()
G_knn.add_nodes_from(range(n_points))
G_knn.add_edges_from(k_edges)
complete_edges = []
complete_weights = {}
for i in range(n_points):
    for j in range(i+1, n_points):
        complete_edges.append((i, j))
        complete_weights[(i, j)] = np.random.rand()  # 随机相似度
complete_edges_sorted = complete_edges.copy()

G_complete = nx.Graph()
G_complete.add_nodes_from(range(n_points))
G_complete.add_edges_from(complete_edges)
nx.set_edge_attributes(G_complete, complete_weights, 'weight')

node_size = 100
node_color = 'steelblue'
interval = 300
fig_epsilon, ax_epsilon = plt.subplots()
ax_epsilon.set_title("ε-近邻图")
ax_epsilon.set_xlim(0, 1)
ax_epsilon.set_ylim(0, 1)
ax_epsilon.set_xticks([])
ax_epsilon.set_yticks([])

n_frames_epsilon = len(epsilon_edges_sorted) + 10

def update_epsilon(frame):
    ax_epsilon.clear()
    ax_epsilon.set_title("ε-近邻图")
    ax_epsilon.set_xlim(0, 1)
    ax_epsilon.set_ylim(0, 1)
    ax_epsilon.set_xticks([])
    ax_epsilon.set_yticks([])
    current_edges = epsilon_edges_sorted[:min(frame, len(epsilon_edges_sorted))]
    nx.draw_networkx_nodes(G_epsilon, pos, node_color=node_color, node_size=node_size, ax=ax_epsilon)
    nx.draw_networkx_edges(G_epsilon, pos, edgelist=current_edges, ax=ax_epsilon, edge_color='gray')
ani_epsilon = FuncAnimation(fig_epsilon, update_epsilon, frames=n_frames_epsilon, interval=interval, repeat=True)

# 2. k-近邻图的动画
fig_knn, ax_knn = plt.subplots()
ax_knn.set_title("k-近邻图")
ax_knn.set_xlim(0, 1)
ax_knn.set_ylim(0, 1)
ax_knn.set_xticks([])
ax_knn.set_yticks([])

n_frames_knn = len(k_edges_sorted) + 10

def update_knn(frame):
    ax_knn.clear()
    ax_knn.set_title("k-近邻图")
    ax_knn.set_xlim(0, 1)
    ax_knn.set_ylim(0, 1)
    ax_knn.set_xticks([])
    ax_knn.set_yticks([])
    current_edges = k_edges_sorted[:min(frame, len(k_edges_sorted))]
    nx.draw_networkx_nodes(G_knn, pos, node_color=node_color, node_size=node_size, ax=ax_knn)
    nx.draw_networkx_edges(G_knn, pos, edgelist=current_edges, ax=ax_knn, edge_color='gray')

ani_knn = FuncAnimation(fig_knn, update_knn, frames=n_frames_knn, interval=interval, repeat=True)

fig_complete, ax_complete = plt.subplots()
ax_complete.set_title("全连通图")
ax_complete.set_xlim(0, 1)
ax_complete.set_ylim(0, 1)
ax_complete.set_xticks([])
ax_complete.set_yticks([])

n_frames_complete = len(complete_edges_sorted) + 10

def update_complete(frame):
    ax_complete.clear()
    ax_complete.set_title("全连通图")
    ax_complete.set_xlim(0, 1)
    ax_complete.set_ylim(0, 1)
    ax_complete.set_xticks([])
    ax_complete.set_yticks([])
    current_edges = complete_edges_sorted[:min(frame, len(complete_edges_sorted))]
    nx.draw_networkx_nodes(G_complete, pos, node_color=node_color, node_size=node_size, ax=ax_complete)
    if current_edges:
        # 根据相似度映射边的颜色
        weights = [G_complete[u][v]['weight'] for u, v in current_edges]
        cmap = plt.cm.viridis
        colors_edge = [cmap(w) for w in weights]
        nx.draw_networkx_edges(G_complete, pos, edgelist=current_edges, ax=ax_complete, edge_color=colors_edge)
    else:
        nx.draw_networkx_edges(G_complete, pos, edgelist=[], ax=ax_complete)

ani_complete = FuncAnimation(fig_complete, update_complete, frames=n_frames_complete, interval=interval, repeat=True)

ani_epsilon.save("epsilon_animation.gif", writer="pillow", fps=10)
ani_knn.save("knn_animation.gif", writer="pillow", fps=10)
ani_complete.save("complete_animation.gif", writer="pillow", fps=10)

plt.show()
