"""
some help :
https://medium.com/@koki_noda/ultimate-guide-to-graph-neural-networks-2-texas-dataset-f70782190f80
"""

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from pprint import pprint
import torch
from torch import nn
from torch import Tensor
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import WebKB
import networkx as nx
from networkx.algorithms import community


def load_wkb_graph(path="", name="texas", main_component_=True, directed=False, draw_=False):
    data_path = os.path.join(path, "data")

    os.makedirs(path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    dataset = WebKB(root=data_path, name=name)
    data = dataset[0]

    label_dict = {
        0: "student",
        1: "project",
        2: "course",
        3: "staff",
        4: "faculty"
    }

    G = to_networkx(data, to_undirected=not directed)
    F = np.concatenate((data.y.reshape(-1, 1), data.x), axis=1)

    # cycle through all nodes (from Gnx) and add their community (from node_data) as attribute
    for node in G.nodes():
        G.nodes[node]["features"] = F[node]
    if main_component_:
        graph_components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(graph_components[0])
    # rename G's nodes to canonical integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)
    F = np.zeros((len(G.nodes()), F.shape[1]))
    for node in G.nodes():
        F[node] = G.nodes[node]["features"]
    # delete the 2nd column of F, node_id
    F = np.delete(F, 1, axis=1)
    # delete G.nodes["features"]
    for node in G.nodes():
        del G.nodes[node]["features"]
    F = torch.from_numpy(F).double()

    if draw_:
        node_color = []
        nodelist = [[], [], [], [], []]

        colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        colorlist = np.array(colorlist)
        labels = data.y
        for n, i in enumerate(labels):
            node_color.append(colorlist[i])
            nodelist[i].append(n)
        labellist = list(label_dict.values())
        pos = nx.spring_layout(G, seed=42)
        print(pos)
        plt.figure(figsize=(10, 10))
        for num, i in enumerate(zip(nodelist, labellist)):
            n, l = i[0], i[1]
            print(n)
            nx.draw_networkx_nodes(G, pos, nodelist=n, node_size=20, node_color=colorlist[num], label=l)
        nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        save_path = os.path.join(path, name + ".png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return G, F, label_dict
