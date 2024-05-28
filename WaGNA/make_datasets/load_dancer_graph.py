import os.path
import networkx as nx
import numpy as np
import torch


def process_features(features: list):
    features_processed = []
    for f in features:
        node_idx, feats, commu = f.split(";")
        node_idx, commu = int(node_idx), int(commu)
        feats = feats.split("|")
        feats, commu = [float(f1) for f1 in feats], commu
        features_processed.append([commu, *feats])
    features_processed = np.array(features_processed)

    return features_processed


def process_edges(edges: list) -> list:
    edges_processed = []
    for e in edges:
        v0, v1 = e.split(";")
        edges_processed.append([int(v0), int(v1)])
    return edges_processed


def load_dancer_graph(path="", **kwargs):
    if not path.endswith(".graph"):
        path = os.path.join(path, "t0.graph")
    with open(path, "r") as file:
        file = file.read().replace("#\n", "").split("\n")
        loc_vertices = file.index("# Vertices")
        loc_edges = file.index("# Edges")
        features = file[loc_vertices + 1:loc_edges]
        edges = file[loc_edges + 1:-1]

    F = process_features(features)
    edges = process_edges(edges)

    F = torch.from_numpy(F)
    G = nx.from_edgelist(edges)

    # align G and its features
    for node in G.nodes():
        G.nodes[node]["features"] = F[node]

    # BEGIN FIX - networkx doesn't keep the order of the nodes
    ## convert G nodes to index and rename them to be sorted
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    sorting_dict = dict()
    for i, node in enumerate(G.nodes):
        sorting_dict[node] = i
    G = nx.relabel_nodes(G, sorting_dict)
    # END FIX

    # sort and rename nodes of G to have communities sorted
    # get the sorting index for the nodes such as having communities sorted
    sorting_dict = []
    for node in G.nodes():
        sorting_dict.append({
            "node" : node,
            "community" : G.nodes[node]["features"][0]
        })
    sorting_dict = sorted(sorting_dict, key=lambda x: x["community"])
    sorting_dict = {x["node"]: i for i,x in enumerate(sorting_dict)}
    G = nx.relabel_nodes(G, sorting_dict)
    G_tmp = nx.Graph()
    G_tmp.add_nodes_from(sorted(G.nodes(data=True)))
    G_tmp.add_edges_from(G.edges(data=True))
    G = G_tmp
    del G_tmp

    F = torch.zeros((len(G.nodes()), F.shape[1]))
    for node in G.nodes():
        F[node] = G.nodes[node]["features"]

    return G, F
