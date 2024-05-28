import math

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx
from torch_scatter import scatter_add

from .seeds import development_seed

DATA_PATH = 'data'

def get_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """ Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing"""
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()

def keep_only_largest_connected_component(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))
    
    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data

    return dataset

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper

def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def set_train_val_test_split(
        seed: int,
        data: Data,
        dataset_name: str,
        split_idx: int = None) -> Data:

    if dataset_name in ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "CoauthorCS", "CoauthorPhysics"]:
        # Use split from "Diffusion Improves Graph Learning" paper, which selects 20 nodes for each class to be in the training set
        num_val = 5000 if dataset_name == "CoauthorCS" else 1500
        data = set_per_class_train_val_test_split(seed=seed, data=data, num_val=num_val, num_train_per_class=20, split_idx=split_idx)
    elif dataset_name in ["OGBN-Arxiv", "OGBN-Products"]:
        # OGBN datasets have pre-assigned split
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['train_test']
    elif dataset_name in ["Twitch", "Deezer-Europe", "FB100", "Actor"]:
        # Datasets from "New Benchmarks for Learning on Non-Homophilous Graphs". They use uniform 50/25/25 split
        data = set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25)
    elif dataset_name == 'Syn-Cora':
        # Datasets from "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs". They use uniform 25/25/50 split
        data = set_uniform_train_val_test_split(seed, data, train_ratio=0.25, val_ratio=0.25)
    elif dataset_name == 'MixHopSynthetic':
        # Datasets from "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing". They use uniform 33/33/33 split
        data = set_uniform_train_val_test_split(seed, data, train_ratio=0.33, val_ratio=0.33)
    else:
        raise ValueError(f"We don't know how to split the data for {dataset_name}")

    return data

def set_per_class_train_val_test_split(
        seed: int,
        data: Data,
        num_val: int = 1500,
        num_train_per_class: int = 20,
        split_idx: int = None) -> Data:

    if split_idx is None: 
        rnd_state = np.random.RandomState(development_seed)
        num_nodes = data.y.shape[0]
        development_idx = rnd_state.choice(num_nodes, num_val, replace=False)
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(seed)
        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_train_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]

        data.train_mask = get_mask(train_idx, num_nodes)
        data.val_mask = get_mask(val_idx, num_nodes)
        data.test_mask = get_mask(test_idx, num_nodes)

    else:
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['train_test']

    return data

def set_uniform_train_val_test_split(
        seed: int,
        data: Data,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    
    # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)
    
    idxs = list(range(num_labeled_nodes))
    # Shuffle in place
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train:num_train+num_val]
    test_idx = idxs[num_train+num_val:]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    data.train_mask = get_mask(train_idx, num_nodes)
    data.val_mask = get_mask(val_idx, num_nodes)
    data.test_mask = get_mask(test_idx, num_nodes)

    # Set labels of unlabeled nodes to 0, otherwise there is an issue in label propagation (which does one-hot encoding of all labels)
    # This labels are not used since these nodes are excluded from all masks, do it doesn't affect any results
    data.y[data.y == -1] = 0

    return data

def get_mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return DAD
     
def get_adj_row_sum(edge_index, edge_weight, n_nodes):
    """
    Get weighted out degree for nodes. This is equivalent to computing the sum of the rows of the weighted adjacency matrix.
    """
    row = edge_index[0]
    return scatter_add(edge_weight, row, dim=0, dim_size=n_nodes)

def get_adj_col_sum(edge_index, edge_weight, n_nodes):
    """
    Get weighted in degree for nodes. This is equivalent to computing the sum of the columns of the weighted adjacency matrix.
    """
    col = edge_index[1]
    return scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)

def row_normalize(edge_index, edge_weight, n_nodes):
    row_sum = get_adj_row_sum(edge_index, edge_weight, n_nodes)
    row_idx = edge_index[0]
    return edge_weight / row_sum[row_idx]

def col_normalize(edge_index, edge_weight, n_nodes):
    col_sum = get_adj_col_sum(edge_index, edge_weight, n_nodes)
    col_idx = edge_index[1]
    return edge_weight / col_sum[col_idx]

def compute_laplacian_eigenvectors(data, normalized_laplacian=True):
    G = to_networkx(data).to_undirected()
    L = nx.normalized_laplacian_matrix(G, list(range(G.number_of_nodes()))) if normalized_laplacian else nx.laplacian_matrix(G)
    
    return np.linalg.eigh(L.todense())

def get_random_graph(n_nodes=20, m=2, n_features=10, k=5):
    G = nx.barabasi_albert_graph(n_nodes, m)
    L = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(G).todense()
    _, eigenvectors = np.linalg.eigh(L)
    
    # Signal is linear combination of first k Laplacian eigenvectors (with random coeffiecients)
    x = torch.Tensor(normalize(eigenvectors[:, :k] @ np.random.uniform(size=(k, n_features)), axis=0))
    
    data = from_networkx(G)
    data.x = x
    
    return data

def compute_neighbor_correlation(edge_index, x):
    # Compute row-normalized adjacency
    edge_weight = torch.ones(edge_index.shape[1])
    edge_weight = row_normalize(edge_index=edge_index, edge_weight=edge_weight, n_nodes=x.shape[0])
    adj_row_norm = torch.sparse.FloatTensor(edge_index, values=edge_weight)
    
    neighbors_mean_x = torch.sparse.mm(adj_row_norm, x)
    # Compute dot product between each node and the mean of its neighbors
    # (which is equal to the mean of the dot product of a node with its neighbors)
    dot_product = torch.sum(x * neighbors_mean_x, axis=1)
    # Compute correlation by dividing by product of stds
    # TODO: This is actually computing dot-product, we need to change it to compute correlation
    correlation = dot_product
    
    # Take the mean over all the nodes
    return torch.mean(correlation).item()

    