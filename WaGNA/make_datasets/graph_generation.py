from itertools import combinations, groupby
import random
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch


def gnp_random_connected_graph(n, p, seed=None):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    if seed:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in tqdm(groupby(edges, key=lambda x: x[0])):
        node_edges = list(node_edges)
        random_edge = rng.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if rng.random() < p:
                G.add_edge(*e)

    i = 0
    for v in tqdm(G.nodes):
        G.nodes[v]["name"], i = i, i + 1
        G.nodes[v]["state"] = "alive"
        G.nodes[v]["size"] = max(rng.normal(500, 200), 0.1)

    return G


def mcar(G, p, na_value=torch.nan, seed=None):
    if seed:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    G = nx.Graph(G)
    for v in tqdm(G.nodes):
        if rng.random() < p:
            G.nodes[v]["state"] = "dead"
            G.nodes[v]["size"] = na_value
    return G


def generate_sbm_graph(sizes=(50, 50, 50, 50),
                       prob=(0.3, 0.025),
                       seed=None,
                       feat_dep=(2, 2, 2),
                       feat_indep=(2, 2, 2),
                       main_component_ony=True,
                       **kwargs):
    '''
    Function to generate a graph following the SBM model
    numb_groups : number of communities
    sizes: number of nodes in each community
    prob: [prob intra, prob extra]
    seed : random seed
    feat_dep : is the number of features of each type (multinomial, integer, continuous) dependent of the community
    feat_indep : is the number of features of each type (multinomial, integer, continuous) independent of the community
    '''

    # if n%num_groups == 0:
    #    sizes = (np.ones(num_groups)*n/num_groups).astype(int)
    # else:
    #    sizes = (int(np.round(np.ones(num_groups)*n/num_groups))).astype(int)
    #    n = np.sum(sizes)
    num_groups = len(sizes)

    if seed:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    n = np.sum(sizes)
    probs = np.ones((num_groups, num_groups)) * prob[1]
    np.fill_diagonal(probs, prob[0])

    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    if main_component_ony:
        graph_components = sorted(nx.connected_components(g), key=len, reverse=True)
        g = g.subgraph(graph_components[0])

        communities = g.graph['partition']
        # remove from communities the nodes that are not in the main component
        for i in range(len(communities)):
            communities[i] = [x for x in communities[i] if x in g.nodes]
        # renumber the nodes of g to have a continuous numbering
        # begin by making a correspondence dictionary between old numbering and new numbering
        # then use the dictionary to renumber the nodes in g and in communities
        correspondence = {}
        for i, node in enumerate(g.nodes):
            correspondence[node] = i
        g = nx.relabel_nodes(g, correspondence)
        for i in range(len(communities)):
            communities[i] = [correspondence[x] for x in communities[i]]
        g.graph['partition'] = communities

        # adjust the constants
        sizes = [len(c) for c in communities]
        n = len(g.nodes)


    # Create p/2 features dependent on the community
    comm = np.zeros(n)
    part_idx = g.graph['partition']
    for i in range(len(part_idx)):
        comm[list(part_idx[i])] = i

    # Shuffle x% of the protected attributes
    X_dep = comm[None, :]

    # Generate multinomials features
    multi, binay, continuous = [], [], []

    # Nombre de modalité pour la multinomiale (on peut le passer en aléatoire)
    k = 3

    # Generation of multinomial features
    for i in range(feat_dep[0]):
        multi = []
        for j in range(num_groups):
            size = sizes[j]
            multi.append(np.where(rng.multinomial(1, rng.dirichlet(np.ones(k), size=1)[0], size) == 1)[1])
        multi = np.concatenate(multi, axis=0)
        X_dep = np.vstack((X_dep, multi))

    # Generation of integer features
    for i in range(feat_dep[1]):
        binary = []
        for j in range(num_groups):
            size = sizes[j]
            binary.append(rng.binomial(1, rng.dirichlet(np.ones(2), size=1)[0][0], size))
        binary = np.concatenate(binary, axis=0)
        X_dep = np.vstack((X_dep, binary))

    # Generation of continious features
    for i in range(feat_dep[2]):
        cont = []
        for j in range(num_groups):
            size = sizes[j]
            cont.append(rng.normal(rng.random(1), 0.2, size))
        cont = np.concatenate(cont, axis=0)
        X_dep = np.vstack((X_dep, cont))

    # We can do the same to generate features not correlated to the community
    X_indep = np.empty(shape=(sum(sizes)))
    # Generation of multinomial features
    for i in range(feat_indep[0]):
        multi = []
        size = sum(sizes)
        multi.append(np.where(rng.multinomial(1, rng.dirichlet(np.ones(k), size=1)[0], size) == 1)[1])
        multi = np.concatenate(multi, axis=0)
        X_indep = np.vstack((X_indep, multi))

    # Generation of integer features
    for i in range(feat_indep[1]):
        binary = []
        size = sum(sizes)
        binary.append(rng.binomial(1, rng.dirichlet(np.ones(2), size=1)[0][0], size))
        binary = np.concatenate(binary, axis=0)
        X_indep = np.vstack((X_indep, binary))

    # Generation of continious features
    for i in range(feat_indep[2]):
        cont = []
        size = sum(sizes)
        cont.append(rng.normal(rng.random(1), 0.2, size))
        cont = np.concatenate(cont, axis=0)
        X_indep = np.vstack((X_indep, cont))

    # X_indep = np.where(rng.multinomial(1, rng.dirichlet(np.ones(k), size=1)[0], n) == 1)[1]

    ### PREVIOUS
    # X_dep = comm
    # for i in np.linspace(0.1,1,9):
    #    _temp = shuffle_part(comm, prop_shuffle=i)
    #    X_dep = np.vstack((X_dep, _temp))
    if sum(feat_indep) > 0:
        F = np.vstack((X_dep, X_indep[1:]))
    else:
        F = np.vstack((X_dep))
    F = torch.from_numpy(F.T)
    return g, F


def shuffle_part(prot_s, prop_shuffle=0.1, seed=None):
    """
    Randomly shuffle some protected attributes
    :param prot_s: the vector to shuffle
    :param prop_shuffle: the proportion of label to shuffle
    :return: the shuffled vector
    """
    if seed:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    prop_shuffle = prop_shuffle
    ix = rng.choice([True, False], size=prot_s.size, replace=True,
                    p=[prop_shuffle, 1 - prop_shuffle])
    prot_s_shuffle = prot_s[ix]
    rng.shuffle(prot_s_shuffle)
    prot_s[ix] = prot_s_shuffle
    return prot_s


if __name__ == "__main__":
    G, F = generate_sbm_graph(sizes=(4, 4),
                              num_groups=2,
                              prob=(0.4, 0.05),
                              seed=42,
                              feat_dep=(0, 0, 0),
                              feat_indep=(0, 0, 0))

    for i, f in enumerate(F):
        if i % 5 == 0:
            print()
        print("[", end="")
        for e in f:
            if e == int(e):
                print(str(int(e)).rjust(5), end=",")
            else:
                print(f"{e:.5f}".rjust(10), end=",")
        print("]")
