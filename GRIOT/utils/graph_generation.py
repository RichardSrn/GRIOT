from itertools import combinations, groupby

from tqdm import tqdm
import networkx as nx
import numpy as np
import torch

from .MissingDataOT_network_utils import MAR_mask, MNAR_mask_logistic, MNAR_mask_quantiles, MNAR_self_mask_logistic

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
                       **kwargs):
    '''
    Function to generate a graph following the SBM model
    numb_groups : number of communities
    sizes: number of nodes in each community
    prob: [prob intra, prob extra]
    seed : random seed
    feat_dep : is the number of features of each type (multinomial, binary, continuous) dependent of the community
    feat_indep : is the number of features of each type (multinomial, binary, continuous) independent of the community
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

    # Generation of binary features
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

    # Generation of binary features
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


# Function produce_NA for generating missing values ------------------------------------------------------
# Taken from Muzellec

def produce_NA(X,
               p_miss,
               preserve_community_features: bool = False,
               mecha="MCAR",
               opt=None,
               p_obs=None,
               q=None,
               seed=None,
               max_missing=0):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    max_missing : maximum number of missing features per sample
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    if preserve_community_features:
        communities = X[:, 0]
        X = X[:, 1:]
    else:
        communities = None

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs, seed=seed).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs, seed=seed).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs, seed=seed).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss, seed=seed).double()
    else:  # MCAR
        if seed:
            torch.manual_seed(seed)
        pre_mask = torch.rand(X.shape)
        if max_missing > 0:
            max_by_row = torch.max(torch.topk(pre_mask, max_missing, dim=1, largest=False)[0][:, -1][:, None], dim=1)
            pre_mask[pre_mask > max_by_row[0][:, None]] = 1
        mask = (pre_mask < p_miss).double()

    # concatenate back the communities column to the X array
    if preserve_community_features:
        X = torch.cat((communities.unsqueeze(1), X), dim=1)
        mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim=1)

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}


def add_missing_features(F_true, p_miss, seed=42, mecha="MCAR", max_missing=0, return_all=False, verbose=False,
                         logging=None):
    if verbose:
        m = f"Generating mask and missing features : {mecha}, " \
            f"missing {p_miss * 100}%" \
            f"{f', max {max_missing}' if max_missing else ''}"
        if logging:
            logging.info(m)
        else:
            print(m)

    F_miss = produce_NA(F_true,
                        p_miss=p_miss,
                        mecha=mecha,
                        seed=seed,
                        preserve_community_features=True,
                        max_missing=max_missing)
    if return_all:
        return F_miss
    else:
        return F_miss['X_incomp']


