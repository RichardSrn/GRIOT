from ..make_datasets.load_dancer_graph import load_dancer_graph
from ..make_datasets.load_cora_graph import load_cora_graph
from ..utils.graph_measures import GraphMeasures
from ..make_datasets.graph_generation import generate_sbm_graph
from ..utils.MissingDataOT_network_utils import MAR_mask, MNAR_mask_logistic, MNAR_mask_quantiles, MNAR_self_mask_logistic
from ..config import cfg

from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import networkx.exception
import numpy as np
import torch
import scipy
import os
import logging
import re

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def get_proximity_matrix(G, show=False, figsize=(7, 7), save=None, title=None, directed=False):
    n = len(G.nodes)
    distance_matrix = torch.zeros((n, n))

    if not directed :
        for i in tqdm(range(n), desc="compute proximity matrix", position=0, leave=True):
            for j in range(n):
                try:
                    l = nx.shortest_path_length(G, i, j)
                except networkx.exception.NetworkXNoPath:
                    l = np.infty
                distance_matrix[i, j] = l
    else:
        for i in tqdm(range(n), desc="compute directed proximity matrix", position=0, leave=True):
            for j in range(i + 1, n):
                try:
                    l = nx.shortest_path_length(G, i, j)
                except networkx.exception.NetworkXNoPath:
                    l = np.infty
                distance_matrix[i, j] = l
                distance_matrix[j, i] = l
    if show or save:
        fig, ax = plt.subplots(figsize=figsize)

        cax = fig.add_axes([0.9, 0.15, 0.02, 0.70])
        im = ax.matshow(distance_matrix)  # , cmap="autumn")
        fig.colorbar(im, cax=cax, orientation="vertical")
        if title:
            title = title.split(" ")
            title_tmp = [title[0]]
            for t in title[1:]:
                if len(title_tmp[-1]) < 50:
                    title_tmp[-1] += " " + t
                else:
                    title_tmp.append(t)
            title = "\n".join(title_tmp)
            ax.set_title(title)
        if save:
            logging.info(f"saved : {save}")
            plt.savefig(save)
        if show:
            plt.show()
        else:
            plt.close()
    return distance_matrix


def add_dependent_features(F, added_features):
    """
    Add features to the features matrix F.
    These new features are dependent to already drawn features (possibly dependent to the graph's structure).
    :param F: feature matrix
    :added_features: list of factors for linear combinations of features.
    :return: the new feature matrix F with added features
    """
    # we check that there are as many element in the linear combination than the number of features +1 for origin shift.
    assert [len(af) == F.shape[1] for af in added_features]
    if type(F) == torch.Tensor:
        F_tmp = F.clone()
    elif type(F) == np.ndarray:
        F_tmp = F.copy()
    else:
        raise ValueError(f"Type of F is not supported : {type(F)}")
    F_tmp[:, 0] = 1
    # create linear combinations of existing features
    new_features = []
    for af in added_features:
        if type(F) == torch.Tensor:
            coefs = torch.tensor(af).double()
            new_features.append(F_tmp.matmul(coefs))
        else:  # type(F) == np.ndarray:
            coefs = np.array(af)
            new_features.append(F_tmp.dot(coefs))
    # concatenate the already existing features with the new ones
    if type(F) == torch.Tensor:
        F = torch.cat([F, torch.stack(new_features).T], dim=1)
    else:  # type(F) == np.ndarray:
        F = np.concatenate([F, np.array(new_features).T], axis=1)
    return F


def get_graph_n_co(graph_parameters=None, dist_save=False, measure_save=True, measure_show=False, UNIQUE="."):
    if 'draw_' in graph_parameters.keys():
        draw_ = graph_parameters['draw_']
    else:
        draw_ = False

    seed = graph_parameters["seed"]

    if "CLASSIC" in graph_parameters["name"]:
        G, F_true = generate_sbm_graph(sizes=graph_parameters["sizes"],
                                       num_groups=len(graph_parameters["sizes"]),
                                       prob=graph_parameters["prob"],
                                       seed=seed,
                                       feat_dep=graph_parameters["feat_dep"],
                                       feat_indep=graph_parameters["feat_indep"])
    elif "DANCER" in graph_parameters["name"]:
        path = os.path.join(graph_parameters["path"], "t0.graph")
        G, F_true = load_dancer_graph(path)
        path_param = os.path.join(graph_parameters["path"], "parameters")
        with open(path_param, "r") as file:
            collected_parameters = {k: v for k, v in [f.split(":") for f in file.read().split("\n")[:-1]]}
    elif "CORA" in graph_parameters["name"]:
        G, F_true = load_cora_graph(graph_parameters["path"])
    else:
        G, F_true = None, None
        logging.error("ERROR : no graph loaded, name does not match anything implemented yet...")
        exit()

    # check if 'feat_added' is not None
    if graph_parameters["feat_added"] is not None:
        logging.info(f"Adding dependent features : {graph_parameters['feat_added']}")
        F_true = add_dependent_features(F_true, graph_parameters["feat_added"])

    if measure_show or measure_save:
        measures = graph_parameters["measures"] if 'measures' in graph_parameters.keys() else "base"
        logging.info(f"Computing Measures : {measures}...")
        Measures = GraphMeasures(compute_choice=measures)
        Measures.compute(G, F_true)
        if measure_save:
            Measures.save(path=f"./{UNIQUE}/graph_measures_{graph_parameters['name']}.txt")
        if measure_show:
            Measures.show()

    title = f"{graph_parameters['name']}_"
    title += f"size{G.number_of_nodes()}_"
    if "CLASSIC" in graph_parameters["name"]:
        title += f"prob{str(graph_parameters['prob'])[1:-1].replace(', ', '-')}_"
        title += f"featuresDep{str(graph_parameters['feat_dep'])[1:-1].replace(', ', '-')}_"
        if graph_parameters['feat_added']:
            title += f"featuresAdd{str(graph_parameters['feat_added'])[1:-1].replace(', ', '-')}_"
        if any(graph_parameters["feat_indep"]):
            title += f"featuresIndep{str(graph_parameters['feat_indep'])[1:-1].replace(', ', '-')}_"
        title += f"seed{seed}_"
    elif "DANCER" in graph_parameters["name"]:
        title += f"prob{collected_parameters['probaMicro']}_"
        title += f"featuresDep{collected_parameters['nbAttributes']}_"
        title += f"seedGen{collected_parameters['seed']}_"
    else:
        pass

    logging.info(f"Getting distance matrix...")
    distance_matrix = get_proximity_matrix(G,
                                           save=f"./{UNIQUE}/Distance_matrix_" + title + f".png" if dist_save else "",
                                           title="Distance matrix " + title.replace("_", " "),
                                           show=False)

    graph_parameters["G"] = G
    graph_parameters["dist_mat"] = distance_matrix
    graph_parameters["F_true"] = F_true
    graph_parameters["draw"] = draw_
    graph_parameters["title"] = title

    return graph_parameters

# Function produce_NA for generating missing values ------------------------------------------------------
# Taken from Muzellec

def produce_NA(X, adjacency_matrix, p_miss, preserve_first_column: bool = False, mecha="MCAR", opt=None, p_obs=None, q=None, seed=None,
               max_missing=0, link_to_degree=True, device=torch.device("cpu")):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    adjacency_matrix : torch.DoubleTensor or np.ndarray, shape (n,n)
        Adjacency matrix or Proximity matrix to account for the network topology.
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
    X = X.clone()
    X = X.to(device)
    if torch.is_tensor(adjacency_matrix) :
        adjacency_matrix.to(device)

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    if preserve_first_column:
        communities = X[:, 0]
        X = X[:, 1:]
    else:
        communities = None

    file_name = f"./WaGNA/masks/{cfg.graph_name}_miss{p_miss}_{mecha}_{opt}_seed{seed}.npy"

    # if the file "./WaGNA/masks/{cfg.graph_name}_{mecha}_{opt}_{seed}.npy" already exists, load it
    if os.path.exists(file_name):
        mask = torch.from_numpy(np.load(file_name))
        cfg.logger.info(f"file '{file_name}' loaded.")
        loaded=True
    else :
        if mecha == "MAR":
            mask = MAR_mask(X, p_miss, p_obs, seed=seed).double()
        elif mecha == "MNAR" and opt == "logistic":
            mask = MNAR_mask_logistic(X, p_miss, p_obs, seed=seed).double()
        elif mecha == "MNAR" and opt == "quantile":
            mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs, seed=seed).double()
        elif mecha == "MNAR" and opt == "selfmasked":
            mask = MNAR_self_mask_logistic(X,
                                           adjacency_matrix,
                                           p_miss,
                                           seed=seed,
                                           link_to_degree=link_to_degree,
                                           device=device).double()
        else:  # MCAR
            if seed:
                torch.manual_seed(seed)
            pre_mask = torch.rand(X.shape)
            if max_missing > 0:
                max_by_row = torch.max(torch.topk(pre_mask, max_missing, dim=1, largest=False)[0][:, -1][:, None], dim=1)
                pre_mask[pre_mask > max_by_row[0][:, None]] = 1
            mask = (pre_mask < p_miss).double()

        # concatenate back the communities column to the X array
        if preserve_first_column:
            X = torch.cat((communities.unsqueeze(1), X), dim=1)
            mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim=1)

        # save mask to "./WaGNA/masks/{cfg.graph_name}_{mecha}_{opt}_{seed}.npy"
        # check if "./masks" directory exists, if not create it
        if not os.path.exists("./WaGNA/masks"):
            os.makedirs("./WaGNA/masks")
        np.save(file_name, mask.numpy())
        cfg.logger.info(f"File '{file_name}' saved")
        loaded=False

    with open(os.path.join(cfg.UNIQUE,"log_missing.log"), "a+") as file:
        file.write(f"mecha : {mecha}, "
                   f"opt : {opt}, "
                   f"p_miss : {p_miss}, "
                   f"p_obs : {p_obs}, "
                   f"q : {q}, "
                   f"seed : {seed}, "
                   f"max_missing : {max_missing}, "
                   f"loaded : {loaded}, "
                   f"file : {file_name}\n")

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}


def add_missing_features(F_true,
                         topology_matrix,
                         p_miss,
                         seed=42,
                         mecha="MCAR",
                         opt=None,
                         max_missing=0,
                         return_all=False,
                         verbose=False,
                         logging=None,
                         preserve_first_column=True,
                         device=torch.device('cpu')):
    if verbose:
        m = f"Generating mask and missing features : {mecha}, " \
            f"missing {p_miss * 100}%" \
            f"{f', max {max_missing}' if max_missing else ''}"
        if logging:
            logging.info(m)
        else:
            print(m)

    topology_matrix = topology_matrix.clone()

    # turn topology matrix into adjacency matrix
    if topology_matrix.max() > 1:
        topology_matrix[topology_matrix > 1] = 0

    F_true = F_true.clone()

    F_miss = produce_NA(F_true,
                        adjacency_matrix=topology_matrix,
                        p_miss=p_miss,
                        preserve_first_column=preserve_first_column,
                        mecha=mecha,
                        opt=opt,
                        seed=seed,
                        max_missing=max_missing,
                        device=device)

    if return_all:
        return F_miss
    else:
        return F_miss['X_incomp']


def load_matrices(data):
    """
    Load numpy matrix from path.
    :param data: dict of paths to load matrix from
    :return:
    """
    data = data.copy()
    def load(path):
        if re.match(r"^(:?\/|\.)?(:?\/[^\/]+)+\.(:?npy|npz)$", path):
            if path.endswith(".npz"):
                matrix =  scipy.sparse.load_npz(path).toarray()
            else: # npy
                matrix = np.load(path)
            matrix = torch.from_numpy(matrix)
            return matrix
        else:
            return path_s
    for key, path_s in data.items():
        try :
            if type(path_s) == str:
                data[key] = load(path_s)
            else:
                data[key] = {label: load(path) for label, path in path_s.items()}
        except FileNotFoundError as err:
            # print content of path_s directory
            print(f"ERROR : {err}")
            print(f"ERROR : {path_s} not found")
            p_ = ""
            for p in path_s.split("/")[:-1] :
                p_ += p + "/"
                print(f"content of {p_} :\n{os.listdir(p_)}")

    return data



def get_edge_index(adjacency_matrix):
    """
    Get edge index from graph G
    """
    if type(adjacency_matrix) == np.ndarray :
        G = nx.from_numpy_array(adjacency_matrix)
    else :
        G = nx.from_numpy_array(adjacency_matrix.detach().numpy())
    # create edge index from G
    # edge_index = torch.tensor([list(e) for e in G.edges]).T
    adj = nx.to_scipy_sparse_array(G).tocoo()
    # adj = adjacency_matrix
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index






















































