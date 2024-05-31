from ..make_datasets.load_planetoid_graph import load_planetoid_graph
from ..make_datasets.graph_generation import generate_sbm_graph
from ..make_datasets.load_dancer_graph import load_dancer_graph
from ..make_datasets.load_cora_graph import load_cora_graph
from ..make_datasets.load_wkb_graph import load_wkb_graph
from ..utils.graph_tool import add_dependent_features, get_proximity_matrix
from ..utils.graph_measures import GraphMeasures
from ..plots.plot_attributes import plot_attributes
from ..plots.graph_drawing import draw2
from scipy.sparse import csr_matrix

import networkx as nx
import numpy as np

import logging
import scipy
import torch
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def plot_all(*F_s, edge_list=None, path="", title_prefix="", communities_names=None):
    """
    Plot all attributes of F.
    :param F: Attribute matrices
    :param path: Path to save plots
    :param title_prefix: Prefix of title
    :return:
    """
    for i, F in enumerate(F_s):
        title = f"{title_prefix}{i if len(F_s) > 1 else ''} - Features ScatterPlot"
        if F[:, 1:].shape[1] > 2:  # more than 2 features
            for kind in ["SEP", "SEP+", "PCA", "TSNE", "UMAP", "MDS", "ISOMAP", "LDA"]:
                for plot_edges in [False, True]:
                    try:
                        plot_attributes(F.clone(),
                                        edge_list=edge_list if plot_edges else None,
                                        kind=kind,
                                        plot_edges=plot_edges,
                                        path=path,
                                        title=title,
                                        communities_names=communities_names)
                    except IndexError:
                        logging.warning(f"Could not plot {kind} for {title}, cause IndexError")
        elif F[:, 1:].shape[1] == 2:  # 2 features
            for plot_edges in [False, True]:
                try:
                    plot_attributes(F.clone(),
                                    edge_list=edge_list if plot_edges else None,
                                    plot_edges=plot_edges,
                                    path=path,
                                    title=title,
                                    communities_names=communities_names)
                except IndexError:
                    logging.warning(f"Could not plot {title}, cause IndexError")
        else:
            logging.warning("Not enough features to plot")


def save_all(path_dir="", **data):
    """
    Save matrices in .npy format.
    :param matrices: Matrices to save
    :return:
    """
    for k, v in data.items():
        if "sparse" in k:
            path = os.path.join(path_dir, f"{k}.npz")
        else:
            path = os.path.join(path_dir, f"{k}.npy")
        logging.info(f"Saving {path}")
        if type(v) == np.ndarray:
            # save numpy array
            np.save(path, v)
        elif type(v) == torch.Tensor:
            # save torch tensor as numpy array
            np.save(path, v.numpy())
        elif type(v) == scipy.sparse.csr.csr_matrix:
            # save scipy sparse matrix as sparse matrix using scipy
            scipy.sparse.save_npz(path, v)
        elif type(v) == dict:
            # save dictionary
            path = os.path.join(path_dir, f"{k}.dict")
            with open(path, "w") as f:
                f.write(f"{k} = ")
                f.write("{\n")
                for k1, v1 in v.items():
                    f.write(f"\t{k1} : {v1},\n")
                f.write("}\n")
        else:
            logging.warning(f"Matrix {k} not saved")

def sbm_graph(path="",
              graph_parameters: dict = None,
              feat_added=(0, 1, 2),
              vanilla=True,
              fit01=True,
              binarize=True,
              draw_graph=False,
              draw_proximity_matrix=False,
              draw_attributes=False,
              measure_show=False,
              measure_save=True,
              measures = "all"):
    """
    Create SBM graph.
    :param measure_save:
    :param measure_show:
    :return F_true: Attribute matrix of SBM
    :return proximity_matrix: Proximity matrix of SBM
    """
    # if path does not exist, make it
    if not os.path.exists(path):
        os.makedirs(path)

    logging.info("loading SBM graph")
    G, F_true = generate_sbm_graph(**graph_parameters)

    logging.info("add dependent feature(s)")
    F_true = add_dependent_features(F_true, feat_added)
    F_true_fit01 = None
    F_true_binarize = None

    # fit F_true within 0-1
    if fit01 or binarize:
        logging.info("fit F_true within 0-1")
        F_true_fit01 = F_true.clone()
        F_true_fit01[:, 1:] = (F_true_fit01[:, 1:] - F_true_fit01[:, 1:].min()) / (
                F_true_fit01[:, 1:].max() - F_true_fit01[:, 1:].min())

        # binarize F_true
        if binarize:
            logging.info("binarize F_true")
            F_true_binarize = F_true_fit01.clone()
            F_true_binarize[:, 1:][F_true_binarize[:, 1:] >= 0.5] = 1
            F_true_binarize[:, 1:][F_true_binarize[:, 1:] < 0.5] = 0

    if measure_show or measure_save:
        logging.info(f"Computing Measures : {measures}...")
        Measures = None
        Measures_fit01 = None
        Measures_binarize = None
        if vanilla:
            Measures = GraphMeasures(compute_choice=measures, G=G, F=F_true)
            Measures.compute()
        if fit01:
            Measures_fit01 = GraphMeasures(compute_choice=measures, G=G, F=F_true_fit01)
            Measures_fit01.compute()
        if binarize:
            Measures_binarize = GraphMeasures(compute_choice=measures, G=G, F=F_true_binarize)
            Measures_binarize.compute()
        if measure_save:
            if vanilla:
                Measures.save(path=f"./{path}/graph_measures_F_true.txt")
            if fit01:
                Measures_fit01.save(path=f"./{path}/graph_measures_F_true_fit01.txt")
            if binarize:
                Measures_binarize.save(path=f"./{path}/graph_measures_F_true_binarize.txt")
        if measure_show or True:
            if vanilla:
                Measures.show()
            if fit01:
                Measures_fit01.show()
            if binarize:
                Measures_binarize.show()

    if draw_graph:
        logging.info("draw graph")
        draw2(G, F_true, save=os.path.join(path, f"SBM.png"), title="SBM graph", show=False)

    if draw_attributes:
        if vanilla and False:
            logging.info("plot attributes F_true")
            plot_all(F_true, edge_list=G.edges(), path=path, title_prefix="SBM_F_true")
        if fit01:
            logging.info("plot attributes F_true_fit01")
            plot_all(F_true_fit01, edge_list=G.edges(), path=path, title_prefix="SBM_F_true_fit01")
        if binarize and False:
            logging.info("plot attributes F_true_binarized")
            plot_all(F_true_binarize, edge_list=G.edges(), path=path, title_prefix="SBM_F_true_binarize")

    logging.info("get adjacency matrix")
    adjacency_matrix = nx.to_numpy_array(G)

    if draw_proximity_matrix:
        save_title = os.path.join(path, f"SBM_proximity_matrix.png")
    else:
        save_title = None
    logging.info("get proximity matrix")
    proximity_matrix = get_proximity_matrix(G,
                                            save=save_title,
                                            title="SBM proximity matrix",
                                            show=False)

    data = {
        "adjacency_matrix": adjacency_matrix,
        "proximity_matrix": proximity_matrix,
    }
    if vanilla:
        data["F_true"] = F_true
    if fit01:
        data["F_true_fit01"] = F_true_fit01
    if binarize:
        data["F_true_binarize"] = F_true_binarize
    return data


def cora_graph(path="",
               draw_graph=False,
               draw_proximity_matrix=False,
               draw_attributes=False,
               measure_show=False,
               measure_save=True,
               measures = "nearly_all"):
    """
    Load Cora graph from path.
    :param path:
    :return:
    """
    logging.info("loading CORA graph")
    G, F_true, community_dict = load_cora_graph(path, directed=False)

    if measure_show or measure_save:
        logging.info(f"Computing Measures : {measures}...")
        Measures = GraphMeasures(compute_choice=measures)
        Measures.compute(G, F_true)
        if measure_save:
            Measures.save(path=f"./{path}/graph_measures_F_true.txt")
        if measure_show:
            Measures.show()

    # logging.info("loading CORA graph directed")
    # G_directed, _, _ = load_cora_graph(path, directed=True)
    if draw_graph:
        logging.info("draw graph")
        draw2(G,
              F_true,
              figsize=(20, 20),
              title="Cora graph",
              save=os.path.join(path, f"Cora.png"),
              show=False,
              communities_name=list(community_dict.keys()))

    if draw_attributes:
        logging.info("plot attributes")
        plot_all(F_true, edge_list=G.edges(), path=path, title_prefix="CORA",
                 communities_names=list(community_dict.keys()))

    logging.info("get adjacency matrix")
    adjacency_matrix = nx.to_numpy_array(G)

    if draw_proximity_matrix:
        save_title = os.path.join(path, f"Cora_proximity_matrix.png")
        # save_title_directed = os.path.join(path, f"Cora_proximity_matrix_directed.png")
    else:
        save_title = None
        # save_title_directed = None
    logging.info("get proximity matrix")
    proximity_matrix = get_proximity_matrix(G,
                                            save=save_title,
                                            title="Cora proximity matrix",
                                            show=False)
    # logging.info("get proximity matrix directed")
    # proximity_matrix_directed = get_proximity_matrix(G_directed,
    #                                                  save=save_title_directed,
    #                                                  title="Cora proximity matrix directed",
    #                                                  show=False,
    #                                                  directed=True)

    # make adjacency a sparse matrix
    adjacency_matrix_sparse = csr_matrix(adjacency_matrix)
    proximity_matrix_sparse = csr_matrix(proximity_matrix)
    # proximity_matrix_directed = csr_matrix(proximity_matrix_directed)
    F_true_sparse = csr_matrix(F_true)

    data = {"adjacency_matrix": adjacency_matrix,
            "adjacency_matrix_sparse": adjacency_matrix_sparse,
            "proximity_matrix": proximity_matrix,
            "proximity_matrix_sparse": proximity_matrix_sparse,
            # "proximity_matrix_directed": proximity_matrix_directed,
            # "proximity_matrix_directed_sparse": proximity_matrix_directed_sparse,
            "F_true": F_true,
            "F_true_sparse": F_true_sparse,
            "community_dict": community_dict}
    return data


def wkb_graph(path="",
              name="",
              draw_graph=True,
              draw_proximity_matrix=True,
              draw_attributes=True,
              measure_show=False,
              measure_save=True,
              measures = "all"):
    """
    Load WebKnowledgeGraphs graphs (Cornell, Texas, Wisconsin) from torch_geometric.datasets.
    :param path:
    :return:
    """
    logging.info("loading WKB graphs")
    #  name in [cornell, wisconsin and texas]
    G, F_true, community_dict = load_wkb_graph(path, name=name, directed=False, draw_=False, main_component_=True)

    if measure_show or measure_save:
        logging.info(f"Computing Measures : {measures}...")
        Measures = GraphMeasures(compute_choice=measures)
        Measures.compute(G, F_true)
        if measure_save:
            Measures.save(path=f"./{path}/graph_measures_F_true.txt")
        if measure_show:
            Measures.show()

    if draw_graph:
        logging.info("draw graph")
        draw2(G,
              F_true,
              figsize=(20, 20),
              title=f"WKB {name}",
              save=os.path.join(path, f"WKB_{name}.png"),
              show=False,
              communities_name=list(community_dict.keys()))

    if draw_attributes:
        logging.info("plot attributes")
        plot_all(F_true, edge_list=G.edges(), path=path, title_prefix="WKB_" + name,
                 communities_names=list(community_dict.keys()))

    logging.info("get adjacency matrix")
    adjacency_matrix = nx.to_numpy_array(G)

    if draw_proximity_matrix:
        save_title = os.path.join(path, f"WKB_{name}_proximity_matrix.png")
    else:
        save_title = None
    logging.info("get proximity matrix")
    proximity_matrix = get_proximity_matrix(G,
                                            save=save_title,
                                            title=f"WKB {name} proximity matrix",
                                            show=False)
    logging.info("get proximity matrix directed")

    data = {"adjacency_matrix": adjacency_matrix,
            "proximity_matrix": proximity_matrix,
            "F_true": F_true,
            "community_dict": community_dict}
    return data


def planetoid_graph(path="",
                    name="",
                    draw_graph=False,
                    draw_proximity_matrix=False,
                    draw_attributes=False,
                    measure_show=True,
                    measure_save=True,
                    measures = "nearly_all"):
    """
    Load Planetoid graphs (CiteSeer, PubMed) from torch_geometric.datasets.Planetoid.
    :param path:
    :return:
    """
    logging.info("loading Planetoid graphs")
    #  name in [cornell, wisconsin and texas]
    G, F_true, community_dict = load_planetoid_graph(path,
                                                     name=name,
                                                     directed=False,
                                                     draw_=False,
                                                     main_component_=True)

    if measure_show or measure_save:
        logging.info(f"Computing Measures : {measures}...")
        Measures = GraphMeasures(compute_choice=measures, G=G, F=F_true)
        Measures.compute()
        if measure_save:
            Measures.save(path=f"./{path}/graph_measures_F_true.txt")
        if measure_show:
            Measures.show()

    if draw_graph:
        logging.info("draw graph")
        draw2(G,
              F_true,
              figsize=(20, 20),
              title=f"PLANETOID {name}",
              save=os.path.join(path, f"PLANETOID_{name}.png"),
              show=False,
              communities_name=None)  # list(community_dict.keys()))

    if draw_attributes:
        logging.info("plot attributes")
        plot_all(F_true, edge_list=G.edges(), path=path, title_prefix="PLANETOID_" + name,
                 communities_names=list(community_dict.keys()))

    logging.info("get adjacency matrix")
    adjacency_matrix = nx.to_numpy_array(G)

    if draw_proximity_matrix:
        save_title = os.path.join(path, f"PLANETOID_{name}_proximity_matrix.png")
    else:
        save_title = None
    logging.info("get proximity matrix")
    proximity_matrix = get_proximity_matrix(G,
                                            save=save_title,
                                            title=f"PLANETOID {name} proximity matrix",
                                            show=False)
    logging.info("get proximity matrix directed")

    data = {"adjacency_matrix": adjacency_matrix,
            "proximity_matrix": proximity_matrix,
            "F_true": F_true,
            "community_dict": community_dict}
    return data


def main():
    # specific values to generate SBM graph as close to DANCER graph as possible
    # for i, prob in enumerate([
    #     (0.0175, 0.0010),
    #     (0.0100, 0.0040),
    #     (0.0030, 0.0100)
    # ]):
    #     graph_parameters = {
    #         "sizes": [100, 100, 200, 400],
    #         "prob": prob,
    #         "seed": 42,
    #         "feat_dep": [0, 0, 2],
    #         "feat_indep": [0, 0, 0],
    #         "main_component_only": True,
    #     }
    #     path_sbm = f"./data/artificial/SBM_{i}"
    #     data_sbm = sbm_graph(path=path_sbm,
    #                          graph_parameters=graph_parameters,
    #                          feat_added=[(0, 1, 2)],
    #                          vanilla=True,
    #                          fit01=True,
    #                          binarize=False,
    #                          draw_attributes=True,
    #                          draw_graph=True,
    #                          draw_proximity_matrix=True,
    #                          measure_show=False,
    #                          measure_save=True)
    #     save_all(path_dir=path_sbm, **data_sbm)
    #     del data_sbm

    #     graph_parameters = {
    #         "sizes": [100, 100, 200, 400],
    #         "prob": prob,
    #         "seed": 42,
    #         "feat_dep": [0, 0, 2],
    #         "feat_indep": [0, 0, 0],
    #         "main_component_only": True,
    #     }
    #
    #     path_sbm = f"./data/artificial/SBM_{i}"
    #     data_sbm = sbm_graph(path=path_sbm,
    #                          graph_parameters=graph_parameters,
    #                          feat_added=[(0, 1, 2)],
    #                          vanilla=True,
    #                          fit01=True,
    #                          binarize=False,
    #                          draw_attributes=True,
    #                          draw_graph=True,
    #                          draw_proximity_matrix=True,
    #                          measure_show=False,
    #                          measure_save=True)
    #     save_all(path_dir=path_sbm, **data_sbm)
    #     del data_sbm
    #
    # for name in ["cornell", "texas", "wisconsin"]:
    for name in ["wisconsin", ]:
        path_wkb = f"./data/real/WKB/{name}"
        data_wkb = wkb_graph(path=path_wkb,
                             name=name,
                             draw_attributes=False,
                             draw_graph=False,
                             draw_proximity_matrix=False)
        save_all(path_dir=path_wkb, **data_wkb)
        del data_wkb

    path_cora = "./data/real/PLANETOID/Cora"
    data_cora = cora_graph(path=path_cora,
                           draw_attributes=False,
                           draw_graph=False,
                           draw_proximity_matrix=False)
    save_all(path_dir=path_cora, **data_cora)
    del data_cora
    #
    # for name in ["PubMed", "CiteSeer"]:
    #     path_planetoid = f"./data/real/PLANETOID/{name}"
    #     data_planetoid = planetoid_graph(path=path_planetoid,
    #                                      name=name,
    #                                      draw_attributes=False,
    #                                      draw_graph=False,
    #                                      draw_proximity_matrix=False)
    #     save_all(path_dir=path_planetoid, **data_planetoid)
    #     del data_planetoid
