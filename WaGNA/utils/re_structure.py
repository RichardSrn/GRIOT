import ot
import numpy as np
import networkx as nx
from typing import Union
from ..losses.MultiW import MultiW
import matplotlib.pyplot as plt
import torch
import torch_geometric
from ..config import cfg


def should_restructure(structure_mat, features_mat, y, fill_nan=True):
    # fill y nan with average
    y = y.clone()

    G = nx.from_numpy_array(structure_mat.detach().numpy())
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    if fill_nan:
        # fill features_mat nan with average by column
        features_mat = features_mat.clone()
        features_mat[torch.isnan(features_mat)] = features_mat[~torch.isnan(features_mat)].mean(dim=0)

        # predict the community of the attributed graph G using the structure and features
        ## add the features to G, each row is the feature of a node
        for i, f in enumerate(features_mat):
            G.nodes[i]['features'] = f.detach().numpy()
        ## predict the communities
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        communities = {i: c for i, c in enumerate(communities)}
        y_pred = torch.zeros_like(y)
        for i, c in communities.items():
            y_pred[list(c)] = i
        # fill y nan with predicted
        y[torch.isnan(y)] = y_pred[torch.isnan(y)]
    else:
        y[torch.isnan(y)] = y[~torch.isnan(y)].mean()

    homophily_community_wise = torch_geometric.utils.homophily(edge_index, y)

    print("\n" * 10)
    print(homophily_community_wise)


def is_connected(structure_mat) -> bool:
    # def dfs(node):
    #     visited.add(node)
    #     for neighbor, edge_exists in enumerate(adjacency_mat[node]):
    #         if edge_exists and neighbor not in visited:
    #             dfs(neighbor)
    #
    # n = len(adjacency_mat)
    # visited = set()
    # dfs(0)
    # return len(visited) == n
    #
    #
    # %timeit is_connected(new_structure>0)
    # 842 ms ± 121 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # %timeit nx.is_connected(nx.from_numpy_array(new_structure))
    # 142 ms ± 18.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    return nx.is_connected(nx.from_numpy_array(structure_mat.detach().cpu().numpy()))


def compute_new_structure(proximity_matrix,
                          features_matrix,
                          true_labels_for_plot=None,
                          ot_plans=None,
                          batch_size: Union[int, float] = 0.25,
                          alpha=0.5,
                          lossfn=None,
                          iters=100,
                          threshold=0.15,
                          threshold_low=0.25,
                          threshold_decay=0.01,
                          allow_disconnected=False,
                          verbose=False,
                          use_CE=True,
                          method="increment",  # ="remake",
                          integer=True):
    """

    :param proximity_matrix:
    :param features_matrix:
    :param ot_plans:
    :param batch_size:
    :param alpha:
    :param lossfn:
    :param iters:
    :param threshold:
    :param threshold_low: only used with method="increment"
    :param threshold_decay:
    :param allow_disconnected:
    :param verbose:
    :param use_CE:
    :param method:
    :param integer:
    :return:
    """
    device = proximity_matrix.device

    new_structure = torch.zeros((proximity_matrix.shape[0], proximity_matrix.shape[0])).to(device)
    if ot_plans is None:
        if isinstance(batch_size, float):
            batch_size = int(batch_size * proximity_matrix.shape[0])

        if lossfn is None:
            lossfn = MultiW(alpha=alpha,
                            epsilon=0.01,
                            p=2,
                            numItermax=1000,
                            stopThr=1e-9,
                            method="sinkhorn",
                            path=".",
                            unique=None,
                            p_unif=True,
                            normalize_F=False,
                            normalize_MF_MC=True,
                            CrossEtpy=use_CE,
                            use_geomloss=False)
        else:
            pass  # already defined

        i = 0
        i_failed = 0
        vertices = np.arange(proximity_matrix.shape[0])
        while i < iters:  # and i_failed < iters / 10:
            idx1 = np.random.choice(vertices, size=batch_size, replace=False)
            idx2 = np.random.choice(vertices, size=batch_size, replace=False)
            idx1Uidx2 = np.union1d(idx1, idx2)
            # select the corresponding sub matrices
            prox1 = proximity_matrix[idx1][:, idx1Uidx2].to(device)
            prox2 = proximity_matrix[idx2][:, idx1Uidx2].to(device)
            feat1 = features_matrix[idx1].to(device)
            feat2 = features_matrix[idx2].to(device)

            ot_plan = lossfn(C1=prox1, C2=prox2,
                             F1=feat1, F2=feat2,
                             return_transport=True,
                             return_distance=False).to(device)

            # ot nan in ot_plan, do not add to new_structure
            if torch.isnan(ot_plan).any():
                i_failed += 1
                continue
            else:
                new_structure[idx1[:, None], idx2] += ot_plan
                new_structure[idx2[:, None], idx1] += ot_plan.T

            if verbose:
                print(f'Iteration {i + 1}/{iters}'.ljust(20), end='\r')
            i += 1
        if verbose:
            cfg.logger(f'Total iteration {i + 1}/{iters} + {i_failed} failed.')
    else:
        for op in ot_plans:
            idx1 = op["idx1"]
            idx2 = op["idx2"]
            ot_plan = op["ot_plan"]
            new_structure[idx1[:, None], idx2] += ot_plan
            new_structure[idx2[:, None], idx1] += ot_plan.T

    # set diagonal to 0
    new_structure = new_structure - torch.diag(new_structure.diag())
    # normalize
    new_structure /= new_structure.max()
    new_structure -= new_structure.min()
    threshold = threshold

    if method == "remake":
        if allow_disconnected:
            # threshold everything above `threshold` to 0
            new_structure[new_structure < threshold] = 0
        else:
            assert is_connected(
                new_structure.detach().cpu().numpy() > 0
            ), 'The graph is not connected, try to increase the iterations or allow disconnected graphs.'

            new_structure_tmp = new_structure.clone()
            new_structure_tmp_ = torch.zeros_like(proximity_matrix)

            percentile_threshold = torch.quantile(new_structure.flatten(), threshold)
            new_structure_tmp_[new_structure < percentile_threshold] = -1
            new_structure_tmp[new_structure_tmp_ == -1] = 0
            threshold_tmp = float(threshold)
            i = 0
            while not is_connected(new_structure_tmp.detach().cpu().numpy() > 0) and threshold_tmp > 0:
                i += 1
                new_structure_tmp_ = torch.zeros_like(proximity_matrix)
                threshold_tmp = threshold - threshold_decay * i
                percentile_threshold = torch.quantile(new_structure.flatten(), threshold_tmp)
                new_structure_tmp_[new_structure < percentile_threshold] = -1
                new_structure_tmp = new_structure.clone()
                new_structure_tmp[new_structure_tmp_ == -1] = 0
                assert threshold_tmp > 0, 'The threshold reached 0, try to lower the threshold decay.'

            if verbose and threshold_tmp < threshold:
                # get all edges from new_structure that are below threshold
                weak_edges = (new_structure == 0).sum() - (new_structure_tmp == 0).sum()
                print(
                    f'Lowered threshold to {threshold_tmp:.2f}.\nNumber of edges kept below the original threshold ({threshold}): {weak_edges}.'
                )
            new_structure = new_structure_tmp

            if integer:
                # round the new_structure to integers 0/1, anything above 0 is 1
                new_structure[new_structure > 0] = 1
                new_structure = new_structure.to(torch.int)

    elif method == "increment":
        new_structure_ = proximity_matrix.clone()

        new_structure_tmp = torch.zeros_like(proximity_matrix)

        ## ADD EDGES
        # get the top `threshold`% of the edges from new_structure and set them to 1 in new_structure_tmp
        percentile_threshold = torch.quantile(new_structure.flatten(), 1 - threshold)
        new_structure_tmp[new_structure > percentile_threshold] = 1
        new_structure_[new_structure_tmp == 1] = 1

        ## DROP EDGES
        # same with the lower values, and set them to -1
        percentile_threshold = torch.quantile(new_structure.flatten(), threshold_low)
        new_structure_tmp[new_structure < percentile_threshold] = -1
        new_structure_tmp_ = new_structure_.clone()
        new_structure_tmp_[new_structure_tmp == -1] = 0
        threshold_tmp = float(threshold_low)
        i = 0
        while not is_connected(new_structure_tmp_ > 0) and threshold_tmp > 0:
            i += 1
            threshold_tmp = threshold_low - threshold_decay * i
            if threshold_tmp > 0:
                percentile_threshold = torch.quantile(new_structure.flatten(), threshold_tmp)
                new_structure_tmp[new_structure < percentile_threshold] = -1
                new_structure_tmp_ = new_structure_.clone()
                new_structure_tmp_[new_structure_tmp == -1] = 0
            else:
                new_structure_tmp_ = new_structure_.clone()
                break

        new_structure = new_structure_tmp_

        if integer:
            # round the new_structure to integers 0/1, anything above 0 is 1
            new_structure = new_structure.to(torch.int)

        if verbose:
            added_edges = (proximity_matrix == 1).sum() - (new_structure_ == 1).sum()
            removed_edges = (new_structure_ == 1).sum() - (new_structure == 1).sum()
            cfg.logger(f"Addes edges: {added_edges} then removed edges: {removed_edges}.")
            if threshold_tmp < threshold_low:
                cfg.logger(f'Lowered threshold from {threshold_low:.2f} to {threshold_tmp:.2f}.')

        mask_min = (new_structure > 0) & (proximity_matrix > 0)
        new_structure = torch.zeros_like(proximity_matrix) + torch.minimum(new_structure,
                                                                           proximity_matrix) * mask_min + torch.maximum(
            new_structure, proximity_matrix) * ~mask_min

    else:
        raise NotImplementedError(f"method should be in {{'increment', 'remake'}} but is {method}")

    plot = True
    if plot:
        G = nx.from_numpy_array(proximity_matrix.detach().cpu().numpy() == 1)
        G_new = nx.from_numpy_array(new_structure.detach().cpu().numpy() == 1)

        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        nx.draw(G, pos,
                with_labels=True,
                node_color=true_labels_for_plot,
                cmap=plt.cm.viridis,
                node_size=50,
                edge_color='black',
                linewidths=1,
                font_size=5)
        if true_labels_for_plot is not None:
            plt.title(f'Original, homophily: {compute_homophily(true_labels_for_plot[:, None], G):.2f}')
        else:
            plt.title(f'Original')
        plt.subplot(122)
        nx.draw(G_new,  # pos,
                with_labels=True,
                node_color=true_labels_for_plot,
                cmap=plt.cm.viridis,
                node_size=50,
                edge_color='black',
                linewidths=1,
                font_size=5)
        if true_labels_for_plot is not None:
            plt.title(f'New, homophily: {compute_homophily(true_labels_for_plot[:, None], G_new):.2f}')
        else:
            plt.title(f'New')
        plt.suptitle('New_structure')
        plt.savefig(f"{cfg.UNIQUE}/new_structure_alpha{alpha}_iters{iters}_threshold{threshold}.png")
        plt.close()

    return new_structure


def compute_homophily(F, G):
    # # HOMOPHILY USING PYTORCH AND COMMUNITIES (instead of attributes)
    # # get the edge_index of the graph
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    homophily_community_wise = torch_geometric.utils.homophily(edge_index, F[:, 0].long())

    return homophily_community_wise
