import os

import networkx as nx
import numpy as np
import pandas as pd
import torch

from time import time, sleep

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from tqdm import tqdm

from ..utils.MissingDataOT_network_utils import MAE, RMSE
from ..utils.graph_tool import load_matrices, add_missing_features
from ..utils.round_ratio import round_ratio
from ..utils.save import save_steps
from ..datasets_class.dataset_class import MyDataset
from ..config import cfg


def compute_baselines(binary=False,
                      data=None,
                      warn=False,
                      device=torch.device("cpu")):
    ## *** AVG ***
    t_start_avg = time()
    # compute the average of each column of F_true
    F_avg = torch.nanmean(data.x_miss, dim=0)
    # F_avg = torch.nanmean(F_true, dim=0)
    if any(F_avg.isnan()):
        print("WARNING nan in F_avg, impute with average over F_avg")
        F_avg = F_avg.nan_to_num(F_avg.nanmean())
    if binary:
        # round average values to the closest binary
        F_avg = round_ratio(F_avg, data.x_miss)
    # copy data.x_miss["X_incomp"] as F_filled_avg
    F_filled_avg = data.x_miss.clone()
    # impute na values of F_filled_avg with the average of each column of F_true
    F_filled_avg = torch.where(torch.isnan(F_filled_avg), F_avg, F_filled_avg)
    elapsed_time_avg = time() - t_start_avg

    ## *** AVG by COMMUNITY ***
    t_start_avg_by_community = time()
    # get the number of different unique favlues of data.x_miss[:, 0]
    number_of_communities = len(torch.unique(data.y))
    # use simple community detection algorithm to estimate communities of F_filled_avg using networkx
    pred_communities_tmp = nx.algorithms.community.greedy_modularity_communities(
        nx.from_numpy_array(data.adjacency_matrix.cpu().detach().numpy()),
        cutoff=number_of_communities,
        best_n=number_of_communities
    )
    pred_communities = torch.zeros(data.x_miss.shape[0], device=device)
    # compute the accuracy score between the true communities and the predicted communities,
    # the score should not consider the label of the community (0, 1, 2, etc.) but only where
    # the splits of the clusters are.

    for i, community in enumerate(pred_communities_tmp):
        pred_communities[list(community)] = i
    del pred_communities_tmp

    scores_community_detection = {
        "ari": adjusted_rand_score(data.y.cpu().detach().numpy(), pred_communities.cpu().detach().numpy()),
        "ami": adjusted_mutual_info_score(data.y.cpu().detach().numpy(), pred_communities.cpu().detach().numpy()),
        "nmi": normalized_mutual_info_score(data.y.cpu().detach().numpy(), pred_communities.cpu().detach().numpy())
    }

    # the first column of F_true represents the community of each node (0 to max(F_true[:, 0]))
    # compute the average of each column of F_true for each community
    F_avg_by_community = torch.vstack([torch.nanmean(data.x_miss[pred_communities == i], dim=0)
                                       for i in range(int(max(pred_communities)) + 1)])
    if any(F_avg_by_community.isnan().reshape(-1)):
        if warn == True:
            print("WARNING nan in F_avg_by_community, impute with average over F_avg_by_community")
        F_avg_by_community = F_avg_by_community.nan_to_num(F_avg_by_community.nanmean())
    if binary:
        # round average values to the closest binary
        F_avg_by_community = round_ratio(F_avg_by_community, data.x_miss)

    # copy data.x_miss["X_incomp"] as F_filled_avg_by_community
    F_filled_avg_by_community = data.x_miss.clone()
    # impute na values of F_filled_avg_by_community with the average by community of each column of F_true
    for i in range(int(max(pred_communities)) + 1):
        commu = (pred_communities == i).repeat(data.x_miss.shape[1], 1).T
        mask_tmp = torch.isnan(F_filled_avg_by_community) & commu
        F_filled_avg_by_community = torch.where(mask_tmp, F_avg_by_community[i], F_filled_avg_by_community)
    elapsed_time_avg_by_community = time() - t_start_avg_by_community

    ## *** RANDOM ***
    t_start_rnd = time()
    # set F_rnd to random values
    F_rnd = torch.rand(size=data.x_miss.shape)
    # multiply each column by the difference between max and min of the corresponding column
    F_rnd = (F_rnd * (np.nanmax(data.x_miss.cpu().detach().numpy(), axis=0) -
                      np.nanmin(data.x_miss.cpu().detach().numpy(), axis=0))
             + np.nanmin(data.x_miss.cpu().detach().numpy(), axis=0)).to(device)
    if binary:
        # round values to the closest binary
        F_rnd = round_ratio(F_rnd, data.x_miss)
    # copy data.x_miss["X_incomp"] as F_filled_rnd
    F_filled_rnd = data.x_miss.clone()
    # impute na values of F_filled_avg with the average of each column of F_true
    F_filled_rnd = torch.where(torch.isnan(F_filled_rnd), F_rnd, F_filled_rnd)
    elapsed_time_rnd = time() - t_start_rnd

    ## *** ZEROS ***
    t_start_zro = time()
    # set F_zro to zeros
    F_zro = torch.zeros(size=data.x_miss.shape, device=device)
    if binary:
        # round values to the closest binary
        F_zro = round_ratio(F_zro, data.x_miss)
    # copy data.x_miss["X_incomp"] as F_filled_zro
    F_filled_zro = data.x_miss.clone()
    # impute na values of F_filled_avg with the average of each column of F_true
    F_filled_zro = torch.where(torch.isnan(F_filled_zro), F_zro, F_filled_zro)
    elapsed_time_zro = time() - t_start_zro

    ## *** AVERAGE NEIGHBORHOOD ***
    t_start_avg_ngb = time()
    # for each node, its neighbors are the nodes connected to it by an edge in the topology matrix
    neighbors_list = [torch.where(data.adjacency_matrix[i, :] == 1)[0]
                      for i in range(data.adjacency_matrix.shape[0])]
    # set F_avg_ngb to be the average of the neighbors of each node
    F_avg_ngb = torch.vstack(
        [torch.nanmean(data.x_miss[neighbors_list[i], :], dim=0) for i in range(len(neighbors_list))])
    # fill nan values of F_avg_ngb with the nanmean of data.x_miss
    if any(F_avg_ngb.isnan().reshape(-1)):
        F_avg_ngb = F_avg_ngb.nan_to_num(torch.nanmean(data.x_miss))
    if binary:
        # round values to the closest binary
        F_avg_ngb = round_ratio(F_avg_ngb, data.x_miss)
    # copy data.x_miss["X_incomp"] as F_filled_avg_ngb
    F_filled_avg_ngb = data.x_miss.clone()
    # impute na values of F_filled_avg with the average of each column of F_true
    F_filled_avg_ngb = torch.where(torch.isnan(F_filled_avg_ngb), F_avg_ngb, F_filled_avg_ngb)
    elapsed_time_avg_ngb = time() - t_start_avg_ngb

    matrices = {"F_true": torch.hstack((data.y[:, None], data.x)),
                "F_filled_avg": torch.hstack((data.y[:, None], F_filled_avg)),
                "F_filled_avg_by_community": torch.hstack(
                    (data.y[:, None], F_filled_avg_by_community)),
                "F_filled_rnd": torch.hstack((data.y[:, None], F_filled_rnd)),
                "F_filled_zro": torch.hstack((data.y[:, None], F_filled_zro)),
                "F_filled_avg_ngb": torch.hstack((data.y[:, None], F_filled_avg_ngb))}

    baselines = {
        # val
        "mae_avg_val": MAE(F_filled_avg, data.x, data.x_miss_mask_val).item(),
        "rmse_avg_val": RMSE(F_filled_avg, data.x, data.x_miss_mask_val).item(),
        "mae_avg_by_community_val": MAE(F_filled_avg_by_community, data.x, data.x_miss_mask_val).item(),
        "rmse_avg_by_community_val": RMSE(F_filled_avg_by_community, data.x, data.x_miss_mask_val).item(),
        "mae_rnd_val": MAE(F_filled_rnd, data.x, data.x_miss_mask_val).item(),
        "rmse_rnd_val": RMSE(F_filled_rnd, data.x, data.x_miss_mask_val).item(),
        "mae_zro_val": MAE(F_filled_zro, data.x, data.x_miss_mask_val).item(),
        "rmse_zro_val": RMSE(F_filled_zro, data.x, data.x_miss_mask_val).item(),
        "mae_avg_ngb_val": MAE(F_filled_avg_ngb, data.x, data.x_miss_mask_val).item(),
        "rmse_avg_ngb_val": RMSE(F_filled_avg_ngb, data.x, data.x_miss_mask_val).item(),
        # train_test
        "mae_avg_test": MAE(F_filled_avg, data.x, data.x_miss_mask_test).item(),
        "rmse_avg_test": RMSE(F_filled_avg, data.x, data.x_miss_mask_test).item(),
        "mae_avg_by_community_test": MAE(F_filled_avg_by_community, data.x, data.x_miss_mask_test).item(),
        "rmse_avg_by_community_test": RMSE(F_filled_avg_by_community, data.x, data.x_miss_mask_test).item(),
        "mae_rnd_test": MAE(F_filled_rnd, data.x, data.x_miss_mask_test).item(),
        "rmse_rnd_test": RMSE(F_filled_rnd, data.x, data.x_miss_mask_test).item(),
        "mae_zro_test": MAE(F_filled_zro, data.x, data.x_miss_mask_test).item(),
        "rmse_zro_test": RMSE(F_filled_zro, data.x, data.x_miss_mask_test).item(),
        "mae_avg_ngb_test": MAE(F_filled_avg_ngb, data.x, data.x_miss_mask_test).item(),
        "rmse_avg_ngb_test": RMSE(F_filled_avg_ngb, data.x, data.x_miss_mask_test).item()
    }

    elapsed_time = {
        "avg": elapsed_time_avg,
        "avg_by_community": elapsed_time_avg_by_community,
        "rnd": elapsed_time_rnd,
        "zro": elapsed_time_zro,
        "avg_ngb": elapsed_time_avg_ngb
    }

    return baselines, matrices, scores_community_detection, elapsed_time


def get_baselines():
    cfg.logger.info(f"baselines_weak - {cfg.UNIQUE}")

    os.makedirs(cfg.UNIQUE, exist_ok=True)

    if cfg.p_miss_s is None:
        start, stop, step = .1, .9, .1
        cfg.p_miss_s = np.arange(start, stop + step, step)

    with open(os.path.join(cfg.UNIQUE, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {cfg.seeds}")
    cfg.logger.info(f"seeds for repetitions = {cfg.seeds}")
    df_s = dict()

    # for graph in tqdm(cfg.graph_parameters,
    #                   desc="Graphs".ljust(20),
    #                   leave=False,
    #                   position=0,
    #                   colour="black"):
    graph = cfg.graph_parameters
    df = pd.DataFrame()
    graph = load_matrices(graph)

    for F_true in tqdm(graph["features"].values(),
                       desc="Features".ljust(20),
                       leave=False,
                       position=1,
                       colour="red"):
        for p_miss in tqdm(cfg.p_miss_s,
                           desc="Missing Features".ljust(20),
                           leave=False,
                           position=2,
                           colour="blue"):
            for i in tqdm(range(cfg.nr),
                          desc="Repetitions".ljust(20),
                          leave=False,
                          position=3,
                          colour="yellow"):
                dataset = MyDataset(proximity_matrix=graph["proximity_matrix"],
                                    features=F_true,
                                    device=cfg.device,
                                    seed=cfg.seeds[i])
                dataset.get_x_miss(p_miss=p_miss, seed=cfg.seeds[i],
                                   max_missing=0, verbose=False,
                                   logging=cfg.logger, preserve_first_column=False,
                                   missing_mechanism=cfg.mecha, opt=cfg.opt)
                # create data object
                data = dataset[0]
                data = data.to(cfg.device)

                binary = True if graph["name"].lower() in ["cora", "cornell", "texas", "wisconsin", "citeseer",
                                                           "pubmed"] else False
                save_attributes = f"{graph['name']}_p_miss{int(p_miss * 100)}_n_rep{i}"
                baselines, matrices, scores_community_detection, elapsed_time = compute_baselines(
                    binary=binary,
                    data=data,
                    device=cfg.device
                )

                unique = f"{graph['name']}_pm{p_miss:.2f}_rep{i}"
                save_steps_path = []
                for k, mat in matrices.items():
                    save_steps_path.append(save_steps(mat,
                                                      cfg.UNIQUE,
                                                      graph["name"],
                                                      unique + f"_{k}"))

                df = pd.concat([df, pd.DataFrame({
                    "graph": [graph["name"]],
                    "elapsed_time": [elapsed_time],
                    "greedy_community_detection_ari": [scores_community_detection["ari"]],
                    "greedy_community_detection_ami": [scores_community_detection["ami"]],
                    "greedy_community_detection_nmi": [scores_community_detection["nmi"]],
                    "p_miss": [p_miss],
                    "n_rep_idx": [i],
                    "save_steps_path": [save_steps_path],
                    # val
                    "mae_avg_val": [baselines["mae_avg_val"]],
                    "rmse_avg_val": [baselines["rmse_avg_val"]],
                    "mae_avg_by_community_val": [baselines["mae_avg_by_community_val"]],
                    "rmse_avg_by_community_val": [baselines["rmse_avg_by_community_val"]],
                    "mae_rnd_val": [baselines["mae_rnd_val"]],
                    "rmse_rnd_val": [baselines["rmse_rnd_val"]],
                    "mae_zro_val": [baselines["mae_zro_val"]],
                    "rmse_zro_val": [baselines["rmse_zro_val"]],
                    "mae_avg_ngb_val": [baselines["mae_avg_ngb_val"]],
                    "rmse_avg_ngb_val": [baselines["rmse_avg_ngb_val"]],
                    # train_test
                    "mae_avg_test": [baselines["mae_avg_test"]],
                    "rmse_avg_test": [baselines["rmse_avg_test"]],
                    "mae_avg_by_community_test": [baselines["mae_avg_by_community_test"]],
                    "rmse_avg_by_community_test": [baselines["rmse_avg_by_community_test"]],
                    "mae_rnd_test": [baselines["mae_rnd_test"]],
                    "rmse_rnd_test": [baselines["rmse_rnd_test"]],
                    "mae_zro_test": [baselines["mae_zro_test"]],
                    "rmse_zro_test": [baselines["rmse_zro_test"]],
                    "mae_avg_ngb_test": [baselines["mae_avg_ngb_test"]],
                    "rmse_avg_ngb_test": [baselines["rmse_avg_ngb_test"]],
                }, index=[0])], ignore_index=True)

    df.to_json(f"./{cfg.UNIQUE}/"
               f"{graph['name']}_nr{cfg.nr}"
               f".json",
               orient="records",
               indent=4)
    df_s[graph["name"]] = df

    path_save = f"./{cfg.UNIQUE}/ALL_nr{cfg.nr}.json"

    df = pd.concat(list(df_s.values()), axis=0, ignore_index=True)
    df.to_json(path_save,
               orient="records",
               indent=4)

    return path_save
