# imports
from time import time

from ..utils.graph_tool import load_matrices
from ..datasets_class.dataset_class import MyDataset
from .imputers import OTimputer, RRimputer
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from ..config import cfg
import os



# fill missing features
def fill_missing(data, niter, save_path=".", unique="", graph_name="", batchsize=16, n_pairs=8,
                 device=torch.device('cpu')):
    model = OTimputer(batchsize=batchsize, niter=niter, n_pairs=n_pairs, device=device)
    unique = unique.replace("εEPS_", f"ε{model.eps:.2f}_")
    unique = unique.replace("lrLR_", f"lr{model.lr:.2f}_")
    binary = True if graph_name.lower() in ["cora", "cornell", "texas", "wisconsin", "citeseer", "pubmed"] else False
    t_start = time()
    F_pred, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path = model.fit_transform(data,
                                                                                                      verbose=False,
                                                                                                      binary=binary,
                                                                                                      save_path=save_path,
                                                                                                      unique=unique,
                                                                                                      graph_name=graph_name)
    elapsed_time = time() - t_start
    return F_pred, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path, elapsed_time


def fill_missing_algo3(data, niter, max_iters,
                       save_path=".", unique="", graph_name="", batchsize=16, n_pairs=8,
                       device=torch.device("cpu")):
    if data.x.shape[1] > 100:
        logging.ERROR("Too many features, algorithm 3 is not running with more than 100 features. (time complexity)")
        return None, None, None, None, None, None
    else:
        models = dict()
        d = data.x.shape[1]
        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = torch.nn.Linear(d - 1, 1).to(device)

        model = RRimputer(models=models, batchsize=batchsize, niter=niter, n_pairs=n_pairs, max_iter=max_iters,
                          device=device)
        unique = unique.replace("εEPS_", f"")
        unique = unique.replace("lrLR_", f"lr{model.lr:.2f}_")
        binary = True if graph_name.lower() in ["cora", "cornell", "texas", "wisconsin", "citeseer",
                                                "pubmed"] else False

        t_start = time()
        F_pred, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path = model.fit_transform(data,
                                                                                                          verbose=False,
                                                                                                          binary=binary,
                                                                                                          save_path=save_path,
                                                                                                          unique=unique,
                                                                                                          graph_name=graph_name)
        elapsed_time = time() - t_start
        return F_pred, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path, elapsed_time



# load data
def load_data(graph_parameters):
    return load_matrices(graph_parameters)



def baseline_muzellec():
    # names = [g["name"] for g in cfg.graphs_parameters]
    name = cfg.graph_parameters["name"]
    # cfg.UNIQUE = cfg.UNIQUE + "_" + "_".join(names)
    cfg.logger.info(f"baseline_OTtab - {cfg.UNIQUE}")

    os.makedirs(cfg.UNIQUE, exist_ok=True)

    if cfg.p_miss_s is None:
        start, stop, step = .1, .9, .1
        cfg.p_miss_s = np.arange(start, stop + step, step)

    with open(os.path.join(cfg.UNIQUE, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {cfg.seeds}")
    cfg.logger.info(f"seeds for repetitions = {cfg.seeds}")

    path_save = os.path.join(cfg.UNIQUE, f'baseline_muzellec_nr{cfg.nr}_np{cfg.np}_ni{cfg.ni}_bs{cfg.batch_size}.json')
    # graphs = load_data(cfg.graph_parameters)

    data_results = []

    # for i, graph in tqdm(list(enumerate(graphs)),
    #                      desc="graph".ljust(20),
    #                      position=1,
    #                      leave=False,
    #                      colour="green"):
    i=0
    graph = load_data(cfg.graph_parameters)
    name = graph["name"]

    for F_true in graph["features"].values():
        k = 0
        for p_miss in tqdm(cfg.p_miss_s,
                           desc="p_miss".ljust(20),
                           position=2,
                           leave=False,
                           colour="blue"):
            for j in tqdm(range(cfg.nr),
                          desc="rep".ljust(20),
                          position=3,
                          leave=False,
                          colour="red"):
                dataset = MyDataset(proximity_matrix=graph["proximity_matrix"],
                                    features=F_true,
                                    device=cfg.device,
                                    seed=cfg.seeds[j],
                                    logging=cfg.logger, )
                dataset.get_x_miss(p_miss=p_miss, seed=cfg.seeds[j],
                                   max_missing=0, verbose=False,
                                   logging=cfg.logger,
                                   preserve_first_column=False,
                                   missing_mechanism=cfg.mecha, opt=cfg.opt)
                # create data object
                data = dataset[0]
                data = data.to(cfg.device)

                unique = f"{name}_{cfg.model}_bs{cfg.batch_size}_εEPS_lrLR_ni{cfg.ni}_np{cfg.np}_" \
                         f"pm{p_miss:.2f}_rep{j}"

                if cfg.model == "basic":

                    _, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path, elapsed_time = fill_missing(
                        data,
                        cfg.ni,
                        save_path=cfg.UNIQUE,
                        unique=unique,
                        graph_name=name,
                        batchsize=cfg.batch_size,
                        n_pairs=cfg.np,
                        device=cfg.device
                    )

                elif cfg.model == "rr":

                    _, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path, elapsed_time = fill_missing_algo3(
                        data,
                        cfg.ni,
                        max_iters=cfg.mi,
                        save_path=cfg.UNIQUE,
                        unique=unique,
                        graph_name=name,
                        batchsize=cfg.batch_size,
                        n_pairs=cfg.np,
                        device=cfg.device
                    )
                else:
                    maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path, elapsed_time = None, None, None, None, None, None, None
                k += 1

                data_results.append({
                    'model': cfg.model,
                    'graph': name,
                    'elapsed_time': elapsed_time,
                    'p_miss': p_miss,
                    'n_rep_idx': j,
                    'label': "Muzellec",
                    'n_iters': cfg.ni,
                    'lossfn': 'OTImputer',
                    'mae_val': maes_val,
                    'rmse_val': rmses_val,
                    'mae_test': maes_test,
                    'rmse_test': rmses_test,
                    'loss': losses,
                    "save_steps_path": save_steps_path
                })

    # save before going to next graph
    df = pd.DataFrame(data_results)
    df.to_json(path_save,
               orient="records",
               indent=4)
    # plot_results(path_save, show=False, save=True)
    return path_save
