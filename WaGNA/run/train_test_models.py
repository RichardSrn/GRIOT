from ..utils.graph_tool import load_matrices, add_missing_features
from ..datasets_class.dataset_class import MyDataset
from ..test_threads.threads import Threads
from ..model_imputers.imputer_GCN import GCN_IMPUTER
from ..model_imputers.imputer_GAT import GAT_IMPUTER
from ..model_imputers.imputer_GCT import GCT_IMPUTER
from ..model_imputers.imputer_RR import GNN_RR
from ..losses.MultiW import MultiW
from ..losses.FGW import FGW
from tqdm import tqdm
import numpy as np
import threading
import itertools
import torch
import os
from ..config import cfg
from ..utils.re_structure import should_restructure

# quieten warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

tqdm.pandas()


def train_test(early_break=False,
               show_progress_bar_outer=True,
               show_progress_bar_inner=True,
               multithreads=False, ):
    cfg.logger.info(f"train_test - {cfg.UNIQUE}")

    assert cfg.model in ["basic", "rr", "gcn", "gat", "gct"]

    if cfg.p_miss_s is None:
        start, stop, step = .1, .9, .1
        cfg.p_miss_s = np.arange(start, stop + step, step)

    if "MultiW" in [cfg.lossfn]:
        p_unif = (cfg.normalization[0] == "u")
        normalize_F = (cfg.normalization[1] == "r")
        normalize_MF_MC = (cfg.normalization[2] == "r")
    else:
        p_unif, normalize_F, normalize_MF_MC = None, None, None

    if cfg.lossfn == "FGW":
        cfg.lossfn = FGW
    elif cfg.lossfn == "MultiW":
        cfg.lossfn = MultiW
    print(cfg.lossfn)

    hyperparameters = list(itertools.product([cfg.batch_size], cfg.epsilons, [cfg.lossfn]))

    with open(os.path.join(cfg.UNIQUE, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {cfg.seeds}")
    cfg.logger.info(f"seeds for repetitions = {cfg.seeds}")

    # total_param = sum([len(g["features"]) for g in cfg.graphs_parameters]) * len(hyperparameters) * \
    #               len(cfg.p_miss_s) * cfg.nr * len(cfg.alphas)
    total_param = len(hyperparameters) * len(cfg.p_miss_s) * cfg.nr * len(cfg.alphas)

    ot_threads = Threads(total=total_param)

    count = 0
    over_count = 0
    # for graph in tqdm(cfg.graphs_parameters,
    #                   desc="Graphs".ljust(25),
    #                   leave=False,
    #                   position=0,
    #                   colour="black",
    #                   disable=not show_progress_bar_outer):
    graph = load_matrices(cfg.graph_parameters)
    name = graph["name"]

    for F_true in tqdm(graph["features"].values(),
                       desc="Features".ljust(25),
                       leave=False,
                       position=1,
                       colour="red",
                       disable=not show_progress_bar_outer):
        for batch_size, epsilon, lossfn in tqdm(hyperparameters,
                                                desc="Hyper-Parameters".ljust(25),
                                                leave=False,
                                                position=2,
                                                colour="magenta",
                                                disable=not show_progress_bar_outer):
            for p_miss in tqdm(list(cfg.p_miss_s),
                               desc="Missing Features".ljust(25),
                               leave=False,
                               position=3,
                               colour="blue",
                               disable=not show_progress_bar_outer):
                if multithreads:
                    threads = dict()
                else:
                    threads = None
                j = 0
                for i in tqdm(range(cfg.nr),
                              desc="Repetitions".ljust(25),
                              leave=False,
                              position=4,
                              colour="yellow",
                              disable=not show_progress_bar_inner):

                    dataset = MyDataset(features=F_true,
                                        proximity_matrix=graph["proximity_matrix"],
                                        device=torch.device("cpu"),  # cfg.device,
                                        keep_proximity_matrix=True,
                                        seed=cfg.seeds[i],
                                        logging=cfg.logger, )
                    dataset.get_x_miss(p_miss=p_miss, seed=cfg.seeds[i],
                                       max_missing=0, verbose=False,
                                       logging=cfg.logger, preserve_first_column=False,
                                       missing_mechanism=cfg.mecha, opt=cfg.opt)
                    # create data object
                    data = dataset[0]
                    data = data.to(torch.device("cpu"))  # cfg.device)
                    p = torch.sum(data.adjacency_matrix == 1, dim=1)

                    for alpha in tqdm(cfg.alphas,
                                      desc="Alpha".ljust(25),
                                      leave=False,
                                      position=5,
                                      colour="cyan",
                                      disable=not show_progress_bar_inner):
                        j += 1
                        loss_fn = lossfn(epsilon=epsilon,
                                         alpha=alpha,
                                         p=cfg.p,
                                         warn=False,
                                         path=cfg.UNIQUE,
                                         p_unif=p_unif,
                                         normalize_F=normalize_F,
                                         normalize_MF_MC=normalize_MF_MC,
                                         use_geomloss=cfg.use_geomloss,
                                         CrossEtpy=cfg.use_ce,
                                         plot=cfg.plot)

                        cfg.restruct_loss_fn = MultiW(alpha=alpha,
                                                      epsilon=0.01,
                                                      p=cfg.p,
                                                      method="sinkhorn",
                                                      path=cfg.UNIQUE,
                                                      p_unif=True,
                                                      normalize_F=False,
                                                      normalize_MF_MC=True,
                                                      CrossEtpy=cfg.use_ce,
                                                      plot=False,
                                                      use_geomloss=False)

                        if cfg.restructure == "auto":
                            # import time
                            #
                            # t = time.time()
                            # cfg.restructure = should_restructure(data.proximity_matrix.clone().to(cfg.device),
                            #                                      data.x_miss.clone().to(cfg.device),
                            #                                      data.y_train,
                            #                                      fill_nan=False)
                            # t1 = time.time() - t
                            # print(f"Time to compute restructure: {t1:.2f} seconds")
                            # cfg.restructure = should_restructure(data.proximity_matrix.clone().to(cfg.device),
                            #                                      data.x_miss.clone().to(cfg.device),
                            #                                      data.y_train,
                            #                                      fill_nan=True)
                            # t2 = time.time() - t - t1
                            # print(f"Time to compute restructure with fill_nan: {t2:.2f} seconds")
                            # exit()
                            cfg.restructure = True

                        kwargs = {
                            "epsilon": epsilon,
                            "batch_size": batch_size,
                            "lr": cfg.lr,
                            "n_iters": cfg.ni,
                            "n_pairs": cfg.np,
                            "tildeC": -1 if loss_fn.name == "FGW" else cfg.tildec,
                            "lossfn": loss_fn,
                            "alpha": alpha,
                            "logging": cfg.logger,
                            "name": name,
                            "p": p,
                            "early_break": early_break,
                            "p_miss": p_miss,
                            "i": i,
                            "show_progress_bar": show_progress_bar_inner,
                            "save_path": cfg.UNIQUE,
                            "data": data,
                            "device": cfg.device,
                            "restructure": cfg.restructure,
                        }

                        n, d = data.x_miss.shape
                        min_max = data.x_min_max

                        kwargs["max_iters"] = cfg.mi
                        if cfg.model == "rr":
                            models = dict()
                            for d_th in range(d):
                                ## predict the ith variable using d-1 others
                                models[d_th] = GNN_RR(d, device=cfg.device)
                            kwargs["models"] = models
                        elif cfg.model == "gcn":
                            kwargs["model"] = GCN_IMPUTER(d,
                                                          dropout=cfg.dropout,
                                                          min_max=min_max,
                                                          device=cfg.device)
                        elif cfg.model == "gat":
                            kwargs["model"] = GAT_IMPUTER(d,
                                                          num_heads=cfg.num_heads,
                                                          dropout=cfg.dropout,
                                                          min_max=min_max,
                                                          device=cfg.device,
                                                          concat=cfg.concat)
                        elif cfg.model == "gct":
                            kwargs["model"] = GCT_IMPUTER(d,
                                                          num_heads=cfg.num_heads,
                                                          dropout=cfg.dropout,
                                                          min_max=min_max,
                                                          device=cfg.device,
                                                          concat=cfg.concat)

                        if multithreads:
                            if cfg.model == "basic":
                                threads[count] = threading.Thread(target=ot_threads.ot_thread,
                                                                  name=f"ot_thread{count}",
                                                                  kwargs=kwargs)
                            elif cfg.model == "rr":
                                threads[count] = threading.Thread(target=ot_threads.rr_thread,
                                                                  name=f"rr_thread{count}",
                                                                  kwargs=kwargs)
                            elif cfg.model == "gcn" or cfg.model == "gat" or cfg.model == "gct":
                                threads[count] = threading.Thread(target=ot_threads.gnn_thread,
                                                                  name=f"gnn_thread{count}",
                                                                  kwargs=kwargs)
                            threads[count].start()
                        else:
                            if cfg.model == "basic":
                                ot_threads.ot_thread(**kwargs)
                            elif cfg.model == "rr":
                                ot_threads.rr_thread(**kwargs)
                            elif cfg.model == "gcn" or cfg.model == "gat" or cfg.model == "gct":
                                ot_threads.gnn_thread(**kwargs)
                        count += 1

                        del loss_fn

                # wait for all current threads to finish
                if multithreads:
                    for key in threads.keys():
                        threads[key].join()

                over_count += j
                # write in a file over.txt the number of threads that are over
                with open(f"./{cfg.UNIQUE}/over.txt", "w") as f:
                    f.write(f"{str(over_count)} / {str(total_param)}")
                # update json file
                path_save = ot_threads.save(cfg.ni, cfg.nr, cfg.np, cfg.lr, count=over_count, main_path=cfg.UNIQUE)

    return path_save
