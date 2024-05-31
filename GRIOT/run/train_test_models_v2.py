from ..utils.graph_tool import load_matrices, add_missing_features
from ..datasets_class.dataset_class import MyDataset
from ..test_threads.threads import Threads
from ..model_imputers.imputer_GCN import GCN_IMPUTER
from ..model_imputers.imputer_GAT import GAT_IMPUTER
from ..model_imputers.imputer_RR import GNN_RR
from ..losses.MultiW import MultiW
from ..losses.FGW import FGW
from tqdm import tqdm
import numpy as np
import threading
import itertools
import torch
import os

# quieten warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

tqdm.pandas()


def train_test2(model,
                graph_parameters,
                unique_path,
                logging,
                seeds=None,
                p_miss_s=None,
                missing_mechanism=None,
                opt=None,
                lr=1e-2,
                max_iters=None,
                n_iters=None,
                n_rep=None,
                n_pairs=None,
                tildeC=True,
                lossfns=None,  # (MultiW, FGW),
                alphas=None,
                epsilons=None,
                p_=None,
                normalization="unn",
                batch_sizes=None,
                early_break=False,
                show_progress_bar_outer=True,
                show_progress_bar_inner=True,
                multithreads=False,
                path_save="",
                device=torch.device("cpu")):
    logging.info(f"train_test - {path_save}")

    assert model in ["basic", "rr", "gcn", "gat"]

    if p_miss_s is None:
        start, stop, step = .1, .9, .1
        p_miss_s = np.arange(start, stop + step, step)

    if "MultiW" in lossfns:
        p_unif = (normalization[0] == "u")
        normalize_F = (normalization[1] == "r")
        normalize_MF_MC = (normalization[2] == "r")
    else:
        p_unif, normalize_F, normalize_MF_MC = None, None, None
    lossfns = [eval(lossfn) for lossfn in lossfns]

    hyperparameters = list(itertools.product(batch_sizes, epsilons, lossfns))

    with open(os.path.join(path_save, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {seeds}")
    logging.info(f"seeds for repetitions = {seeds}")

    # total_param = sum([len(g["features"]) for g in graphs_parameters]) * len(hyperparameters) * \
    #               len(p_miss_s) * n_rep * len(alphas)
    total_param = len(hyperparameters) * len(p_miss_s) * n_rep * len(alphas)

    ot_threads = Threads(total=total_param)

    count = 0
    over_count = 0
    # for graph in tqdm(graphs_parameters,
    #                   desc="Graphs".ljust(25),
    #                   leave=False,
    #                   position=0,
    #                   colour="black",
    #                   disable=not show_progress_bar_outer):
    graph = load_matrices(graph_parameters)
    graph = load_matrices(graph)
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
            for p_miss in tqdm(list(p_miss_s),
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
                for i in tqdm(range(n_rep),
                              desc="Repetitions".ljust(25),
                              leave=False,
                              position=4,
                              colour="yellow",
                              disable=not show_progress_bar_inner):

                    dataset = MyDataset(features=F_true,
                                        proximity_matrix=graph["proximity_matrix"],
                                        device=device,
                                        keep_proximity_matrix=True,
                                        seed=seeds[i],
                                        logging=logging, )
                    dataset.get_x_miss(p_miss=p_miss, seed=seeds[i],
                                       max_missing=0, verbose=False,
                                       logging=logging, preserve_first_column=False,
                                       missing_mechanism=missing_mechanism, opt=opt)
                    # create data object
                    data = dataset[0]
                    data = data.to(device)
                    p = torch.sum(data.adjacency_matrix == 1, dim=1)

                    for alpha in tqdm(alphas,
                                      desc="Alpha".ljust(25),
                                      leave=False,
                                      position=5,
                                      colour="cyan",
                                      disable=not show_progress_bar_inner):
                        j += 1
                        loss_fn = lossfn(epsilon=epsilon,
                                         alpha=alpha,
                                         p=p_,
                                         warn=False,
                                         path=unique_path,
                                         p_unif=p_unif,
                                         normalize_F=normalize_F,
                                         normalize_MF_MC=normalize_MF_MC)

                        kwargs = {
                            "epsilon": epsilon,
                            "batch_size": batch_size,
                            "lr": lr,
                            "n_iters": n_iters,
                            "n_pairs": n_pairs,
                            "tildeC": False if loss_fn.name == "FGW" else tildeC,
                            "lossfn": loss_fn,
                            "alpha": alpha,
                            "logging": logging,
                            "name": name,
                            "p": p,
                            "early_break": early_break,
                            "p_miss": p_miss,
                            "i": i,
                            "show_progress_bar": show_progress_bar_inner,
                            "save_path": unique_path,
                            "data": data,
                            "device": device,
                        }

                        n, d = data.x_miss.shape
                        min_max = data.x_min_max

                        kwargs["max_iters"] = max_iters
                        if model == "gcn":
                            kwargs["model"] = GCN_IMPUTER(d, min_max, device=device)
                        elif model == "gat":
                            kwargs["model"] = GAT_IMPUTER(d, min_max, device=device)

                        if multithreads:
                            threads[count] = threading.Thread(target=ot_threads.griot2_thread,
                                                              name=f"gnn_thread{count}",
                                                              kwargs=kwargs)
                            threads[count].start()
                        else:
                            ot_threads.griot2_thread(**kwargs)
                        count += 1

                        del loss_fn

                # wait for all current threads to finish
                if multithreads:
                    for key in threads.keys():
                        threads[key].join()

                over_count += j
                # write in a file over.txt the number of threads that are over
                with open(f"./{unique_path}/over.txt", "w") as f:
                    f.write(f"{str(over_count)} / {str(total_param)}")
                # update json file
                path_save = ot_threads.save(n_iters, n_rep, n_pairs, lr, count=over_count, main_path=unique_path)

    return path_save
