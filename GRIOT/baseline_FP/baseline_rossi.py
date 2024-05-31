from ..utils.graph_tool import load_matrices
from ..datasets_class.dataset_class import MyDataset
from ..utils.MissingDataOT_network_utils import MAE, RMSE
from ..utils.save import save_steps
from ..baseline_FP.filling_strategies import filling
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import torch
import os
from ..config import cfg

# quieten warnings
import warnings


class Args():
    """
    Default values from Rossi et al. Feature Propagation
    """

    def __init__(self):
        self.dataset_name = "Photo"
        self.mask_type = "uniform"
        self.filling_method = "dirichlet_diffusion"
        self.model = "gcn"
        self.missing_rate = 0.1
        self.patience = 200
        self.lr = 0.005
        self.epochs = 10000
        self.n_runs = 10
        self.hidden_dim = 64
        self.attention_type = "transformer"
        self.attention_dim = 64
        self.num_layers = 2
        self.num_iterations = 40
        self.lp_alpha = 0.9
        self.dropout = 0.5
        self.batch_size = 1024
        self.reconstruction_only = False
        self.graph_sampling = False
        self.log = "WARNING"
        self.homophily = None
        self.gpu_idx = 0


# fill missing features
def fill_missing_fp(data, save_path=".", unique="", graph_name=""):
    args = Args()
    x = data.x.clone()
    maes_val = []
    rmses_val = []
    maes_test = []
    rmses_test = []
    feature_mask = ~data.x_miss_mask.bool().clone()
    # feature_mask = torch.zeros(data.x.shape).bool() +1
    t_start = time()
    filled_features, lfp = filling(args.filling_method,
                                   data.edge_index,
                                   x,
                                   feature_mask,
                                   args.num_iterations,
                                   args.attention_dim,
                                   args.attention_type) if args.model not in ["gcnmf", "pagnn", "missing_gat"] \
        else (torch.full_like(x, float('nan')), None)
    elapsed_time = time() - t_start
    # print(filled_features)
    maes_val.append(MAE(filled_features, data.x.float(), data.val_mask.bool()).item())
    rmses_val.append(RMSE(filled_features, data.x.float(), data.val_mask.bool()).item())
    maes_test.append(MAE(filled_features, data.x.float(), data.test_mask.bool()).item())
    rmses_test.append(RMSE(filled_features, data.x.float(), data.test_mask.bool()).item())

    # # CLASSIFICATION
    # graph_sampling = True if args.graph_sampling or args.dataset_name == "OGBN-Products" else False
    # train_loader = NeighborSampler(data.edge_index,
    #                                node_idx=np.where(data.train_mask)[0],  # split_idx['train'],
    #                                sizes=[15, 10],
    #                                batch_size=args.batch_size,
    #                                shuffle=True,
    #                                num_workers=12) if graph_sampling else None
    # model = get_model(model_name=args.model,
    #                   num_features=data.num_features,
    #                   num_classes=data.num_classes,
    #                   edge_index=data.edge_index,
    #                   x=x,
    #                   mask=feature_mask,
    #                   args=args).to(device)
    # params = list(model.parameters())
    # optimizer = torch.optim.Adam(params, lr=args.lr)
    # critereon = torch.nn.NLLLoss()
    # t_start = time()
    # for epoch in tqdm(range(args.epochs),
    #                   desc="epochs".ljust(25),
    #                   colour="green"):
    #     x = torch.where(feature_mask, data.x, filled_features)
    #     train(model,
    #           x,
    #           data,
    #           optimizer,
    #           critereon,
    #           train_loader=train_loader,
    #           device=device)
    #     maes.append(MAE(x, data.x.float(), data.test_mask.bool()).item())
    #     rmses.append(RMSE(x, data.x.float(), data.test_mask.bool()).item())
    #     print(maes[-1], rmses[-1])
    # elapsed_time = time() - t_start

    # save last step
    save_steps_path = [save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None],
                                             x.cpu().detach().numpy())),
                                  save_path,
                                  graph_name,
                                  unique + f"_FINAL")]
    # return F_pred, maes, rmses, save_steps_path, elapsed_time
    return x, maes_val, rmses_val, maes_test, rmses_test, save_steps_path, elapsed_time


def baseline_rossi(show_progress_bar_outer=True,
                   show_progress_bar_inner=True,):
    cfg.logger.info(f"baseline_rossi - {cfg.UNIQUE}")

    assert cfg.model in ["fp"]

    if cfg.p_miss_s is None:
        start, stop, step = .1, .9, .1
        cfg.p_miss_s = np.arange(start, stop + step, step)

    with open(os.path.join(cfg.UNIQUE, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {cfg.seeds}")
    cfg.logger.info(f"seeds for repetitions = {cfg.seeds}")

    # total_param = sum([len(g["features"]) for g in cfg.graphs_parameters]) * len(cfg.p_miss_s) * cfg.nr
    total_param = len(cfg.p_miss_s) * cfg.nr

    over_count = 0
    data_results = []
    path_save_old = ""
    # for i, graph in tqdm(list(enumerate(cfg.graphs_parameters)),
    #                      desc="Graphs".ljust(25),
    #                      leave=False,
    #                      position=0,
    #                      colour="black",
    #                      disable=not show_progress_bar_outer):
    i = 0
    graph = load_matrices(cfg.graph_parameters)
    name = graph["name"]

    for F_true in tqdm(graph["features"].values(),
                       desc="Features".ljust(25),
                       leave=False,
                       position=1,
                       colour="red",
                       disable=not show_progress_bar_outer):
        k = 0
        for p_miss in tqdm(list(cfg.p_miss_s),
                           desc="Missing Features".ljust(25),
                           leave=False,
                           position=3,
                           colour="blue",
                           disable=not show_progress_bar_outer):
            for j in tqdm(range(cfg.nr),
                          desc="Repetitions".ljust(25),
                          leave=False,
                          position=4,
                          colour="yellow",
                          disable=not show_progress_bar_inner):
                dataset = MyDataset(proximity_matrix=graph["proximity_matrix"],
                                    features=F_true,
                                    device=cfg.device,
                                    seed=cfg.seeds[j],
                                    logging=cfg.logger,)
                dataset.get_x_miss(p_miss=p_miss, seed=cfg.seeds[j],
                                   max_missing=0, verbose=False,
                                   logging=cfg.logger, preserve_first_column=False,
                                   missing_mechanism=cfg.mecha, opt=cfg.opt)
                # create data object
                data = dataset[0]
                data = data.to(cfg.device)
                unique = f"{name}_{cfg.model}_pm{p_miss:.2f}_rep{j}"
                _, maes_val, rmses_val, maes_test, rmses_test, save_steps_path, elapsed_time = fill_missing_fp(data=data,
                                                                                save_path=cfg.UNIQUE,
                                                                                unique=unique,
                                                                                graph_name=name)
                k += 1

                data_results.append({
                    'model': cfg.model,
                    'graph': name,
                    'elapsed_time': elapsed_time,
                    'p_miss': p_miss,
                    'n_rep_idx': j,
                    'label': "Rossi",
                    'mae_val': maes_val,
                    'rmse_val': rmses_val,
                    'mae_test': maes_test,
                    'rmse_test': rmses_test,
                    "save_steps_path": save_steps_path
                })
        over_count += k

        # save before going to next graph
        path_save_new = f"./{cfg.UNIQUE}/data_nr{cfg.nr}_{over_count}o{total_param}.json"
        df = pd.DataFrame(data_results)
        df.to_json(path_save_new,
                   orient="records",
                   indent=4)

        if path_save_old != "" and path_save_old != path_save_new:
            os.remove(path_save_old)
        path_save_old = path_save_new

    return path_save_new
