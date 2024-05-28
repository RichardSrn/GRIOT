import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from ..config import cfg
from ..utils.graph_tool import load_matrices
from ..datasets_class.dataset_class import MyDataset
from ..baseline_FP.baseline_rossi import fill_missing_fp
from ..utils.round_ratio import round_ratio
from ..utils.save import save_steps
from ..utils.MissingDataOT_network_utils import MAE, RMSE
from ..models_training.GraphGCNimputer import GCNimputer
from ..model_imputers.imputer_GCN import GCN_IMPUTER
from ..model_imputers.imputer_GAT import GAT_IMPUTER
from ..losses.MultiW import MultiW
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
import torch.nn.functional as F
import time
import itertools

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score


def fill_rand(data, binary=False, unique="", save_path="", graph_name=""):
    t_start_rnd = time.time()
    # set F_rnd to random values
    F_rnd = torch.rand(size=data.x_miss.shape)
    # multiply each column by the difference between max and min of the corresponding column
    F_rnd = (F_rnd * (np.nanmax(data.x_miss.cpu().detach().numpy(), axis=0) -
                      np.nanmin(data.x_miss.cpu().detach().numpy(), axis=0))
             + np.nanmin(data.x_miss.cpu().detach().numpy(), axis=0)).to(cfg.device)
    if binary:
        # round values to the closest binary
        F_rnd = round_ratio(F_rnd, data.x_miss)
    # copy data.x_miss["X_incomp"] as F_filled_rnd
    F_filled_rnd = data.x_miss.clone()
    # impute na values of F_filled_avg with the average of each column of F_true
    F_filled_rnd = torch.where(torch.isnan(F_filled_rnd), F_rnd, F_filled_rnd)
    elapsed_time_rnd = time.time() - t_start_rnd

    x = torch.hstack((data.y[:, None], F_filled_rnd))
    mae_rnd_val = MAE(F_filled_rnd, data.x, data.x_miss_mask_val).item()
    rmse_rnd_val = RMSE(F_filled_rnd, data.x, data.x_miss_mask_val).item()
    mae_rnd_test = MAE(F_filled_rnd, data.x, data.x_miss_mask_test).item()
    rmse_rnd_test = RMSE(F_filled_rnd, data.x, data.x_miss_mask_test).item()

    save_steps_path = []
    save_steps_path.append(save_steps(x,
                                      save_path,
                                      graph_name,
                                      unique,
                                      "FINAL"))

    return x, mae_rnd_val, rmse_rnd_val, mae_rnd_test, rmse_rnd_test, save_steps_path, elapsed_time_rnd


def fill_gcn(data,
             model="",
             unique="",
             save_path="",
             graph_name="",
             logging=None,
             n_iters=25,
             n_pairs=8,
             lr=None,
             loss_fn=None):
    n, d = data.x_miss.shape
    p = torch.sum(data.adjacency_matrix == 1, dim=1)
    name = graph_name

    # define model
    if model == "gcn":
        model = GCN_IMPUTER(d, device=cfg.device)
    elif model == "gat":
        model = GCN_IMPUTER(d, device=cfg.device)
    else:
        raise ValueError(f"model '{model}' not defined, use 'gcn' or 'gat'.")

    # load model
    ot_imputer = GCNimputer(
        model=model,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        max_iters=cfg.mi,
        n_iters=cfg.ni,
        n_pairs=cfg.np,
        tildeC=cfg.tildec,
        lossfn=loss_fn,
        logging=cfg.logger,
        device=cfg.device,
        # show_progress_bar=show_progress_bar
    )
    # fit model
    t_start = time.time()
    ot_imp, ot_metrics, ot_loss, save_steps_path = ot_imputer.fit_transform(
        data=data,
        p=p,
        verbose=False,
        report_interval=n_iters + 1,
        binary=True if name.lower() in ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin"] else False,
        # early_break=early_break,
        unique=unique,
        save_path=save_path,
        graph_name=name
    )
    elapsed_time = time.time() - t_start

    mae_val = ot_metrics["mae_val"]
    rmse_val = ot_metrics["rmse_val"]
    mae_test = ot_metrics["mae_test"]
    rmse_test = ot_metrics["rmse_test"]

    return ot_imp, mae_val, rmse_val, mae_test, rmse_test, save_steps_path, elapsed_time


def fill_missing_for_pagnn(model,
                           data,
                           save_path=".",
                           unique="",
                           graph_name="",
                           binary=False,
                           logging=None,
                           n_iters=None,
                           n_pairs=None,
                           lr=None,
                           loss_fn=None):
    if model == "fp":
        return fill_missing_fp(data, save_path, unique, graph_name)
    elif model == "rand":
        return fill_rand(data, binary, unique, save_path, graph_name)
    elif model == "gcn" or model == "gat":
        return fill_gcn(data=data,
                        model=model,
                        unique=unique,
                        save_path=save_path,
                        graph_name=graph_name,
                        logging=logging,
                        n_iters=n_iters,
                        n_pairs=n_pairs,
                        lr=lr,
                        loss_fn=loss_fn)


def pagnn_scores(path_json=None,
                 final_only=None):
    def get_scores(x):
        # gr = next((item for item in graphs_parameters if item["name"] == x["graph"]), None)
        gr = cfg.graph_parameters
        proximity_matrix = torch.from_numpy(np.load(gr["proximity_matrix"])).to(cfg.device)
        scores = None
        mask_train = torch.from_numpy(np.load(x["save_path_train_mask"])).to(cfg.device)
        mask_val = torch.from_numpy(np.load(x["save_path_val_mask"])).to(cfg.device)
        mask_test = torch.from_numpy(np.load(x["save_path_test_mask"])).to(cfg.device)
        for i, step_path in enumerate(x["save_steps_path"]):
            if final_only and not step_path.endswith("FINAL_.npy"):
                continue
            F_set = torch.from_numpy(np.load(step_path)).to(cfg.device)
            scores_gnn = train_test_pagnn(proximity_matrix,
                                          F_set.clone(),
                                          mask_train=mask_train,
                                          mask_val=mask_val,
                                          mask_test=mask_test,
                                          seed=cfg.seeds[x["n_rep_idx"]],
                                          save_path="/".join(step_path.split("/")[:-1]),
                                          save_name=step_path.split("/")[-1].replace(".npy", ""),
                                          device=cfg.device)
            if scores is None:
                scores = {k: [v] for k, v in scores_gnn.items()}
            else:
                for k, v in scores_gnn.items():
                    scores[k].append(v)
        return scores

    # load json file
    df = pd.read_json(path_json)
    df = pd.concat([df, df[["graph",
                            "save_steps_path",
                            "save_path_train_mask",
                            "save_path_val_mask",
                            "save_path_test_mask",
                            "n_rep_idx"]].progress_apply(get_scores,
                                                         axis=1,
                                                         result_type="expand")
                    ], axis=1)
    path_save = path_json.replace(".json", "_classifiers_scores.json")
    df.to_json(path_save,
               orient="records",
               indent=4)
    return path_save


def train_test_pagnn(topology_matrix,
                     F_set,
                     mask_train,
                     mask_val,
                     mask_test,
                     return_pred=False,
                     seed=42,
                     epochs=251,
                     show_progress_bar=False,
                     show=False,
                     save_path="",
                     save_name="",
                     device=None,
                     **kwargs):
    # args = Args()
    dataset = MyDataset(proximity_matrix=topology_matrix,
                        features=F_set,
                        device=cfg.device,
                        seed=seed,
                        **kwargs)

    # create data object
    data = dataset[0]
    data = data.to(device)

    num_features = data.num_features
    num_classes = data.num_classes
    if torch.is_tensor(mask_train):
        feature_mask = mask_train.clone().bool().to(device)
    else:
        feature_mask = torch.from_numpy(mask_train).clone().bool().to(device)
    edge_index = data.edge_index.to(device)

    # create model
    # model = Net(data, dataset, lr=lr, device=device, **kwargs).to(device)
    model = PaGNN(num_features=num_features,
                  num_classes=num_classes,
                  hidden_dim=64,  # args.hidden_dim,
                  dropout=0.5,  # args.dropout,
                  mask=feature_mask,
                  edge_index=edge_index,
                  device=cfg.device)

    os.makedirs(os.path.join(save_path, "model_summary"), exist_ok=True)
    # model.summary(show=False, save_path=os.path.join(save_path, "model_summary", save_name) + "_MODEL_SUMMARY.log")
    metrics = []
    losses = []

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.005)  # args.lr)
    critereon = torch.nn.NLLLoss()
    if torch.is_tensor(F_set):
        x = F_set[:, 1:].clone().to(device)
    else:
        x = torch.from_numpy(F_set[:, 1:]).clone().to(device)
    split_idx = dataset.get_idx_split() if hasattr(dataset, 'get_idx_split') else None

    graph_sampling = False  # True if args.graph_sampling else False
    train_loader = None
    # NeighborSampler(dataset.data.edge_index,
    # node_idx=split_idx['train'],
    # sizes=[15, 10],
    # batch_size=1024,  # args.batch_size,
    # shuffle=True,
    # num_workers=12) if graph_sampling else None
    inference_loader = None
    # NeighborSampler(dataset.data.edge_index,
    # node_idx=None,
    # sizes=[-1],
    # batch_size=1024,
    # shuffle=False,
    # num_workers=12) if graph_sampling else None

    for epoch in tqdm(range(epochs),
                      desc="training",
                      position=0 if type(show_progress_bar) == bool else show_progress_bar,
                      leave=False,
                      disable=not show_progress_bar):
        # loss = model.train_model()
        loss = model.train_model(x,
                                 data,
                                 mask_train.any(1),
                                 optimizer,
                                 critereon,
                                 train_loader=train_loader,
                                 device=cfg.device)

        if show or save_path:
            losses.append({"epoch": epoch,
                           "loss": loss})
            if epoch % 10 and epoch < epochs - 1:
                # scores = model.train_test()
                scores = model.test(x=x,
                                    data=data,
                                    mask_train=mask_train,
                                    mask_val=mask_val,
                                    mask_test=mask_test,
                                    # evaluator=None,  # evaluator,
                                    inference_loader=inference_loader,
                                    device=cfg.device)
                metrics.append({"epoch": epoch, **scores})

    # scores = model.train_test(training_data=False)
    scores = model.test(x=x,
                        data=data,
                        mask_train=mask_train,
                        mask_val=mask_val,
                        mask_test=mask_test,
                        # evaluator=None,  # evaluator,
                        inference_loader=inference_loader,
                        device=cfg.device)
    if return_pred:
        pred = model().max(1)[1].cpu().detach().numpy()
        return {**scores, "pred": pred}
    else:
        return scores


class PaGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0., mask=None, edge_index=None,
                 device=torch.device("cpu")):
        super(PaGNN, self).__init__()
        # NOTE: It is not specified in their paper (https://arxiv.org/pdf/2003.10130.pdf), but the only way for their model to work is to have only the first layer
        # to be what they describe and the others to be standard GCN layers. Otherwise, the feature matrix would change dimensionality, and it couldn't be
        # multiplied elmentwise with the mask anymore
        self.convs = ModuleList([PaGNNConv(num_features, hidden_dim, mask, edge_index, device=cfg.device)])
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim).to(device))
        self.convs.append(GCNConv(hidden_dim, num_classes).to(device))
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.convs[-1](x, edge_index)
        return torch.nn.functional.log_softmax(out, dim=1)

    def train_model(self, x, data, mask_train, optimizer, critereon, train_loader=None, device="cuda"):
        self.train()

        return self.train_sampled(train_loader,
                                  x,
                                  data,
                                  mask_train,
                                  optimizer,
                                  critereon,
                                  device) if train_loader \
            else self.train_full_batch(x,
                                       data,
                                       mask_train,
                                       optimizer,
                                       critereon)

    def train_full_batch(self, x, data, mask_train, optimizer, critereon):
        self.train()

        optimizer.zero_grad()

        y_pred = self(x, data.edge_index)[mask_train]
        y_true = data.y[mask_train].squeeze()

        loss = critereon(y_pred, y_true)
        loss.backward()
        optimizer.step()

        return loss

    def train_sampled(self, train_loader, x, data, mask_train, optimizer, critereon, device):
        self.train()

        total_loss = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            x_batch = x[n_id]

            optimizer.zero_grad()
            y_pred = self(x_batch, adjs=adjs, full_batch=False)
            y_true = data.y[n_id[:batch_size]].squeeze()
            loss = critereon(y_pred, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # logging.debug(f"Batch loss: {loss.item():.2f}")

        return total_loss / len(train_loader)

    @torch.no_grad()
    def test(self,
             x,
             data,
             mask_train,
             mask_val,
             mask_test,
             inference_loader=None,
             device="cuda",
             training_data=True):

        # flatten masks by turning them into a single column
        # such that a row with at least one True value is True
        mask_train = mask_train.any(axis=1)
        mask_val = mask_val.any(axis=1)
        mask_test = mask_test.any(axis=1)

        self.eval()
        logits = self.inference_sampled(x,
                                        inference_loader,
                                        device) if inference_loader \
            else self.inference_full_batch(x,
                                           data.edge_index)
        # pred = logits[mask].max(1)[1]
        # y_true = data.y[mask]
        # y_pred = pred.unsqueeze(1)
        pred = logits.max(1)[1]
        y_true = data.y
        y_pred = pred  # .unsqueeze(1)
        y_pred_proba = logits

        scores = {}
        cycle = []

        if training_data:
            # mask_train = data.train_mask
            pred_train = y_pred[mask_train]
            pred_train_proba = y_pred_proba[mask_train]
            acc_train = accuracy_score(data.y[mask_train].cpu(), pred_train.cpu())
            scores["gnn_accuracy_train"] = acc_train
            cycle.append([data.y[mask_train], pred_train, pred_train_proba, "train"])

        # mask_val = data.val_mask
        pred_val = y_pred[mask_val]
        pred_val_proba = y_pred_proba[mask_val]
        acc_val = accuracy_score(data.y[mask_val].cpu(), pred_val.cpu())
        scores["gnn_accuracy_val"] = acc_val
        cycle.append([data.y[mask_val], pred_val, pred_val_proba, "val"])

        # mask_test = data.test_mask
        pred_test = y_pred[mask_test]
        pred_test_proba = y_pred_proba[mask_test]
        acc_test = accuracy_score(data.y[mask_test].cpu(), pred_test.cpu())
        scores["gnn_accuracy_test"] = acc_test
        cycle.append([data.y[mask_test], pred_test, pred_test_proba, "test"])

        for data_true, data_pred, data_pred_proba, data_type in cycle:
            for average in ["micro", "macro", "weighted"]:
                s = precision_recall_fscore_support(data_true.cpu(),
                                                    data_pred.cpu(),
                                                    average=average,
                                                    zero_division=0)
                scores[f"gnn_precision{'_' + data_type}_{average}"] = s[0]
                scores[f"gnn_recall{'_' + data_type}_{average}"] = s[1]
                scores[f"gnn_f1{'_' + data_type}_{average}"] = s[2]

            scores[f"gnn_ari{'_' + data_type}"] = adjusted_rand_score(
                data_true.cpu(),
                data_pred.cpu()
            )
            scores[f"gnn_ami{'_' + data_type}"] = adjusted_mutual_info_score(
                data_true.cpu(),
                data_pred.cpu()
            )
            scores[f"gnn_nmi{'_' + data_type}"] = normalized_mutual_info_score(
                data_true.cpu(),
                data_pred.cpu()
            )

            # get roc_auc_score
            data_true_proba = torch.zeros(data_pred_proba.shape)
            data_true_proba[torch.arange(data_pred_proba.shape[0]), data_true] = 1
            try:
                score = roc_auc_score(
                    data_true_proba.cpu(),
                    data_pred_proba.cpu(),
                    multi_class="ovo",
                    average="macro"
                )
            except ValueError:
                score = 0
            scores[f"gnn_roc_auc_score{'_' + data_type}"] = score

        return scores

    def inference_full_batch(self, x, edge_index):
        out = self(x, edge_index)
        return out

    def inference_sampled(self, x, inference_loader, device):
        return self.inference(x, inference_loader, device)


class PaGNNConv(torch.nn.Module):
    def __init__(self, in_features, out_features, mask, edge_index, device):
        super(PaGNNConv, self).__init__()
        self.lin = torch.nn.Linear(in_features, out_features).to(device)
        self.mask = mask.to(torch.float64).to(device)
        edge_weight = get_symmetrically_normalized_adjacency(edge_index, mask.shape[0]).to(device)
        self.adj = torch.sparse_coo_tensor(edge_index,
                                           values=edge_weight,
                                           dtype=torch.float64,
                                           device=edge_index.device)
        self.device = device

    def forward(self, x, edge_index):
        x[x.isnan()] = 0
        numerator = torch.sparse.mm(self.adj, torch.ones_like(x, device=self.device)) * torch.sparse.mm(self.adj,
                                                                                                        self.mask * x)
        denominator = torch.sparse.mm(self.adj, self.mask)
        ratio = torch.nan_to_num(numerator / denominator)
        x = self.lin(ratio)

        return x


def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return DAD


def baseline_pagnn(binary=False,
                   show_progress_bar_outer=True,
                   show_progress_bar_inner=True):
    cfg.logger.info(f"baseline_rossi - {cfg.UNIQUE}")

    assert cfg.model in ["rand", "fp", "gcn"], f"model {cfg.model} not in ['rand', 'fp', 'gcn']"

    if cfg.p_miss_s is None:
        start, stop, step = .1, .9, .1
        cfg.p_miss_s = np.arange(start, stop + step, step)

    with open(os.path.join(cfg.UNIQUE, "seeds_missing_features.log"), "w") as file:
        file.write(f"seeds : {cfg.seeds}")
    cfg.logger.info(f"seeds for repetitions = {cfg.seeds}")

    if cfg.model == "gcn":
        hyperparameters = list(itertools.product([cfg.batch_size], cfg.epsilons, [eval(cfg.lossfn)]))
    else:
        cfg.alphas = [None]
        hyperparameters = [[None, None, None]]

    # total_param = sum([len(g["features"]) for g in cfg.graphs_parameters]) * len(hyperparameters) * \
    #               len(cfg.p_miss_s) * cfg.nr * len(cfg.alphas)
    total_param = len(hyperparameters) * len(cfg.p_miss_s) * cfg.nr * len(cfg.alphas)

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
        for batch_size, epsilon, lossfn in tqdm(hyperparameters,
                                                desc="Hyper-Parameters".ljust(25),
                                                leave=False,
                                                position=2,
                                                colour="magenta",
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
                                        logging=cfg.logger,
                                        keep_proximity_matrix=(cfg.model == "gcn"))
                    dataset.get_x_miss(p_miss=p_miss, seed=cfg.seeds[j],
                                       max_missing=0, verbose=False,
                                       logging=cfg.logger, preserve_first_column=False,
                                       missing_mechanism=cfg.mecha, opt=cfg.opt)
                    # create data object
                    data = dataset[0]
                    data = data.to(cfg.device)

                    for alpha in tqdm(cfg.alphas,
                                      desc="Alpha".ljust(25),
                                      leave=False,
                                      position=5,
                                      colour="cyan",
                                      disable=not show_progress_bar_inner):

                        unique = f"{name}_{cfg.model}_pm{p_miss:.2f}_rep{j}"
                        if cfg.model == "gcn":
                            loss_fn = lossfn(
                                epsilon=epsilon,
                                alpha=alpha,
                                p=cfg.p,
                                warn=False,
                                path=cfg.UNIQUE,

                                p_unif=(cfg.normalization[0] == "u"),
                                normalize_F=(cfg.normalization[1] == "r"),
                                normalize_MF_MC=(cfg.normalization[2] == "r"),
                                use_geomloss=cfg.use_geomloss,
                                CrossEtpy=cfg.use_ce,
                                plot=cfg.plot
                            )
                            unique = unique + f"ε{epsilon:.2f}_α{alpha:.2f}_reg{cfg.normalization}_bs{batch_size}_lr{cfg.lr:.2f}_ni{cfg.ni}_np{cfg.np}_ls{loss_fn.name}"
                        else:
                            loss_fn = None

                        _, maes_val, rmses_val, maes_test, rmses_test, save_steps_path, elapsed_time = fill_missing_for_pagnn(
                            model=cfg.model,
                            data=data,
                            save_path=cfg.UNIQUE,
                            unique=unique,
                            graph_name=name,
                            binary=binary,
                            logging=cfg.logger,
                            n_iters=cfg.ni,
                            n_pairs=cfg.np,
                            loss_fn=loss_fn,
                            lr=cfg.lr
                        )
                        k += 1

                        save_path_train_mask = save_steps_path[-1].replace("FINAL_", "TRAIN_MASK_")
                        save_path_val_mask = save_steps_path[-1].replace("FINAL_", "VAL_MASK_")
                        save_path_test_mask = save_steps_path[-1].replace("FINAL_", "TEST_MASK_")

                        np.save(save_path_train_mask, data.x_miss_mask_train.cpu().detach().numpy())
                        np.save(save_path_val_mask, data.x_miss_mask_val.cpu().detach().numpy())
                        np.save(save_path_test_mask, data.x_miss_mask_test.cpu().detach().numpy())

                        data_results.append({
                            'model': cfg.model,
                            'graph': name,
                            'elapsed_time': elapsed_time,
                            'p_miss': p_miss,
                            'lossfn': loss_fn.name if loss_fn else None,
                            "alpha": f'α={alpha:.2f}' if alpha is not None else None,
                            "epsilon": epsilon,
                            "reg": cfg.normalization,
                            'n_rep_idx': j,
                            'label': "PaGNN",
                            'mae_val': maes_val,
                            'rmse_val': rmses_val,
                            'mae_test': maes_test,
                            'rmse_test': rmses_test,
                            "save_steps_path": save_steps_path,
                            "save_path_train_mask": save_path_train_mask,
                            "save_path_val_mask": save_path_val_mask,
                            "save_path_test_mask": save_path_test_mask
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
