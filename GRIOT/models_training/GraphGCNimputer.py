from ..utils.MissingDataOT_network_utils import nanmean, MAE, RMSE
from ..utils.save import save_steps
from ..utils.round_ratio import round_ratio
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from ..utils.re_structure import compute_new_structure
from ..utils.graph_tool import get_edge_index
from ..config import cfg


class GCNimputer():
    """
    Parameters
    ----------
    model:  torch.nn.Module. A model that takes as input all the features and output all the features.
    eps: float, default=0.01
        Sinkhorn regularization parameter.

    lr : float, default = 0.01
        Learning rate.
    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    max_iters : int, default=10
        Maximum number of round-robin cycles for imputation.
    n_iters : int, default=15
        Number of gradient updates for each model within a cycle.
    batch_size : int, default=128
        Size of the batches on which the sinkhorn divergence is evaluated.
    n_pairs : int, default=10
        Number of batch pairs used per gradient update.
    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.
    weight_decay : float, default = 1e-5
        L2 regularization magnitude.
    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.
    unsymmetrize: bool, default=True
        If True, sample one batch with no missing
        data in each pair during training.
    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"
    """

    def __init__(self,
                 model,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 max_iters=10,
                 n_iters=15,
                 batch_size=128,
                 n_pairs=10,
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=False,  # True,
                 tildeC=False,
                 scaling=.9,
                 logging=None,
                 device=None,
                 lossfn=None,
                 restructure: bool = None):

        self.model = model.to(cfg.device)
        # self.sk = SamplesLoss("sinkhorn", p=2, blur=eps,
        #                       scaling=scaling, backend="auto")
        self.lossfn = lossfn
        self.lr = lr
        self.opt = opt
        self.tildeC = tildeC
        self.max_iters = max_iters
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.logging = self.log(logging)
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.restructure = restructure
        self.is_fitted = False

    def log(self, logging):
        if logging is None:
            class p:
                def __init__(self):
                    pass

                def info(self, *args, **kwargs):
                    print("INFO:", *args, **kwargs)

                def error(self, *args, **kwargs):
                    print("ERROR:", *args, **kwargs)

            return p()
        else:
            return logging

    def fit_transform(self,
                      data,
                      p,
                      binary=False,
                      save_path="",
                      graph_name="",
                      unique=None,
                      p_miss="",
                      alpha=0.5,
                      verbose=True,
                      report_interval=1000):
        """
        Fits the imputer on a dataset with missing data, and returns the
        imputations.
        Parameters
        ----------
        data: torch_geometric.data.Data
            graph data

        p: List[float]
            graph nodes weights

        binary: bool, default=False
            If True, round up F_filled to 0 or 1.

        save_path: str, default=""
            Path to save the imputations.

        graph_name: str, default=""
            Name of the graph.

        unique: int, default=None
            Unique path.

        p_miss: str, default=""
            Missing data percentage.

        verbose: bool, default=False
            If True, prints the imputation metrics at each iteration.

        report_interval: int, default=100
            Interval at which the imputation metrics are printed.
        Returns
        -------
        F_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).
        """

        F = data.x_miss.clone().cpu()  # self.device)
        F_true = data.x.clone().cpu()  # self.device)
        C = data.proximity_matrix.clone().cpu()  # self.device)
        edge_index = data.edge_index.clone().cpu()  # self.device)
        n, d = F.shape
        mask = torch.isnan(F).double().cpu()  # self.device)
        rand = torch.randn(mask.shape).double().cpu()  # self.device)

        normalized_tol = self.tol * torch.max(torch.abs(F[~mask.bool()]))

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e
            if verbose:
                self.logging.info(f"Batchsize larger that half size = {len(F) // 2}."
                                  f" Setting batch_size to {self.batch_size}.")

        # order_ = torch.argsort(mask.sum(0))
        optimizer = self.opt(self.model.parameters(),
                             lr=self.lr, weight_decay=self.weight_decay)

        imps = (self.noise * rand + nanmean(F, 0))
        F[mask.bool()] = imps[mask.bool()]
        F_filled = F.clone().cpu()  # .to(cfg.device)
        # mask = mask.to(cfg.device)

        if self.restructure == True:
            C_backup = C.clone()
            # C = compute_new_structure(C, F_filled,
            #                           alpha=alpha,
            #                           iters=cfg.restruct_iters,
            #                           threshold=0.5,
            #                           threshold_decay=0.05,
            #                           allow_disconnected=False,
            #                           verbose=verbose)
        else:
            C_backup = None

        if F_true is not None:
            metrics = pd.DataFrame(columns=["iteration", "mae_val", "rmse_val", "mae_test", "rmse_test"])
        else:
            metrics = None

        loss_steps = dict()

        save_steps_path = []

        del imps, rand

        # import pynvml
        # # Initialize the NVML library
        # pynvml.nvmlInit()
        #
        # # Get the number of available GPUs
        # num_gpus = pynvml.nvmlDeviceGetCount()

        i=0
        for i in tqdm(range(self.max_iters),
                      desc="outer_loop".ljust(20),
                      position=8,
                      leave=False):
            # Iterate over each GPU
            # for i_ in range(num_gpus):
            #     # Get the handle of the GPU
            #     handle = pynvml.nvmlDeviceGetHandleByIndex(i_)
            #     # Query the utilization of the GPU
            #     utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            #     print(f" {i} - GPU {i_}: Utilization - {utilization.gpu}%")
            #     # Query the memory information of the GPU
            #     memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            #     # Print the total memory (maximum capacity) of the GPU
            #     print(f" {i} - GPU {i_}: Total Memory - {memory_info.total / (1024**2)} MiB")

            F_old = F_filled.clone().detach().cpu()

            if self.restructure == True:
                with torch.device("cpu"):  # cfg.device:
                    C = compute_new_structure(proximity_matrix=C_backup.cpu(),
                                              features_matrix=F_filled.cpu(),
                                              lossfn=cfg.restruct_loss_fn,
                                              iters=cfg.restruct_iters,
                                              threshold=cfg.restruct_thld,
                                              threshold_low=cfg.restruct_thld_low,
                                              method=cfg.restruct_method,
                                              integer=cfg.restruct_integer,
                                              true_labels_for_plot=data.y,
                                              allow_disconnected=False,
                                              verbose=verbose,)
                A_ = C.clone()
                A_[A_ != 1] = 0
                edge_index = get_edge_index(A_).cpu()  # to(cfg.device)
                if cfg.restruct_for_loss:
                    C = C_backup.clone()

            for k in tqdm(range(self.n_iters),
                          desc="inner_loop".ljust(20),
                          position=9,
                          leave=False):

                optimizer.zero_grad()
                self.loss = torch.tensor(0.0, requires_grad=True).to(cfg.device)

                with cfg.device:
                    F_filled = F_filled.detach()
                    F_filled[mask.bool()] = self.model(
                        F_filled.clone().to(cfg.device),
                        edge_index.to(cfg.device)
                    ).squeeze()[mask.bool()].cpu()

                for _ in tqdm(range(self.n_pairs),
                              desc="n_pairs".ljust(20),
                              position=10,
                              leave=False,
                              disable=True):

                    # idx1 = np.random.choice(np.where(data.train_mask.cpu())[0], self.batch_size, replace=False)
                    # idx2 = np.random.choice(np.where(data.train_mask.cpu())[0], self.batch_size, replace=False)
                    # same as above but pick idx1 and idx2 using torch
                    idx1 = torch.randint(high=data.train_vertex_index.size(0), size=(self.batch_size,))
                    idx1 = data.train_vertex_index[idx1]
                    idx2 = torch.randint(high=data.train_vertex_index.size(0), size=(self.batch_size,))
                    idx2 = data.train_vertex_index[idx2]

                    F1 = F_filled[idx1]
                    F2 = F_filled[idx2]

                    p1 = p[idx1]
                    p2 = p[idx2]

                    if self.tildeC == 0:
                        C1 = C[idx1, :][:, list(set(idx1) | set(idx2))]
                        C2 = C[idx2, :][:, list(set(idx1) | set(idx2))]
                    elif self.tildeC > 0:
                        C1 = C[idx1]
                        C2 = C[idx2]
                    else:
                        C1 = C[idx1, :][:, idx1]
                        C2 = C[idx2, :][:, idx2]

                    with cfg.device:
                        l = self.lossfn(C1=C1.to(cfg.device), C2=C2.to(cfg.device),
                                        F1=F1.to(cfg.device), F2=F2.to(cfg.device),
                                        p1=p1.to(cfg.device), p2=p2.to(cfg.device)).to(cfg.device)
                    self.loss = self.loss + l / self.n_pairs

                if cfg.plot:
                    if k == (self.n_iters - 1) \
                            and i == (self.max_iters - 1) \
                            and self.lossfn.name.lower() == "multiw":
                        with cfg.device:
                            with torch.no_grad():
                                self.lossfn(C1=C1.to(cfg.device), C2=C2.to(cfg.device),
                                            F1=F1.to(cfg.device), F2=F2.to(cfg.device),
                                            p1=p1.to(cfg.device), p2=p2.to(cfg.device),
                                            plot=cfg.plot,
                                            i=i, unique=unique)

                optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                self.loss.backward()
                optimizer.step()

            # Impute with last parameters
            with torch.no_grad():
                F_filled[mask.bool()] = self.model(
                    F_filled.clone().to(cfg.device),
                    edge_index.to(cfg.device)
                ).squeeze().cpu()[mask.bool()]

            metrics = pd.concat([metrics,
                                 pd.DataFrame({
                                     "iteration": i,
                                     "mae_train": MAE(F_filled.detach(), F_true, data.train_mask.bool()).item(),
                                     "rmse_train": RMSE(F_filled.detach(), F_true, data.train_mask.bool()).item(),
                                     "mae_val": MAE(F_filled.detach(), F_true, data.val_mask.bool()).item(),
                                     "rmse_val": RMSE(F_filled.detach(), F_true, data.val_mask.bool()).item(),
                                     "mae_test": MAE(F_filled.detach(), F_true, data.test_mask.bool()).item(),
                                     "rmse_test": RMSE(F_filled.detach(), F_true, data.test_mask.bool()).item()
                                 },
                                     index=[0])],
                                ignore_index=True)

            if verbose and (i % report_interval == 0):
                if F_true is not None:
                    m = f'Iteration {i}:\t' \
                        f'Loss: {self.loss.item() / self.n_pairs:.4f}\t' \
                        f'Validation MAE: {metrics["mae_val"][i]:.4f}\t' \
                        f'RMSE: {metrics["rmse_val"][i]:.4f}' \
                        f'Test MAE: {metrics["mae_test"][i]:.4f}\t' \
                        f'RMSE: {metrics["rmse_test"][i]:.4f}'
                    if self.logging:
                        self.logging.info(m)
                    else:
                        print(m)
                else:
                    m = f'Iteration {i}:\t Loss: {self.loss.item() / self.n_pairs:.4f}'
                    if self.logging:
                        self.logging.info(m)
                    else:
                        print(m)

            loss_steps[i] = self.loss.item()

            if i % report_interval == 0 and i > 0:
                save_steps_path.append(
                    save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None], F_filled.cpu().detach().numpy())),
                               save_path,
                               graph_name,
                               unique + f"_it{i}")
                )
                if torch.norm(F_filled - F_old, p=np.inf) < normalized_tol:
                    break

            if torch.isnan(self.loss).any() or torch.isinf(self.loss).any():
                # Catch numerical errors/overflows (should not happen)
                if self.logging:
                    self.logging.error("Nan or inf loss")
                else:
                    print("Nan or inf loss")
                loss_steps[i] = -1
                break

        if i == (self.max_iters - 1) and verbose:
            self.logging.info('Early stopping criterion not reached')

        self.is_fitted = True

        F_filled = F_filled.cpu().detach().numpy()

        # if binary is True, round imps to the closest whole number
        if binary:
            F_filled = round_ratio(F_filled, F)

        save_steps_path.append(
            save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None], F_filled)),
                       save_path,
                       graph_name,
                       unique + f"_FINAL")
        )

        loss_steps = pd.DataFrame(loss_steps,
                                  index=[self.lossfn.name]).T.reset_index().rename(columns={"index": "iteration"})

        # # Shutdown the NVML library
        # pynvml.nvmlShutdown()

        return F_filled, metrics, loss_steps, save_steps_path