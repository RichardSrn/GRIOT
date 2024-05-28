import matplotlib.pyplot as plt
from ..utils.MissingDataOT_network_utils import nanmean, MAE, RMSE
from ..utils.save import save_steps
from ..utils.round_ratio import round_ratio
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

class OTimputer():
    """
    'One parameter equals one imputed value' model (Algorithm 1. in the paper)

    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.

    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    n_iter : int, default=15
        Number of gradient updates for each model within a cycle.

    batch_size : int, defatul=128
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
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 batch_size=128,
                 n_iters=2000,
                 n_pairs=1,
                 noise=0.1,
                 threshold=0.0045,
                 tildeC=False,
                 lossfn=None,
                 logging=None,
                 show_progress_bar=True,
                 device=torch.device('cpu')):
        self.lr = lr
        self.opt = opt
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.tildeC = tildeC
        self.lossfn = lossfn
        self.threshold = threshold
        self.logging = logging
        self.show_progress_bar = show_progress_bar
        self.device=device

    def fit_transform(self,
                      data,
                      p=None,
                      verbose=True,
                      report_interval=500,
                      binary=False,
                      early_break=False,
                      unique=None,
                      save_path="",
                      graph_name=""):
        """
        Imputes missing values using a batched OT loss

        Parameters
        ----------
        data: torch.DoubleTensor or torch.cuda.DoubleTensor
            Graph data.

        p: graph nodes weights

        verbose: bool, default=True
            If True, output loss to log during iterations.

        report_interval: int, default=500
            Interval at which to report MAE, RMSE, loss to log.

        binary: bool, default=False
            If True, round the values of X_filled to 0 or 1.

        early_break: bool, default=False
            If True, break the training loop when the loss evolution is below the threshold.

        unique: int, default=None
            Unique name to which save the steps.

        save_path: str, default=""
            Path to save the steps.

        graph_name: str, default=""
            Name of the current graph.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).
        """

        C = data.proximity_matrix.clone().to(self.device)
        p = p.clone().to(self.device)
        F = data.x_miss.clone().to(self.device)
        F_true = data.x.clone().to(self.device)
        n, d = F.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2 ** e
            if verbose:
                m = f"Batchsize larger that half size = {len(F) // 2}. Setting batch_size to {self.batch_size}."
                if self.logging:
                    self.logging.info(m)
                else:
                    print(m)
        mask = torch.isnan(F).double().to(self.device)
        imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + nanmean(F, 0))[mask.bool()]
        imps = imps.to(self.device)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            m = f"batch_size = {self.batch_size}, loss_fn = {self.lossfn.name}"
            if self.logging:
                self.logging.info(m)
            else:
                print(m)

        if F_true is not None:
            metrics = pd.DataFrame(columns=["iteration", "mae_val", "rmse_val", "mae_test", "rmse_test"])
        else:
            metrics = None

        loss_steps = dict()

        save_steps_path = []

        for i in tqdm(range(self.n_iters),
                      position=6,
                      desc="n_iters".ljust(25),
                      leave=False,
                      colour='green') if self.show_progress_bar and self.n_iters >= 64 else range(self.n_iters):
            F_filled = F.detach().clone()
            F_filled[mask.bool()] = imps
            self.loss = 0

            for _ in tqdm(range(self.n_pairs),
                          position=7,
                          desc="n_pairs".ljust(25),
                          leave=False,
                          colour='blue') if self.show_progress_bar and self.n_pairs >= 16 else range(self.n_pairs):

                idx1 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                         self.batch_size,
                                                         replace=False)).to(F.device)
                idx2 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                         self.batch_size,
                                                         replace=False)).to(F.device)

                # fill attributes
                F1 = F_filled[idx1]
                F2 = F_filled[idx2]

                # sample p1, p2
                p1 = p[idx1] if p is not None else None
                p2 = p[idx2] if p is not None else None

                # sample C1, C2
                if self.tildeC and self.lossfn.name.lower() != "fgw":
                    C1 = C[idx1, :]
                    C2 = C[idx2, :]
                else:
                    C1 = C[idx1, :][:, idx1]
                    C2 = C[idx2, :][:, idx2]


                self.loss = self.loss + self.lossfn(C1=C1, C2=C2,
                                                    F1=F1, F2=F2,
                                                    p1=p1, p2=p2)

            if i == (self.n_iters - 1) and i > 0 and self.lossfn.name.lower() == "multiw":
                self.lossfn(C1=C1, C2=C2,
                            F1=F1, F2=F2,
                            p1=p1, p2=p2,
                            plot=True, i=i, unique=unique)

            if torch.isnan(self.loss).any() or torch.isinf(self.loss).any():
                ### Catch numerical errors/overflows (should not happen)
                if self.logging:
                    self.logging.error("Nan or inf loss")
                else:
                    print("Nan or inf loss")
                loss_steps[i] = -1
                break

            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()

            if F_true is not None:
                metrics = pd.concat([metrics,
                                        pd.DataFrame({
                                            "iteration": i,
                                            "mae_val": MAE(F_filled.detach(), F_true, data.val_mask.bool()).item(),
                                            "rmse_val": RMSE(F_filled.detach(), F_true, data.val_mask.bool()).item(),
                                            "mae_test": MAE(F_filled.detach(), F_true, data.test_mask.bool()).item(),
                                            "rmse_test": RMSE(F_filled.detach(), F_true, data.test_mask.bool()).item()
                                        },
                                            index=[0])],
                                        ignore_index=True)
            loss_steps[i] = self.loss.item() / self.n_pairs

            # check whether to stop early or not, compute svm metrics, etc.
            if i % report_interval == 0 and i > 0:
                save_steps_path.append(
                    save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None],
                                          F_filled.cpu().detach().numpy())),
                               save_path,
                               graph_name,
                               unique + f"_it{i}")
                )
                if i > 5 * report_interval and early_break:
                    if np.std(metrics[metrics["iteration"] > i - report_interval * 4]["mae_val"].to_numpy()) \
                            < self.threshold:
                        break

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

        F_filled = F.detach().clone()

        # if binary is True, round imps to the closest whole number
        if binary:
            imps = round_ratio(imps, F)

        F_filled[mask.bool()] = imps

        save_steps_path.append(
            save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None],
                                  F_filled.cpu().detach().numpy())),
                       save_path,
                       graph_name,
                       unique + f"_FINAL")
        )

        loss_steps = pd.DataFrame(loss_steps,
                                  index=[self.lossfn.name]).T.reset_index().rename(columns={"index": "iteration"})
        return F_filled, metrics, loss_steps, save_steps_path
