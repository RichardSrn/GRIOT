#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..utils.save import save_steps
from ..utils.round_ratio import round_ratio

import numpy as np
import torch
from geomloss import SamplesLoss

from tqdm import tqdm

from ..utils.MissingDataOT_network_utils import nanmean, MAE, RMSE

import logging


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
    niter : int, default=15
        Number of gradient updates for each model within a cycle.
    batchsize : int, defatul=128
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
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9,
                 device=torch.device('cpu')):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
        self.device = device

    def fit_transform(self, data, verbose=True, report_interval=500,
                      binary=False, save_path=".", unique="", graph_name=""):
        """
        Imputes missing values using a batched OT loss
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).
        mask : torch.DoubleTensor or torch.cuda.DoubleTensor
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.
        verbose: bool, default=True
            If True, output loss to log during iterations.
        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training, and return score arrays.
            For validation/debugging only.
        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).
        """

        X = data.x_miss.clone()
        X_true = data.x.clone()
        n, d = X.shape

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()
        imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + nanmean(X, 0))[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            logging.info(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        maes_val = np.zeros(self.niter)
        rmses_val = np.zeros(self.niter)
        maes_test = np.zeros(self.niter)
        rmses_test = np.zeros(self.niter)
        losses = np.zeros(self.niter)

        save_steps_path = []
        for i in tqdm(range(self.niter),
                      disable=False,
                      position=4,
                      colour='black',
                      desc='OT imputation'.ljust(20),
                      leave=False):

            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0

            for _ in range(self.n_pairs):
                idx1 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                         self.batchsize,
                                                         replace=False)).to(self.device)
                idx2 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                         self.batchsize,
                                                         replace=False)).to(self.device)

                X1 = X_filled[idx1].clone()
                X2 = X_filled[idx2].clone()

                loss = loss + self.sk(X1, X2)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            maes_val[i] = MAE(X_filled, X_true, data.val_mask.bool()).item()
            rmses_val[i] = RMSE(X_filled, X_true, data.val_mask.bool()).item()
            maes_test[i] = MAE(X_filled, X_true, data.test_mask.bool()).item()
            rmses_test[i] = RMSE(X_filled, X_true, data.test_mask.bool()).item()
            losses[i] = loss.item()

            if verbose and (i % report_interval == 0):
                logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
                             f'Validation MAE: {maes_val[i]:.4f}\t'
                             f'RMSE: {rmses_val[i]:.4f}')
                logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
                             f'train_test MAE: {maes_test[i]:.4f}\t'
                             f'RMSE: {rmses_test[i]:.4f}')

            if i % report_interval == 0 and i > 0:
                save_steps_path.append(
                    save_steps(X_filled.detach().numpy(),
                               save_path,
                               graph_name,
                               unique + f"_it{i}")
                )

        X_filled = X.detach().clone()
        # if binary is True, round imps to the closest whole number
        if binary:
            imps = round_ratio(imps, X)
        X_filled[mask.bool()] = imps
        X_filled = X_filled.cpu().detach().numpy()

        save_steps_path.append(
            save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None], X_filled)),
                       save_path,
                       graph_name,
                       unique + f"_FINAL")
        )

        return X_filled, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path


class RRimputer():
    """
    Round-Robin imputer with a batch sinkhorn loss
    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.
    eps: float, default=0.01
        Sinkhorn regularization parameter.

    lr : float, default = 0.01
        Learning rate.
    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.
    niter : int, default=15
        Number of gradient updates for each model within a cycle.
    batchsize : int, defatul=128
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
                 models,
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 max_iter=10,
                 niter=15,
                 batchsize=128,
                 n_pairs=10,
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=False,
                 scaling=.9,
                 device=torch.device("cpu")):

        self.models = models
        for layer in self.models.values():
            layer.to(device)
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps,
                              scaling=scaling, backend="auto").to(device)
        self.lr = lr
        self.opt = opt
        self.max_iter = max_iter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize

        self.is_fitted = False
        self.device = device

    def fit_transform(self, data, verbose=True, report_interval=100,
                      binary=False, save_path=".", unique="", graph_name=""):
        """
        Fits the imputer on a dataset with missing data, and returns the
        imputations.
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).
        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.
        verbose : bool, default=True
            If True, output loss to log during iterations.

        report_interval : int, default=1
            Interval between loss reports (if verbose).
        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training. For debugging only.
        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).
        """

        X = data.x_miss.clone()
        X_true = data.x.clone()
        n, d = X.shape

        mask = torch.isnan(X).double()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}."
                             f" Setting batchsize to {self.batchsize}.")

        order_ = torch.argsort(mask.sum(0))

        optimizers = [self.opt(self.models[i].parameters(),
                               lr=self.lr, weight_decay=self.weight_decay) for i in range(d)]

        imps = (self.noise * torch.randn(mask.shape, device=self.device).double() + nanmean(X, 0))[mask.bool()]
        X[mask.bool()] = imps
        X_filled = X.clone()

        maes_val = np.zeros(self.max_iter)
        rmses_val = np.zeros(self.max_iter)
        maes_test = np.zeros(self.max_iter)
        rmses_test = np.zeros(self.max_iter)
        losses = np.zeros(self.max_iter)

        save_steps_path = []
        for i in tqdm(range(self.max_iter),
                      desc="max_iters".ljust(20),
                      position=7,
                      leave=False):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            loss = 0

            for l in tqdm(range(d),
                          desc="d".ljust(20),
                          position=8,
                          leave=False):
                j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()

                if n - n_not_miss == 0:
                    continue  # no missing value on that coordinate

                for k in tqdm(range(self.niter),
                              desc="n_iters".ljust(20),
                              position=9,
                              leave=False):

                    loss = 0

                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()

                    for _ in tqdm(range(self.n_pairs),
                                  desc="n_pairs".ljust(20),
                                  position=10,
                                  leave=False,
                                  disable=True):

                        idx1 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                                 self.batchsize,
                                                                 replace=False)).to(self.device)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                                     self.batchsize,
                                                                     replace=self.batchsize > n_miss)).to(self.device)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = torch.from_numpy(np.random.choice(np.where(data.train_mask.cpu())[0],
                                                                     self.batchsize,
                                                                     replace=False)).to(self.device)
                            X2 = X_filled[idx2]

                        loss = loss + self.sk(X1, X2)

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with last parameters
                with torch.no_grad():
                    imps = self.models[j](
                        X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
                    if binary:
                        # todo: check imps dimensions
                        imps = round_ratio(imps, X)
                    X_filled[mask[:, j].bool(), j] = imps

            maes_val[i] = MAE(X_filled, X_true, data.val_mask).item()
            rmses_val[i] = RMSE(X_filled, X_true, data.val_mask).item()
            maes_test[i] = MAE(X_filled, X_true, data.test_mask).item()
            rmses_test[i] = RMSE(X_filled, X_true, data.test_mask).item()
            losses[i] = loss.item()

            if verbose and (i % report_interval == 0):
                logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t'
                             f'Validation MAE: {maes_val[i]:.4f}\t'
                             f'RMSE: {rmses_val[i]: .4f}')
                logging.info(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t'
                             f'Test MAE: {maes_test[i]:.4f}\t'
                             f'RMSE: {rmses_test[i]: .4f}')

            if i % report_interval == 0 and i > 0:
                save_steps_path.append(
                    save_steps(X_filled.detach().numpy(),
                               save_path,
                               graph_name,
                               unique + f"_it{i}")
                )

                if np.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                    break

            if np.isnan(losses).any() or np.isinf(losses).any():
                print("Nan or inf loss")
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        self.is_fitted = True

        X_filled = X_filled.cpu().detach().numpy()

        save_steps_path.append(
            save_steps(np.hstack((data.y.cpu().detach().numpy()[:, None], X_filled)),
                       save_path,
                       graph_name,
                       unique + f"_FINAL")
        )

        return X_filled, maes_val, rmses_val, maes_test, rmses_test, losses, save_steps_path

    # def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
    #     """
    #     Impute missing values on new data. Assumes models have been previously
    #     fitted on other data.
    #
    #     Parameters
    #     ----------
    #     X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
    #         Contains non-missing and missing data at the indices given by the
    #         "mask" argument. Missing values can be arbitrarily assigned
    #         (e.g. with NaNs).
    #     mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
    #         mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.
    #     verbose: bool, default=True
    #         If True, output loss to log during iterations.
    #
    #     report_interval : int, default=1
    #         Interval between loss reports (if verbose).
    #     X_true: torch.DoubleTensor or None, default=None
    #         Ground truth for the missing values. If provided, will output a
    #         validation score during training. For debugging only.
    #     Returns
    #     -------
    #     X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
    #         Imputed missing data (plus unchanged non-missing data).
    #     """
    #
    #     assert self.is_fitted, "The model has not been fitted yet."
    #
    #     n, d = X.shape
    #     normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
    #
    #     order_ = torch.argsort(mask.sum(0))
    #
    #     X[mask] = nanmean(X)
    #     X_filled = X.clone()
    #
    #     for i in range(self.max_iter):
    #
    #         if self.order == 'random':
    #             order_ = np.random.choice(d, d, replace=False)
    #         X_old = X_filled.clone().detach()
    #
    #         for l in range(d):
    #             j = order_[l].item()
    #
    #             with torch.no_grad():
    #                 X_filled[mask[:, j].bool(), j] = self.models[j](
    #                     X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
    #
    #         if verbose and (i % report_interval == 0):
    #             if X_true is not None:
    #                 logging.info(f'Iteration {i}:\t '
    #                              f'Validation MAE: {MAE(X_filled, X_true, mask).item():.4f}\t'
    #                              f'RMSE: {RMSE(X_filled, X_true, mask).item():.4f}')
    #
    #         if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
    #             break
    #
    #     if i == (self.max_iter - 1) and verbose:
    #         logging.info('Early stopping criterion not reached')
    #
    #     return X_filled
