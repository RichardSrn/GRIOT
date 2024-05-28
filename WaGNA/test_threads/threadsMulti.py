from time import time
import pandas as pd
import os
from ..models_training.GraphOTimputer import OTimputer
from ..models_training.GraphRRimputer import RRimputer
from ..models_training.GraphGCNimputerMulti import GCNimputerMulti
from ..models_training.GraphImputer2Multi import Imputer2TrainerMulti


class ThreadsMulti():
    def __init__(self, total=0):
        self.data = pd.DataFrame(columns=[
            "graph", "elapsed_time", "features_label", "p_miss", "label", "epsilon", "n_iters", "n_rep_idx",
            "lossfn", "mae_val", "rmse_val", "mae_test", "rmse_test", "loss",
        ])
        self.total = total
        self.prev_title = ""

    def ot_thread_multi(self,
                  epsilon,
                  batch_size,
                  lr,
                  n_iters,
                  n_pairs,
                  tildeC,
                  lossfn,
                  alpha,
                  logging,
                  data,
                  name,
                  p,
                  early_break,
                  p_miss,
                  i,
                  device,
                  show_progress_bar=True,
                  save_path=""):
        raise NotImplementedError
        # load model
        ot_imputer = OTimputer(batch_size=batch_size,
                               lr=lr,
                               n_iters=n_iters,
                               n_pairs=n_pairs,
                               tildeC=tildeC,
                               lossfn=lossfn,
                               logging=logging,
                               show_progress_bar=show_progress_bar)
        # fit model
        t_start = time()
        ot_imp, ot_metrics, ot_loss, save_steps_path = ot_imputer.fit_transform(
            data=data,
            p=p,
            verbose=False,
            report_interval=n_iters + 1,
            binary=True if name.lower() in ["cora", "citeseer", "pubmed", "cornell", "texas",
                                                     "wisconsin"] else False,
            early_break=early_break,
            unique=f"{name}_ε{epsilon:.2f}_α{alpha:.2f}_bs{batch_size}_lr{lr:.2f}_ni{n_iters}_"
                   f"np{n_pairs}_ls{lossfn.name}_pm{p_miss:.2f}_rep{i}",
            save_path=save_path,
            graph_name=name
        )
        elapsed_time = time() - t_start


        # add last infos to data
        self.data = self.data.append({
            "graph": name,
            "elapsed_time": elapsed_time,
            "p_miss": p_miss,
            "label": f'α={alpha:.2f}',
            "epsilon": epsilon,
            "n_iters": ot_loss["iteration"].max(),
            "n_rep_idx": i,
            "lossfn": lossfn.name,
            "mae_val": ot_metrics["mae_val"],
            "rmse_val": ot_metrics["rmse_val"],
            "mae_test": ot_metrics["mae_test"],
            "rmse_test": ot_metrics["rmse_test"],
            "loss": ot_loss,
            "save_steps_path": save_steps_path
        }, ignore_index=True)

    def rr_thread_multi(self,
                  models,
                  epsilon,
                  batch_size,
                  lr,
                  max_iters,
                  n_iters,
                  n_pairs,
                  tildeC,
                  lossfn,
                  alpha,
                  logging,
                  name,
                  data,
                  p,
                  early_break,
                  device,
                  p_miss,
                  i,
                  show_progress_bar=True,
                  save_path=""):
        raise NotImplementedError
        # load model
        ot_imputer = RRimputer(
            models=models,
            batch_size=batch_size,
            lr=lr,
            max_iters=max_iters,
            n_iters=n_iters,
            n_pairs=n_pairs,
            tildeC=tildeC,
            lossfn=lossfn,
            logging=logging,
            device=device,
            # show_progress_bar=show_progress_bar
        )
        # fit model
        t_start = time()
        ot_imp, ot_metrics, ot_loss, save_steps_path = ot_imputer.fit_transform(
            data=data,
            p=p,
            verbose=False,
            report_interval=n_iters + 1,
            binary=True if name.lower() in ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin"] else False,
            # early_break=early_break,
            unique=f"{name}_ε{epsilon:.2f}_α{alpha:.2f}_bs{batch_size}_lr{lr:.2f}_ni{n_iters}_"
                   f"np{n_pairs}_ls{lossfn.name}_pm{p_miss:.2f}_rep{i}",
            save_path=save_path,
            graph_name=name
        )
        elapsed_time = time() - t_start


        # add last infos to data
        self.data = self.data.append({
            "graph": name,
            "elapsed_time": elapsed_time,
            "p_miss": p_miss,
            "label": f'α={alpha:.2f}',
            "epsilon": epsilon,
            "n_iters": ot_loss["iteration"].max(),
            "n_rep_idx": i,
            "lossfn": lossfn.name,
            "mae_val": ot_metrics["mae_val"],
            "rmse_val": ot_metrics["rmse_val"],
            "mae_test": ot_metrics["mae_test"],
            "rmse_test": ot_metrics["rmse_test"],
            "loss": ot_loss,
            "save_steps_path": save_steps_path
        }, ignore_index=True)

    def gnn_thread_multi(self,
                         model,
                         epsilon,
                         batch_size,
                         lr,
                         max_iters,
                         n_iters,
                         n_pairs,
                         tildeC,
                         lossfn,
                         alpha,
                         logging,
                         name,
                         data,
                         p,
                         early_break,
                         device,
                         p_miss,
                         i,
                         show_progress_bar=True,
                         save_path=""):
        # load model
        ot_imputer = GCNimputerMulti(
            model=model,
            batch_size=batch_size,
            lr=lr,
            max_iters=max_iters,
            n_iters=n_iters,
            n_pairs=n_pairs,
            tildeC=tildeC,
            lossfn=lossfn,
            logging=logging,
            device=device,
            # show_progress_bar=show_progress_bar
        )
        # fit model
        t_start = time()
        ot_imp, ot_metrics, ot_loss, save_steps_path = ot_imputer.fit_transform(
            data=data,
            p=p,
            verbose=False,
            report_interval=n_iters + 1,
            binary=True if name.lower() in ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin"] else False,
            # early_break=early_break,
            unique=f"{name}_ε{epsilon:.2f}"
                   f"_α{str([f'{a:.2f}' for a in alpha]).replace(' ','')[1:-1]}"
                   f"_bs{batch_size}_lr{lr:.2f}_ni{n_iters}_"
                   f"np{n_pairs}_ls{lossfn.name}_pm{p_miss:.2f}_rep{i}",
            save_path=save_path,
            graph_name=name
        )
        elapsed_time = time() - t_start


        # add last infos to data
        self.data = pd.concat([self.data, pd.DataFrame({
            "graph": [name],
            "elapsed_time": [elapsed_time],
            "p_miss": [p_miss],
            "label": [f"_α{str([f'{a:.2f}' for a in alpha]).replace(' ','')[1:-1]}"],
            "epsilon": [epsilon],
            "n_iters": [ot_loss["iteration"].max()],
            "n_rep_idx": [i],
            "lossfn": [lossfn.name],
            "mae_val": [ot_metrics["mae_val"]],
            "rmse_val": [ot_metrics["rmse_val"]],
            "mae_test": [ot_metrics["mae_test"]],
            "rmse_test": [ot_metrics["rmse_test"]],
            "loss": [ot_loss],
            "save_steps_path": [save_steps_path]
        })], ignore_index=True)

    def wagna2_thread_multi(self,
                         model,
                         epsilon,
                         batch_size,
                         lr,
                         max_iters,
                         n_iters,
                         n_pairs,
                         tildeC,
                         lossfn,
                         alpha,
                         logging,
                         name,
                         data,
                         p,
                         early_break,
                         device,
                         p_miss,
                         i,
                         show_progress_bar=True,
                         save_path=""):
        # load model
        ot_imputer = Imputer2TrainerMulti(
            model=model,
            batch_size=batch_size,
            lr=lr,
            max_iters=max_iters,
            n_iters=n_iters,
            n_pairs=n_pairs,
            tildeC=tildeC,
            lossfn=lossfn,
            logging=logging,
            device=device,
            # show_progress_bar=show_progress_bar
        )
        # fit model
        t_start = time()
        ot_imp, ot_metrics, ot_loss, save_steps_path = ot_imputer.fit_transform(
            data=data,
            p=p,
            verbose=False,
            report_interval=n_iters + 1,
            binary=True if name.lower() in ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin"] else False,
            # early_break=early_break,
            unique=f"{name}_ε{epsilon:.2f}"
                   f"_α{str([f'{a:.2f}' for a in alpha]).replace(' ','')[1:-1]}"
                   f"_bs{batch_size}_lr{lr:.2f}_ni{n_iters}_"
                   f"np{n_pairs}_ls{lossfn.name}_pm{p_miss:.2f}_rep{i}",
            save_path=save_path,
            graph_name=name
        )
        elapsed_time = time() - t_start


        # add last infos to data
        self.data = pd.concat([self.data, pd.DataFrame({
            "graph": [name],
            "elapsed_time": [elapsed_time],
            "p_miss": [p_miss],
            "label": [f"_α{str([f'{a:.2f}' for a in alpha]).replace(' ','')[1:-1]}"],
            "epsilon": [epsilon],
            "n_iters": [ot_loss["iteration"].max()],
            "n_rep_idx": [i],
            "lossfn": [lossfn.name],
            "mae_val": [ot_metrics["mae_val"]],
            "rmse_val": [ot_metrics["rmse_val"]],
            "mae_test": [ot_metrics["mae_test"]],
            "rmse_test": [ot_metrics["rmse_test"]],
            "loss": [ot_loss],
            "save_steps_path": [save_steps_path]
        })], ignore_index=True)

    def reset_data_tmp(self):
        self.data_tmp = dict()

    def save(self, n_iters, n_rep, n_pairs, lr, count, main_path=".", path_save=""):
        if not path_save:
            path_save = f"./{main_path}/data_ni{n_iters}_nr{n_rep}_np{n_pairs}_lr{lr}_{count}o{self.total}.json"
        self.data.to_json(path_save,
                          orient="records",
                          indent=4)
        if self.prev_title != "" and self.prev_title != path_save:
            os.remove(self.prev_title)
        self.prev_title = path_save
        return path_save
