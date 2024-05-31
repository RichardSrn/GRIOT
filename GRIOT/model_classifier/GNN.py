import os
import networkx as nx
import numpy as np
import random
import torch
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from torch_geometric.transforms import NormalizeFeatures
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, TAGConv, ChebConv
import torch.nn as nn
from torch_geometric.nn import GCN
from torchsummary import summary
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

from ..datasets_class.dataset_class import MyDataset
from ..utils.metrics import purity_score

# quieten warnings
import warnings

from ..config import cfg

warnings.filterwarnings("ignore", category=FutureWarning)


# GCN model with 2 layers
class Net(torch.nn.Module):
    def __init__(self, data, dataset, lr, weight_decay=5e-4, show_summary=False, layers_size: list = None,
                 device="cpu", nntype="CHEB", **kwargs):
        super(Net, self).__init__()
        self.data = data
        self.dataset = dataset
        # hidden_channel = max((2 ** round(np.log2(self.dataset.num_features ** 0.5))), int(self.dataset.num_classes))
        hidden_channel = int(self.dataset.num_features ** .5)
        # hidden_channel = int(self.dataset.num_features//2)
        self.nntype = nntype

        if self.nntype == "GCN":
            # GCN START ------------------
            self.conv1 = GCNConv(self.dataset.num_features, hidden_channel).to(device)
            # self.conv2 = GCNConv(hidden_channel, self.dataset.num_classes).to(device)
            self.conv2 = GCNConv(hidden_channel, hidden_channel).to(device)
            self.out = torch.nn.Linear(hidden_channel, self.dataset.num_classes).to(device)
            # GCN END --------------------
        elif self.nntype == "GIN":
            # GIN START ------------------
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(self.dataset.num_features, hidden_channel),
                nn.ReLU(),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU()
            )).to(device)
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU()
            )).to(device)
            self.out = torch.nn.Linear(hidden_channel, self.dataset.num_classes).to(device)
            # GIN END --------------------
        elif self.nntype == "SAGE":
            # SAGE START -----------------
            self.conv1 = SAGEConv(self.dataset.num_features, hidden_channel, aggr='mean').to(device)
            self.conv2 = SAGEConv(hidden_channel, hidden_channel, aggr='mean').to(device)
            self.out = torch.nn.Linear(hidden_channel, self.dataset.num_classes).to(device)
            # SAGE END -------------------
        elif self.nntype == "TAG":
            # TAG START ------------------
            self.conv1 = TAGConv(self.dataset.num_features, hidden_channel, K=2).to(device)
            self.conv2 = TAGConv(hidden_channel, hidden_channel, K=2).to(device)
            self.out = torch.nn.Linear(hidden_channel, self.dataset.num_classes).to(device)
            # TAG END --------------------
        elif self.nntype == "CHEB":
            # CHEB START -----------------
            self.conv1 = ChebConv(self.dataset.num_features, hidden_channel, K=2).to(device)
            self.conv2 = ChebConv(hidden_channel, hidden_channel, K=2).to(device)
            self.out = torch.nn.Linear(hidden_channel, self.dataset.num_classes).to(device)
            # CHEB END -------------------
        else:
            raise ValueError("nntype must be in ['GCN', 'GIN', 'SAGE', 'TAG', 'CHEB']")

        self.layers_size = [self.dataset.num_features, hidden_channel, self.dataset.num_classes]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.lossfn = torch.nn.CrossEntropyLoss().to(device)
        self.loss = 0
        self.device = device
        if show_summary:
            self.summary()

    def summary(self, show=False, save_path=""):
        text = "\n"
        text += str("=" * 20 + "model summary" + "=" * 20 + "\n")
        nb_parameters = []
        for i, conv in enumerate([self.conv1, self.conv2]):
            layer_param = 0
            for p in conv.parameters():
                nn = 1
                for s in list(p.size()):
                    nn *= s
                layer_param += nn
            nb_parameters.append(layer_param)
            text += str(f"conv_{i} =".ljust(9) + f"{conv}".ljust(20) +
                        ";" + f"{nb_parameters[i]}".rjust(10) + "parameters" + "\n")
        text += str("-" * 55 + "\n")
        text += str(" " * 12 + f"total : {sum(nb_parameters)}".ljust(18) + "parameters" + "\n")
        text += str("=" * 55 + "\n")
        text += "\n"
        if show:
            print(text)
        if save_path:
            with open(save_path, "w") as file:
                file.write(text)

    def forward(self, x, edge_index):
        if self.nntype == "GCN":
            # GCN START ------------------
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            # x = F.dropout(x, p=0.25, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.dropout(x, p=.5, training=self.training)
            x = self.out(x)
            # GCN END --------------------
        elif self.nntype == "GIN":
            # GIN START ------------------
            x = self.conv1(x, edge_index)
            x = F.dropout(x, p=.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.dropout(x, p=.5, training=self.training)
            x = self.out(x)
            # GIN END --------------------
        else:
            # SAGE START ----------------−
            # TAG START ------------------
            # CHEB START -----------------
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=.5, training=self.training)
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=.5, training=self.training)
            x = self.out(x)
            # SAGE END -------------------
            # TAG END --------------------
            # CHEB END -------------------

        return x

    def train_model(self):
        self.train()
        self.optimizer.zero_grad()
        y_pred = self(self.data.x, self.data.edge_index)[self.data.train_mask]
        y_true = self.data.y[self.data.train_mask]
        self.loss = self.lossfn(y_pred, y_true)
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    @torch.no_grad()
    def test(self, training_data=True):
        self.eval()
        y_pred = self(self.data.x, self.data.edge_index).argmax(dim=1)
        y_pred_proba = self(self.data.x, self.data.edge_index)

        scores = {}
        cycle = []

        if training_data:
            mask_train = self.data.train_mask
            pred_train = y_pred[mask_train]
            pred_train_proba = y_pred_proba[mask_train]
            acc_train = accuracy_score(self.data.y[mask_train].cpu(), pred_train.cpu())
            scores["gnn_accuracy_train"] = acc_train
            cycle.append([self.data.y[mask_train], pred_train, pred_train_proba, "train"])

        mask_val = self.data.val_mask
        pred_val = y_pred[mask_val]
        pred_val_proba = y_pred_proba[mask_val]
        acc_val = accuracy_score(self.data.y[mask_val].cpu(), pred_val.cpu())
        scores["gnn_accuracy_val"] = acc_val
        cycle.append([self.data.y[mask_val], pred_val, pred_val_proba, "val"])

        mask_test = self.data.test_mask
        pred_test = y_pred[mask_test]
        pred_test_proba = y_pred_proba[mask_test]
        acc_test = accuracy_score(self.data.y[mask_test].cpu(), pred_test.cpu())
        scores["gnn_accuracy_test"] = acc_test
        cycle.append([self.data.y[mask_test], pred_test, pred_test_proba, "test"])

        for data_true, data_pred, data_pred_proba, data_type in cycle:
            for average in ["micro", "macro", "weighted"]:
                s = precision_recall_fscore_support(data_true.cpu(),
                                                    data_pred.cpu(),
                                                    average=average,
                                                    zero_division=0)
                scores[f"gnn_precision{'_' + data_type}_{average}"] = s[0]
                scores[f"gnn_recall{'_' + data_type}_{average}"] = s[1]
                scores[f"gnn_f1{'_' + data_type}_{average}"] = s[2]

            # get roc_auc_score
            data_true_proba = torch.zeros(data_pred_proba.shape)
            data_true_proba[torch.arange(data_pred_proba.shape[0]), data_true] = 1
            try:
                # TODO : get the columns where data_true_proba == 0 everywhere
                score = roc_auc_score(
                    data_true_proba.cpu(),
                    data_pred_proba.cpu(),
                    multi_class="ovo",
                    average="macro"
                )
            except ValueError:
                score = 0
            scores[f"gnn_roc_auc_score{'_' + data_type}"] = score

        if cfg.save_pred:
            os.makedirs(os.path.join(cfg.UNIQUE, "predictions"), exist_ok=True)
            np.save(os.path.join(cfg.UNIQUE, "predictions", "pred.npy"), y_pred.cpu().detach().numpy())
            np.save(os.path.join(cfg.UNIQUE, "predictions", "pred_proba.npy"), y_pred_proba.cpu().detach().numpy())
            np.save(os.path.join(cfg.UNIQUE, "predictions", "true.npy"), self.data.y.cpu().detach().numpy())
            if training_data:
                np.save(os.path.join(cfg.UNIQUE, "predictions", "mask_train.npy"), mask_train.cpu().detach().numpy())
            np.save(os.path.join(cfg.UNIQUE, "predictions", "mask_val.npy"), mask_val.cpu().detach().numpy())
            np.save(os.path.join(cfg.UNIQUE, "predictions", "mask_test.npy"), mask_test.cpu().detach().numpy())

        return scores


def train_test_gnn(topology_matrix,
                   F_set,
                   *args,
                   return_pred=False,
                   seed=42,
                   lr=1e-2,
                   epochs=251,
                   show_progress_bar=False,
                   show=False,
                   save_path="",
                   save_name="",
                   device=torch.device("cpu"),
                   **kwargs):
    dataset = MyDataset(proximity_matrix=topology_matrix,
                        features=F_set,
                        device=device,
                        seed=seed,
                        **kwargs)

    # create data object
    data = dataset[0]
    data = data.to(device)

    # create model
    model = Net(data, dataset, lr=lr, device=device, **kwargs).to(device)
    os.makedirs(os.path.join(save_path, "model_summary"), exist_ok=True)
    model.summary(show=False, save_path=os.path.join(save_path, "model_summary", save_name) + "_MODEL_SUMMARY.log")
    metrics = []
    losses = []
    start = time.time()
    for epoch in tqdm(range(epochs),
                      desc="training",
                      position=0 if type(show_progress_bar) == bool else show_progress_bar,
                      leave=False,
                      disable=not show_progress_bar):
        loss = model.train_model()
        if show or save_path:
            losses.append({"epoch": epoch,
                           "loss": loss})
            if epoch % 10 and epoch < epochs - 1:
                scores = model.test()
                metrics.append({"epoch": epoch, **scores})

    if show or save_path:
        losses = pd.DataFrame(losses)
        metrics = pd.DataFrame(metrics)
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        sns.lineplot(data=losses,
                     x="epoch",
                     y="loss",
                     label=f"loss {losses.iloc[-1]['loss']:.2f}")
        plt.ylabel("loss")
        plt.subplot(2, 1, 2)
        colors = ["lime", "magenta", "blue", "orange"]
        for i, m in enumerate(["gnn_accuracy_train", "gnn_accuracy_test", "gnn_f1_train_macro", "gnn_f1_test_macro"]):
            sns.lineplot(data=metrics,
                         x="epoch",
                         y=m,
                         label=f"{m} {metrics.iloc[-1][m]:.2f}",
                         alpha=0.5,
                         color=colors[i])
        plt.ylabel("scores")
        plt.ylim([0, 1])
        plt.legend(loc="upper left")
        plt.suptitle(f"Loss - lr={lr} - epochs={epochs} - layers={model.layers_size} - time={time.time() - start:.2f}s")
        if show:
            plt.show()
        if save_path:
            os.makedirs(os.path.join(save_path, "GNN_plots"), exist_ok=True)
            plt.savefig(os.path.join(save_path, "GNN_plots", save_name) + ".png")
            plt.close()

    scores = model.test(training_data=False)
    if return_pred:
        pred = model().max(1)[1].cpu().detach().numpy()
        return {**scores, "pred": pred}
    else:
        return scores
