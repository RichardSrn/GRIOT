from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from ..utils.graph_tool import load_matrices, add_missing_features
import numpy as np
import networkx as nx


# custom dataset class #RR and ONE
class MyDatasetMulti(InMemoryDataset):
    def __init__(self,
                 features,
                 proximity_matrix,
                 keep_proximity_matrix=False,
                 transform=None,
                 seed=42,
                 device=None,
                 max_precision_16=True,
                 logging=None,
                 **kwargs):

        if type(proximity_matrix) == torch.Tensor:
            adjacency_matrix = proximity_matrix.clone()
        else:  # np.ndarray
            adjacency_matrix = proximity_matrix.copy()
        if adjacency_matrix.max() > 1:
            adjacency_matrix[adjacency_matrix > 1] = 0

        if type(adjacency_matrix) == torch.Tensor:
            G = nx.from_numpy_array(adjacency_matrix.detach().numpy())
        else:
            G = nx.from_numpy_array(adjacency_matrix)

        if type(features) == torch.Tensor:
            Features = features.clone()
        else:  # np.ndarray
            Features = features.copy()
        if type(Features) == np.ndarray:
            Features = torch.from_numpy(Features)

        # create edge index from G
        # edge_index = torch.tensor([list(e) for e in G.edges]).T
        adj = nx.to_scipy_sparse_array(G).tocoo()
        # adj = adjacency_matrix
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        labels = Features[:, 0].to(int)
        embeddings = Features[:, 1:]

        super(MyDatasetMulti, self).__init__('.', transform, None, None)

        data = Data(edge_index=edge_index)
        data.edge_index = edge_index

        data.num_nodes = G.number_of_nodes()

        # embedding
        data.x = embeddings
        # set edge_attr_dim for grape baseline
        data.edge_attr_dim = data.x.shape[1]

        # if max_precision_16:
        #     # if precision of embeddings is too high, lower it
        #     if data.x.dtype == torch.float64:  # or data.x.dtype == torch.float32 :
        #         if logging:
        #             logging.warning("Lowering precision of embeddings to torch.float16")
        #         else:
        #             print("Lowering precision of embeddings to torch.float16")
        #         data.x = data.x.to(torch.float32)

        # labels
        data.y = labels.clone().detach()
        data.num_classes = len(np.unique(data.y))

        # set data.num_classes to unique number of labels
        data.num_classes = len(torch.unique(labels))

        X_train, X_test, y_train, y_test = train_test_split(list(G.nodes()),
                                                            labels,
                                                            test_size=0.15,
                                                            random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                          y_train,
                                                          test_size=0.30,
                                                          random_state=seed)

        n_nodes = G.number_of_nodes()

        # create train, val, and train_test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train] = True
        val_mask[X_val] = True
        test_mask[X_test] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data.train_vertex_index = torch.tensor(X_train)
        data.val_vertex_index = torch.tensor(X_val)
        data.test_vertex_index = torch.tensor(X_test)

        data.adjacency_matrix = adjacency_matrix
        if keep_proximity_matrix:
            data.proximity_matrix = proximity_matrix

        # make the laplacian matrix
        data.laplacian_matrix = torch.tensor(nx.laplacian_matrix(G).todense())

        if device is not None:
            self.device = device

        data.to(self.device)

        self.data, self.slices = self.collate([data])



    def get_x_miss(self,
                   p_miss=0.5,
                   max_missing=0,
                   verbose=False,
                   logging=None,
                   preserve_first_column=True,
                   missing_mechanism=None,
                   seed=42):
        # missing features
        x_miss = add_missing_features(self._data.x, self._data.adjacency_matrix.cpu(),
                                      p_miss=p_miss, seed=seed,
                                      max_missing=max_missing, return_all=True,
                                      verbose=verbose, logging=logging,
                                      preserve_first_column=preserve_first_column,
                                      **missing_mechanism)
        self._data.x_miss = x_miss["X_incomp"].to(self.device)
        self._data.x_miss_mask = x_miss["mask"].to(self.device)
        self._data.x_miss_mask_train = self._data.x_miss_mask.clone()
        self._data.x_miss_mask_train[self._data.test_mask] = False
        self._data.x_miss_mask_train[self._data.val_mask] = False
        self._data.x_miss_mask_val = self._data.x_miss_mask.clone()
        self._data.x_miss_mask_val[self._data.train_mask] = False
        self._data.x_miss_mask_val[self._data.test_mask] = False
        self._data.x_miss_mask_test = self._data.x_miss_mask.clone()
        self._data.x_miss_mask_test[self._data.train_mask] = False
        self._data.x_miss_mask_test[self._data.val_mask] = False

        # save masks
        if type(self._data.x_miss_mask_train) == torch.Tensor:
            mask_train = self._data.x_miss_mask_train.cpu().detach().numpy()
        if type(self._data.x_miss_mask_val) == torch.Tensor:
            mask_val = self._data.x_miss_mask_val.cpu().detach().numpy()
        if type(self._data.x_miss_mask_test) == torch.Tensor:
            mask_test = self._data.x_miss_mask_test.cpu().detach().numpy()
        # np.save("mask_train.npy", mask_train)
        # np.save("mask_val.npy", mask_val)
        # np.save("mask_test.npy", mask_test)

    def to(self, device=None):
        if self.device:
            device = self.device
        self._data.to(device)
        self._data.x.to(device)
        self._data.y.to(device)
        self._data.edge_index.to(device)
        self._data.train_mask.to(device)
        self._data.test_mask.to(device)

        return self

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
