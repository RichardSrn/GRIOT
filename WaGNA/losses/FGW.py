import networkx as nx
import ot
import torch
import numpy as np


class FGW():
    def __init__(self, alpha=0.5, log=False, **kwargs):
        self.alpha = alpha
        self.name = "FGW"
        self.log = log
        self.kwargs = kwargs

    def __call__(self,
                 C1: torch.Tensor = None,
                 C2: torch.Tensor = None,
                 F1: torch.Tensor = None,
                 F2: torch.Tensor = None,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 **kwargs):
        # get histograms (nodes' weight)
        if p1 is None:
            p1 = torch.ones(C1.shape[0], dtype=torch.float32)
        if p2 is None:
            p2 = torch.ones(C2.shape[0], dtype=torch.float32)

        if C1.max() > 1 or C2.max() > 1:
            C1[C1 > 1] = 0
            C2[C2 > 1] = 0

        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        # Compute Euclidean distance matrix between F1 and F2
        M = torch.cdist(F1, F2, p=2)

        # Make sure everything is float and not double to avoid error
        M = M.to(torch.float32)
        C1 = C1.to(torch.float32)
        C2 = C2.to(torch.float32)

        # compute FGW between G1 and G2
        fgw_dist = ot.gromov.fused_gromov_wasserstein2(M,
                                                       C2,
                                                       C1,
                                                       p1,
                                                       p2,
                                                       alpha=self.alpha,
                                                       log=self.log)
        return fgw_dist
