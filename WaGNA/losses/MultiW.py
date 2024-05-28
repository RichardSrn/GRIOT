import os.path

import ot.bregman
import torch
import os
import matplotlib.pyplot as plt
from geomloss.sinkhorn_divergence import scaling_parameters, sinkhorn_loop, log_weights, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized
from ..config import cfg


class MultiW:
    def __init__(self,
                 alpha=0.5,
                 epsilon=0.1,
                 p=2,
                 numItermax=1000,
                 stopThr=1e-9,
                 method="sinkhorn",
                 path=".",
                 plot=False,
                 unique=None,
                 p_unif=True,
                 normalize_F=False,
                 normalize_MF_MC=False,
                 use_geomloss=True,
                 CrossEtpy=False,
                 **kwargs):
        self.alpha = alpha
        self.epsilon = epsilon
        self.p = p
        self.numItermax = numItermax
        self.name = "MultiW"
        self.stopThr = stopThr
        self.method = method
        self.path = path
        self.plot = plot
        self.unique = unique
        self.p_unif = p_unif
        self.normalize_F = normalize_F
        self.normalize_MF_MC = normalize_MF_MC
        self.use_geomloss = use_geomloss
        self.CrossEtpy = CrossEtpy
        self.kwargs = kwargs
        self.sumup()

    def sumup(self):
        """
        Write a summary of the class parameters to a file.
        """
        path = cfg.UNIQUE
        file_name = f"{path}/MultiW.txt"

        l_k = min(max([len(k) for k in self.__dict__.keys()]), 25)
        s = ""
        for k, v in self.__dict__.items():
            s += f"{k.ljust(l_k)}: {v}" + "\n"
        s = "MultiW CONFIG\n" + s + "-" * 25 + "\n"

        if os.path.isfile(file_name):
            with open(file_name, "a") as f:
                f.write(s)
        else:
            with open(file_name, "w") as f:
                f.write(s)



    @staticmethod
    def cross_entropy_distance(A, B):
        M = torch.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                M[i, j] = torch.nn.functional.cross_entropy(A[i], B[j])
        return M

    def __call__(self,
                 C1: torch.Tensor = None,
                 C2: torch.Tensor = None,
                 F1: torch.Tensor = None,
                 F2: torch.Tensor = None,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 i=None,
                 total_iterations=None,
                 return_transport=False,
                 return_distance=True,
                 **kwargs):

        C1 = C1.clone()
        C2 = C2.clone()
        F1 = F1.clone()
        F2 = F2.clone()

        # get histograms (nodes' weight)
        if p1 is None or self.p_unif:
            p1 = torch.ones(C1.shape[0], dtype=torch.float32)
        if p2 is None or self.p_unif:
            p2 = torch.ones(C2.shape[0], dtype=torch.float32)

        p1 = p1.clone()
        p2 = p2.clone()

        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        p1 = p1.to(F1.device)
        p2 = p2.to(F2.device)

        ### GEOMLOSS - LEGACY CODE - START ###
        if self.use_geomloss:
            # compute pairwise distance between rows of F1 and F2 according to the L2 norm compute M_F and M_C
            if not self.CrossEtpy:
                M_F_12 = torch.cdist(F1, F2, p=self.p) ** 2 / 2
                M_F_21 = M_F_12.T
                M_F_11 = torch.cdist(F1, F1, p=self.p) ** 2 / 2
                M_F_22 = torch.cdist(F2, F2, p=self.p) ** 2 / 2
            else:  # compute the crossentropy between the vectors of F1 and F1, F1 and F1, F2 and F2, etc.
                M_F_12 = self.cross_entropy_distance(F1, F2).to(F1.device)
                M_F_21 = M_F_12.T
                M_F_11 = self.cross_entropy_distance(F1, F1).to(F1.device)
                M_F_22 = self.cross_entropy_distance(F2, F2).to(F1.device)

            M_C_12 = torch.cdist(C1, C2, p=self.p) ** 2 / 2
            M_C_21 = M_C_12.T
            M_C_11 = torch.cdist(C1, C1, p=self.p) ** 2 / 2
            M_C_22 = torch.cdist(C2, C2, p=self.p) ** 2 / 2

            # # normalize M_F and M_C
            if self.normalize_MF_MC:
                M_F_11 = (M_F_11 - M_F_11.min()) / (M_F_11.max() - M_F_11.min())
                M_C_11 = (M_C_11 - M_C_11.min()) / (M_C_11.max() - M_C_11.min())

                M_F_12 = (M_F_12 - M_F_12.min()) / (M_F_12.max() - M_F_12.min())
                M_C_12 = (M_C_12 - M_C_12.min()) / (M_C_12.max() - M_C_12.min())

                M_F_22 = (M_F_22 - M_F_22.min()) / (M_F_22.max() - M_F_22.min())
                M_C_22 = (M_C_22 - M_C_22.min()) / (M_C_22.max() - M_C_22.min())

            # compute M
            M_12 = (1 - self.alpha) * M_F_12 + (self.alpha) * M_C_12
            M_21 = (1 - self.alpha) * M_F_21 + (self.alpha) * M_C_21
            M_11 = (1 - self.alpha) * M_F_11 + (self.alpha) * M_C_11
            M_22 = (1 - self.alpha) * M_F_22 + (self.alpha) * M_C_22

            diameter, eps, eps_list, _ = scaling_parameters(x=F1.unsqueeze(0),
                                                            y=F2.unsqueeze(0),
                                                            p=2,
                                                            blur=self.epsilon,
                                                            reach=None,
                                                            diameter=None,
                                                            scaling=0.9)

            # Use an optimal transport solver to retrieve the dual potentials:
            f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
                softmin_tensorized,
                log_weights(p1),
                log_weights(p2),
                M_11.unsqueeze(0),
                M_22.unsqueeze(0),
                M_12.unsqueeze(0),
                M_21.unsqueeze(0),
                eps_list,
                rho=None,
                debias=True,
            )

            if return_transport:
                ot_plan = None
            if return_distance:
                w = sinkhorn_cost(
                    eps=eps,
                    rho=None,
                    a=p1.unsqueeze(0),
                    b=p2.unsqueeze(0),
                    f_aa=f_aa,
                    g_bb=g_bb,
                    g_ab=g_ab,
                    f_ba=f_ba,
                    batch=True,
                    debias=True,
                    potentials=False,
                )
        ### GEOMLOSS - LEGACY CODE - END ###

        # Compute the Wasserstein distance between F1 and F2
        # Create the cost matrix
        # normalize F_k by columns
        else :
            if self.normalize_F:
                F1 = (F1 - F1.mean(dim=0)) / (F1.max(dim=0).values - F1.min(dim=0).values + 1e-5)
                F2 = (F2 - F2.mean(dim=0)) / (F2.max(dim=0).values - F2.min(dim=0).values + 1e-5)
            # normalize C_k
            C1 = (C1 - C1.min()) / (C1.max() - C1.min())
            C2 = (C2 - C2.min()) / (C2.max() - C2.min())

            # compute pairwise distance between rows of F1 and F2 according to the L2 norm compute M_F and M_C
            if not self.CrossEtpy:
                M_F = torch.cdist(F1, F2, p=self.p) ** 2 / 2
            else:  # compute the crossentropy between the vectors of F1 and F1, F1 and F1, F2 and F2, etc.
                M_F = self.cross_entropy_distance(F1, F2).to(F1.device)

            M_C = torch.cdist(C1, C2, p=self.p) ** 2 / 2

            # # normalize M_F and M_C
            if self.normalize_MF_MC:
                M_F = (M_F - M_F.min()) / (M_F.max() - M_F.min())
                M_C = (M_C - M_C.min()) / (M_C.max() - M_C.min())

            # compute M
            M = (1 - self.alpha) * M_F + (self.alpha) * M_C

            # get OT
            if return_transport:
                ot_plan = ot.sinkhorn(p1, p2, M, reg=self.epsilon)
            if return_distance:
                w = ot.sinkhorn2(p1, p2, M, reg=self.epsilon)

        if self.plot:
            if self.use_geomloss:
                M_F_11_ = M_F_11.clone().cpu()
                M_F_12_ = M_F_12.clone().cpu()
                M_F_22_ = M_F_22.clone().cpu()
                M_C_11_ = M_C_11.clone().cpu()
                M_C_12_ = M_C_12.clone().cpu()
                M_C_22_ = M_C_22.clone().cpu()
                M_11_ = M_11.clone().cpu()
                M_12_ = M_12.clone().cpu()
                M_22_ = M_22.clone().cpu()
            else:
                M_F_ = M_F.clone().cpu()
                M_C_ = M_C.clone().cpu()
                M_ = M.clone().cpu()

            os.makedirs(os.path.join(self.path, 'multiw'), exist_ok=True)

            fig, ax = plt.subplot_mosaic("ABCZ;EFGZ;IJKZ",
                                         figsize=(20, 10),
                                         gridspec_kw={'width_ratios': [1, 1, 1, 4]})

            def imshow_with_colorbar(ax_name, title, data):
                ax[ax_name].imshow(data.detach().numpy())
                fig.colorbar(ax[ax_name].imshow(data.detach().numpy()), ax=ax[ax_name])
                ax[ax_name].set_title(title)

            if self.use_geomloss:
                imshow_with_colorbar('A', "M_F_11", M_F_11_)
                imshow_with_colorbar('B', "M_F_12", M_F_12_)
                imshow_with_colorbar('C', "M_F_22", M_F_22_)

                imshow_with_colorbar('E', "M_C_11", M_C_11_)
                imshow_with_colorbar('F', "M_C_12", M_C_12_)
                imshow_with_colorbar('G', "M_C_22", M_C_22_)

                imshow_with_colorbar('I', "M_11", M_11_)
                imshow_with_colorbar('J', "M_12", M_12_)
                imshow_with_colorbar('K', "M_22", M_22_)
            else:
                imshow_with_colorbar('B', "M_F_11", M_F_)
                imshow_with_colorbar('F', "M_C_12", M_C_)
                imshow_with_colorbar('J', "M_12", M_)

            if self.use_geomloss:
                # reshape the data M_F, M_C and M to 1D.
                # sort M by ascending order and sort M_F and M_C by the same order.
                # then plot them all on a single plot.
                # Plot each distribution as a line plot.
                M_F_11_ = M_F_11_.reshape(-1)
                M_F_12_ = M_F_12_.reshape(-1)
                M_F_22_ = M_F_22_.reshape(-1)

                M_C_11_ = M_C_11_.reshape(-1)
                M_C_12_ = M_C_12_.reshape(-1)
                M_C_22_ = M_C_22_.reshape(-1)

                M_11_ = M_11_.reshape(-1)
                M_12_ = M_12_.reshape(-1)
                M_22_ = M_22_.reshape(-1)

                M_F_11_ = M_F_11_[M_11_.argsort()]
                M_C_11_ = M_C_11_[M_11_.argsort()]
                M_11_ = M_11_[M_11_.argsort()]

                M_F_12_ = M_F_12_[M_12_.argsort()]
                M_C_12_ = M_C_12_[M_12_.argsort()]
                M_12_ = M_12_[M_12_.argsort()]

                M_F_22_ = M_F_22_[M_22_.argsort()]
                M_C_22_ = M_C_22_[M_22_.argsort()]
                M_22_ = M_22_[M_22_.argsort()]
            else:
                M_F_ = M_F_.reshape(-1)
                M_C_ = M_C_.reshape(-1)
                M_ = M_.reshape(-1)
                M_F_ = M_F_[M_.argsort()]
                M_C_ = M_C_[M_.argsort()]
                M_ = M_[M_.argsort()]

            if self.use_geomloss:
                if self.alpha < 1:
                    ax['Z'].plot(M_F_11_.detach().numpy(),
                                 label="M_F_11",
                                 color="red",
                                 alpha=0.75,
                                 linestyle=':',
                                 linewidth=1.5)
                if self.alpha > 0:
                    ax['Z'].plot(M_C_11_.detach().numpy(),
                                 label="M_C_11",
                                 color="blue",
                                 alpha=0.75,
                                 linestyle='-.',
                                 linewidth=1.5)
                ax['Z'].plot(M_11_.detach().numpy(),
                             label="M_11",
                             color="violet",
                             alpha=0.75,
                             linestyle="--",
                             linewidth=3)

            if self.alpha < 1:
                ax['Z'].plot(M_F_12_.detach().numpy() if self.use_geomloss else M_F_.detach().numpy(),
                # ax['Z'].plot(M_F_.detach().numpy(),
                             label="M_F_12" if self.use_geomloss else "M_F",
                             color="pink",
                             alpha=0.75,
                             linestyle=':',
                             linewidth=1.5)
            if self.alpha > 0:
                ax['Z'].plot(M_C_12_.detach().numpy() if self.use_geomloss else M_C_.detach().numpy(),
                # ax['Z'].plot(M_C_.detach().numpy(),
                             label="M_C_12" if self.use_geomloss else "M_C",
                             color="cyan",
                             alpha=0.75,
                             linestyle='-.',
                             linewidth=1.5)
            ax['Z'].plot(M_12_.detach().numpy() if self.use_geomloss else M_.detach().numpy(),
            # ax['Z'].plot(M_.detach().numpy(),
                         label="M_12" if self.use_geomloss else "M",
                         color="magenta",
                         alpha=0.75,
                         linestyle='--',
                         linewidth=3)

            if self.use_geomloss:
                if self.alpha < 1:
                    ax['Z'].plot(M_F_22_.detach().numpy(),
                                 label="M_F_22",
                                 color="brown",
                                 alpha=0.75,
                                 linestyle=':',
                                 linewidth=1.5)
                if self.alpha > 0:
                    ax['Z'].plot(M_C_22_.detach().numpy(),
                                 label="M_C_22",
                                 color="teal",
                                 alpha=0.75,
                                 linestyle='-.',
                                 linewidth=1.5)
                ax['Z'].plot(M_22_.detach().numpy(),
                             label="M_22",
                             color="purple",
                             alpha=0.75,
                             linestyle='--',
                             linewidth=3)

            ax['Z'].legend(bbox_to_anchor=(1.01, 1),
                           loc='upper left',
                           borderaxespad=0.)

            fig.suptitle(f"MultiW, w={w.item():.2f}, iteration={i}/{total_iterations}, alpha={self.alpha}",
                         fontsize=16,
                         y=.99)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(os.path.join(self.path, 'multiw'), f"multiw_bar_{i}o{total_iterations}_"
                                                                        f"{self.unique}.png"))
            plt.close()

            if self.use_geomloss:
                del M_F_11_, M_F_12_, M_F_22_, M_C_11_, M_C_12_, M_C_22_, M_11_, M_12_, M_22_
            else:
                del M_F_, M_C_, M_

        del C1, C2
        del F1, F2
        del p1, p2
        if self.use_geomloss:
            del M_F_11, M_F_12, M_F_22, M_C_11, M_C_12, M_C_22, M_C_21, M_F_21, M_11, M_12, M_22, M_21
            del f_aa, g_bb, g_ab, f_ba
            del diameter, eps, eps_list
        else:
            del M_F, M_C, M
        del kwargs
        del i
        del total_iterations

        if return_distance and return_transport:
            return w, ot_plan
        elif return_transport:
            return ot_plan
        elif return_distance:
            return w
        else:
            return None



#### LEGACY CODE ####
"""

        if self.use_geomloss:
            # compute pairwise distance between rows of F1 and F2 according to the L2 norm compute M_F and M_C
            if not self.CrossEtpy:
                M_F_12 = torch.cdist(F1, F2, p=self.p) ** 2 / 2
                M_F_21 = M_F_12.T
                M_F_11 = torch.cdist(F1, F1, p=self.p) ** 2 / 2
                M_F_22 = torch.cdist(F2, F2, p=self.p) ** 2 / 2
            else:  # compute the crossentropy between the vectors of F1 and F1, F1 and F1, F2 and F2, etc.
                M_F_12 = self.cross_entropy_distance(F1, F2).to(F1.device)
                M_F_21 = M_F_12.T
                M_F_11 = self.cross_entropy_distance(F1, F1).to(F1.device)
                M_F_22 = self.cross_entropy_distance(F2, F2).to(F1.device)

            M_C_12 = torch.cdist(C1, C2, p=self.p) ** 2 / 2
            M_C_21 = M_C_12.T
            M_C_11 = torch.cdist(C1, C1, p=self.p) ** 2 / 2
            M_C_22 = torch.cdist(C2, C2, p=self.p) ** 2 / 2

            # # normalize M_F and M_C
            if self.normalize_MF_MC:
                M_F_11 = (M_F_11 - M_F_11.min()) / (M_F_11.max() - M_F_11.min())
                M_C_11 = (M_C_11 - M_C_11.min()) / (M_C_11.max() - M_C_11.min())

                M_F_12 = (M_F_12 - M_F_12.min()) / (M_F_12.max() - M_F_12.min())
                M_C_12 = (M_C_12 - M_C_12.min()) / (M_C_12.max() - M_C_12.min())

                M_F_22 = (M_F_22 - M_F_22.min()) / (M_F_22.max() - M_F_22.min())
                M_C_22 = (M_C_22 - M_C_22.min()) / (M_C_22.max() - M_C_22.min())

            # compute M
            M_12 = (1 - self.alpha) * M_F_12 + (self.alpha) * M_C_12
            M_21 = (1 - self.alpha) * M_F_21 + (self.alpha) * M_C_21
            M_11 = (1 - self.alpha) * M_F_11 + (self.alpha) * M_C_11
            M_22 = (1 - self.alpha) * M_F_22 + (self.alpha) * M_C_22

            diameter, eps, eps_list, _ = scaling_parameters(x=F1.unsqueeze(0),
                                                            y=F2.unsqueeze(0),
                                                            p=2,
                                                            blur=self.epsilon,
                                                            reach=None,
                                                            diameter=None,
                                                            scaling=0.9)

            # Use an optimal transport solver to retrieve the dual potentials:
            f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
                softmin_tensorized,
                log_weights(p1),
                log_weights(p2),
                M_11.unsqueeze(0),
                M_22.unsqueeze(0),
                M_12.unsqueeze(0),
                M_21.unsqueeze(0),
                eps_list,
                rho=None,
                debias=True,
            )

            if return_transport:
                ot_plan = None
            if return_distance:
                w = sinkhorn_cost(
                    eps=eps,
                    rho=None,
                    a=p1.unsqueeze(0),
                    b=p2.unsqueeze(0),
                    f_aa=f_aa,
                    g_bb=g_bb,
                    g_ab=g_ab,
                    f_ba=f_ba,
                    batch=True,
                    debias=True,
                    potentials=False,
                )
        else:
        
"""