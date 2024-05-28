import os.path
import torch
import os
import matplotlib.pyplot as plt
from geomloss.sinkhorn_divergence import scaling_parameters, sinkhorn_loop, log_weights, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized
from typing import List


class MultiW2():
    def __init__(self,
                 alpha=(0.34,0.33,0.33),
                 epsilon=0.1,
                 p=2,
                 numItermax=1000,
                 stopThr=1e-9,
                 method="sinkhorn",
                 path=".",
                 unique=None,
                 normalize_p=True,
                 normalize_views=False,
                 normalize_distances=False,
                 **kwargs):
        self.alpha = alpha
        self.epsilon = epsilon
        self.p = p
        self.numItermax = numItermax
        self.name = "MultiW"
        self.stopThr = stopThr
        self.method = method
        self.path = path
        self.unique = unique
        self.normalize_p = normalize_p
        self.normalize_views = normalize_views
        self.normalize_distances = normalize_distances
        self.kwargs = kwargs

    def __call__(self,
                 C1 : torch.Tensor,
                 C2 : torch.Tensor,
                 view1: List[torch.Tensor],
                 view2: List[torch.Tensor],
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 plot=False,
                 i=None,
                 total_iterations=None,
                 **kwargs):

        # clone Cs
        C1 = C1.clone()
        C2 = C2.clone()
        # clone views
        view1 = [v.clone() for v in view1]
        view2 = [v.clone() for v in view2]

        # get histograms (nodes' weight)
        if p1 is None or self.normalize_p:
            p1 = torch.ones(view1[0].shape[0], dtype=torch.float32)
        if p2 is None or self.normalize_p:
            p2 = torch.ones(view2[0].shape[0], dtype=torch.float32)

        p1 = p1.clone()
        p2 = p2.clone()

        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        p1 = p1.to(C1.device)
        p2 = p2.to(C2.device)

        # normalize C_k
        C1 = (C1 - C1.min()) / (C1.max() - C1.min())
        C2 = (C2 - C2.min()) / (C2.max() - C2.min())
        # normalize views
        if self.normalize_views:
            for i in range(len(view1)):
                view1[i] = (view1[i] - view1[i].min(dim=0)) / (view1[i].max(dim=0) - view1[i].min(dim=0) + 1e-5)
                view2[i] = (view2[i] - view2[i].min(dim=0)) / (view2[i].max(dim=0) - view2[i].min(dim=0) + 1e-5)

        # compute pairwise distance between rows of C1 and C2 according to the Lp norm
        M_C_11 = torch.cdist(C1, C1, p=self.p) ** 2 / 2
        M_C_12 = torch.cdist(C1, C2, p=self.p) ** 2 / 2
        M_C_21 = M_C_12.clone().T
        M_C_22 = torch.cdist(C2, C2, p=self.p) ** 2 / 2
        # compute pairwise distance between rows of each views according to the Lp norm
        M_v_11, M_v_12, M_v_21, M_v_22 = [], [], [], []
        for v1, v2 in zip(view1, view2):
            M_v_11.append(torch.cdist(v1, v1, p=self.p) ** 2 / 2)
            M_v_12.append(torch.cdist(v1, v2, p=self.p) ** 2 / 2)
            M_v_21.append(M_v_12[-1].clone().T)
            M_v_22.append(torch.cdist(v2, v2, p=self.p) ** 2 / 2)


        # normalize M_C_xy and M_v_xy
        if self.normalize_distances:
            # M_C_xy
            M_C_11 = (M_C_11 - M_C_11.min()) / (M_C_11.max() - M_C_11.min())
            M_C_12 = (M_C_12 - M_C_12.min()) / (M_C_12.max() - M_C_12.min())
            M_C_21 = (M_C_21 - M_C_21.min()) / (M_C_21.max() - M_C_21.min())
            M_C_22 = (M_C_22 - M_C_22.min()) / (M_C_22.max() - M_C_22.min())
            # M_v_xy
            for i in range(len(M_v_11)):
                M_v_11[i] = (M_v_11[i] - M_v_11[i].min()) / (M_v_11[i].max() - M_v_11[i].min())
                M_v_12[i] = (M_v_12[i] - M_v_12[i].min()) / (M_v_12[i].max() - M_v_12[i].min())
                M_v_21[i] = (M_v_21[i] - M_v_21[i].min()) / (M_v_21[i].max() - M_v_21[i].min())
                M_v_22[i] = (M_v_22[i] - M_v_22[i].min()) / (M_v_22[i].max() - M_v_22[i].min())

        # compute $ M = \alpha_0 * (M_C_xy) + \sum_{i=1} \alpha_i * (M_v_xy)_{i-1} $
        M_11 = self.alpha[0] * M_C_11 + sum([self.alpha[i] * M_v_11[i-1] for i in range(1, len(self.alpha))])
        M_12 = self.alpha[0] * M_C_12 + sum([self.alpha[i] * M_v_12[i-1] for i in range(1, len(self.alpha))])
        M_21 = self.alpha[0] * M_C_21 + sum([self.alpha[i] * M_v_21[i-1] for i in range(1, len(self.alpha))])
        M_22 = self.alpha[0] * M_C_22 + sum([self.alpha[i] * M_v_22[i-1] for i in range(1, len(self.alpha))])

        diameter, eps, eps_list, _ = scaling_parameters(x=C1.unsqueeze(0),
                                                        y=C2.unsqueeze(0),
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

        if plot:
            os.makedirs(os.path.join(self.path, 'multiw'), exist_ok=True)

            fig, ax = plt.subplot_mosaic("ABCZ;EFGZ;IJKZ",
                                         figsize=(20, 10),
                                         gridspec_kw={'width_ratios': [1, 1, 1, 4]})

            def imshow_with_colorbar(ax_name, title, data):
                ax[ax_name].imshow(data.detach().numpy())
                fig.colorbar(ax[ax_name].imshow(data.detach().numpy()), ax=ax[ax_name])
                ax[ax_name].set_title(title)

            imshow_with_colorbar('A', "M_F_11", M_v_11[0])
            imshow_with_colorbar('B', "M_F_12", M_v_12[0])
            imshow_with_colorbar('C', "M_F_22", M_v_22[0])

            imshow_with_colorbar('E', "M_C_11", M_C_11)
            imshow_with_colorbar('F', "M_C_12", M_C_12)
            imshow_with_colorbar('G', "M_C_22", M_C_22)

            imshow_with_colorbar('I', "M_11", M_11)
            imshow_with_colorbar('J', "M_12", M_12)
            imshow_with_colorbar('K', "M_22", M_22)

            # reshape the data M_F, M_C and M to 1D.
            # sort M by ascending order and sort M_F and M_C by the same order.
            # then plot them all on a single plot.
            # Plot each distribution as a line plot.
            M_F_11 = M_v_11[0].reshape(-1)
            M_F_12 = M_v_12[0].reshape(-1)
            M_F_22 = M_v_22[0].reshape(-1)

            M_C_11 = M_C_11.reshape(-1)
            M_C_12 = M_C_12.reshape(-1)
            M_C_22 = M_C_22.reshape(-1)

            M_11 = M_11.reshape(-1)
            M_12 = M_12.reshape(-1)
            M_22 = M_22.reshape(-1)

            M_F_11 = M_F_11[M_11.argsort()]
            M_C_11 = M_C_11[M_11.argsort()]
            M_11 = M_11[M_11.argsort()]

            M_F_12 = M_F_12[M_12.argsort()]
            M_C_12 = M_C_12[M_12.argsort()]
            M_12 = M_12[M_12.argsort()]

            M_F_22 = M_F_22[M_22.argsort()]
            M_C_22 = M_C_22[M_22.argsort()]
            M_22 = M_22[M_22.argsort()]

            if self.alpha < 1:
                ax['Z'].plot(M_F_11.detach().numpy(),
                             label="M_F_11",
                             color="red",
                             alpha=0.75,
                             linestyle=':',
                             linewidth=1.5)
            if self.alpha > 0:
                ax['Z'].plot(M_C_11.detach().numpy(),
                             label="M_C_11",
                             color="blue",
                             alpha=0.75,
                             linestyle='-.',
                             linewidth=1.5)
            ax['Z'].plot(M_11.detach().numpy(),
                         label="M_11",
                         color="violet",
                         alpha=0.75,
                         linestyle="--",
                         linewidth=3)

            if self.alpha < 1:
                ax['Z'].plot(M_F_12.detach().numpy(),
                             label="M_F_12",
                             color="pink",
                             alpha=0.75,
                             linestyle=':',
                             linewidth=1.5)
            if self.alpha > 0:
                ax['Z'].plot(M_C_12.detach().numpy(),
                             label="M_C_12",
                             color="cyan",
                             alpha=0.75,
                             linestyle='-.',
                             linewidth=1.5)
            ax['Z'].plot(M_12.detach().numpy(),
                         label="M_12",
                         color="magenta",
                         alpha=0.75,
                         linestyle='--',
                         linewidth=3)

            if self.alpha < 1:
                ax['Z'].plot(M_F_22.detach().numpy(),
                             label="M_F_22",
                             color="brown",
                             alpha=0.75,
                             linestyle=':',
                             linewidth=1.5)
            if self.alpha > 0:
                ax['Z'].plot(M_C_22.detach().numpy(),
                             label="M_C_22",
                             color="teal",
                             alpha=0.75,
                             linestyle='-.',
                             linewidth=1.5)
            ax['Z'].plot(M_22.detach().numpy(),
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

        del C1, C2
        del view1, view2
        del p1, p2
        del M_v_11, M_v_12, M_v_21, M_v_22, M_C_11, M_C_12, M_C_22, M_C_21, M_11, M_12, M_22, M_21
        del f_aa, g_bb, g_ab, f_ba
        del diameter, eps, eps_list
        del kwargs
        del plot
        del i
        del total_iterations

        return w
