from .SEP import SEP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import logging
import re

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def plot_attributes(F, edge_list=None,
                    kind="PCA", plot_edges=False,
                    path=".", title="",
                    communities_names=None,
                    s=100,
                    **kwargs):
    """
    Plot the attributes of tensor F.
    The first columns of F (F[:,0]) is the community of the nodes, it gives the color of each point of the scatter.
    The other features are to be plotted spatially.
    Distinguish two cases :
    1) if there are 2 features to F (beside the community), scatter plot the features F1 and F2 and community as color.
    2) if there are more than 2 features (beside community), plot a PCA of the features F1, F2, F3, ... and community as color.
    :param F: tensor of attributes
    :return: None

        # TODO : continuous cmap plot missing attributes

    """

    assert F.shape[1] > 1, "Not enough features to plot"
    assert not (edge_list is None and plot_edges), "G must be different than None if plot_edges is True"

    F = F.clone()

    # colors = ["blue", "orange", "green", "red", "yellow", "purple", "magenta", "brown", "yellow", "grey", "black"]
    colors = list(plt.cm.get_cmap("tab10").colors[:F[:, 0].max().int().item() + 1])

    df = pd.DataFrame(F[:, 0], columns=["community"])
    if communities_names is None:
        df["community"] = df["community"].apply(int)
    else:
        df["community"] = df["community"].apply(lambda x: communities_names[int(x)])

    if title == "":
        title = f"Features ScatterPlot"
    if plot_edges:
        title += " - edges"

    try :
        if F[:, 1:].shape[1] == 2:
            x_label, y_label = "F1", "F2"
            F = F[:, 1:]
        else:
            if kind == "PCA":
                pca = PCA(n_components=2)
                F = pca.fit_transform(F[:, 1:])
                title += " - PCA"
                x_label, y_label = "PC1", "PC2"
            elif kind == "TSNE":
                tsne = TSNE(n_components=2)
                F = tsne.fit_transform(F[:, 1:])
                title += " - TSNE"
                x_label, y_label = "TSNE1", "TSNE2"
            elif kind == "UMAP":
                umap = UMAP(n_components=2)
                F = umap.fit_transform(F[:, 1:])
                title += " - UMAP"
                x_label, y_label = "UMAP1", "UMAP2"
            elif kind == "MDS":
                mds = MDS(n_components=2)
                F = mds.fit_transform(F[:, 1:])
                title += " - MDS"
                x_label, y_label = "MDS1", "MDS2"
            elif kind == "ISOMAP":
                isomap = Isomap(n_components=2)
                F = isomap.fit_transform(F[:, 1:])
                title += " - ISOMAP"
                x_label, y_label = "ISOMAP1", "ISOMAP2"
            elif kind == "LDA":
                lda = LinearDiscriminantAnalysis(n_components=2)
                F = lda.fit_transform(F[:, 1:], F[:, 0])
                title += " - LDA"
                x_label, y_label = "LDA1", "LDA2"
            elif kind == "SEP":
                sep = SEP(n_components=2, r2=-1)
                F = sep.fit_transform(F[:, 1:], F[:, 0])
                title += " - SEP"
                x_label, y_label = "SEP1", "SEP2"
            elif kind == "SEP+":
                sep = SEP(n_components=2, r2=1)
                F = sep.fit_transform(F[:, 1:], F[:, 0])
                title += " - SEP_p"
                x_label, y_label = "SEP1", "SEP2"
            else:
                title = None
                x_label, y_label = None, None
                raise("kind not implemented.")

        df[x_label] = F[:, 0]
        df[y_label] = F[:, 1]

        # plt.figure(figsize=(10, 10))
        plt.figure(figsize=(3.5, 3.5))
        sns.set_theme()
        if plot_edges :
            # plot edges between coordinates of nodes df[x_label] and df[y_label]
            for i, j in edge_list:
                plt.plot([df[x_label][i], df[x_label][j]], [df[y_label][i], df[y_label][j]],
                         'k-',
                         lw=0.3,
                         # alpha=0.5,
                         alpha=1,
                         zorder=1)
        sns.scatterplot(data=df,
                        x=x_label,
                        y=y_label,
                        hue="community",
                        # style="missing",
                        style="community",
                        # s=s,#50,
                        s=50,#s,
                        # alpha=0.75,
                        alpha=0.9,
                        palette=colors[:len(df["community"].unique())],
                        zorder=2,
                        legend=False,
                        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, alpha=1, zorder=0, color="white")
        # hide x and y axis numbers
        plt.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelbottom=False,
                        labelleft=False)
        # plt.legend(loc='upper right')
        # plt.title(title)
        plt.tight_layout()
        # replace any number of space with underscore
        title = re.sub(r"(?:\s|-)+", "_", title)
        logging.info(f"save : ./{path}/{title}.png")
        plt.savefig(f"./{path}/{title}.png", dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()
    except Exception as e :
        print(e)
