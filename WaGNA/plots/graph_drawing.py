import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def draw_legacy(G, F=None, pos=None, title=None, save=None, show=True, figsize=(10, 10)):
    if F is not None:
        degrees = dict(nx.degree(G))
        for i in range(len(G.nodes)):
            G.nodes[i]["name"] = str(i)
            if any(np.isnan(F[i])):
                G.nodes[i]["state"] = "dead"
            else:
                G.nodes[i]["state"] = "alive"
            G.nodes[i]["size"] = (degrees[i] / 2) ** 2

    if pos is None:
        # pos = nx.shell_layout(G)
        pos = nx.kamada_kawai_layout(G)
    node_color = []
    for v in nx.get_node_attributes(G, 'state').values():
        if v == "alive":
            node_color.append((0, 1, 0))
        elif v == "saved":
            node_color.append((0, 0, 1))
        else:  # dead
            node_color.append((1, 0, 0))
    node_size = np.array(list(nx.get_node_attributes(G, "size").values()))
    node_size[np.isnan(node_size)] = 250.0
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    nx.draw(G, pos, node_size=node_size, node_color=node_color, ax=ax)
    node_labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'state')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    if (0, 1, 0) in node_color:
        ax.scatter([], [], label="alive", color=(0, 1, 0))
    if (0, 0, 1) in node_color:
        ax.scatter([], [], label="saved", color=(0, 0, 1))
    if (1, 0, 0) in node_color:
        ax.scatter([], [], label="missing", color=(1, 0, 0))  # dead
    fig.legend()
    if title is not None:
        title = title.split(" ")
        title_tmp = [title[0]]
        for t in title[1:]:
            if len(title_tmp[-1]) < 50:
                title_tmp[-1] += " " + t
            else:
                title_tmp.append(t)
        title = "\n".join(title_tmp)
        fig.suptitle(title)
    if save:
        print("saved :", save)
        plt.savefig(save)
    if show:
        plt.show()
    else:
        plt.close()
    if pos is None:
        return pos

def draw2_legacy(G, F:torch.Tensor=None, pos=None, title=None, save=None, show=True, figsize=(10, 10)):
    node_colors = []
    unique_colors = dict()
    max_na_in_a_row = max(torch.isnan(F).sum(axis=1)).item()
    for f in F :
        if not any(np.isnan(f)):
            node_colors.append([0,1,0])
        else :
            e = 0.8/max_na_in_a_row*(torch.isnan(f).sum().item()-1)
            node_colors.append([1-e,e/2,e/2])
        if tuple(node_colors[-1]) not in unique_colors.keys() :
            unique_colors[tuple(node_colors[-1])] = torch.isnan(f).sum().item()
    node_colors = np.array(node_colors)


    degrees = dict(nx.degree(G))
    for i in range(len(G.nodes)):
        G.nodes[i]["name"] = str(i)
        if any(np.isnan(F[i])):
            G.nodes[i]["state"] = "dead"
        else:
            G.nodes[i]["state"] = "alive"
        G.nodes[i]["size"] = (degrees[i]/2) ** 2

    if pos is None:
        # pos = nx.shell_layout(G)
        pos = nx.kamada_kawai_layout(G)
    node_size = np.array(list(nx.get_node_attributes(G, "size").values()))
    node_size[np.isnan(node_size)] = 250.0
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    nx.draw(G, pos, node_size=node_size, node_color=node_colors, ax=ax)
    node_labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'state')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    for k,v in unique_colors.items() :
        if v == 0 :
            ax.scatter([], [], label="alive", color=k)
        else :
            ax.scatter([], [], label=f"missing {v} values", color=k)
    fig.legend()

    if title is not None:
        title = title.split(" ")
        title_tmp = [title[0]]
        for t in title[1:] :
            if len(title_tmp[-1]) < 50 :
                title_tmp[-1] += " " + t
            else :
                title_tmp.append(t)
        title = "\n".join(title_tmp)
        fig.suptitle(title)
    if save:
        print("saved :", save)
        plt.savefig(save)
    if show:
        plt.show()
    else:
        plt.close()
    if pos is None:
        return pos

def draw2(G,
          F: torch.Tensor = None,
          communities_name=None,
          pos=None,
          title=None,
          save=None,
          show=True,
          figsize=(3, 3)):
    """
    Draw graph, color as community, size as weight
    :param G:
    :param F:
    :param communities_name:
    :param pos:
    :param title:
    :param save:
    :param show:
    :param figsize:
    :return:
    """
    node_size = np.array([G.degree[i] for i in range(len(G.nodes))]) ** 1.30
    node_size[node_size < 20] = 20

    # set node color to community
    colors = plt.cm.get_cmap("tab10").colors[:F[:, 0].max().int().item() + 1]
    node_color = [colors[int(i)] for i in F[:, 0]]

    if communities_name:
        assert len(communities_name) == F[:, 0].max().int().item() + 1

    # set node_label to node index
    node_label = {i:i for i in G.nodes}

    if pos is None:
        # pos = nx.shell_layout(G)
        # pos = nx.kamada_kawai_layout(G, weight=None, dim=2)
        # pos = nx.spring_layout(G, k=0.5, iterations=100)
        pos = nx.nx_agraph.graphviz_layout(G, prog="fdp")

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # set figure title
    if title is not None:
        title = title.split(" ")
        title_tmp = [title[0]]
        for t in title[1:]:
            if len(title_tmp[-1]) < 50:
                title_tmp[-1] += " " + t
            else:
                title_tmp.append(t)
        title = "\n".join(title_tmp)
        # fig.suptitle(title)
    nx.draw(G,
            pos=pos,
            # node_size=node_size,#50,
            node_size=50,#node_size,
            node_color=node_color,
            labels=node_label if len(node_label) < 200 else None,#None,
            # labels=None,
            with_labels=True if len(node_label) < 200 else False,#False,
            # with_labels=False,
            font_color="gray",
            font_size=8,
            alpha=0.9,
            width=0.33,
            ax=ax)

    for i in range(len(colors)):
        label = f"{i}" if communities_name is None else f"{communities_name[i]}"
        ax.scatter([], [], label=label, color=colors[i], alpha=0.8)
    # fig.legend()
    plt.tight_layout()
    if save:
        logging.info(f"saved : {save}")
        plt.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    if pos is None:
        return pos
