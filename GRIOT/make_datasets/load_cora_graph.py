import logging
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def check(edgelist, node_data):
    try:
        # check if edgelist if a pandas dataframe
        if isinstance(edgelist, pd.DataFrame):
            # check if every node in edgelist["source"] is in node_data["node_id"]
            assert set(edgelist["source"].unique()).issubset(set(node_data["node_id"].unique()))
            # check if every node in edgelist["target"] is in node_data["node_id"]
            assert set(edgelist["target"].unique()).issubset(set(node_data["node_id"].unique()))
            # check if the sets of nodes are the same in edgelist and node_data
            assert (set(edgelist["target"].unique()) | set(edgelist["source"].unique()) == set(
                node_data["node_id"].unique()))
        # check if edgelist is a numpy ndarray
        elif isinstance(edgelist, np.ndarray):
            # check if every node in edgelist[:,0] is in node_data["node_id"]
            assert set(edgelist[:, 0]).issubset(set(np.arange(len(node_data))))
            # check if every node in edgelist[:,1] is in node_data["node_id"]
            assert set(edgelist[:, 1]).issubset(set(np.arange(len(node_data))))
    except AssertionError:
        logging.error("ERROR : one of the following occured :"
                      "- some nodes in edgelist are not in node_data"
                      "- some nodes in edgelist are not in node_data"
                      "- the sets of nodes are not the same in edgelist and node_data")
        exit()

    return True


def load_cora_graph(path="../data/graph_real/CORA",
                    directed=False,
                    draw_=False,
                    main_component_=True,
                    checks_=True,
                    return_community_dict=True):
    data_dir = os.path.expanduser(path)

    # get edgelist
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])

    # get features
    column_names = ["w_{}".format(ii) for ii in range(1433)] + ["subject"]
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
    node_data = node_data.reset_index().rename(columns={"index": "node_id"})

    if checks_:
        check(edgelist, node_data)

    # attribute an integer to each community
    community_dict = {label: i for i, label in enumerate(node_data["subject"].unique())}
    node_data['community'] = node_data["subject"].apply(lambda x: community_dict[x]).to_numpy()

    # drop node_data["subject"]
    node_data = node_data.drop(columns=["subject"])

    # keep in edgelist only rows such that edgelist["source"] is in node_data["node_id"]
    # edgelist = edgelist[edgelist["source"].isin(node_data["node_id"])]

    # # sort node_data according to node_id
    # node_data = node_data.sort_values(by="node_id")
    #
    # # make a dictionary to match every node in node_data["node_id"] with a minimal unique integer id
    # node_id = {label: i for i, label in enumerate(node_data["node_id"])}
    # # apply the dictionary to node_data["node_id"]
    # node_data["node_id"] = node_data["node_id"].apply(lambda x: node_id[x])
    # # apply the dictionary to edgelist["source"] and edgelist["target"]
    # edgelist["source"] = edgelist["source"].apply(lambda x: node_id[x])
    # edgelist["target"] = edgelist["target"].apply(lambda x: node_id[x])
    #
    # # sort edgelist according to target
    # edgelist = edgelist.sort_values(by="source")

    # make "community" the first column of node_data and drop "node_id"
    node_data = node_data[["community", "node_id"] + [col for col in node_data.columns if col not in ["node_id", "community"]]]

    # if checks_:
    #     check(edgelist, node_data)

    # # drop "node_id" from node_data
    # node_data = node_data.drop(columns=["node_id"])

    # make node_data a numpy array and call it F (feature matrix)
    F = node_data.to_numpy()
    # make F tensor
    F = torch.from_numpy(F).double()
    # make edgelist a numpy array
    edgelist = edgelist.to_numpy()

    # if checks_:
    #     check(edgelist, F)

    # # if directed is True, load G from edge list, as a directed graph
    # if directed:
    #     G = nx.from_edgelist(edgelist, create_using=nx.DiGraph())
    # else:
    #     G = nx.from_edgelist(edgelist)
    G = nx.from_edgelist(edgelist)


    # cycle through all nodes (from Gnx) and add their community (from node_data) as attribute
    for node in G.nodes():
        G.nodes[node]["features"] = F[F[:,1]==node]

    if main_component_:
        # make graph from edgelist numpy array
        # G = nx.from_pandas_edgelist(edgelist)
        # remove nodes which are not connected to main component
        graph_components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(graph_components[0])
        # remove nodes from node_data that are not in G
        # node_data = node_data[node_data["node_id"].isin(list(G.nodes()))]
        # del G

    # rename G's nodes to canonical integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)

    F = np.zeros((len(G.nodes()), F.shape[1]))
    for node in G.nodes():
        F[node] = G.nodes[node]["features"]

    # delete the 2nd column of F (node_id)
    F = np.delete(F, 1, axis=1)

    # delete G.nodes["features"]
    for node in G.nodes():
        del G.nodes[node]["features"]

    # import matplotlib.pyplot as plt
    # import json
    # # pos = nx.spring_layout(G, k=0.15, iterations=200, seed=42)
    # # # save dictionary of positions pos as pos.json by making is a pandas dataframe
    # # pd.DataFrame(pos).T.reset_index().to_csv('pos_5.csv', index=False)
    # # load dictionary of positions pos from pos.json
    # pos = pd.read_csv('pos_5.csv').set_index('index').T.to_dict('list')
    # plt.figure(figsize=(20, 20))
    # nx.draw(G,
    #         pos=pos,
    #         alpha=0.75,
    #         node_size=[75 + v ** 1.1 for v in dict(G.degree()).values()],
    #         node_color=F[:,0],  # [G.nodes[node]["features"][0] for node in G.nodes()],
    #         with_labels=False,
    #         width=0.25)
    # plt.show()
    # exit()

    if draw_:
        # cycle through all nodes (from Gnx) and add their community (from node_data) as attribute
        for node in G.nodes():
            G.nodes[node]["community"] = F[node, 0]

        # plot Gnx using nx.draw
        # set nodes color to their community Gnx.nodes[node]["community"]
        # set nodes size to their degree
        plt.figure(figsize=(35, 20))
        plt.title("Cora graph, only main component ({} nodes),"
                  " color=community, size=degree".format(len(G.nodes)))
        # improve the layout
        # pos = nx.spring_layout(Gnx, k=0.15, iterations=20)
        pos = nx.nx_agraph.graphviz_layout(G, prog="fdp")
        nx.draw(G,
                pos=pos,
                node_color=[G.nodes[node]["community"] for node in G.nodes()],
                node_size=[G.degree(node) ** 1.05 * 30 for node in G.nodes()],
                with_labels=False,
                alpha=0.80,
                width=0.33,
                cmap=plt.cm.tab20)
        # make a legend for the community using the dictionary community_dict
        plt.legend(
            handles=[plt.scatter([], [], color=plt.cm.tab20(i), label=label) for label, i in community_dict.items()])
        plt.tight_layout()
        plt.savefig(f"cora_graph.png")
        # plt.show()

    if return_community_dict:
        return G, F, community_dict
    else:
        return G, F
