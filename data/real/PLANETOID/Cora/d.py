import logging
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns


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


def load_cora_graph(path=".",
                    draw_=True,
                    main_component_=True,
                    checks_=False):
    print("0")
    data_dir = os.path.expanduser(path)

    print("get edgelist")
    # get edgelist
    edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])

    print("get features")
    # get features
    column_names = ["w_{}".format(ii) for ii in range(1433)] + ["subject"]
    node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None, names=column_names)
    node_data = node_data.reset_index().rename(columns={"index": "node_id"})

    if checks_:
        print("check1")
        check(edgelist, node_data)

    print("attribute an integer to each community")
    # attribute an integer to each community
    community_dict = {label: i for i, label in enumerate(node_data["subject"].unique())}
    node_data['community'] = node_data["subject"].apply(lambda x: community_dict[x]).to_numpy()
    # drop node_data["subject"]
    node_data = node_data.drop(columns=["subject"])

    if main_component_:
        print("keep main component")
        # make graph from edgelist numpy array
        G = nx.from_pandas_edgelist(edgelist)
        # remove nodes which are not connected to main component
        graph_components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(graph_components[0])
        # remove nodes from node_data that are not in G
        node_data = node_data[node_data["node_id"].isin(list(G.nodes()))]
        del G

    print("filter edge list")
    # keep in edgelist only rows such that edgelist["source"] is in node_data["node_id"]
    edgelist = edgelist[edgelist["source"].isin(node_data["node_id"])]

    print("assign index to communities")
    # make a dictionary to match every node in node_data["node_id"] with a minimal unique integer id
    node_id = {label: i for i, label in enumerate(node_data["node_id"].unique())}
    # apply the dictionary to node_data["node_id"]
    node_data["node_id"] = node_data["node_id"].apply(lambda x: node_id[x])
    # apply the dictionary to edgelist["source"] and edgelist["target"]
    edgelist["source"] = edgelist["source"].apply(lambda x: node_id[x])
    edgelist["target"] = edgelist["target"].apply(lambda x: node_id[x])

    print("simplify nodes id")
    # sort node_data according to node_id
    node_data = node_data.sort_values(by="node_id")
    # sort edgelist according to target
    edgelist = edgelist.sort_values(by="target")
    # make "community" the first column of node_data and drop "node_id"
    node_data = node_data[["community"] + [col for col in node_data.columns if col not in ["node_id", "community"]]]

    if checks_:
        print("check2")
        check(edgelist, node_data)

    print("make attribute matrix and edges matrix")
    # make node_data a numpy array and call it F (feature matrix)
    F = node_data.to_numpy()
    del node_data
    # make F dtype float
    F = np.array(F, dtype=float)
    # make edgelist a numpy array
    edgelist = edgelist.to_numpy()

    if checks_:
        print("check3")
        check(edgelist, F)

    print("make graph G from edge list")
    G = nx.from_edgelist(edgelist)
    del edgelist

    if draw_:
        print("draw START")
        print("assign to each node its community attribute")
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
        print("get position using graphviz")
        prog="neato"
        pos = nx.nx_agraph.graphviz_layout(G, prog=prog)
        print("draw using nx.draw")
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
        print("save figure")
        plt.savefig(f"cora_graph_{prog}.png")
        # plt.show()
        print("draw END")


    return G, F

load_cora_graph()