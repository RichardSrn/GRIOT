import networkx as nx
import numpy as np


def enforce_graph_size(g, prob, rng, verbatim=True):
    """
    Function to enforce the size of a graph
    :param g: networkx graph
    :param prob: [prob intra, prob extra]
    :param rng: numpy random state
    :param verbatim: print detailed progress ("DEBUG" for more detailed progress)
    :return:
    """

    print("enforcing graph size")
    sizes = [len(g.graph['partition'][i]) for i in range(len(g.graph['partition']))]
    num_groups = len(sizes)
    n = np.sum(sizes)

    # make sure each partition is connected
    for i in range(num_groups):
        g_sub = g.subgraph(list(g.graph['partition'][i]))
        # check if g_sub is connected
        if not nx.is_connected(g_sub):
            # if not connected, add edges until connected
            nodes = list(g_sub.nodes)
            while not nx.is_connected(g_sub):
                # get two random nodes
                node1, node2 = rng.choice(nodes, 2, replace=False)
                if not g_sub.has_edge(node1, node2):
                    g.add_edge(node1, node2)
    num_edges = 0
    num_edges_max = 0
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            # check if there is an edge between the two partitions
            if not g.has_edge(list(g.graph['partition'][i])[0], list(g.graph['partition'][j])[0]):
                # if not connected, add edges until connected
                nodes1 = list(g.graph['partition'][i])
                nodes2 = list(g.graph['partition'][j])
                # count the number of edges between the two partitions
                num_edges = 0
                num_edges_max = 0
                for node1 in nodes1:
                    for node2 in nodes2:
                        # skip if the two nodes are the same
                        if node1 == node2:
                            continue
                        if g.has_edge(node1, node2):
                            num_edges += 1
                        num_edges_max += 1
                # add edges until the proportion of edges between the two partitions is at least prob[1]
                # allow 25% margin
                while num_edges / num_edges_max < prob[1] * 0.75 or num_edges == 0:
                    if verbatim :
                        print("adding edge between partitions ",
                              i, " and ", j, " : ", num_edges / num_edges_max, " < ", prob[1] * 0.9)
                    # get two random nodes
                    node1 = rng.choice(nodes1)
                    # node2 = rng.choice(nodes2)
                    # pick node2 among nodes that are not connected to node1 within nodes2
                    node2 = rng.choice([node for node in nodes2 if not g.has_edge(node1, node)])
                    if not g.has_edge(node1, node2):
                        g.add_edge(node1, node2)
                        num_edges += 1
                # remove edges until the proportion of edges between the two partitions is at most prob[1]
                # allow 25% margin
                while num_edges / num_edges_max > prob[1] * 1.25:
                    if verbatim :
                        print("removing edge between partitions ",
                              i, " and ", j, " : ", num_edges / num_edges_max, " > ", prob[1] * 1.1)
                    # get two random nodes
                    node1 = rng.choice(nodes1)
                    # pick node2 among nodes that are connected to node1 within nodes2
                    node2 = rng.choice([node for node in nodes2 if g.has_edge(node1, node)])
                    if g.has_edge(node1, node2):
                        g.remove_edge(node1, node2)
                        num_edges -= 1

    # set num_edges the number of edges in the graph and num_edges_max the maximum number of edges
    num_edges = len(g.edges) - num_edges
    num_edges_max = n * (n - 1) / 2 - num_edges_max
    # if the probability is too low, add edges until the probability is at least prob[0] (allow 10% margin)
    while num_edges / num_edges_max < prob[0] * 0.1:
        if verbatim :
            # print a progress of desired probability vs actual probability
            print(f"adding edge : {num_edges / num_edges_max:.6f}".ljust(25), " < ", f"{prob[0] * 0.5:.6f}", end="\r")
        # pick a random node, node1,
        node1 = rng.choice(list(g.nodes))
        # get the partition of node1
        partition1 = [i for i in range(num_groups) if node1 in g.graph['partition'][i]][0]
        # pick node2 among nodes that are not connected to node1, from the same partition
        node2 = rng.choice([node for node in g.graph['partition'][partition1] if not g.has_edge(node1, node)])

        if not g.has_edge(node1, node2):
            g.add_edge(node1, node2)
            num_edges += 1
    if verbatim :
        print()
    # if the probability is too high, remove edges until the probability is at most prob[0] (allow 10% margin)
    stop = False
    blacklist_nodes = []
    list_nodes = []
    objective_remove_edges = int(np.ceil(num_edges - prob[0] * 1.1 * num_edges_max))
    # copy num_edges
    num_edges_when_started = int(num_edges)
    while num_edges / num_edges_max > prob[0] * 1.1 and not stop:
        if verbatim :
            # print a progress of desired probability vs actual probability
            print(f"removing edge : {num_edges / num_edges_max:.6f}".ljust(25), " > ", f"{prob[0] * 1.1:.6f}",
                  f" ; len(list_nodes) : {len(list_nodes)}".rjust(30),
                  f" ; len(blacklist) : {len(blacklist_nodes)}".rjust(30),
                  f" ; removed edges : {num_edges_when_started - num_edges}".rjust(30),
                  f" ; objective : {objective_remove_edges}",
                  f" ; edges left : {num_edges}".ljust(25),
                  end="\r")
        # make an empty list of nodes
        list_nodes = []
        # cycle through all partitions,
        # get all nodes that have more than 1 edge within each partition
        # add them to list_nodes
        for partition in range(num_groups):
            # list_nodes += [node for node in g.graph['partition'][partition] if len(list(g.neighbors(node))) > 1]
            # similar to above line, but count only the neighbors within the current partition
            list_nodes_tmp = [node for node in g.graph['partition'][partition] if
                              len([nd for nd in g.neighbors(node) if nd in g.graph['partition'][partition]]) > 1]
            # remove from list_nodes_tmp, all nodes with less than 1 edge within the current partition
            list_nodes_tmp = [node for node in list_nodes_tmp if
                              len([nd for nd in g.neighbors(node) if nd in g.graph['partition'][partition]]) > 1]
            list_nodes += list_nodes_tmp
        # remove all blacklisted nodes from list_nodes
        list_nodes = [node for node in list_nodes if node not in blacklist_nodes]

        assert len(list_nodes) > 0, "ERROR : list_nodes is empty"

        # pick a random node, node1, from list_nodes
        node1 = rng.choice(list_nodes)
        # get the partition of node1
        partition1 = [i for i in range(num_groups) if node1 in g.graph['partition'][i]][0]
        # list all nodes connected to node1 with more than 1 edge within the same partition
        list_nodes1 = [node for node in g.neighbors(node1) if node in g.graph['partition'][partition1]]
        # remove from list_nodes1, all nodes with less than 1 edge within the same partition
        list_nodes1 = [node for node in list_nodes1 if
                       len([nd for nd in g.neighbors(node) if nd in g.graph['partition'][partition1]]) > 1]

        if len(list_nodes1) == 0:
            # add node1 to blacklist_nodes
            blacklist_nodes.append(node1)
        else:
            # randomize list_node1
            rng.shuffle(list_nodes1)
            # pick node2
            i_node2 = 0
            node2 = list_nodes1[i_node2]
            # check if node1 and node2 are connected without link (node1, node2)
            g_backup = g.copy()
            assert g.has_edge(node1, node2), "ERROR : edge does not exist"
            g.remove_edge(node1, node2)
            stop_ = False
            while not nx.is_connected(g) and not stop_:
                # if not connected, restore g and continue
                g = g_backup.copy()
                i_node2 += 1
                if i_node2 == len(list_nodes1):
                    stop_ = True
                    break
                node2 = list_nodes1[i_node2]
                assert g.has_edge(node1, node2), "ERROR : edge does not exist"
                g.remove_edge(node1, node2)
            # if stop_ it means that node1 is not linked to any suitable node2
            if stop_:
                # add node1 to blacklist_nodes
                blacklist_nodes.append(node1)
                g = g_backup.copy()
                continue
            else:
                num_edges -= 1

                if verbatim == "DEBUG":
                    print("\nremoving edge in partition ", partition1, " : ", num_edges / num_edges_max, " > ",
                          prob[0] * 1.1)
                    # print the nodes chosen, their partition and the number of link they have
                    print("node1 : ", node1, " partition : ", partition1, " num_edges : ", len(list(g.neighbors(node1))))
                    # print the number of edges between node1 and each partitions
                    for partition in range(num_groups):
                        print("num_edges between node1 and partition ", partition, " : ",
                              len([nd for nd in g.neighbors(node1) if nd in g.graph['partition'][partition]]))
                    print("node2 : ", node2, " partition : ", partition1, " num_edges : ", len(list(g.neighbors(node2))))
                    # print the number of edges between node2 and each partitions
                    for partition in range(num_groups):
                        print("num_edges between node2 and partition ", partition, " : ",
                              len([nd for nd in g.neighbors(node2) if nd in g.graph['partition'][partition]]))

        try:
            assert nx.is_connected(g), "ERROR : graph is disconnected"
        except AssertionError:
            # draw the partition partition1 of g and g_backup in two subplots with a rectangular figure
            # draw node1 and node2 in red
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # make subplot of g for partition partition1
            g_sub = g.subgraph(list(g.graph['partition'][partition1]))
            pos = nx.spring_layout(g_sub)
            nx.draw(g_sub, pos, node_size=20, node_color="blue", ax=ax1)
            nx.draw_networkx_nodes(g_sub, pos, nodelist=[node1, node2], node_color="red", ax=ax1)
            # make subplot of g_backup for partition partition1
            g_sub = g_backup.subgraph(list(g.graph['partition'][partition1]))
            nx.draw(g_sub, pos, node_size=20, node_color="blue", ax=ax2)
            nx.draw_networkx_nodes(g_sub, pos, nodelist=[node1, node2], node_color="red", ax=ax2)
            # show the plot
            plt.show()
        finally:
            assert nx.is_connected(g), "ERROR : graph is disconnected"
    if verbatim:
        print()
    return g