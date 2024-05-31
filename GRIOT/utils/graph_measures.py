from time import time
import torch_geometric
from tqdm import tqdm
import networkx as nx
import numpy as np
import logging
import torch

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class GraphMeasures():
    def __init__(self, compute_choice, G=None, F=None):
        self.compute_choice = compute_choice
        self.G = G
        if F is None or type(F) == torch.Tensor:
            self.F = F
        elif type(F) == np.ndarray:
            self.F = torch.tensor(F)

    def compute(self, G=None, F=None):
        if G is not None:
            self.G = G
        if F is not None:
            if type(F) == torch.Tensor:
                self.F = F
            elif type(F) == np.ndarray:
                self.F = torch.tensor(F)

        self.measures = dict()

        logging.info("Number of vertices")
        t_start = time()
        self.measures["nb_vertices"] = self.get_nb_vertices()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Number of edges")
        t_start = time()
        self.measures["nb_edges"] = self.get_nb_edges()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Number of features")
        t_start = time()
        self.measures["nb_features"] = self.F.size(1) - 1
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Number of classes")
        t_start = time()
        self.measures["nb_classes"] = len(set(self.F[:, 0].tolist()))
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Edges probability")
        t_start = time()
        self.measures["edges_proba"] = self.get_edges_proba()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("average degree")
        t_start = time()
        self.measures["avg_degree"] = self.get_average_degree()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Number of components")
        t_start = time()
        self.measures["components"] = self.get_components()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Number of communities")
        t_start = time()
        self.measures["nb_community"] = self.get_nb_community()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Communities size")
        t_start = time()
        self.measures["community_size"] = self.get_community_size()
        logging.info(f"DONE. {time() - t_start:.2f}s.")
        if self.compute_choice == "base":  # compute low resources demanding measures
            return self.measures

        logging.info("Diameter")
        t_start = time()
        self.measures["diameter"] = self.get_diameter()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Modularity")
        t_start = time()
        self.measures["modularity"] = self.get_modularity()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Inertia")
        t_start = time()
        self.measures["inertia"] = self.get_inertia()
        logging.info(f"DONE. {time() - t_start:.2f}s.")

        logging.info("Clustering coefficient")
        t_start = time()
        self.measures["clustering_coefficient"] = self.get_clustering_coefficient()
        logging.info(f"DONE. {time() - t_start:.2f}s.")
        if self.compute_choice == "most":  # compute medium resources demanding measures
            return self.measures

        logging.info("Homophily")
        t_start = time()
        self.measures["homophily"] = self.get_homophily()
        logging.info(f"DONE. {time() - t_start:.2f}s.")
        if self.compute_choice == "nearly_all":  # compute high resources demanding measures
            return self.measures

        logging.info("Average shortest pass")
        t_start = time()
        self.measures["avg_shortest_pass"] = self.get_avg_shortest_pass()
        logging.info(f"DONE. {time() - t_start:.2f}s.")
        if self.compute_choice == "all":  # compute high resources demanding measures
            return self.measures

        return self.measures

    def save(self, path=""):
        with open(path, "w") as f:
            for key, value in self.measures.items():
                f.write(str(key) + ": " + str(value) + "\n")

    def show(self):
        for key, value in self.measures.items():
            logging.info(f"{str(key).ljust(25)}:{str(value)}")

    def get_communities_indexes(self):
        """
        Compute the communities indexes of the graph
        :return:
        """
        return [np.where(self.F[:, 0] == x)[0] for x in range(int(max(self.F[:, 0])) + 1)]

    def get_nb_vertices(self):
        """
        Compute the number of vertices in the graph
        :return:
        """
        return len(self.G.nodes())

    def get_nb_edges(self):
        """
        Compute the number of edges in the graph
        :return:
        """
        return len(self.G.edges)

    def get_edges_proba(self):
        """
        Compute the probability of edges of the graph withing communities and between communities
        :return:
        """
        communities = self.get_communities_indexes()
        p_communities = [len(community) for community in communities]
        max_edges_within = 1
        for i in range(len(p_communities)):
            max_edges_within += p_communities[i] * (p_communities[i] - 1) / 2
        max_edges_between = 1
        for i in range(len(p_communities)):
            for j in range(i + 1, len(p_communities)):
                max_edges_between += p_communities[i] * p_communities[j]
        nb_edges_within = [0] * len(communities)
        nb_edges_between = 0
        for i, j in self.G.edges:
            c1, c2 = int(self.F[i, 0].item()), int(self.F[j, 0].item())
            if c1 == c2:
                nb_edges_within[c1] += 1
            else:
                nb_edges_between += 1
        p_edges_within = sum(nb_edges_within) / max_edges_within
        p_edges_between = nb_edges_between / max_edges_between
        return {"within": p_edges_within, "between": p_edges_between}

    def get_average_degree(self):
        """
        Compute the average degree of the graph
        :return:
        """
        return np.array(nx.degree(self.G))[:, 1].mean()

    def get_components(self):
        """
        Compute the number of components in the graph
        :return:
        """
        return len(list(nx.connected_components(self.G)))

    def get_nb_community(self):
        """
        Compute the number of communities in the graph
        :return:
        """
        return len(set(self.F[:, 0].tolist()))

    def get_community_size(self):
        """
        Compute the community size of the graph
        :return:
        """
        return [sum(self.F[:, 0] == x).tolist() for x in range(int(max(self.F[:, 0])) + 1)]

    def get_diameter(self):
        """
        Compute the diameter of the graph
        :return:
        """
        try:
            return nx.diameter(self.G)
        except nx.NetworkXError:
            return np.inf

    def get_modularity(self):
        """
        Compute the modularity of the graph
        :return:
        """
        communities = self.get_communities_indexes()
        return nx.algorithms.community.modularity(self.G, communities)

    def get_inertia(self):
        """
        Compute the inertia of the graph
        :return:
        """
        k = self.get_community_size()
        p_C_k = torch.tensor(k) / sum(k)

        # total inertia
        center_of_gravity_total = torch.mean(self.F[:, 1:], dim=0)
        inertia_total = torch.square(torch.linalg.norm(self.F[:, 1:]
                                                       - center_of_gravity_total, dim=1)).sum()

        # within inertia
        center_of_gravity_within = torch.tensor([torch.mean(self.F[:, 1:][self.F[:, 0] == i], dim=0).tolist()
                                                 for i in range(int(max(self.F[:, 0])) + 1)])
        inertia_within = torch.tensor(
            [torch.square(torch.linalg.norm(self.F[:, 1:][self.F[:, 0] == i]
                                            - center_of_gravity_within[i], dim=1)).sum()
             for i in range(int(max(self.F[:, 0])) + 1)]
        )
        inertia_within_total = sum(inertia_within) / inertia_total

        # between inertia
        inertia_between = sum([p_C_k[i] * torch.square(torch.linalg.norm(center_of_gravity_within[i]
                                                                         - center_of_gravity_total))
                               for i in range(len(center_of_gravity_within))]) / inertia_total
        # LaTeX - corresponding math formula of the inertia between :
        # $$ inertia\_between = \sum_{i=1}^{k} p(C_k) \times \left\| \bar{F}_{C_k} - \bar{F}_{total} \right\|^2 $$
        # where p(C_k) = \frac{|C_k|}{\sum_{i=1}^{k} |C_k|} is the probability of the community C_k

        def p():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 10))
            colors = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])[self.F[:, 0].tolist()]
            plt.scatter(self.F[:, 1], self.F[:, 2], c=colors)
            plt.scatter(center_of_gravity_total[0], center_of_gravity_total[1],
                        s=150,
                        marker="o",
                        c="black")
            plt.scatter(center_of_gravity_within[:, 0], center_of_gravity_within[:, 1],
                        s=150,
                        marker="x",
                        c=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
            draw_circle = plt.Circle((center_of_gravity_total[0], center_of_gravity_total[1]), 1, color="black",
                                     fill=False,
                                     label="total inertia")
            draw_circle_w = [
                plt.Circle((center_of_gravity_within[i, 0], center_of_gravity_within[i, 1]), inertia_within[i],
                           color=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]][i], fill=False,
                           label=f"community {i} inertia") for i in range(4)
            ]
            ax.add_artist(draw_circle)
            for d in draw_circle_w:
                ax.add_artist(d)
            plt.legend()
            plt.show()

        # p()

        inertia_within_total = inertia_within_total.item()
        inertia_between = inertia_between.item()

        return {"within": inertia_within_total,
                "between": inertia_between}

    def get_homophily(self):
        """
        Compute the homophily of the graph
        :return:
        """

        sigma = self.F[:, 1:].std(dim=0)
        k = self.get_nb_community()
        bound = 2 * sigma / k
        n = len(self.G.nodes)

        try:
            # Compute similarity matrix using broadcasting
            F_expanded = self.F[:, 1:].unsqueeze(1)  # Shape: (n, 1, num_attributes)
            similarity_matrix_ = torch.min(torch.abs(F_expanded - F_expanded.transpose(0, 1)) - bound, dim=2).values
            # Fill the upper triangular part of the matrix
            tri_mask = torch.triu(torch.ones_like(similarity_matrix_), diagonal=1)
            similarity_matrix_ = similarity_matrix_ * tri_mask
            # make the matrix symmetric
            similarity_matrix_ = similarity_matrix_ + similarity_matrix_.transpose(0, 1)
            # Fill the diagonal with -bound
            similarity_matrix_ = similarity_matrix_ + torch.diag(torch.ones(n)) * min(-bound)
        except RuntimeError as e:
            logging.warning('RuntimeError handled.')
            print(e)
            # compute similarity matrix
            # we considered that two vertices v and v′ are similar iff. ∃ A ∈ \mathcal{A} s.t. |vA − v′A| < 2σ_A/k
            similarity_matrix_ = torch.zeros((n, n))
            for i in tqdm(range(n), desc="n", leave=False):
                for j in tqdm(range(i, n), desc="i", leave=False):
                    if j > i:
                        similarity_matrix_[i, j] = similarity_matrix_[j, i] \
                            = min(torch.abs(self.F[i, 1:] - self.F[j, 1:]) - bound)
                    elif j == i:
                        similarity_matrix_[i, j] = min(torch.abs(self.F[i, 1:] - self.F[j, 1:]) - bound)

        similarity_matrix_ = similarity_matrix_ < 0
        homophily = (sum([similarity_matrix_[i, j] for i, j in self.G.edges]) / len(self.G.edges)).item()
        homophily_expected = (similarity_matrix_.sum() / (n * (n - 1))).item()
        homophily_ratio = homophily / homophily_expected

        # # HOMOPHILY USING PYTORCH AND COMMUNITIES (instead of attributes)
        # # get the edge_index of the graph
        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        homophily_community_wise = torch_geometric.utils.homophily(edge_index, self.F[:, 0].long())

        return {
            "expected": homophily_expected,
            "actual": homophily,
            "ratio": homophily_ratio,
            "community-wise": homophily_community_wise
        }

    def get_clustering_coefficient(self):
        """
        Compute the clustering coefficient of the graph
        The network average clustering coefficient is a measure of the clustering tendency of the network.
        This observed value can be compared with the expected value computed on a random graph having the
        same vertex set: An observed value higher than the expected value confirms the community structure.
        :return:
        """
        # Average clustering coefficient : is given as an indication of the transitivity of connections in the network
        avg_clust_coef = nx.algorithms.approximation.average_clustering(self.G, seed=42)
        # Random clustering coefficient : gives the clustering coefficient in a Erdös–Renyi random graph having the
        # same number of vertices and edges
        n = len(self.G.nodes)
        G_random = nx.erdos_renyi_graph(n, len(self.G.edges) / (n * (n - 1)) * 2)
        random_clust_coef = nx.algorithms.approximation.average_clustering(G_random, seed=42)
        return {"average": avg_clust_coef, "random": random_clust_coef}

    def get_avg_shortest_pass(self):
        """
        Compute the average shortest path length of the graph
        :return:
        """
        return nx.average_shortest_path_length(self.G)

    def get_dyadicity(self):
        """
        Compute the dyadicity of the graph

        Dyadicity measures the number of same label edges compared to what is expected in a random configuration of
        the network, in other words, if the labels were randomly distributed. Say ng represents the number of green
        nodes. The expected number of green label edges then becomes: the number of combinations of 2 out of ng times
        p, which, as you recall, is the probability of two nodes being connected.

        If the dyadicity is greater than 1 we say that the network is dyadic because nodes with the same label are
        more connected amongst themselves.

        With two classes, we note
        $$ dyadicity = {n\choose 2}p = \frac{n(n-1)}{2}p $$
        where n is the number of nodes and p is the probability of two nodes being connected.

        We can extend this to k classes by summing over all classes.
        $$ dyadicity = \sum_{i=1}^{k} {n_i\choose 2}p_i = \sum_{i=1}^{k} \frac{n_i(n_i-1)}{2}p_i $$
        where n_i is the number of nodes in class i and p_i is the probability of two nodes being connected within
        class i.
        :return:
        """

        dyadicity = 0
        k = self.get_nb_community()
        for i in range(k):
            # n_i is the number of nodes in class i
            n_i = sum(self.F[:, 0] == i)
            # p_i is the probability of two nodes being connected within class i.
            p_i = 2 * sum([1 for i, j in self.G.edges if self.F[i, 0] == self.F[j, 0]]) / (n_i*(n_i-1))
            dyadicity += (n_i * (n_i - 1)) / 2 * p_i
        return dyadicity

