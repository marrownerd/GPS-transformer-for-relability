import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_networkx
from utils import monte_carlo_atr

class ZincReliabilityDataset(InMemoryDataset):
    def __init__(self, root="data", version="zinc_atr_v1", num_graphs=3000):
        self.version = version
        self.num_graphs = num_graphs
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self): 
        return [f"data_{self.version}.pt"]

    def process(self):
        print("[INFO] Скачивание и обработка оригинального бенчмарка ZINC (V1)...")
        raw_dataset = ZINC(root=self.root, subset=True, split='train')
        data_list = []

        limit = min(self.num_graphs, len(raw_dataset))
        for i in tqdm(range(limit), desc="Расчет Монте-Карло для ZINC"):
            pyg_graph = raw_dataset[i]
            G = to_networkx(pyg_graph, to_undirected=True)

            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)
            n = G.number_of_nodes()
            if n < 5: 
                continue

            ebc = nx.edge_betweenness_centrality(G)
            bridge_edges = set(map(tuple, map(sorted, nx.bridges(G))))
            ebc_vals = list(ebc.values())
            ebc_90 = np.percentile(ebc_vals, 90) if ebc_vals else 0

            global_health = random.choice(["perfect", "normal", "critical"])

            for u, v in G.edges():
                key = tuple(sorted((u, v)))
                is_b = key in bridge_edges
                is_hub = ebc[(u, v)] >= ebc_90

                if global_health == "perfect": 
                    base_p = random.uniform(0.85, 0.99)
                elif global_health == "normal": 
                    base_p = random.uniform(0.60, 0.90)
                else: 
                    base_p = random.uniform(0.40, 0.70)

                penalty = 0.0
                if is_b: 
                    penalty += random.uniform(0.15, 0.35)
                if is_hub: 
                    penalty += random.uniform(0.05, 0.20)

                final_p = np.clip(base_p - penalty, 0.1, 0.99)
                G[u][v]["p"] = final_p
                G[u][v]["is_b"] = 1.0 if is_b else 0.0
                G[u][v]["ebc"] = ebc[(u, v)]

            mu, var = monte_carlo_atr(G, num_sims=1000)

            bc = nx.betweenness_centrality(G)
            cc = nx.clustering(G)
            deg = dict(G.degree())
            x = torch.tensor([[bc[i], cc[i], deg[i] / n] for i in range(n)], dtype=torch.float)
            edge_idx, edge_attr = [], []
            for u, v, d in G.edges(data=True):
                feats = [d["p"], d["is_b"], d["ebc"], d["p"] * d["ebc"], 1.0 - d["p"]]
                edge_idx.extend([[u, v], [v, u]])
                edge_attr.extend([feats, feats])

            data_list.append(Data(x=x, edge_index=torch.tensor(edge_idx).t().long(),
                                  edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                                  y_mu=torch.tensor([[mu]]), y_var=torch.tensor([[var]]),
                                  bounds=torch.tensor([[max(0.0, mu - 0.08), min(1.0, mu + 0.08)]])))
        torch.save(self.collate(data_list), self.processed_paths[0])