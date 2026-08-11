import numpy as np
import networkx as nx
import torch

def gaussian_kl_loss(mu_pred, var_pred, mu_true, var_true):
    var_true = var_true.clamp(min=1e-6)
    var_pred = var_pred.clamp(min=1e-6)
    kl = torch.log(var_pred.sqrt() / var_true.sqrt()) + (var_true + (mu_true - mu_pred)**2) / (2 * var_pred) - 0.5
    return kl.mean()

def monte_carlo_atr(G, num_sims=500):
    edges = list(G.edges(data='p'))
    success_count = 0
    for _ in range(num_sims):
        rands = np.random.rand(len(edges))
        survived = [(u, v) for i, (u, v, p) in enumerate(edges) if rands[i] <= p]
        G_sim = nx.Graph()
        G_sim.add_nodes_from(G.nodes())
        G_sim.add_edges_from(survived)
        if nx.is_connected(G_sim): 
            success_count += 1
    mu = success_count / num_sims
    return mu, max((mu * (1 - mu)) / num_sims, 1e-6)