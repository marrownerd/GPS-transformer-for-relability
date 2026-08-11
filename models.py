import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, LayerNorm, Sigmoid, Dropout
from torch_geometric.nn import GINEConv, TransformerConv, global_mean_pool, global_max_pool
from config import HIDDEN_DIM_BASE, HIDDEN_DIM_SOTA

class BaselineGINEModel(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=5, hidden_dim=128):
        super().__init__()
        self.node_emb = Linear(node_in_dim, hidden_dim)
        self.edge_emb = Linear(edge_in_dim, hidden_dim)
        self.convs = nn.ModuleList([GINEConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)), aggr='mean') for _ in range(4)])
        self.head_mu = Sequential(Linear(hidden_dim, 64), ReLU(), Linear(64, 1), Sigmoid())
        self.head_var = Sequential(Linear(hidden_dim, 64), ReLU(), Linear(64, 1))

    def forward(self, x, edge_index, edge_attr, batch, bounds):
        h = F.relu(self.node_emb(x))
        e = F.relu(self.edge_emb(edge_attr))
        for conv in self.convs: 
            h = F.relu(conv(h, edge_index, edge_attr=e))
        g = global_mean_pool(h, batch)
        mu = bounds[:, 0:1] + self.head_mu(g) * (bounds[:, 1:2] - bounds[:, 0:1])
        var = torch.clamp(F.softplus(self.head_var(g)) + 1e-6, max=1.0)
        return mu, var

class GPSLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.local_mpnn = GINEConv(Sequential(Linear(hidden_dim, hidden_dim * 2), LeakyReLU(0.1), Linear(hidden_dim * 2, hidden_dim)))
        self.global_attn = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.ffn = Sequential(Linear(hidden_dim, hidden_dim * 2), LeakyReLU(0.1), Dropout(0.1), Linear(hidden_dim * 2, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        h = x + F.leaky_relu(self.local_mpnn(x, edge_index, edge_attr)) + F.leaky_relu(self.global_attn(x, edge_index, edge_attr))
        h = self.norm1(h)
        return self.norm2(h + self.ffn(h))

class PINNReliabilityModel(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=5, hidden_dim=128):
        super().__init__()
        self.node_emb = Linear(node_in_dim, hidden_dim)
        self.edge_emb = Linear(edge_in_dim, hidden_dim)
        self.v_node_embed = nn.Parameter(torch.randn(1, hidden_dim))
        self.num_layers = 4
        self.layers = nn.ModuleList([GPSLayer(hidden_dim) for _ in range(self.num_layers)])
        jk_dim = hidden_dim * (self.num_layers + 1)
        self.pool_comp = Sequential(Linear(jk_dim * 2, hidden_dim), LayerNorm(hidden_dim), LeakyReLU(0.1))
        self.tf_gate = nn.Parameter(torch.tensor([0.5]))
        self.head_mu = Sequential(Linear(hidden_dim, 64), LeakyReLU(0.1), Dropout(0.1), Linear(64, 1))
        self.head_var = Sequential(Linear(hidden_dim, 64), LeakyReLU(0.1), Dropout(0.1), Linear(64, 1))

    def forward(self, x, edge_index, edge_attr, batch, bounds):
        h = F.leaky_relu(self.node_emb(x))
        e = F.leaky_relu(self.edge_emb(edge_attr))
        v_node = self.v_node_embed.expand(batch.max() + 1, -1)
        xs = [h]
        for layer in self.layers:
            h = layer(h + v_node[batch], edge_index, e)
            v_node = v_node + global_max_pool(h, batch)
            xs.append(h)

        h_all = torch.cat(xs, dim=-1)
        pool = torch.cat([global_mean_pool(h_all, batch), global_max_pool(h_all, batch)], dim=-1)
        embeds = global_mean_pool(h, batch) + self.tf_gate * self.pool_comp(pool)

        raw_mu = self.head_mu(embeds)
        scale = (bounds[:, 1:2] - bounds[:, 0:1]) / 2.0
        center = (bounds[:, 1:2] + bounds[:, 0:1]) / 2.0
        mu = center + scale * torch.tanh(raw_mu)
        var = torch.clamp(F.softplus(self.head_var(embeds)) + 1e-6, max=1.0)
        return mu, var

class GatedReliabilityModel(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=5):
        super().__init__()
        self.base_expert = BaselineGINEModel(node_in_dim, edge_in_dim, HIDDEN_DIM_BASE)
        self.adv_expert = PINNReliabilityModel(node_in_dim, edge_in_dim, HIDDEN_DIM_SOTA)
        self.gate = Sequential(Linear(node_in_dim, 16), LeakyReLU(0.1), Linear(16, 1), Sigmoid())

    def forward(self, x, edge_index, edge_attr, batch, bounds):
        mu_b, var_b = self.base_expert(x, edge_index, edge_attr, batch, bounds)
        mu_a, var_a = self.adv_expert(x, edge_index, edge_attr, batch, bounds)
        alpha = self.gate(global_mean_pool(x, batch))
        return alpha * mu_b + (1.0 - alpha) * mu_a, alpha * var_b + (1.0 - alpha) * var_a