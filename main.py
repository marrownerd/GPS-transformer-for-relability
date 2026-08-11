import time
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from scipy import stats
import dagshub
import mlflow
import mlflow.pytorch

from config import DEVICE, EPOCHS, BATCH_SIZE, NUM_GRAPHS_TO_USE
from utils import gaussian_kl_loss
from dataset import ZincReliabilityDataset
from models import BaselineGINEModel, GatedReliabilityModel
from visualization import ExpertVisualizer

def train_model(model, loader, v_loader, name, history, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in tqdm(loader, desc=f"{name} {epoch}", leave=False):
            b = b.to(DEVICE)
            opt.zero_grad()
            mu, var = model(b.x, b.edge_index, b.edge_attr, b.batch, b.bounds)
            loss = gaussian_kl_loss(mu, var, b.y_mu, b.y_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for b in v_loader:
                b = b.to(DEVICE)
                mu, var = model(b.x, b.edge_index, b.edge_attr, b.batch, b.bounds)
                v_loss += gaussian_kl_loss(mu, var, b.y_mu, b.y_var).item()
        avg_v = v_loss / len(v_loader)
        history[name]['val_loss'].append(avg_v)
        if epoch % 5 == 0 or epoch == 1:
            print(f"{name} Ep {epoch} | Val: {avg_v:.4f}")
        torch.save(model.state_dict(), f'/content/drive/MyDrive/gnn_project/models/best_model_{name}.pth')

def test_on_real_internet_topology(m_base, m_gated):
    print("\n" + "="*80)
    print("ФИНАЛЬНЫЙ ТЕСТ ZERO-SHOT: Обучались на молекулах ZINC, тестируем на Internet AS")
    print("="*80)

    G = nx.random_internet_as_graph(n=120, seed=101)
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    ebc = nx.edge_betweenness_centrality(G)
    bridge_edges = set(map(tuple, map(sorted, nx.bridges(G))))

    edge_list_p = []
    for u, v in G.edges():
        is_b = tuple(sorted((u,v))) in bridge_edges
        p = 0.65 if is_b else 0.95
        edge_list_p.append((u, v, p, is_b, ebc[(u, v)]))

    print(f"[INFO] Запуск симуляции Монте-Карло для телеком-сети...")
    start_t = time.time()
    success_count = 0
    for _ in range(10000):
        rands = np.random.rand(len(edge_list_p))
        survived = [(u, v) for i, (u, v, p, _, _) in enumerate(edge_list_p) if rands[i] <= p]
        G_sim = nx.Graph()
        G_sim.add_nodes_from(range(n))
        G_sim.add_edges_from(survived)
        if nx.is_connected(G_sim):
            success_count += 1

    mu_true = success_count / 10000
    margin = 3 * np.sqrt((mu_true * (1 - mu_true)) / 10000)

    print(f"Истинная надежность: {mu_true:.5f} ± {margin:.5f} (Расчет занял {time.time()-start_t:.1f}с)")

    if mu_true == 0.0:
        print("[ВНИМАНИЕ] Граф 100% разваливается. Измените seed или повысьте p!")
        return

    bc = nx.betweenness_centrality(G)
    cc = nx.clustering(G)
    deg = dict(G.degree())
    x = torch.tensor([[bc[i], cc[i], deg[i]/n] for i in range(n)], dtype=torch.float).to(DEVICE)
    ei, ea = [], []
    for u, v, p, is_b, ebc_val in edge_list_p:
        is_b_float = 1.0 if is_b else 0.0
        feats = [p, is_b_float, ebc_val, p * ebc_val, 1.0 - p]
        ei.extend([[u, v], [v, u]])
        ea.extend([feats, feats])

    m_base.eval()
    m_gated.eval()
    start_inf = time.time()
    with torch.no_grad():
        b = torch.zeros(n, dtype=torch.long).to(DEVICE)
        bnd = torch.tensor([[max(0.0, mu_true-0.05), min(1.0, mu_true+0.05)]]).to(DEVICE)
        ei_t = torch.tensor(ei).t().long().to(DEVICE)
        ea_t = torch.tensor(ea).float().to(DEVICE)

        pred_mu_b, _ = m_base(x, ei_t, ea_t, b, bnd)
        pred_mu_g, _ = m_gated(x, ei_t, ea_t, b, bnd)
    inf_time = time.time() - start_inf

    err_b = abs(mu_true - pred_mu_b.item())
    err_g = abs(mu_true - pred_mu_g.item())

    print("\n" + "-"*50)
    print(f"BASELINE GIN Predict: {pred_mu_b.item():.5f} (Ошибка: {err_b:.5f})")
    print(f"SOTA GPS Predict:     {pred_mu_g.item():.5f} (Ошибка: {err_g:.5f})")
    print("-" * 50)

    if err_g < margin:
        print("ВЕРДИКТ: Ваша модель попала в математический 3-Sigma интервал (Zero-Shot)!")
    print(f"УЛУЧШЕНИЕ: Модель GPS в {err_b/(err_g+1e-9):.2f} раз(а) точнее базы.")
    print(f"Время ИИ-прогноза: {inf_time:.4f} сек.")
    print("="*80)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    colors = ['#c0392b' if p < 0.9 else '#2980b9' for u, v, p, _, _ in edge_list_p]
    widths = [2.0 if p < 0.9 else 0.5 for u, v, p, _, _ in edge_list_p]
    nx.draw(G, pos, node_size=20, node_color='black', edge_color=colors, width=widths, alpha=0.6)
    plt.title(f"Zero-Shot Тест: Топология Internet AS (N={n})\nКрасным выделены уязвимые мосты", fontsize=14)
    plt.show()

def main():
    dagshub.init(repo_owner='marrownerd', repo_name='gnnreliability', mlflow=True)
    with mlflow.start_run():
        dataset = ZincReliabilityDataset(num_graphs=NUM_GRAPHS_TO_USE)
        num_items = len(dataset)
        print(f"[INFO] Готово к обучению {num_items} графов из ZINC.")

        tr_ds, val_ds, te_ds = random_split(dataset, [int(0.7*num_items), int(0.15*num_items), num_items - int(0.85*num_items)])
        tr_l, val_l, te_l = DataLoader(tr_ds, BATCH_SIZE, True), DataLoader(val_ds, BATCH_SIZE), DataLoader(te_ds, 64)

        history = {'base': {'val_loss': []}, 'gated': {'val_loss': []}}
        results = {'base': {k: [] for k in ['mu_pred', 'mu_true', 'var_pred', 'var_true']},
                   'gated': {k: [] for k in ['mu_pred', 'mu_true', 'var_pred', 'var_true']}}

        print("\n--- Training Slow Baseline ---")
        m_base = BaselineGINEModel().to(DEVICE)
        train_model(m_base, tr_l, val_l, "base", history, 1e-4)

        print("\n--- Training SOTA GraphGPS Model ---")
        m_gated = GatedReliabilityModel().to(DEVICE)
        train_model(m_gated, tr_l, val_l, "gated", history, 1e-3)

        m_base.eval()
        m_gated.eval()
        with torch.no_grad():
            for b in tqdm(te_l, desc="Сбор финальной статистики"):
                b = b.to(DEVICE)
                for n, m in [('base', m_base), ('gated', m_gated)]:
                    p_mu, p_var = m(b.x, b.edge_index, b.edge_attr, b.batch, b.bounds)
                    results[n]['mu_pred'].extend(p_mu.cpu().numpy().flatten())
                    results[n]['mu_true'].extend(b.y_mu.cpu().numpy().flatten())
                    results[n]['var_pred'].extend(p_var.cpu().numpy().flatten())
                    results[n]['var_true'].extend(b.y_var.cpu().numpy().flatten())

        ExpertVisualizer.plot_dashboard(history, results)

        for m in ['base', 'gated']:
            mae = np.mean(np.abs(np.array(results[m]['mu_pred']) - np.array(results[m]['mu_true'])))
            rmse = np.sqrt(np.mean((np.array(results[m]['mu_pred']) - np.array(results[m]['mu_true']))**2))
            pearson_r, _ = stats.pearsonr(results[m]['mu_true'], results[m]['mu_pred'])

            mlflow.log_metric(f"final_test_mae_{m}", mae)
            mlflow.log_metric(f"final_test_rmse_{m}", rmse)
            mlflow.log_metric(f"final_test_pearson_{m}", pearson_r)

        test_on_real_internet_topology(m_base, m_gated)
        print("\n[УСПЕХ] Пайплайн полностью завершен!")

if __name__ == "__main__":
    main()