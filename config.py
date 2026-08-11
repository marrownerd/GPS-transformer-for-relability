import os
import torch
from google.colab import drive

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
HIDDEN_DIM_BASE = 128
HIDDEN_DIM_SOTA = 128
BATCH_SIZE = 128
NUM_GRAPHS_TO_USE = 5000
REAL_FILES = ["en_net.txt", "en_net_kaofo.txt", "GEANT.txt", "GEANT2004_EdgeList.txt"]

print(f"[INFO] Используемое устройство: {DEVICE}")

drive.mount('/content/drive')
save_path = '/content/drive/MyDrive/gnn_project'
os.makedirs(f'{save_path}/models', exist_ok=True)
os.makedirs(f'{save_path}/data', exist_ok=True)