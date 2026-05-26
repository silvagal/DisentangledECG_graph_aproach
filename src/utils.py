import random
from typing import Any, Dict

import math
import numpy as np
import torch
import yaml
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sinusoidal_positional_encodings(
    num_positions: int, embedding_dim: int
) -> torch.Tensor:
    """Generate sinusoidal positional encodings.

    This follows the formulation from "Attention Is All You Need" and returns
    a tensor of shape ``(num_positions, embedding_dim)``.
    """

    pe = torch.zeros(num_positions, embedding_dim, dtype=torch.float)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float)
        * (-math.log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def compute_laplacian_pe(num_nodes: int, edge_index: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float)
    i, j = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    data = np.ones(len(i), dtype=np.float32)
    A = sp.coo_matrix((data, (i, j)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).flatten()
    D = sp.diags(deg)
    L = D - A
    try:
        vals, vecs = eigsh(L, k=min(k, max(1, num_nodes - 1)), which='SM')
        pe = torch.from_numpy(vecs[:, :k]).float()
    except Exception:
        pe = torch.zeros((num_nodes, k), dtype=torch.float)
    return pe


def compute_rwse(num_nodes: int, edge_index: torch.Tensor, steps: list[int]) -> torch.Tensor:
    if not steps or edge_index.numel() == 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float)
    i, j = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    data = np.ones(len(i), dtype=np.float32)
    A = sp.coo_matrix((data, (i, j)), shape=(num_nodes, num_nodes)).tocsr()
    deg = np.asarray(A.sum(1)).flatten()
    deg[deg == 0] = 1.0
    Dinv = sp.diags(1.0 / deg)
    P = Dinv @ A
    feats = []
    M = sp.eye(num_nodes, dtype=np.float32)
    max_s = max(steps)
    for s in range(1, max_s + 1):
        M = M @ P
        if s in steps:
            feats.append(torch.from_numpy(M.diagonal()).float().unsqueeze(1))
    return torch.cat(feats, dim=1) if feats else torch.zeros((num_nodes, 0), dtype=torch.float)
