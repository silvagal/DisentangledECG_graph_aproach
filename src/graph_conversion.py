import numpy as np
import os
import torch
from torch_geometric.data import Data
from ts2vg import NaturalVG
try:
    from ts2vg import HorizontalVG
except Exception:  # pragma: no cover
    HorizontalVG = None
from typing import Optional, List, Tuple

try:  # pragma: no cover - optional dependency
    from ts2vvg_v2.graph import build_graph as build_vvg
except Exception:  # pragma: no cover
    build_vvg = None


_VG_SINGLETON = NaturalVG()
_HVG_SINGLETON = HorizontalVG() if HorizontalVG is not None else None
_VVG_BACKEND_LOGGED = False


def _build_edges(signal: np.ndarray, graph_type: str = "vg", k_limit: Optional[int] = None):
    gt = (graph_type or "vg").lower()
    if gt == "vg":
        graph = _VG_SINGLETON.build(signal)
        edges = graph.edges
    elif gt == "hvg" and _HVG_SINGLETON is not None:
        graph = _HVG_SINGLETON.build(signal)
        edges = graph.edges
    else:
        graph = _VG_SINGLETON.build(signal)
        edges = graph.edges
    if k_limit is not None and k_limit > 0:
        edges = [(u, v) for (u, v) in edges if abs(u - v) <= k_limit]
    return edges


def _edge_features(signal: np.ndarray, edges: List[Tuple[int, int]], series2: Optional[np.ndarray],
                   feats: Optional[List[str]]) -> Optional[torch.Tensor]:
    if not feats:
        return None
    n_edges = len(edges)
    if n_edges == 0:
        return None
    feats = [f.lower() for f in feats]
    cols = []
    x = signal.astype(float)
    for name in feats:
        if name == "delta_t":
            cols.append(np.array([abs(j - i) for (i, j) in edges], dtype=float))
        elif name == "slope":
            cols.append(np.array([(x[j] - x[i]) / (j - i) if j != i else 0.0 for (i, j) in edges], dtype=float))
        elif name == "length":
            cols.append(np.array([np.sqrt((j - i) ** 2 + (x[j] - x[i]) ** 2) for (i, j) in edges], dtype=float))
        elif name == "direction" and series2 is not None:
            y = series2.astype(float)
            cols.append(np.array([np.sign((y[j] - y[i])) for (i, j) in edges], dtype=float))
        else:
            cols.append(np.zeros(n_edges, dtype=float))
    edge_attr = np.stack(cols, axis=1)
    return torch.tensor(edge_attr, dtype=torch.float)


def signal_to_vg(signal: np.ndarray, *, graph_type: str = "vg", k_limit: Optional[int] = None,
                 edge_features: Optional[List[str]] = None) -> Data:
    edges = _build_edges(signal, graph_type=graph_type, k_limit=k_limit)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if len(edges) else torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(signal, dtype=torch.float).unsqueeze(-1)
    edge_attr = _edge_features(signal, edges, None, edge_features)
    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    return data


def signal_to_vvg(series: Tuple[np.ndarray, np.ndarray], time_direction: bool = False,
                  edge_features: Optional[List[str]] = None) -> Data:
    """Convert a 2D signal into a vector visibility graph.

    If the optional ts2vvg dependency is unavailable, gracefully fall back to
    building a standard VG on the first series while still computing any
    requested edge features that depend on the second series (e.g., 'direction').
    This keeps experiments running without hard failing, albeit with edges
    defined by VG instead of true VVG.
    """
    global _VVG_BACKEND_LOGGED
    if build_vvg is None:
        # Fallback: use NaturalVG on the first series to define edges, but still
        # compute edge features that may leverage the second series.
        s0, s1 = series
        edges = _build_edges(s0, graph_type="vg", k_limit=None)
        edges_py = [(int(u), int(v)) for (u, v) in edges]
    else:
        env_flag = os.environ.get("ECG_VVG_USE_GPU")
        use_gpu = torch.cuda.is_available() if env_flag is None else (env_flag.strip().lower() in {"1", "true", "yes"})
        if not _VVG_BACKEND_LOGGED:
            backend_msg = "GPU" if use_gpu else "CPU (Numba)"
            print(f"[VVG] ts2vvg_v2 backend: {backend_msg}")
            _VVG_BACKEND_LOGGED = True
        try:
            adj = build_vvg(series, time_direction, use_gpu=use_gpu)
        except (TypeError, RuntimeError, ModuleNotFoundError) as e:
            try:
                adj = build_vvg(series=series, time_direction=time_direction, use_gpu=use_gpu)
            except (TypeError, RuntimeError, ModuleNotFoundError):
                if use_gpu:
                    print(f"[VVG] GPU path unavailable ({e}) — falling back to CPU (Numba)")
                try:
                    adj = build_vvg(series, time_direction, use_gpu=False)
                except TypeError:
                    adj = build_vvg(series=series, time_direction=time_direction)
        if isinstance(adj, dict):
            edges_py = [(int(u), int(v)) for u, nbrs in adj.items() for v in nbrs]
        else:
            edges_py = [(int(u), int(v)) for (u, v) in adj]
    edge_index = torch.tensor(edges_py, dtype=torch.long).t().contiguous() if len(edges_py) else torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(series[0], dtype=torch.float).unsqueeze(-1)
    edge_attr = _edge_features(series[0], edges_py, series[1], edge_features)
    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    return data
