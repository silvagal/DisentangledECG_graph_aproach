import os
from typing import List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import torch
try:
    # Prefer file_system sharing to avoid /dev/shm exhaustion in mp workers
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from .graph_conversion import signal_to_vg
from .utils import generate_sinusoidal_positional_encodings


_LABEL_MAP = {"N": 0, "S": 1, "V": 2}
_VALID_TYPES = "NLRejAaJSVEFP/fUQ"


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _full_dataset_enabled(data_cfg: dict) -> bool:
    env_flag = _to_bool(os.environ.get("ECG_FULL_DATASET"), default=False)
    cfg_flag = _to_bool(data_cfg.get("use_full_dataset"), default=False)
    return env_flag or cfg_flag


class ECGGraphDataset(Dataset):
    """Dataset that loads ECG beats and provides graph and raw signal.

    Supports optional undersampling of the majority class (N) either randomly
    or by deterministic stride, with the option to also apply it to the test
    split to mirror legacy protocol.
    """

    def __init__(
        self,
        root: str,
        split: str,
        beat_size_before: int,
        beat_size_after: int,
        pe_dim: int = 0,
        undersampling_ratio: float = 1.0,
        undersampling_method: str = "random",
        undersample_test: bool = False,
        feature_set: str = "raw",
        include_files: Optional[List[str]] = None,
        graph_constructor: str = "vg",
        k_limit: Optional[int] = None,
        edge_features: Optional[List[str]] = None,
    ) -> None:
        self.root = os.path.join(root, split)
        self.split = split
        self.beat_size_before = beat_size_before
        self.beat_size_after = beat_size_after
        self.pe_dim = pe_dim
        self.undersampling_ratio = float(undersampling_ratio)
        self.undersampling_method = (undersampling_method or "random").lower()
        self.undersample_test = bool(undersample_test)
        self.feature_set = (feature_set or "raw").lower()
        self.include_files = include_files
        self.graph_constructor = (graph_constructor or "vg").lower()
        self.k_limit = int(k_limit) if k_limit is not None else None
        self.edge_features = edge_features
        self.items: List[dict] = []
        self._process()
        self._maybe_undersample()

    def _process(self) -> None:
        all_files = sorted([f for f in os.listdir(self.root) if f.endswith(".mat")])
        files = [f for f in all_files if (self.include_files is None or f in set(self.include_files))]
        workers = int(os.environ.get("ECG_PREPROCESS_WORKERS", "0"))
        if workers > 1:
            args = [(
                self.root,
                file,
                self.beat_size_before,
                self.beat_size_after,
                self.pe_dim,
                self.feature_set,
                self.graph_constructor,
                self.k_limit,
                self.edge_features,
            ) for file in files]
            try:
                with mp.Pool(processes=workers) as pool:
                    for items in pool.imap_unordered(_process_mat_file, args, chunksize=1):
                        self.items.extend(items)
            except Exception:
                # Fallback to serial if multiprocessing fails (e.g., low shared memory)
                for file in files:
                    self.items.extend(_process_mat_file((
                        self.root,
                        file,
                        self.beat_size_before,
                        self.beat_size_after,
                        self.pe_dim,
                        self.feature_set,
                        self.graph_constructor,
                        self.k_limit,
                        self.edge_features,
                    )))
        else:
            for file in files:
                self.items.extend(_process_mat_file((
                    self.root,
                    file,
                    self.beat_size_before,
                    self.beat_size_after,
                    self.pe_dim,
                    self.feature_set,
                    self.graph_constructor,
                    self.k_limit,
                    self.edge_features,
                )))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    def _maybe_undersample(self) -> None:
        apply = self.split.lower() == "train" or (
            self.split.lower() == "test" and self.undersample_test
        )
        if not apply or self.undersampling_ratio >= 1.0:
            return
        labels = [int(item["graph"].y.item()) for item in self.items]
        majority_label = 0
        majority_idx = [i for i, l in enumerate(labels) if l == majority_label]
        if not majority_idx:
            return
        if self.undersampling_method == "stride":
            k = max(1, int(round(1.0 / max(1e-12, self.undersampling_ratio))))
            keep_majority = set(majority_idx[::k])
            keep_indices = [
                i for i in range(len(self.items))
                if labels[i] != majority_label or i in keep_majority
            ]
        else:
            keep = max(1, int(len(majority_idx) * self.undersampling_ratio))
            sampled = set(np.random.choice(majority_idx, keep, replace=False))
            keep_indices = [
                i for i in range(len(self.items))
                if labels[i] != majority_label or i in sampled
            ]
        self.items = [self.items[i] for i in keep_indices]


class SequentialBeatDataset(Dataset):
    """Dataset returning sequences of beats for seq2seq models."""

    def __init__(
        self, beats: List[torch.Tensor], labels: List[int], sequence_length: int
    ) -> None:
        self.beats = beats
        self.labels = labels
        self.sequence_length = sequence_length
        self.start_indices = list(range(len(beats) - sequence_length + 1))

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.start_indices[idx]
        end = start + self.sequence_length
        inputs = torch.stack(self.beats[start:end])
        targets = torch.tensor(self.labels[start:end], dtype=torch.long)
        return {"inputs": inputs, "targets": targets}


class SequentialBeatPerRecordDataset(Dataset):
    """Dataset returning sequences of beats grouped per record/patient.

    Windows are generated with stride=1 within each record. Optional
    undersampling by stride can be applied based on the center label to mimic
    the class balance protocol (reduce majority class N).
    """

    def __init__(
        self,
        items: List[dict],
        sequence_length: int,
        split: str,
        undersampling_ratio: float = 1.0,
        undersampling_method: str = "stride",
        undersample_test: bool = False,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.split = split
        self.undersampling_ratio = float(undersampling_ratio)
        self.undersampling_method = (undersampling_method or "stride").lower()
        self.undersample_test = bool(undersample_test)

        # Group items by record_id preserving order
        groups: dict[str, List[dict]] = {}
        for it in items:
            rid = it.get("record_id", "")
            groups.setdefault(str(rid), []).append(it)
        self._windows: List[tuple[str, int]] = []  # (record_id, start)
        self._groups = groups

        for rid, seq in groups.items():
            n = len(seq)
            L = self.sequence_length
            for start in range(0, max(0, n - L + 1)):
                self._windows.append((rid, start))

        # Apply stride undersampling on majority center label (N=0)
        apply = (split.lower() == "train") or (split.lower() == "test" and self.undersample_test)
        if apply and self.undersampling_ratio < 1.0 and len(self._windows) > 0:
            if self.undersampling_method == "stride":
                k = max(1, int(round(1.0 / max(1e-12, self.undersampling_ratio))))
                kept: List[tuple[str, int]] = []
                keep_counter = 0
                for w in self._windows:
                    rid, start = w
                    center = start + (self.sequence_length // 2)
                    label = int(self._groups[rid][center]["graph"].y.item())
                    if label != 0:
                        kept.append(w)
                    else:
                        if keep_counter % k == 0:
                            kept.append(w)
                        keep_counter += 1
                self._windows = kept
            else:
                # Random undersampling on windows based on center label
                import random
                maj = [w for w in self._windows if int(self._groups[w[0]][w[1] + (self.sequence_length // 2)]["graph"].y.item()) == 0]
                mino = [w for w in self._windows if w not in maj]
                keep = max(1, int(len(maj) * self.undersampling_ratio))
                maj_kept = random.sample(maj, keep) if len(maj) > keep else maj
                self._windows = mino + maj_kept

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        rid, start = self._windows[idx]
        seq = self._groups[rid]
        L = self.sequence_length
        sl = seq[start:start+L]
        inputs = torch.stack([it["raw_tensor"].squeeze(0) for it in sl])  # [L, beat_len]
        targets = torch.tensor([int(it["graph"].y.item()) for it in sl], dtype=torch.long)
        return {"inputs": inputs, "targets": targets}


class SequentialGraphDataset(Dataset):
    """Dataset yielding sequences of graph Data objects.

    Supports undersampling the majority class (label 0) either randomly or
    deterministically by stride, and optionally applying it to the test split.
    """

    def __init__(
        self,
        graphs: List[Data],
        sequence_length: int,
        split: str,
        undersampling_ratio: float = 1.0,
        undersampling_method: str = "random",
        undersample_test: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.split = split
        self.undersampling_ratio = float(undersampling_ratio)
        self.undersampling_method = (undersampling_method or "random").lower()
        self.undersample_test = bool(undersample_test)

        apply_undersampling = (
            split.lower() == "train" or (split.lower() == "test" and self.undersample_test)
        )
        if apply_undersampling and self.undersampling_ratio < 1.0:
            if self.undersampling_method == "stride":
                self.graphs = self._undersample_stride(graphs)
            else:
                self.graphs = self._undersample_random(graphs)
        else:
            self.graphs = graphs

    def _undersample_random(self, graphs: List[Data]) -> List[Data]:
        labels = [int(g.y.item()) for g in graphs]
        majority_label = 0
        majority_indices = [i for i, l in enumerate(labels) if l == majority_label]
        if not majority_indices:
            return graphs
        keep = max(1, int(len(majority_indices) * self.undersampling_ratio))
        sampled_majority = set(np.random.choice(majority_indices, keep, replace=False))
        selected_indices = [
            i for i in range(len(graphs)) if labels[i] != majority_label or i in sampled_majority
        ]
        return [graphs[i] for i in selected_indices]

    def _undersample_stride(self, graphs: List[Data]) -> List[Data]:
        """Deterministically keep every k-th majority example to approximate
        the target ratio (e.g., ratio=0.1 -> k≈10), matching the paper.
        """
        labels = [int(g.y.item()) for g in graphs]
        majority_label = 0
        majority_indices = [i for i, l in enumerate(labels) if l == majority_label]
        if not majority_indices:
            return graphs
        k = max(1, int(round(1.0 / max(1e-12, self.undersampling_ratio))))
        keep_majority = set(majority_indices[::k])
        selected_indices = [
            i for i in range(len(graphs)) if labels[i] != majority_label or i in keep_majority
        ]
        return [graphs[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.graphs) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> dict:
        seq = self.graphs[idx : idx + self.sequence_length]
        label = int(seq[self.sequence_length // 2].y.item())
        return {"graph_sequence": seq, "label": label}


def ecg_collate_fn(batch: List[dict]) -> dict:
    graphs = [item["graph"] for item in batch]
    raw_tensors = [item["raw_tensor"] for item in batch]
    batch_graph = Batch.from_data_list(graphs)
    raw_tensor = torch.stack(raw_tensors)
    return {"graph": batch_graph, "raw_tensor": raw_tensor}


def sequential_graph_collate(batch_list: List[dict]) -> dict:
    all_graphs: List[Data] = []
    labels: List[int] = []
    for item in batch_list:
        all_graphs.extend(item["graph_sequence"])
        labels.append(item["label"])
    batch_graph = Batch.from_data_list(all_graphs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {"batch_graph": batch_graph, "labels": labels_tensor}


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config.get("data") or config.get("dataset_params", {})
    model_cfg = config.get("model") or config.get("model_params", {})
    use_full_dataset = _full_dataset_enabled(data_cfg)
    pe_dim = model_cfg.get("pe_dim", 0) if model_cfg.get("use_pe") or model_cfg.get("use_positional_encoding") else 0

    # Apply default subsampling for non-sequential pipelines as in the paper
    undersample_ratio = data_cfg.get("undersampling_ratio", 0.1)
    undersample_method = data_cfg.get("undersampling_method", "stride")
    undersample_test = data_cfg.get("undersample_test", True)
    if use_full_dataset:
        undersample_ratio = 1.0
        undersample_test = False
    feature_set = data_cfg.get("feature_set", "raw")
    graph_constructor = (data_cfg.get("graph_constructor", "vg") or "vg").lower()
    k_limit = data_cfg.get("k_limit")
    edge_features = data_cfg.get("edge_features")
    paradigm = (data_cfg.get("paradigm", "inter") or "inter").lower()
    val_from_train_ratio = float(data_cfg.get("val_from_train_ratio", 0.0))
    if "preprocess_workers" in data_cfg:
        os.environ["ECG_PREPROCESS_WORKERS"] = str(data_cfg["preprocess_workers"])
    # Optional positional encoding mode
    pe_type = (data_cfg.get("pe_type", "sin") or "sin").lower()
    os.environ["ECG_PE_TYPE"] = pe_type
    if pe_type == "rwse":
        steps = data_cfg.get("rwse_steps", [1, 2, 3, 4])
        if isinstance(steps, list):
            os.environ["ECG_RWSE_STEPS"] = ",".join(str(int(s)) for s in steps)

    if paradigm == "inter":
        if val_from_train_ratio > 0.0:
            train_root = os.path.join(data_cfg["path"], data_cfg.get("train_split", "Train"))
            files_all = sorted([f for f in os.listdir(train_root) if f.endswith(".mat")])
            k = int(len(files_all) * (1.0 - val_from_train_ratio))
            include_train = files_all[:k]
            include_val = files_all[k:]
            train_dataset = ECGGraphDataset(
                root=data_cfg["path"],
                split=data_cfg.get("train_split", "Train"),
                beat_size_before=data_cfg["beat_size_before"],
                beat_size_after=data_cfg["beat_size_after"],
                pe_dim=pe_dim,
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=False,
                feature_set=feature_set,
                include_files=include_train,
                graph_constructor=graph_constructor,
                k_limit=k_limit,
                edge_features=edge_features,
            )
            val_dataset = ECGGraphDataset(
                root=data_cfg["path"],
                split=data_cfg.get("train_split", "Train"),
                beat_size_before=data_cfg["beat_size_before"],
                beat_size_after=data_cfg["beat_size_after"],
                pe_dim=pe_dim,
                undersampling_ratio=1.0,
                undersampling_method=undersample_method,
                undersample_test=False,
                feature_set=feature_set,
                include_files=include_val,
                graph_constructor=graph_constructor,
                k_limit=k_limit,
                edge_features=edge_features,
            )
        else:
            train_dataset = ECGGraphDataset(
                root=data_cfg["path"],
                split=data_cfg.get("train_split", "Train"),
                beat_size_before=data_cfg["beat_size_before"],
                beat_size_after=data_cfg["beat_size_after"],
                pe_dim=pe_dim,
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=False,
                feature_set=feature_set,
                graph_constructor=graph_constructor,
                k_limit=k_limit,
                edge_features=edge_features,
            )
            val_dataset = None
        # Support ds1-ds1: if test_split equals train_split and a train/val split is defined,
        # use the held-out partition as the test set.
        test_include = None
        if (
            val_from_train_ratio > 0.0
            and data_cfg.get("test_split", "Test") == data_cfg.get("train_split", "Train")
        ):
            test_include = include_val
        test_dataset = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("test_split", "Test"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=pe_dim,
            undersampling_ratio=undersample_ratio,
            undersampling_method=undersample_method,
            undersample_test=undersample_test,
            feature_set=feature_set,
            include_files=test_include,
            graph_constructor=graph_constructor,
            k_limit=k_limit,
            edge_features=edge_features,
        )
    elif paradigm == "intra":
        # Build combined items from Train and Test, then split by beat
        ds_train = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("train_split", "Train"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=pe_dim,
            undersampling_ratio=1.0,
            feature_set=feature_set,
            graph_constructor=graph_constructor,
            k_limit=k_limit,
            edge_features=edge_features,
        )
        ds_test = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("test_split", "Test"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=pe_dim,
            undersampling_ratio=1.0,
            feature_set=feature_set,
            graph_constructor=graph_constructor,
            k_limit=k_limit,
            edge_features=edge_features,
        )
        all_items = ds_train.items + ds_test.items
        rng = np.random.default_rng(seed=config.get("seed", 42))
        idx = np.arange(len(all_items))
        rng.shuffle(idx)
        test_ratio = float(data_cfg.get("intra_test_ratio", 0.2))
        cut = int(len(all_items) * (1.0 - test_ratio))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        train_dataset = _ItemsDataset([all_items[i] for i in train_idx])
        val_dataset = None
        test_dataset = _ItemsDataset([all_items[i] for i in test_idx])
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    batch_size = config.get("training", {}).get(
        "batch_size", config.get("batch_size", 32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ecg_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ecg_collate_fn,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ecg_collate_fn)
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader


def _process_mat_file(args) -> List[dict]:
    root, file, beat_size_before, beat_size_after, pe_dim, feature_set, graph_constructor, k_limit, edge_features = args
    from scipy.io import loadmat
    import numpy as np
    import torch
    from .graph_conversion import signal_to_vg, signal_to_vvg
    from .utils import generate_sinusoidal_positional_encodings, compute_laplacian_pe, compute_rwse

    path = os.path.join(root, file)
    struct = loadmat(path)
    data = struct["individual"][0][0]
    ecg = data["signal_r"]
    ecg_ii = ecg[:, 0]
    ecg_v1 = ecg[:, 1]
    if file == "114.mat":
        ecg_ii, ecg_v1 = ecg_v1, ecg_ii
    peaks = data["anno_anns"]
    types = data["anno_type"]
    items: List[dict] = []
    for it, (peak, beat_type) in enumerate(zip(peaks, types)):
        if beat_type not in _VALID_TYPES:
            continue
        start = int(peak - beat_size_before)
        end = int(peak + beat_size_after)
        if start < 0 or end > len(ecg_ii):
            continue
        seg_ii = ecg_ii[start:end].flatten()
        seg_v1 = ecg_v1[start:end].flatten()
        if beat_type in "NLRej":
            label = _LABEL_MAP["N"]
        elif beat_type in "AaJS":
            label = _LABEL_MAP["S"]
        elif beat_type in "VE":
            label = _LABEL_MAP["V"]
        else:
            continue
        if (graph_constructor or "vg").lower() == "vvg":
            graph = signal_to_vvg((np.asarray(seg_ii), np.asarray(seg_v1)), edge_features=edge_features)
        else:
            k_lim = int(k_limit) if k_limit is not None else None
            graph = signal_to_vg(np.asarray(seg_ii), graph_type=graph_constructor, k_limit=k_lim, edge_features=edge_features)
        # Build feature sets
        t = np.linspace(0.0, 1.0, num=len(seg_ii), dtype=float)
        # RR intervals
        try:
            rr_pre = float((peak - peaks[it - 1])[0]) if it - 1 >= 0 else 0.0
        except Exception:
            rr_pre = 0.0
        try:
            rr_pos = float((peaks[it + 1] - peak)[0]) if it + 1 < len(peaks) else 0.0
        except Exception:
            rr_pos = 0.0
        ii = seg_ii.astype(float)
        v1 = seg_v1.astype(float)
        if feature_set == "raw":
            feats = ii.reshape(-1, 1)
        else:
            cols = [ii, v1, t]
            if feature_set in {"rr", "difii", "avgii", "stdii", "stats"}:
                cols.append(np.full_like(ii, rr_pre, dtype=float))
                cols.append(np.full_like(ii, rr_pos, dtype=float))
            if feature_set in {"difii", "avgii", "stdii", "stats"}:
                cols.append(v1 - ii)
            if feature_set in {"avgii", "stdii", "stats"}:
                m = float(np.nanmean(ii)) if np.isfinite(ii).any() else 1.0
                cols.append(v1 / (m if m != 0.0 else 1.0))
            if feature_set in {"stdii", "stats"}:
                s = float(np.nanstd(ii)) if np.isfinite(ii).any() else 1.0
                cols.append(v1 / (s if s != 0.0 else 1.0))
            feats = np.stack(cols, axis=1)
            if feature_set == "stats":
                # 14 stats from lead II, repeated across nodes
                from scipy import stats as _stats
                from collections import Counter as _Counter
                n5 = float(np.nanpercentile(ii, 5))
                n25 = float(np.nanpercentile(ii, 25))
                n75 = float(np.nanpercentile(ii, 75))
                n95 = float(np.nanpercentile(ii, 95))
                median = float(np.nanpercentile(ii, 50))
                mean = float(np.nanmean(ii))
                stdv = float(np.nanstd(ii))
                var = float(np.nanvar(ii))
                rms = float(np.nanmean(np.sqrt(ii ** 2)))
                kurt = float(_stats.kurtosis(ii)) if len(ii) > 3 else 0.0
                skew = float(_stats.skew(ii)) if len(ii) > 2 else 0.0
                zero_cross = int(np.sum(np.diff(ii > 0)))
                mean_cross = int(np.sum(np.diff(ii > mean)))
                counts = _Counter(ii.tolist())
                probs = np.array([c / len(ii) for _, c in counts.most_common()], dtype=float)
                from scipy.stats import entropy as _entropy
                ent = float(_entropy(probs)) if probs.size > 0 else 0.0
                stats_vec = np.array([
                    ent,
                    zero_cross,
                    mean_cross,
                    n5,
                    n25,
                    n75,
                    n95,
                    median,
                    mean,
                    stdv,
                    var,
                    rms,
                    kurt,
                    skew,
                ], dtype=float)
                stats_mat = np.repeat(stats_vec.reshape(1, -1), repeats=len(ii), axis=0)
                feats = np.concatenate([feats, stats_mat], axis=1)
        graph.x = torch.tensor(feats, dtype=torch.float)
        if pe_dim > 0:
            pe_type = os.environ.get("ECG_PE_TYPE", "sin").lower()
            if pe_type == "lap":
                pe = compute_laplacian_pe(len(seg_ii), graph.edge_index, pe_dim)
            elif pe_type == "rwse":
                steps_env = os.environ.get("ECG_RWSE_STEPS", "1,2,3,4")
                steps = [int(s) for s in steps_env.split(",") if s.strip().isdigit()]
                pe = compute_rwse(len(seg_ii), graph.edge_index, steps)
                if pe.size(1) < pe_dim:
                    pad = torch.zeros((pe.size(0), pe_dim - pe.size(1)))
                    pe = torch.cat([pe, pad], dim=1)
                pe = pe[:, :pe_dim]
            else:
                pe = generate_sinusoidal_positional_encodings(len(seg_ii), pe_dim)
            graph.pos_encoding = pe
        graph.y = torch.tensor([label], dtype=torch.long)
        raw_tensor = torch.tensor(seg_ii, dtype=torch.float32).unsqueeze(0)
        # Attach record id to the graph and item for sequential grouping
        graph.record_id = file
        items.append({"graph": graph, "raw_tensor": raw_tensor, "record_id": file})
    return items


class _ItemsDataset(Dataset):
    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


class SequentialGraphPerRecordDataset(Dataset):
    """Dataset yielding sequences of graph Data objects within each record.

    Similar to SequentialGraphDataset, but sequences never cross record/patient
    boundaries. Optional undersampling is applied on the central label of each
    window using stride or random strategies.
    """

    def __init__(
        self,
        graphs: list,
        sequence_length: int,
        split: str,
        undersampling_ratio: float = 1.0,
        undersampling_method: str = "stride",
        undersample_test: bool = False,
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.split = split
        self.undersampling_ratio = float(undersampling_ratio)
        self.undersampling_method = (undersampling_method or "stride").lower()
        self.undersample_test = bool(undersample_test)
        # Group graphs by record_id
        groups = {}
        for g in graphs:
            rid = getattr(g, "record_id", "")
            groups.setdefault(str(rid), []).append(g)
        self._groups = groups
        self._windows = []  # list of (rid, start)
        for rid, seq in groups.items():
            n = len(seq)
            L = self.sequence_length
            for start in range(0, max(0, n - L + 1)):
                self._windows.append((rid, start))

        # Apply undersampling on center label
        apply = (split.lower() == "train") or (split.lower() == "test" and self.undersample_test)
        if apply and self.undersampling_ratio < 1.0 and len(self._windows) > 0:
            if self.undersampling_method == "stride":
                k = max(1, int(round(1.0 / max(1e-12, self.undersampling_ratio))))
                kept = []
                keep_counter = 0
                for rid, start in self._windows:
                    center = start + (self.sequence_length // 2)
                    label = int(self._groups[rid][center].y.item())
                    if label != 0:
                        kept.append((rid, start))
                    else:
                        if keep_counter % k == 0:
                            kept.append((rid, start))
                        keep_counter += 1
                self._windows = kept
            else:
                import random
                maj = [w for w in self._windows if int(self._groups[w[0]][w[1] + (self.sequence_length // 2)].y.item()) == 0]
                mino = [w for w in self._windows if w not in maj]
                keep = max(1, int(len(maj) * self.undersampling_ratio))
                maj_kept = random.sample(maj, keep) if len(maj) > keep else maj
                self._windows = mino + maj_kept

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        rid, start = self._windows[idx]
        seq = self._groups[rid]
        L = self.sequence_length
        out = seq[start:start+L]
        label = int(out[L // 2].y.item())
        return {"graph_sequence": out, "label": label}
