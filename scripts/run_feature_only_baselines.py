#!/usr/bin/env python3
"""Run feature-only RR+stats baselines (MLP and DNN) for MIT-BIH.

This script addresses the reviewer request for non-graph baselines using only
handcrafted rhythm/statistical descriptors.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy import stats as scipy_stats
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.utils import load_config


VALID_TYPES = "NLRejAaJSVEFP/fUQ"
N_TYPES = set("NLRej")
S_TYPES = set("AaJS")
V_TYPES = set("VE")
CLASS_NAMES = ("N", "S", "V")


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RR+stats feature-only baselines")
    ap.add_argument("--config", type=str, default="config/Part1_Exp1_3_NodeFeat_RR_seed42.yaml")
    ap.add_argument("--seeds", type=str, default="42,123,456,999")
    ap.add_argument("--output-root", type=str, default="output")
    ap.add_argument("--respect-val-split", action="store_true", default=True)
    ap.add_argument("--no-respect-val-split", dest="respect_val_split", action="store_false")
    ap.add_argument("--undersample-test", dest="undersample_test", action="store_true", default=False,
                    help="Undersample the test set (legacy protocol). Default: False (full DS2).")
    return ap.parse_args()


def _peak_to_int(v) -> int:
    arr = np.asarray(v).reshape(-1)
    return int(arr[0]) if arr.size else 0


def _compute_rr(peaks: Sequence, idx: int) -> Tuple[float, float]:
    peak = _peak_to_int(peaks[idx])
    if idx - 1 >= 0:
        rr_pre = float(peak - _peak_to_int(peaks[idx - 1]))
    else:
        rr_pre = 0.0
    if idx + 1 < len(peaks):
        rr_pos = float(_peak_to_int(peaks[idx + 1]) - peak)
    else:
        rr_pos = 0.0
    return rr_pre, rr_pos


def _compute_stats_features(ii: np.ndarray) -> np.ndarray:
    n5 = float(np.nanpercentile(ii, 5))
    n25 = float(np.nanpercentile(ii, 25))
    n75 = float(np.nanpercentile(ii, 75))
    n95 = float(np.nanpercentile(ii, 95))
    median = float(np.nanpercentile(ii, 50))
    mean = float(np.nanmean(ii))
    stdv = float(np.nanstd(ii))
    var = float(np.nanvar(ii))
    rms = float(np.nanmean(np.sqrt(ii ** 2)))
    kurt = float(scipy_stats.kurtosis(ii)) if len(ii) > 3 else 0.0
    skew = float(scipy_stats.skew(ii)) if len(ii) > 2 else 0.0
    zero_cross = float(np.sum(np.diff(ii > 0)))
    mean_cross = float(np.sum(np.diff(ii > mean)))
    counts = Counter(ii.tolist())
    probs = np.array([c / len(ii) for _, c in counts.most_common()], dtype=float)
    ent = float(entropy(probs)) if probs.size > 0 else 0.0
    return np.array(
        [
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
        ],
        dtype=np.float32,
    )


def _label_from_type(bt: str) -> int | None:
    if bt in N_TYPES:
        return 0
    if bt in S_TYPES:
        return 1
    if bt in V_TYPES:
        return 2
    return None


def _extract_split_features(
    root: str,
    split: str,
    beat_before: int,
    beat_after: int,
    include_files: Sequence[str] | None = None,
) -> SplitData:
    split_dir = os.path.join(root, split)
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".mat")])
    if include_files is not None:
        include = set(include_files)
        files = [f for f in files if f in include]

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for file in files:
        struct = loadmat(os.path.join(split_dir, file))
        data = struct["individual"][0][0]
        ecg = data["signal_r"]
        ecg_ii = ecg[:, 0]
        peaks = data["anno_anns"]
        types = data["anno_type"]
        for idx, (peak, bt) in enumerate(zip(peaks, types)):
            if bt not in VALID_TYPES:
                continue
            center = _peak_to_int(peak)
            start = center - beat_before
            end = center + beat_after
            if start < 0 or end > len(ecg_ii):
                continue
            label = _label_from_type(bt)
            if label is None:
                continue
            seg_ii = ecg_ii[start:end].flatten().astype(float)
            rr_pre, rr_pos = _compute_rr(peaks, idx)
            stats_vec = _compute_stats_features(seg_ii)
            feat = np.concatenate(([rr_pre, rr_pos], stats_vec), axis=0)
            xs.append(feat.astype(np.float32))
            ys.append(label)
    if not xs:
        return SplitData(x=np.zeros((0, 16), dtype=np.float32), y=np.zeros((0,), dtype=np.int64))
    return SplitData(x=np.vstack(xs), y=np.asarray(ys, dtype=np.int64))


def _undersample_indices(labels: np.ndarray, ratio: float, method: str, seed: int) -> np.ndarray:
    if ratio >= 1.0:
        return np.arange(len(labels))
    majority_idx = np.where(labels == 0)[0]
    if majority_idx.size == 0:
        return np.arange(len(labels))
    method = (method or "stride").lower()
    if method == "stride":
        k = max(1, int(round(1.0 / max(1e-12, ratio))))
        keep_majority = set(majority_idx[::k].tolist())
    else:
        keep = max(1, int(majority_idx.size * ratio))
        rng = np.random.default_rng(seed)
        keep_majority = set(rng.choice(majority_idx, size=keep, replace=False).tolist())
    keep_idx = [i for i in range(len(labels)) if labels[i] != 0 or i in keep_majority]
    return np.asarray(keep_idx, dtype=np.int64)


def _subset(data: SplitData, idx: np.ndarray) -> SplitData:
    return SplitData(x=data.x[idx], y=data.y[idx])


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(1, total))
    payload = {
        "accuracy": acc,
        "per_class": [],
    }
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        payload["per_class"].append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "fpr": 1.0 - specificity,
                "support": int(cm[i, :].sum()),
            }
        )
    return payload


def _format_mean_std(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean() * 100:.2f} ± {arr.std(ddof=0) * 100:.2f}"


def _aggregate_seed_metrics(seed_payloads: Sequence[Dict]) -> Dict:
    out: Dict[str, Dict[str, float]] = {}
    out["accuracy"] = {
        "mean": float(np.mean([p["accuracy"] for p in seed_payloads])),
        "std": float(np.std([p["accuracy"] for p in seed_payloads], ddof=0)),
    }
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        for key in ("f1", "precision", "recall", "specificity"):
            vals = [p["per_class"][cls_idx][key] for p in seed_payloads]
            out[f"{cls_name}_{key}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }
    return out


def _write_metrics_dir(
    output_dir: str,
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with open(os.path.join(output_dir, "confusion_matrix.csv"), "w", encoding="utf-8") as f:
        f.write(",".join([" "] + list(CLASS_NAMES)) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([CLASS_NAMES[i]] + [str(int(x)) for x in row.tolist()]) + "\n")
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=list(CLASS_NAMES), zero_division=0))


def _make_model(name: str, seed: int) -> MLPClassifier:
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    if name == "DNN":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=8e-4,
            max_iter=350,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=seed,
        )
    raise ValueError(f"Unknown model name: {name}")


def _split_train_files(path: str, split: str, val_ratio: float) -> Tuple[List[str], List[str]]:
    files = sorted([f for f in os.listdir(os.path.join(path, split)) if f.endswith(".mat")])
    k = int(len(files) * (1.0 - val_ratio))
    return files[:k], files[k:]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data") or cfg.get("dataset_params", {})

    path = data_cfg["path"]
    train_split = data_cfg.get("train_split", "Train")
    test_split = data_cfg.get("test_split", "Test")
    beat_before = int(data_cfg.get("beat_size_before", 100))
    beat_after = int(data_cfg.get("beat_size_after", 180))
    ratio = float(data_cfg.get("undersampling_ratio", 0.1))
    method = str(data_cfg.get("undersampling_method", "stride")).lower()
    undersample_test = args.undersample_test  # CLI overrides config; default=False (full DS2)
    val_ratio = float(data_cfg.get("val_from_train_ratio", 0.0))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    train_include = None
    if args.respect_val_split and val_ratio > 0.0:
        train_include, _ = _split_train_files(path, train_split, val_ratio)

    train_full = _extract_split_features(
        root=path,
        split=train_split,
        beat_before=beat_before,
        beat_after=beat_after,
        include_files=train_include,
    )
    test_full = _extract_split_features(
        root=path,
        split=test_split,
        beat_before=beat_before,
        beat_after=beat_after,
        include_files=None,
    )

    model_names = ("MLP", "DNN")
    seed_payloads: Dict[str, List[Dict]] = {m: [] for m in model_names}
    for seed in seeds:
        train_idx = _undersample_indices(train_full.y, ratio=ratio, method=method, seed=seed)
        test_idx = _undersample_indices(test_full.y, ratio=ratio, method=method, seed=seed) if undersample_test else np.arange(len(test_full.y))
        train_data = _subset(train_full, train_idx)
        test_data = _subset(test_full, test_idx)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_data.x)
        x_test = scaler.transform(test_data.x)
        y_train = train_data.y
        y_test = test_data.y

        for model_name in model_names:
            model = _make_model(model_name, seed=seed)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            payload = _compute_metrics(y_true=y_test, y_pred=y_pred)
            payload["seed"] = seed
            payload["train_size"] = int(y_train.size)
            payload["test_size"] = int(y_test.size)
            seed_payloads[model_name].append(payload)

            out_dir = os.path.join(args.output_root, f"Part2_Exp2_1_FeatureOnly_{model_name}_seed{seed}")
            _write_metrics_dir(output_dir=out_dir, metrics=payload, y_true=y_test, y_pred=y_pred)

    for model_name in model_names:
        agg = _aggregate_seed_metrics(seed_payloads[model_name])
        summary_dir = os.path.join(args.output_root, f"Part2_Exp2_1_FeatureOnly_{model_name}")
        os.makedirs(summary_dir, exist_ok=True)
        with open(os.path.join(summary_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": args.config,
                    "seeds": seeds,
                    "respect_val_split": args.respect_val_split,
                    "undersample_test": undersample_test,
                    "seed_metrics": seed_payloads[model_name],
                    "aggregate": agg,
                },
                f,
                indent=2,
            )

        print(f"\n[{model_name}]")
        print(f"Overall Acc.: {_format_mean_std([p['accuracy'] for p in seed_payloads[model_name]])}")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            f1 = _format_mean_std([p['per_class'][cls_idx]['f1'] for p in seed_payloads[model_name]])
            prec = _format_mean_std([p['per_class'][cls_idx]['precision'] for p in seed_payloads[model_name]])
            rec = _format_mean_std([p['per_class'][cls_idx]['recall'] for p in seed_payloads[model_name]])
            spec = _format_mean_std([p['per_class'][cls_idx]['specificity'] for p in seed_payloads[model_name]])
            print(f"{cls_name}: F1 {f1} | Prec {prec} | Sens {rec} | Spec {spec}")


if __name__ == "__main__":
    main()
