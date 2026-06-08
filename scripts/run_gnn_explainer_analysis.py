#!/usr/bin/env python3
"""Run a GNNExplainer-based interpretability analysis for GINE+VG+RR."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_loader import ECGGraphDataset, create_dataloaders, ecg_collate_fn
from src.models import get_model
from src.trainer import Trainer
from src.utils import load_config, set_seed


CLASS_NAMES = ("N", "S", "V")
CLASS_LABELS = ("Normal (N)", "Supraventricular (S)", "Ventricular (V)")
BEAT_PRE_R = 100   # samples before R-peak
BEAT_R_END = 141   # R-peak region ends at sample 141


@dataclass
class Candidate:
    graph: Data
    raw_signal: np.ndarray
    true_label: int
    pred_label: int
    confidence: float


class _GINEWrapper(torch.nn.Module):
    """Adapts the project model signature to Explainer's x/edge_index API."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        return self.model(data)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GNNExplainer analysis for GINE+VG+RR")
    ap.add_argument("--config", type=str, default="config/Part1_Exp1_3_NodeFeat_RR_seed42.yaml")
    ap.add_argument("--output-dir", type=str, default="output/Part2_Interpretability_GNNExplainer")
    ap.add_argument("--load-model", type=str, default="",
                    help="Path to model_state.pt. If provided, skip training.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--train-epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples-per-class", type=int, default=2)
    ap.add_argument("--min-confidence", type=float, default=0.70,
                    help="Minimum model confidence for sample selection.")
    ap.add_argument("--explainer-epochs", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--max-train-files", type=int, default=0)
    ap.add_argument("--max-test-files", type=int, default=0)
    ap.add_argument("--train-files", type=str, default="")
    ap.add_argument("--test-files", type=str, default="")
    return ap.parse_args()


def _node_region_importance(node_imp: np.ndarray) -> Dict[str, float]:
    total = float(node_imp.sum()) + 1e-12
    pre = float(node_imp[:BEAT_PRE_R].sum() / total)
    around_r = float(node_imp[BEAT_PRE_R:BEAT_R_END].sum() / total)
    post = float(node_imp[BEAT_R_END:].sum() / total)
    return {"pre_r": pre, "around_r": around_r, "post_r": post}


def _pick_candidates(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    samples_per_class: int,
    min_confidence: float,
) -> Dict[int, List[Candidate]]:
    model.eval()
    selected: Dict[int, List[Candidate]] = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for batch in loader:
            batch_graph = batch["graph"].to(device)
            logits = model(batch_graph)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch_graph.y.cpu().numpy()
            data_list = batch_graph.to_data_list()
            raw = batch["raw_tensor"].cpu().numpy()
            for i in range(len(data_list)):
                y_true = int(labels[i])
                y_pred = int(preds[i])
                if y_true != y_pred:
                    continue
                if len(selected[y_true]) >= samples_per_class:
                    continue
                conf = float(probs[i, y_pred].item())
                if conf < min_confidence:
                    continue
                selected[y_true].append(
                    Candidate(
                        graph=data_list[i].cpu(),
                        raw_signal=raw[i].reshape(-1),
                        true_label=y_true,
                        pred_label=y_pred,
                        confidence=conf,
                    )
                )
            if all(len(v) >= samples_per_class for v in selected.values()):
                break
    return selected


def _compute_fidelity(
    explainer: "Explainer",
    explanation,
    model: torch.nn.Module,
    candidate: Candidate,
    device: torch.device,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (fidelity_plus, fidelity_minus, characterization_score).

    Tries PyG's built-in metric first; falls back to manual node-feature masking.

    Fidelity+  = P(ŷ|G) − P(ŷ|G\\S)  — how much removing S hurts.
    Fidelity−  = P(ŷ|G) − P(ŷ|S)    — how much keeping only S suffices.
    Char. score = harmonic mean of Fidelity+ and (1 − Fidelity−).
    """
    try:
        from torch_geometric.explain.metric import (
            characterization_score as pyg_char_score,
            fidelity as pyg_fidelity,
        )
        pos_fid, neg_fid = pyg_fidelity(explainer, explanation)
        char_score = float(pyg_char_score(pos_fid, neg_fid))
        return float(pos_fid), float(neg_fid), char_score
    except Exception as exc:
        print(f"[warn] PyG fidelity failed ({exc}); falling back to manual masking")

    # ── Manual fallback: soft node-feature masking ──────────────────────────
    try:
        model.eval()
        g = candidate.graph
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        edge_attr = g.edge_attr.to(device) if getattr(g, "edge_attr", None) is not None else None
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        pred_class = candidate.pred_label

        nm = explanation.node_mask.to(device)
        nm_norm = nm.abs().mean(dim=1, keepdim=True) if nm.ndim == 2 else nm.abs().unsqueeze(1)
        max_val = nm_norm.max()
        if max_val > 0:
            nm_norm = nm_norm / max_val

        with torch.no_grad():
            data_orig = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            p_orig = torch.softmax(model(data_orig), dim=1)[0, pred_class].item()

            # Fidelity+: remove important features (complement mask)
            data_comp = Data(x=x * (1.0 - nm_norm), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            p_comp = torch.softmax(model(data_comp), dim=1)[0, pred_class].item()

            # Fidelity−: keep only important features (subgraph mask)
            data_sub = Data(x=x * nm_norm, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            p_sub = torch.softmax(model(data_sub), dim=1)[0, pred_class].item()

        pos_fid = float(p_orig - p_comp)
        neg_fid = float(p_orig - p_sub)
        pos_f = max(0.0, pos_fid)
        one_minus_neg = max(0.0, 1.0 - neg_fid)
        denom = 0.5 * pos_f + 0.5 * one_minus_neg
        char_score = float(pos_f * one_minus_neg / denom) if denom > 0 else 0.0
        return pos_fid, neg_fid, char_score
    except Exception as exc2:
        print(f"[warn] Manual fidelity also failed: {exc2}")
        return None, None, None


def _run_explainer(
    model: torch.nn.Module,
    candidate: Candidate,
    explainer_epochs: int,
    top_k: int,
    output_prefix: str,
    device: torch.device,
) -> Tuple[Dict, np.ndarray]:
    """Returns (payload_dict, node_imp_array)."""
    wrapped = _GINEWrapper(model).to(device)
    wrapped.eval()
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )

    g = candidate.graph
    x = g.x.to(device)
    edge_index = g.edge_index.to(device)
    edge_attr = getattr(g, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
    )

    node_mask = explanation.node_mask.detach().cpu().numpy()
    if node_mask.ndim == 2:
        node_imp = np.mean(np.abs(node_mask), axis=1)
    else:
        node_imp = np.abs(node_mask).reshape(-1)
    node_imp = node_imp / (node_imp.sum() + 1e-12)

    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    edge_imp = np.abs(edge_mask)
    edge_imp = edge_imp / (edge_imp.sum() + 1e-12)
    edges_np = edge_index.detach().cpu().numpy().T

    top_nodes = np.argsort(-node_imp)[:top_k]
    top_edges = np.argsort(-edge_imp)[:top_k]
    region = _node_region_importance(node_imp)

    pos_fid, neg_fid, char_score = _compute_fidelity(
        explainer, explanation, model, candidate, device
    )

    # Per-sample plot (kept for reference)
    _plot_single(candidate, node_imp, output_prefix)

    payload = {
        "true_label": CLASS_NAMES[candidate.true_label],
        "pred_label": CLASS_NAMES[candidate.pred_label],
        "confidence": candidate.confidence,
        "region_importance": region,
        "fidelity_plus": pos_fid,
        "fidelity_minus": neg_fid,
        "characterization_score": char_score,
        "top_nodes": [
            {"index": int(i), "importance": float(node_imp[i])}
            for i in top_nodes.tolist()
        ],
        "top_edges": [
            {
                "source": int(edges_np[i][0]),
                "target": int(edges_np[i][1]),
                "importance": float(edge_imp[i]),
                "span": int(abs(int(edges_np[i][1]) - int(edges_np[i][0]))),
            }
            for i in top_edges.tolist()
        ],
    }
    with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload, node_imp


def _plot_single(candidate: Candidate, node_imp: np.ndarray, output_prefix: str) -> None:
    sig = candidate.raw_signal
    x_axis = np.arange(sig.shape[0])
    plt.figure(figsize=(10, 3))
    plt.plot(x_axis, sig, color="#c9c9c9", linewidth=1.0, label="ECG")
    sc = plt.scatter(x_axis, sig, c=node_imp, cmap="YlOrRd", s=12, vmin=0)
    plt.axvline(BEAT_PRE_R, color="#3b82f6", linestyle="--", linewidth=1.0)
    plt.axvline(BEAT_R_END, color="#3b82f6", linestyle="--", linewidth=1.0)
    plt.title(
        f"Node importance — true/pred: {CLASS_NAMES[candidate.true_label]} "
        f"(conf. {candidate.confidence:.3f})"
    )
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    cbar = plt.colorbar(sc)
    cbar.set_label("Norm. node importance")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_signal_importance.png", dpi=200)
    plt.close()


def _make_combined_figure(
    class_data: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, float]]],
    output_dir: str,
) -> None:
    """Generate a publication-quality 3-panel figure (N, S, V) for the paper."""
    # Global vmax across all classes for a consistent colormap
    all_imps = [d[1] for d in class_data.values() if d is not None]
    if not all_imps:
        return
    vmax = max(float(imp.max()) for imp in all_imps)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

    region_patches = [
        mpatches.Patch(facecolor="#93c5fd", alpha=0.55, label="Pre-R"),
        mpatches.Patch(facecolor="#f9a8d4", alpha=0.55, label="QRS"),
        mpatches.Patch(facecolor="#86efac", alpha=0.55, label="Post-R"),
    ]

    scatter_refs = []
    for cls_idx, ax in enumerate(axes):
        data = class_data.get(cls_idx)
        if data is None:
            ax.set_title(CLASS_LABELS[cls_idx] + "\n(no sample)", fontsize=10)
            ax.axis("off")
            continue

        sig, node_imp, conf = data
        n = len(sig)
        x = np.arange(n)

        # Region shading
        ax.axvspan(0, BEAT_PRE_R, alpha=0.18, color="#93c5fd", zorder=0)
        ax.axvspan(BEAT_PRE_R, BEAT_R_END, alpha=0.20, color="#f9a8d4", zorder=0)
        ax.axvspan(BEAT_R_END, n, alpha=0.14, color="#86efac", zorder=0)

        ax.plot(x, sig, color="#9ca3af", linewidth=0.8, zorder=1)
        sc = ax.scatter(x, sig, c=node_imp, cmap="YlOrRd", s=16,
                        vmin=0, vmax=vmax, zorder=2)
        scatter_refs.append(sc)

        ax.axvline(BEAT_PRE_R, color="#3b82f6", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axvline(BEAT_R_END, color="#3b82f6", linestyle="--", linewidth=0.9, alpha=0.8)

        ax.set_title(f"{CLASS_LABELS[cls_idx]}\n(conf. {conf:.2f})", fontsize=10)
        ax.set_xlabel("Sample index", fontsize=9)
        if cls_idx == 0:
            ax.set_ylabel("Amplitude (norm.)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend on the first panel
    axes[0].legend(handles=region_patches, fontsize=7.5, loc="upper left",
                   framealpha=0.75, handlelength=1.2)

    # Shared colorbar
    if scatter_refs:
        cbar = fig.colorbar(scatter_refs[0], ax=axes, orientation="vertical",
                            fraction=0.018, pad=0.02)
        cbar.set_label("Node importance (norm.)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("GNNExplainer — node importance per beat class (GINE+VG+RR)",
                 fontsize=11, y=1.01)

    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"gnnexplainer_combined.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[interpretability] combined figure saved: {path}")
    plt.close(fig)


def _aggregate_interpretability(results: Dict[int, List[Dict]]) -> Dict:
    out = {"per_class": {}, "overall": {}}
    all_regions: List[Dict[str, float]] = []
    all_pos_fid: List[float] = []
    all_neg_fid: List[float] = []
    all_char: List[float] = []

    for cls_idx, payloads in results.items():
        if not payloads:
            continue
        regs = [p["region_importance"] for p in payloads]
        all_regions.extend(regs)

        pos_fids = [p["fidelity_plus"] for p in payloads if p.get("fidelity_plus") is not None]
        neg_fids = [p["fidelity_minus"] for p in payloads if p.get("fidelity_minus") is not None]
        chars = [p["characterization_score"] for p in payloads if p.get("characterization_score") is not None]
        all_pos_fid.extend(pos_fids)
        all_neg_fid.extend(neg_fids)
        all_char.extend(chars)

        entry: Dict = {
            "pre_r_mean": float(np.mean([r["pre_r"] for r in regs])),
            "around_r_mean": float(np.mean([r["around_r"] for r in regs])),
            "post_r_mean": float(np.mean([r["post_r"] for r in regs])),
        }
        if pos_fids:
            entry["fidelity_plus_mean"] = float(np.mean(pos_fids))
        if neg_fids:
            entry["fidelity_minus_mean"] = float(np.mean(neg_fids))
        if chars:
            entry["characterization_score_mean"] = float(np.mean(chars))
        out["per_class"][CLASS_NAMES[cls_idx]] = entry

    if all_regions:
        overall: Dict = {
            "pre_r_mean": float(np.mean([r["pre_r"] for r in all_regions])),
            "around_r_mean": float(np.mean([r["around_r"] for r in all_regions])),
            "post_r_mean": float(np.mean([r["post_r"] for r in all_regions])),
        }
        if all_pos_fid:
            overall["fidelity_plus_mean"] = float(np.mean(all_pos_fid))
        if all_neg_fid:
            overall["fidelity_minus_mean"] = float(np.mean(all_neg_fid))
        if all_char:
            overall["characterization_score_mean"] = float(np.mean(all_char))
        out["overall"] = overall
    return out


def main() -> None:
    args = parse_args()
    print("[interpretability] starting", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[interpretability] output_dir={args.output_dir}", flush=True)

    cfg = load_config(args.config)
    print(f"[interpretability] config={args.config}", flush=True)
    set_seed(args.seed)
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = int(args.train_epochs)
    cfg["training"]["batch_size"] = int(args.batch_size)

    model_cfg = cfg.get("model") or cfg.get("model_params", {})
    if not model_cfg:
        raise ValueError("Model configuration not found in config.")

    data_cfg = cfg.get("data") or cfg.get("dataset_params", {})
    print("[interpretability] preparing dataloaders...", flush=True)
    explicit_train = [s.strip() for s in str(args.train_files).split(",") if s.strip()]
    explicit_test = [s.strip() for s in str(args.test_files).split(",") if s.strip()]
    if explicit_train or explicit_test or int(args.max_train_files) > 0 or int(args.max_test_files) > 0:
        train_root = os.path.join(data_cfg["path"], data_cfg.get("train_split", "Train"))
        test_root = os.path.join(data_cfg["path"], data_cfg.get("test_split", "Test"))
        train_files = sorted([f for f in os.listdir(train_root) if f.endswith(".mat")])
        test_files = sorted([f for f in os.listdir(test_root) if f.endswith(".mat")])
        if explicit_train:
            train_files = explicit_train
        if explicit_test:
            test_files = explicit_test
        if int(args.max_train_files) > 0:
            train_files = train_files[: int(args.max_train_files)]
        if int(args.max_test_files) > 0:
            test_files = test_files[: int(args.max_test_files)]

        train_dataset = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("train_split", "Train"),
            beat_size_before=int(data_cfg["beat_size_before"]),
            beat_size_after=int(data_cfg["beat_size_after"]),
            pe_dim=0,
            undersampling_ratio=float(data_cfg.get("undersampling_ratio", 0.1)),
            undersampling_method=str(data_cfg.get("undersampling_method", "stride")),
            undersample_test=False,
            feature_set=str(data_cfg.get("feature_set", "rr")),
            include_files=train_files,
            graph_constructor=str(data_cfg.get("graph_constructor", "vg")),
            k_limit=data_cfg.get("k_limit"),
            edge_features=data_cfg.get("edge_features"),
        )
        test_dataset = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("test_split", "Test"),
            beat_size_before=int(data_cfg["beat_size_before"]),
            beat_size_after=int(data_cfg["beat_size_after"]),
            pe_dim=0,
            undersampling_ratio=float(data_cfg.get("undersampling_ratio", 0.1)),
            undersampling_method=str(data_cfg.get("undersampling_method", "stride")),
            undersample_test=bool(data_cfg.get("undersample_test", True)),
            feature_set=str(data_cfg.get("feature_set", "rr")),
            include_files=test_files,
            graph_constructor=str(data_cfg.get("graph_constructor", "vg")),
            k_limit=data_cfg.get("k_limit"),
            edge_features=data_cfg.get("edge_features"),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=ecg_collate_fn,
        )
        val_loader = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=ecg_collate_fn,
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
    print(
        f"[interpretability] train={len(train_loader.dataset)} test={len(test_loader.dataset)}",
        flush=True,
    )
    device = torch.device(args.device)

    model = get_model(model_cfg).to(device)

    if args.load_model:
        # Load pre-trained weights — skip training entirely
        state = torch.load(args.load_model, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"[interpretability] loaded model from {args.load_model}", flush=True)
        test_acc = None
    else:
        lr = float(cfg.get("training", {}).get("learning_rate", cfg.get("learning_rate", 0.001)))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_type=str(model_cfg.get("model_type", "gnn")).lower(),
            num_classes=int(model_cfg.get("num_classes", 3)),
            sequence_length=int(model_cfg.get("sequence_length", 1)),
        )
        trainer.fit(train_loader, val_loader, epochs=int(args.train_epochs))
        print("[interpretability] training done", flush=True)
        test_acc = float(trainer.evaluate(test_loader))
        print(f"[interpretability] test_acc={test_acc:.4f}", flush=True)

    candidates = _pick_candidates(
        model=model,
        loader=test_loader,
        device=device,
        samples_per_class=int(args.samples_per_class),
        min_confidence=float(args.min_confidence),
    )
    for cls_idx, cands in candidates.items():
        print(f"[interpretability] class {CLASS_NAMES[cls_idx]}: {len(cands)} candidate(s) found")

    per_class_results: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    # Maps cls_idx → (signal, node_imp, confidence) for the combined figure
    combined_data: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, float]]] = {}

    for cls_idx in [0, 1, 2]:
        combined_data[cls_idx] = None
        for j, cand in enumerate(candidates[cls_idx]):
            prefix = os.path.join(args.output_dir, f"{CLASS_NAMES[cls_idx]}_sample{j+1}")
            payload, node_imp = _run_explainer(
                model=model,
                candidate=cand,
                explainer_epochs=int(args.explainer_epochs),
                top_k=int(args.top_k),
                output_prefix=prefix,
                device=device,
            )
            per_class_results[cls_idx].append(payload)
            # Use first sample per class for the combined figure
            if j == 0:
                combined_data[cls_idx] = (cand.raw_signal, node_imp, cand.confidence)

    # Generate the publication-quality combined figure
    _make_combined_figure(combined_data, args.output_dir)

    summary = {
        "config": args.config,
        "load_model": args.load_model or None,
        "seed": args.seed,
        "device": args.device,
        "train_epochs": args.train_epochs if not args.load_model else "skipped",
        "test_accuracy": test_acc,
        "min_confidence": args.min_confidence,
        "samples_found": {CLASS_NAMES[k]: len(v) for k, v in candidates.items()},
        "interpretability_summary": _aggregate_interpretability(per_class_results),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[interpretability] summary saved", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
