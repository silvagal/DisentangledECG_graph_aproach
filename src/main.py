import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .data_loader import (
    ECGGraphDataset,
    SequentialBeatDataset,
    SequentialGraphDataset,
    sequential_graph_collate,
    create_dataloaders,
)
from .models import get_model
from .trainer import Trainer
from .utils import load_config, set_seed


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
    return _to_bool(os.environ.get("ECG_FULL_DATASET"), default=False) or _to_bool(
        data_cfg.get("use_full_dataset"), default=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG Graph Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Disable undersampling globally and run with full DS1/DS2 splits.",
    )
    args = parser.parse_args()
    if args.full_dataset:
        os.environ["ECG_FULL_DATASET"] = "1"

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda":
        cudnn.enabled = bool(config.get("use_cudnn", False))

    model_cfg = config.get("model") or config.get("model_params", {})
    model_type = model_cfg.get("model_type", "gnn").lower()
    global_data_cfg = config.get("data") or config.get("dataset_params", {})
    use_full_dataset = _full_dataset_enabled(global_data_cfg)
    model_name = (model_cfg.get("model_name") or model_cfg.get("name") or "").lower()
    if use_full_dataset and model_type == "gnn" and model_name in {
        "gcn", "gcnmodel", "gcn2", "gcn7", "gcn60", "gcn120", "gcn240", "gin", "ginmodel", "gine", "ginemodel"
    }:
        model_cfg.setdefault("dropout", 0.2)

    if model_type == "seq2seq_cnn":
        data_cfg = config.get("data") or config.get("dataset_params", {})
        dl_cfg = config.get("data_loader_params", {})
        seq_len = dl_cfg.get("sequence_length", 10)
        batch_size = config.get("training", {}).get(
            "batch_size", config.get("batch_size", 32)
        )
        # Default subsampling parameters (applied to both splits by default)
        undersample_ratio = data_cfg.get("undersampling_ratio", 0.1)
        undersample_method = data_cfg.get("undersampling_method", "stride")
        undersample_test = data_cfg.get("undersample_test", True)
        if use_full_dataset:
            undersample_ratio = 1.0
            undersample_test = False
        paradigm = (data_cfg.get("paradigm", "inter") or "inter").lower()
        val_from_train_ratio = float(data_cfg.get("val_from_train_ratio", 0.0))
        if "preprocess_workers" in data_cfg:
            os.environ["ECG_PREPROCESS_WORKERS"] = str(data_cfg["preprocess_workers"])
        cnn_cfg = model_cfg.setdefault("cnn", {})
        if isinstance(cnn_cfg, dict):
            beat_len = int(data_cfg["beat_size_before"]) + int(data_cfg["beat_size_after"])
            cnn_cfg.setdefault("input_depth", beat_len)

        # Support ds1-ds1 by splitting Train into train/test partitions when requested
        use_ds1_ds1 = (
            data_cfg.get("train_split", "Train") == data_cfg.get("test_split", "Test")
            and paradigm == "inter"
            and val_from_train_ratio > 0.0
        )
        include_train = include_test = None
        if use_ds1_ds1:
            train_root = os.path.join(data_cfg["path"], data_cfg.get("train_split", "Train"))
            files_all = sorted([f for f in os.listdir(train_root) if f.endswith(".mat")])
            k = int(len(files_all) * (1.0 - val_from_train_ratio))
            include_train = files_all[:k]
            include_test = files_all[k:]

        train_base = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("train_split", "Train"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=0,
            undersampling_ratio=undersample_ratio,
            undersampling_method=undersample_method,
            undersample_test=False,
            include_files=include_train,
        )
        test_base = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("test_split", "Test"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=0,
            undersampling_ratio=undersample_ratio,
            undersampling_method=undersample_method,
            undersample_test=undersample_test,
            include_files=include_test,
        )
        from .data_loader import SequentialBeatPerRecordDataset
        train_dataset = SequentialBeatPerRecordDataset(
            train_base.items,
            sequence_length=seq_len,
            split="train",
            undersampling_ratio=undersample_ratio,
            undersampling_method=undersample_method,
            undersample_test=False,
        )
        test_dataset = SequentialBeatPerRecordDataset(
            test_base.items,
            sequence_length=seq_len,
            split="test",
            undersampling_ratio=undersample_ratio,
            undersampling_method=undersample_method,
            undersample_test=undersample_test,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    elif model_type in ("gnn_transformer", "seq2seq_gnn"):
        data_cfg = config.get("dataset_params", {})
        dl_cfg = config.get("data_loader_params", {})
        seq_len = dl_cfg.get("sequence_length", 10)
        undersample_ratio = data_cfg.get("undersampling_ratio", 1.0)
        undersample_method = data_cfg.get("undersampling_method", "random")
        undersample_test = data_cfg.get("undersample_test", False)
        if use_full_dataset:
            undersample_ratio = 1.0
            undersample_test = False
        paradigm = (data_cfg.get("paradigm", "inter") or "inter").lower()
        val_from_train_ratio = float(data_cfg.get("val_from_train_ratio", 0.0))
        # Optional preprocess workers
        if "preprocess_workers" in data_cfg:
            os.environ["ECG_PREPROCESS_WORKERS"] = str(data_cfg["preprocess_workers"])

        batch_size = config.get("training", {}).get(
            "batch_size", config.get("batch_size", 32)
        )
        # Encoder config: different key names depending on the model
        if model_type in ("gnn_transformer", "seq2seq_gnn", "seq2seq_cnn"):
            gnn_cfg = model_cfg.get("gnn_encoder_config", {})
            use_pe = gnn_cfg.get("use_pe") or gnn_cfg.get("use_positional_encoding", False)
            pe_dim = gnn_cfg.get("pe_dim", 0) if use_pe else 0
        else:  # seq2seq_gnn (GINE-based)
            gnn_cfg = model_cfg.get("gine_config", {})
            use_pe = gnn_cfg.get("use_pe") or gnn_cfg.get("use_positional_encoding", False)
            pe_dim = gnn_cfg.get("pe_dim", 0) if use_pe else 0

        # For GNN-Transformer we handle subsampling in the sequential dataset.
        # If ds1-ds1 protocol is requested (train==test split and val_from_train_ratio>0),
        # split the Train folder into train/test by patient order.
        use_ds1_ds1 = (
            data_cfg.get("train_split", "Train") == data_cfg.get("test_split", "Test")
            and paradigm == "inter"
            and val_from_train_ratio > 0.0
        )
        include_train = include_test = None
        if use_ds1_ds1:
            train_root = os.path.join(data_cfg["path"], data_cfg.get("train_split", "Train"))
            files_all = sorted([f for f in os.listdir(train_root) if f.endswith(".mat")])
            k = int(len(files_all) * (1.0 - val_from_train_ratio))
            include_train = files_all[:k]
            include_test = files_all[k:]

        train_base = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("train_split", "Train"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=pe_dim,
            undersampling_ratio=1.0,
            undersampling_method=data_cfg.get("undersampling_method", "random"),
            undersample_test=False,
            feature_set=data_cfg.get("feature_set", "raw"),
            include_files=include_train,
            graph_constructor=data_cfg.get("graph_constructor", "vg"),
            k_limit=data_cfg.get("k_limit"),
            edge_features=data_cfg.get("edge_features"),
        )
        test_base = ECGGraphDataset(
            root=data_cfg["path"],
            split=data_cfg.get("test_split", "Test"),
            beat_size_before=data_cfg["beat_size_before"],
            beat_size_after=data_cfg["beat_size_after"],
            pe_dim=pe_dim,
            undersampling_ratio=1.0,
            undersampling_method=data_cfg.get("undersampling_method", "random"),
            undersample_test=undersample_test,
            feature_set=data_cfg.get("feature_set", "raw"),
            include_files=include_test,
            graph_constructor=data_cfg.get("graph_constructor", "vg"),
            k_limit=data_cfg.get("k_limit"),
            edge_features=data_cfg.get("edge_features"),
        )
        train_graphs = [item["graph"] for item in train_base.items]
        test_graphs = [item["graph"] for item in test_base.items]
        if model_type in ("seq2seq_gnn", "gnn_transformer"):
            from .data_loader import SequentialGraphPerRecordDataset
            train_dataset = SequentialGraphPerRecordDataset(
                train_graphs,
                sequence_length=seq_len,
                split="train",
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=False,
            )
            test_dataset = SequentialGraphPerRecordDataset(
                test_graphs,
                sequence_length=seq_len,
                split="test",
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=undersample_test,
            )
        else:
            train_dataset = SequentialGraphDataset(
                train_graphs,
                sequence_length=seq_len,
                split="train",
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=False,
            )
            test_dataset = SequentialGraphDataset(
                test_graphs,
                sequence_length=seq_len,
                split="test",
                undersampling_ratio=undersample_ratio,
                undersampling_method=undersample_method,
                undersample_test=undersample_test,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=sequential_graph_collate,
        )
        val_loader = None
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=sequential_graph_collate,
        )
        # For GNN-Transformer, we use input_dim; for GINE, in_channels
        if model_type in ("gnn_transformer", "seq2seq_gnn", "seq2seq_cnn"):
            input_dim = gnn_cfg.get("input_dim", 0)
            if use_pe:
                input_dim += gnn_cfg.get("pe_dim", 0)
            gnn_cfg["input_dim"] = input_dim
            gnn_cfg["use_pe"] = use_pe
        else:
            in_channels = gnn_cfg.get("in_channels", 0)
            if use_pe:
                in_channels += gnn_cfg.get("pe_dim", 0)
            gnn_cfg["in_channels"] = in_channels
            gnn_cfg["use_pe"] = use_pe
    else:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        use_pe = model_cfg.get("use_pe") or model_cfg.get("use_positional_encoding", False)
        # Do not add pe_dim here to avoid double counting; get_model will handle it once.
        in_channels = model_cfg.get("in_channels", model_cfg.get("input_dim", 0))
        model_cfg["in_channels"] = in_channels
        model_cfg["use_pe"] = use_pe

    model = get_model(model_cfg)
    exp_name = config.get("experiment_name", model_cfg.get("model_name", "experiment"))
    
    # Respeitar variável de ambiente SUITE_OUTPUT_DIR se definida
    base_output_dir = os.environ.get("SUITE_OUTPUT_DIR", "output")
    output_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    train_cfg = config.get("training", {})
    debug_one_batch = bool(train_cfg.get("debug_one_batch", False))
    lr = train_cfg.get(
        "learning_rate", config.get("learning_rate", 0.001)
    )
    epochs = train_cfg.get("epochs", config.get("epochs", 10))
    if debug_one_batch:
        epochs = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if model_type in ("gnn_transformer", "seq2seq_gnn", "seq2seq_cnn"):
        weight_decay = float(train_cfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Linear warmup + decay
        try:
            total_steps = len(train_loader) * epochs
            warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
            warmup_steps = int(total_steps * warmup_ratio)
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(current_step: int):
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
            scheduler = LambdaLR(optimizer, lr_lambda)
        except Exception:
            scheduler = None
    num_classes = (
        model_cfg.get("num_classes")
        or model_cfg.get("classifier", {}).get("num_classes")
        or 0
    )
    import numpy as _np
    weights_tensor = None
    cw_mode = train_cfg.get("class_weights_mode")
    if cw_mode is None and use_full_dataset and model_type in {"gnn", "gnn_transformer", "seq2seq_gnn"}:
        cw_mode = "effective"
    if cw_mode in {"inverse", "effective"} and num_classes > 0:
        counts = _np.zeros((num_classes,), dtype=_np.int64)
        max_batches = int(train_cfg.get("class_weights_scan_batches", 200))
        scanned = 0
        with torch.no_grad():
            for batch in train_loader:
                if model_type == "gnn":
                    labels = batch["graph"].y
                elif model_type == "cnn":
                    labels = batch["graph"].y
                elif model_type == "seq2seq_cnn":
                    t = batch["targets"]
                    labels = t[:, t.size(1)//2]
                elif model_type in {"seq2seq_gnn", "gnn_transformer"}:
                    labels = batch["labels"]
                else:
                    labels = None
                if labels is None:
                    continue
                import numpy as __np
                hist = __np.bincount(labels.cpu().numpy(), minlength=num_classes)
                counts[:len(hist)] += hist
                scanned += 1
                if scanned >= max_batches:
                    break
        if cw_mode == "inverse":
            w = 1.0 / _np.maximum(counts, 1)
        else:
            beta = float(train_cfg.get("class_weights_beta", 0.9999))
            w = (1.0 - beta) / (1.0 - _np.power(beta, _np.maximum(counts, 1)))
        w = w / (w.sum() + 1e-12) * float(num_classes)
        weights_tensor = torch.tensor(w, dtype=torch.float)

    loss_name = str(train_cfg.get("loss", "ce")).lower()
    if loss_name in {"focal", "focal_loss"}:
        from .losses import FocalLoss
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        criterion = FocalLoss(gamma=gamma, alpha=weights_tensor.to(device) if weights_tensor is not None else None)
    else:
        ls = float(train_cfg.get("label_smoothing", 0.0) or 0.0)
        try:
            criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device) if weights_tensor is not None else None,
                                                 label_smoothing=ls)
        except TypeError:
            # Older torch fallback without label smoothing
            criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device) if weights_tensor is not None else None)

    num_classes = (
        model_cfg.get("num_classes")
        or model_cfg.get("classifier", {}).get("num_classes")
        or 0
    )
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        model_type=model_type,
        num_classes=num_classes,
        sequence_length=model_cfg.get("sequence_length", config.get("data_loader_params", {}).get("sequence_length", 1)),
        debug_max_train_batches=(1 if debug_one_batch else None),
        debug_max_eval_batches=(1 if debug_one_batch else None),
        scheduler=scheduler,
        grad_clip_max_norm=float(train_cfg.get("grad_clip_max_norm", 1.0)) if model_type in ("gnn_transformer", "seq2seq_gnn", "seq2seq_cnn") else None,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs)
    model_path = os.path.join(output_dir, "model_state.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    test_acc = trainer.evaluate_with_report(test_loader, output_dir=output_dir)
    print(f"Test accuracy: {test_acc:.4f}. Report saved to {output_dir}")


if __name__ == "__main__":
    main()
