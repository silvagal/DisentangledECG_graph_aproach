from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import json
import os


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    device: torch.device
    model_type: str = "gnn"
    num_classes: int = 0
    sequence_length: int = 1
    # Debug controls
    debug_max_train_batches: Optional[int] = None
    debug_max_eval_batches: Optional[int] = None
    scheduler: Optional[__import__('torch').optim.lr_scheduler.LRScheduler] = None
    grad_clip_max_norm: Optional[float] = None

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> None:
        self.model.to(self.device)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for ib, batch in enumerate(train_loader, 1):
                if self.model_type == "gnn":
                    batch["graph"] = batch["graph"].to(self.device)
                    outputs = self.model(batch["graph"])
                    labels = batch["graph"].y
                    batch_size = batch["graph"].num_graphs
                    loss = self.criterion(outputs, labels)
                elif self.model_type == "cnn":
                    batch["raw_tensor"] = batch["raw_tensor"].to(self.device)
                    outputs = self.model(batch["raw_tensor"])
                    labels = batch["graph"].y.to(self.device)
                    batch_size = batch["raw_tensor"].size(0)
                    loss = self.criterion(outputs, labels)
                elif self.model_type == "seq2seq_cnn":
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    outputs = self.model(inputs, targets=targets)
                    batch_size = inputs.size(0)
                    loss = self.criterion(
                        outputs.view(-1, self.num_classes), targets.view(-1)
                    )
                elif self.model_type == "seq2seq_gnn":
                    batch_graph = batch["batch_graph"].to(self.device)
                    labels = batch.get("labels")
                    if labels is None:
                        raise ValueError("seq2seq_gnn expects central labels in batch['labels']")
                    labels = labels.to(self.device)
                    batch_size = labels.size(0)
                    seq_len = int(self.sequence_length)
                    outputs = self.model(batch_graph, sequence_length=seq_len)
                    # Loss only on the central step
                    logits_c = outputs[:, seq_len // 2, :]
                    loss = self.criterion(logits_c, labels)
                elif self.model_type == "gnn_transformer":
                    batch_graph = batch["batch_graph"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    # Infer sequence length from the batch if mismatched
                    seq_len = int(self.sequence_length)
                    if hasattr(batch_graph, "num_graphs") and labels.numel() > 0:
                        if batch_graph.num_graphs % labels.size(0) == 0:
                            inferred = batch_graph.num_graphs // labels.size(0)
                            if inferred != seq_len:
                                seq_len = inferred
                    outputs = self.model(batch_graph, sequence_length=seq_len)
                    batch_size = labels.size(0)
                    loss = self.criterion(outputs, labels)
                else:
                    raise ValueError("Unsupported model type")
                self.optimizer.zero_grad()
                loss.backward()
                import torch as _torch
                from torch.nn.utils import clip_grad_norm_ as _clip
                if self.grad_clip_max_norm is not None:
                    _clip(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_loss += loss.item() * batch_size
                if self.debug_max_train_batches is not None and ib >= self.debug_max_train_batches:
                    break
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")
            if val_loader is not None:
                acc = self.evaluate(val_loader)
                print(f"  Val accuracy: {acc:.4f}")

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for ib, batch in enumerate(loader, 1):
                if self.model_type == "gnn":
                    batch["graph"] = batch["graph"].to(self.device)
                    outputs = self.model(batch["graph"])
                    labels = batch["graph"].y
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += batch["graph"].num_graphs
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                elif self.model_type == "cnn":
                    batch["raw_tensor"] = batch["raw_tensor"].to(self.device)
                    outputs = self.model(batch["raw_tensor"])
                    labels = batch["graph"].y.to(self.device)
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += batch["raw_tensor"].size(0)
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                elif self.model_type == "seq2seq_cnn":
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    outputs = self.model(inputs)
                    seq_len = outputs.size(1)
                    pred = outputs.argmax(dim=2)[:, seq_len // 2]
                    labels = targets[:, seq_len // 2]
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                elif self.model_type == "seq2seq_gnn":
                    batch_graph = batch["batch_graph"].to(self.device)
                    outputs = self.model(batch_graph, sequence_length=self.sequence_length)
                    seq_len = outputs.size(1)
                    pred = outputs.argmax(dim=2)[:, seq_len // 2]
                    labels = batch["labels"].to(self.device)
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                elif self.model_type == "gnn_transformer":
                    batch_graph = batch["batch_graph"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    seq_len = int(self.sequence_length)
                    if hasattr(batch_graph, "num_graphs") and labels.numel() > 0:
                        if batch_graph.num_graphs % labels.size(0) == 0:
                            inferred = batch_graph.num_graphs // labels.size(0)
                            if inferred != seq_len:
                                seq_len = inferred
                    outputs = self.model(batch_graph, sequence_length=seq_len)
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                    all_preds.extend(pred.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                else:
                    raise ValueError("Unsupported model type")
                if self.debug_max_eval_batches is not None and ib >= self.debug_max_eval_batches:
                    break
        return correct / total if total > 0 else 0.0

    def evaluate_with_report(
        self,
        loader: DataLoader,
        output_dir: str,
        class_names: Tuple[str, ...] = ("N", "S", "V"),
    ) -> float:
        """Evaluate and save detailed metrics, confusion matrix and a report.

        Saves files in `output_dir`:
        - metrics.json: overall accuracy, per-class precision/recall/f1/FPR, weighted avg
        - confusion_matrix.csv
        - classification_report.txt
        """
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        preds: List[int] = []
        labels_all: List[int] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for ib, batch in enumerate(loader, 1):
                if self.model_type == "gnn":
                    batch["graph"] = batch["graph"].to(self.device)
                    outputs = self.model(batch["graph"])
                    labels = batch["graph"].y
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += batch["graph"].num_graphs
                elif self.model_type == "cnn":
                    batch["raw_tensor"] = batch["raw_tensor"].to(self.device)
                    outputs = self.model(batch["raw_tensor"])
                    labels = batch["graph"].y.to(self.device)
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += batch["raw_tensor"].size(0)
                elif self.model_type == "seq2seq_cnn":
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["targets"].to(self.device)
                    outputs = self.model(inputs)
                    seq_len = outputs.size(1)
                    pred = outputs.argmax(dim=2)[:, seq_len // 2]
                    labels = targets[:, seq_len // 2]
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                elif self.model_type == "seq2seq_gnn":
                    batch_graph = batch["batch_graph"].to(self.device)
                    outputs = self.model(batch_graph, sequence_length=self.sequence_length)
                    seq_len = outputs.size(1)
                    pred = outputs.argmax(dim=2)[:, seq_len // 2]
                    labels = batch["labels"].to(self.device)
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                elif self.model_type == "gnn_transformer":
                    batch_graph = batch["batch_graph"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    seq_len = int(self.sequence_length)
                    if hasattr(batch_graph, "num_graphs") and labels.numel() > 0:
                        if batch_graph.num_graphs % labels.size(0) == 0:
                            inferred = batch_graph.num_graphs // labels.size(0)
                            if inferred != seq_len:
                                seq_len = inferred
                    outputs = self.model(batch_graph, sequence_length=seq_len)
                    pred = outputs.argmax(dim=1)
                    correct += int((pred == labels).sum())
                    total += labels.size(0)
                else:
                    raise ValueError("Unsupported model type")
                preds.extend(pred.detach().cpu().tolist())
                labels_all.extend(labels.detach().cpu().tolist())
                if self.debug_max_eval_batches is not None and ib >= self.debug_max_eval_batches:
                    break

        acc = correct / total if total > 0 else 0.0
        # Compute metrics
        cm = confusion_matrix(labels_all, preds, labels=list(range(len(class_names))))
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_all, preds, labels=list(range(len(class_names))), zero_division=0
        )
        # FPR per class: FP / (FP + TN)
        fpr = []
        tn_fp = cm.sum(axis=1)  # TP+FN per true class (not used)
        for i in range(len(class_names)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fpr.append(rate)
        weighted_avg = {
            "precision": float(np.average(precision, weights=support)),
            "recall": float(np.average(recall, weights=support)),
            "f1": float(np.average(f1, weights=support)),
            "fpr": float(np.average(fpr, weights=support)),
        }
        # Save files
        with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(
                classification_report(
                    labels_all,
                    preds,
                    labels=list(range(len(class_names))),
                    target_names=list(class_names),
                    zero_division=0,
                )
            )
        with open(os.path.join(output_dir, "confusion_matrix.csv"), "w", encoding="utf-8") as f:
            f.write(",".join([" "] + list(class_names)) + "\n")
            for i, row in enumerate(cm):
                f.write(",".join([class_names[i]] + [str(int(x)) for x in row.tolist()]) + "\n")
        metrics_payload = {
            "accuracy": acc,
            "per_class": [
                {
                    "class": class_names[i],
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "fpr": float(fpr[i]),
                    "support": int(support[i]),
                }
                for i in range(len(class_names))
            ],
            "weighted_avg": weighted_avg,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        return acc
