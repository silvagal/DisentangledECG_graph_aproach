import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, List, Tuple
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    SAGEConv,
    GINConv,
    GINEConv,
    TransformerConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    Set2Set,
    global_sort_pool,
)
from torch_geometric.nn.models import GIN
from .utils import generate_sinusoidal_positional_encodings


class FoundationGIN(nn.Module):
    """Wrapper around a pre-trained GIN model used as a frozen embedder."""

    def __init__(self, pretrained_name: str = "ogbg-molpcba") -> None:
        super().__init__()
        # Load pre-trained model from the PyG model zoo
        self.gin = GIN.from_pretrained(pretrained_name)
        # Freeze all parameters to prevent any training
        for param in self.gin.parameters():
            param.requires_grad = False
        # Remove original classification head so that forward returns embeddings
        self.gin.lin = nn.Identity()

        # Expose expected input dimensionality for convenience
        try:
            self.in_dim = self.gin.convs[0].nn[0].in_features
        except Exception:
            # Fallback to the canonical dimension of the ogbg-molpcba dataset
            self.in_dim = 9

    def forward(self, data):
        """Encode a graph and return a single embedding vector."""
        return self.gin(data.x, data.edge_index, data.batch)


class GCNModel(nn.Module):
    """Simple Graph Convolutional Network for graph classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        use_pe: bool = False,
        readout: str = "mean",
        readout_params: Optional[dict] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_pe = use_pe
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.dropout = float(dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        ro_dim = hidden_channels
        if self.readout == "set2set":
            self.set2set = Set2Set(hidden_channels, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = hidden_channels * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = hidden_channels * k
        self.lin = nn.Linear(ro_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_pe and hasattr(data, "pos_encoding"):
            x = torch.cat([x, data.pos_encoding], dim=-1)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            self.set2set = self.set2set.to(x.device)
            self.set2set = self.set2set.to(x.device)
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return self.lin(x)


class GCNVariant(nn.Module):
    """Named GCN architectures (GCN2, GCN7, GCN60, GCN120, GCN240) with configurable readout."""

    def __init__(self, in_channels: int, num_classes: int, arch: str, use_pe: bool = False,
                 readout: str = "mean", readout_params: Optional[dict] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.use_pe = use_pe
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.dropout = float(dropout)
        arch = arch.lower()
        if arch == "gcn2":
            layer_specs: List[Tuple[str, int]] = [("gcn", 20), ("gcn", num_classes)]
        elif arch == "gcn7":
            layer_specs = [("sage", 20), ("sage", 40), ("sage", 30), ("sage", 20), ("sage", 10), ("gcn", 5), ("gcn", num_classes)]
        elif arch == "gcn60":
            layer_specs = [("sage", 60), ("sage", 50), ("sage", 35), ("sage", 20), ("sage", num_classes)]
        elif arch == "gcn120":
            layer_specs = [("sage", 120), ("sage", 40), ("sage", 20), ("sage", num_classes)]
        elif arch == "gcn240":
            layer_specs = [("sage", 240), ("sage", 140), ("sage", 40), ("sage", num_classes)]
        else:
            raise ValueError(
                f"Unknown GCN variant: {arch} (expected: gcn2, gcn7, gcn60, gcn120, gcn240)"
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = in_channels
        for layer_idx, (ltype, outc) in enumerate(layer_specs):
            if ltype == "gcn":
                self.convs.append(GCNConv(prev, outc))
            elif ltype == "sage":
                self.convs.append(SAGEConv(prev, outc))
            else:
                raise ValueError(f"Unsupported layer type: {ltype}")
            if layer_idx != len(layer_specs) - 1:
                self.norms.append(nn.BatchNorm1d(outc))
            prev = outc
        self.out_dim = prev
        ro_dim = self.out_dim
        if self.readout == "set2set":
            self.set2set = Set2Set(self.out_dim, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = self.out_dim * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = self.out_dim * k
        self.lin = nn.Linear(ro_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_pe and hasattr(data, "pos_encoding"):
            x = torch.cat([x, data.pos_encoding], dim=-1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return self.lin(x)


class GATModel(nn.Module):
    """Graph Attention Network for graph classification with optional PEs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        use_pe: bool = False,
        readout: str = "mean",
        readout_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.use_pe = use_pe
        self.dropout = dropout
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.convs.append(
                    GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
                )
            elif layer == num_layers - 1:
                self.convs.append(
                    GATv2Conv(
                        hidden_dim * heads,
                        hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        concat=False,
                    )
                )
            else:
                self.convs.append(
                    GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
                )
        ro_dim = hidden_dim
        if self.readout == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = hidden_dim * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = hidden_dim * k
        self.lin = nn.Linear(ro_dim, output_dim)

    def encode_graph(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_pe and hasattr(data, "pos_encoding"):
            x = torch.cat([x, data.pos_encoding], dim=-1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        if self.readout == "sum":
            return global_add_pool(x, batch)
        if self.readout == "max":
            return global_max_pool(x, batch)
        if self.readout == "set2set":
            return self.set2set(x, batch)
        if self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            return global_sort_pool(x, batch, k)
        return global_mean_pool(x, batch)

    def forward(self, data, return_embeddings: bool = False):
        x = self.encode_graph(data)
        if return_embeddings:
            return x
        return self.lin(x)


class GINModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, num_classes: int,
                 eps: float = 0.0, train_eps: bool = True, readout: str = "mean",
                 readout_params: Optional[dict] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(max(0, num_layers - 1)):
            mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        ro_dim = hidden_channels
        if self.readout == "set2set":
            self.set2set = Set2Set(hidden_channels, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = hidden_channels * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = hidden_channels * k
        self.lin = nn.Linear(ro_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[idx](x)
            x = F.relu(x)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return self.lin(x)


class GINEModel(nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int, num_layers: int, num_classes: int,
                 readout: str = "mean", readout_params: Optional[dict] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(max(0, num_layers - 1)):
            mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        ro_dim = hidden_channels
        if self.readout == "set2set":
            self.set2set = Set2Set(hidden_channels, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = hidden_channels * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = hidden_channels * k
        self.lin = nn.Linear(ro_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.norms[idx](x)
            x = F.relu(x)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return self.lin(x)


class EmbeddingGINE(GINEModel):
    """GINE encoder that returns pooled graph embeddings (no classifier).

    Mirrors GINEModel architecture but exposes the pooled embedding instead of
    class logits. Provides `emb_dim` with the embedding dimensionality that
    depends on the readout choice.
    """

    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int, num_layers: int,
                 readout: str = "mean", readout_params: Optional[dict] = None,
                 dropout: float = 0.0) -> None:
        super().__init__(in_channels, edge_dim, hidden_channels, num_layers, num_classes=1,
                         readout=readout, readout_params=readout_params, dropout=dropout)
        # Store dims and compute embedding size based on readout
        self.hidden_channels = hidden_channels
        self.emb_dim = hidden_channels
        if self.readout == "set2set":
            self.emb_dim = hidden_channels * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            self.emb_dim = hidden_channels * k

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.norms[idx](x)
            x = F.relu(x)
            if idx != len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return x


class GINESeq2Seq(nn.Module):
    """Hybrid GINE encoder + Seq2Seq (BiLSTM encoder + LSTM decoder).

    The GINE encoder produces graph embeddings per beat, which are then fed as
    the input sequence to a Seq2Seq module inspired by Mousavi & Afghah (2019).
    """

    def __init__(
        self,
        gine_config: dict,
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        classifier: Optional[dict] = None,
        teacher_forcing: Optional[bool] = None,
    ) -> None:
        super().__init__()
        # Defaults
        enc_hidden = (encoder or {}).get("hidden_size", 128)
        enc_layers = (encoder or {}).get("num_layers", 1)
        dec_hidden = (decoder or {}).get("hidden_size", 128)
        dec_layers = (decoder or {}).get("num_layers", 1)
        num_classes = (classifier or {}).get("num_classes", 3)
        teacher_forcing = (decoder or {}).get("teacher_forcing", teacher_forcing)
        teacher_forcing = True if teacher_forcing is None else bool(teacher_forcing)

        # GINE encoder for embeddings
        self.gine_encoder = EmbeddingGINE(
            in_channels=gine_config.get("in_channels", 0),
            edge_dim=gine_config.get("edge_dim", 0),
            hidden_channels=gine_config.get("hidden_channels", 128),
            num_layers=gine_config.get("num_layers", 5),
            readout=gine_config.get("readout", "mean"),
            readout_params=gine_config.get("readout_params", {}),
            dropout=gine_config.get("dropout", 0.0),
        )
        self.emb_dim = self.gine_encoder.emb_dim

        # Seq2Seq components
        self.encoder = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_to_dec_h = nn.Linear(enc_hidden * 2, dec_hidden)
        self.decoder = nn.LSTM(
            input_size=dec_hidden,
            hidden_size=dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
        )
        self.embedding = nn.Embedding(num_classes + 1, dec_hidden)
        self.classifier = nn.Linear(dec_hidden, num_classes)
        self.num_classes = num_classes
        self.teacher_forcing = teacher_forcing

    def forward(self, batch_graph, targets: Optional[torch.Tensor] = None, sequence_length: int = 1) -> torch.Tensor:
        # Encode all graphs, reshape into sequences
        graph_embeddings = self.gine_encoder(batch_graph)
        feats = graph_embeddings.view(-1, sequence_length, self.emb_dim)

        batch = feats.size(0)
        seq_len = feats.size(1)

        enc_out, (h_n, c_n) = self.encoder(feats)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        dec_h0 = torch.tanh(self.enc_to_dec_h(h_cat)).unsqueeze(0)
        dec_c0 = torch.zeros_like(dec_h0)

        device = feats.device
        go_token = torch.full((batch,), self.num_classes, dtype=torch.long, device=device)
        inp = self.embedding(go_token).unsqueeze(1)
        hidden = (dec_h0, dec_c0)
        outputs = []
        for t in range(seq_len):
            out, hidden = self.decoder(inp, hidden)
            logits = self.classifier(out.squeeze(1))
            outputs.append(logits)
            if self.training and self.teacher_forcing and targets is not None:
                next_in = targets[:, t]
            else:
                next_in = logits.argmax(dim=1)
            inp = self.embedding(next_in).unsqueeze(1)
        return torch.stack(outputs, dim=1)


class TransformerConvModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, heads: int,
                 num_classes: int, dropout: float = 0.1, readout: str = "mean",
                 readout_params: Optional[dict] = None) -> None:
        super().__init__()
        self.readout = (readout or "mean").lower()
        self.readout_params = readout_params or {}
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        in_dim = hidden_channels * heads
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(TransformerConv(in_dim, hidden_channels, heads=heads, dropout=dropout))
            in_dim = hidden_channels * heads
        ro_dim = in_dim
        if self.readout == "set2set":
            self.set2set = Set2Set(ro_dim, processing_steps=self.readout_params.get("iters", 3))
            ro_dim = ro_dim * 2
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            ro_dim = ro_dim * k
        self.lin = nn.Linear(ro_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            if i != len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "set2set":
            x = self.set2set(x, batch)
        elif self.readout == "sortpool":
            k = int(self.readout_params.get("k", 30))
            x = global_sort_pool(x, batch, k)
        else:
            x = global_mean_pool(x, batch)
        return self.lin(x)
class CNNModel(nn.Module):
    """Baseline 1D CNN model."""

    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)
        self.input_length = input_length

    def forward(self, x):
        # x: (batch, input_length)
        x = x.view(-1, 1, self.input_length)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    """A very simple 1D CNN baseline."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main_path = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main_path(x)
        identity = self.shortcut(x)
        if out.size(-1) != identity.size(-1):
            min_len = min(out.size(-1), identity.size(-1))
            out = out[..., :min_len]
            identity = identity[..., :min_len]
        out = out + identity
        return F.relu(out)


class PreActivationResidualBlock(nn.Module):
    """Residual block with pre-activation as used in HannunResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main_path = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main_path(x)
        identity = self.shortcut(x)
        # Align temporal dimensions in case of off-by-one due to even kernel sizes/strides
        if out.size(-1) != identity.size(-1):
            min_len = min(out.size(-1), identity.size(-1))
            out = out[..., :min_len]
            identity = identity[..., :min_len]
        out = out + identity
        return out


class ResNet1DModel(nn.Module):
    """1D ResNet-style CNN for ECG classification."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        layers_config,
    ) -> None:
        super().__init__()
        if not layers_config:
            raise ValueError("layers_config must not be empty")
        first_channels = layers_config[0][0]
        self.input_layer = nn.Sequential(
            nn.Conv1d(input_channels, first_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for cfg in layers_config:
            in_c, out_c, k = cfg[:3]
            stride = cfg[3] if len(cfg) > 3 else 1
            blocks.append(ResidualBlock1D(in_c, out_c, k, stride))
        self.layers = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        final_channels = layers_config[-1][1]
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class HannunResNet(nn.Module):
    """Replication of Hannun et al. (2019) 34-layer residual network."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(
            input_channels, 32, kernel_size=16, stride=2, padding=8, bias=False
        )
        blocks = []
        channels = [32] * 4 + [64] * 4 + [128] * 4 + [256] * 4
        in_c = 32
        for idx, out_c in enumerate(channels):
            stride = 2 if idx in {4, 8, 12} else 1
            blocks.append(
                PreActivationResidualBlock(
                    in_c, out_c, kernel_size=16, stride=stride
                )
            )
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm1d(in_c)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.blocks(x)
        x = F.relu(self.final_bn(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class _CNNExtractor(nn.Module):
    """Per-beat CNN matching the reference Mousavi preprocessing shape."""

    def __init__(self, n_channels: int = 10, input_depth: int = 280):
        super().__init__()
        self.n_channels = int(n_channels)
        self.input_depth = int(input_depth)
        if self.input_depth % self.n_channels != 0:
            raise ValueError(
                f"input_depth ({self.input_depth}) must be divisible by n_channels ({self.n_channels})"
            )
        self.sub_len = self.input_depth // self.n_channels
        self.conv1 = nn.Conv1d(self.sub_len, 32, kernel_size=2, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=0)
        with torch.no_grad():
            dummy = torch.zeros(1, self.sub_len, self.n_channels)
            out = self._forward_conv(dummy)
            self.output_dim = int(out.flatten(1).size(1))

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # "same" padding behavior for kernel_size=2 as in the TF reference.
        x = self.conv1(F.pad(x, (0, 1)))
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(F.pad(x, (0, 1)))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(F.pad(x, (0, 1)))
        x = F.relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.ndim != 2:
            raise ValueError(f"_CNNExtractor expects [N, beat_len], got shape={tuple(x.shape)}")
        if x.size(1) != self.input_depth:
            raise ValueError(
                f"_CNNExtractor expected beat length {self.input_depth}, got {x.size(1)}."
            )
        beats = x.reshape(-1, self.n_channels, self.sub_len).transpose(1, 2).contiguous()
        z = self._forward_conv(beats)
        return z.flatten(1)


class MousaviSeq2Seq(nn.Module):
    """Seq2Seq CNN + BiLSTM baseline from Mousavi & Afghah (2019)."""

    def __init__(
        self,
        enc_hidden: Optional[int] = None,
        enc_layers: Optional[int] = None,
        bidirectional: Optional[bool] = None,
        dec_hidden: Optional[int] = None,
        dec_layers: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        teacher_forcing: Optional[bool] = None,
        *,
        cnn: Optional[dict] = None,
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        classifier: Optional[dict] = None,
    ) -> None:
        super().__init__()

        n_channels = 10
        input_depth = 280
        if cnn is not None:
            n_channels = int(cnn.get("n_channels", n_channels))
            input_depth = int(cnn.get("input_depth", input_depth))
        if encoder is not None:
            enc_hidden = encoder.get("hidden_size", enc_hidden)
            enc_layers = encoder.get("num_layers", enc_layers)
            bidirectional = encoder.get("bidirectional", bidirectional)
        if decoder is not None:
            dec_hidden = decoder.get("hidden_size", dec_hidden)
            dec_layers = decoder.get("num_layers", dec_layers)
            teacher_forcing = decoder.get("teacher_forcing", teacher_forcing)
            embedding_dim = decoder.get("embedding_dim", embedding_dim)
        if classifier is not None:
            num_classes = classifier.get("num_classes", num_classes)

        enc_hidden = enc_hidden if enc_hidden is not None else 128
        enc_layers = enc_layers if enc_layers is not None else 1
        bidirectional = True if bidirectional is None else bool(bidirectional)
        enc_factor = 2 if bidirectional else 1
        dec_hidden = dec_hidden if dec_hidden is not None else enc_hidden * enc_factor
        dec_layers = dec_layers if dec_layers is not None else 1
        embedding_dim = embedding_dim if embedding_dim is not None else 10
        num_classes = num_classes if num_classes is not None else 3
        teacher_forcing = teacher_forcing if teacher_forcing is not None else True

        self.cnn = _CNNExtractor(n_channels=n_channels, input_depth=input_depth)
        self.encoder = nn.LSTM(
            self.cnn.output_dim,
            enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.h_bridge = nn.Linear(enc_hidden * enc_factor, dec_hidden)
        self.c_bridge = nn.Linear(enc_hidden * enc_factor, dec_hidden)
        self.decoder = nn.LSTM(
            embedding_dim, dec_hidden, num_layers=dec_layers, batch_first=True
        )
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.classifier = nn.Linear(dec_hidden, num_classes)
        self.num_classes = num_classes
        self.teacher_forcing = teacher_forcing
        self.bidirectional = bidirectional
        self.dec_layers = dec_layers

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        batch, seq_len, beat_len = x.shape
        feats = x.reshape(batch * seq_len, beat_len)
        feats = self.cnn(feats)
        feats = feats.view(batch, seq_len, -1)

        _, (h_n, c_n) = self.encoder(feats)
        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
            c_last = torch.cat((c_n[-2], c_n[-1]), dim=1)
        else:
            h_last = h_n[-1]
            c_last = c_n[-1]
        dec_h0 = torch.tanh(self.h_bridge(h_last)).unsqueeze(0).repeat(self.dec_layers, 1, 1)
        dec_c0 = torch.tanh(self.c_bridge(c_last)).unsqueeze(0).repeat(self.dec_layers, 1, 1)
        go_token = torch.full((batch, 1), self.num_classes, dtype=torch.long, device=x.device)

        if self.training and self.teacher_forcing and targets is not None:
            teacher_tokens = torch.cat([go_token, targets[:, :-1]], dim=1)
            dec_inputs = self.embedding(teacher_tokens)
            dec_out, _ = self.decoder(dec_inputs, (dec_h0, dec_c0))
            return self.classifier(dec_out)

        outputs = []
        hidden = (dec_h0, dec_c0)
        next_token = go_token.squeeze(1)
        for _ in range(seq_len):
            dec_in = self.embedding(next_token).unsqueeze(1)
            dec_out, hidden = self.decoder(dec_in, hidden)
            logits = self.classifier(dec_out.squeeze(1))
            outputs.append(logits)
            next_token = logits.argmax(dim=1)
        return torch.stack(outputs, dim=1)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = generate_sinusoidal_positional_encodings(max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class GNNTransformerModel(nn.Module):
    """Hybrid model combining a GNN encoder with a Transformer over sequences."""

    def __init__(
        self,
        gnn_encoder_config: dict,
        transformer_config: dict,
        classifier_config: dict,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self.gnn_hidden_dim = gnn_encoder_config["hidden_dim"]
        self.gnn_encoder = GATModel(
            input_dim=gnn_encoder_config["input_dim"],
            hidden_dim=self.gnn_hidden_dim,
            output_dim=self.gnn_hidden_dim,
            num_layers=gnn_encoder_config["num_layers"],
            heads=gnn_encoder_config["heads"],
            dropout=gnn_encoder_config["dropout"],
            use_pe=gnn_encoder_config.get("use_pe", False),
        )
        self.positional_encoder = SinusoidalPositionalEncoding(self.gnn_hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.gnn_hidden_dim,
            nhead=transformer_config["num_heads"],
            dim_feedforward=transformer_config.get("ff_dim", 2048),
            dropout=transformer_config.get("dropout", 0.1),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=transformer_config["num_layers"]
        )
        self.classifier = nn.Linear(
            self.gnn_hidden_dim, classifier_config["num_classes"]
        )

    def forward(self, batch_graph, sequence_length: int) -> torch.Tensor:
        graph_embeddings = self.gnn_encoder(batch_graph, return_embeddings=True)
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] graph_embeddings: shape={tuple(graph_embeddings.shape)} mean={graph_embeddings.mean().item():.6f} std={graph_embeddings.std().item():.6f}")

        sequences = graph_embeddings.view(-1, sequence_length, self.gnn_hidden_dim)
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] sequences (view): shape={tuple(sequences.shape)} mean={sequences.mean().item():.6f} std={sequences.std().item():.6f}")

        sequences_with_pe = self.positional_encoder(sequences)
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] sequences_with_pe: shape={tuple(sequences_with_pe.shape)} mean={sequences_with_pe.mean().item():.6f} std={sequences_with_pe.std().item():.6f}")

        transformer_output = self.transformer_encoder(sequences_with_pe)
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] transformer_output: shape={tuple(transformer_output.shape)} mean={transformer_output.mean().item():.6f} std={transformer_output.std().item():.6f}")

        central_embedding = transformer_output[:, sequence_length // 2, :]
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] central_embedding: shape={tuple(central_embedding.shape)} mean={central_embedding.mean().item():.6f} std={central_embedding.std().item():.6f}")

        logits = self.classifier(central_embedding)
        if self.debug:
            with torch.no_grad():
                print(f"[DBG] logits: shape={tuple(logits.shape)} mean={logits.mean().item():.6f} std={logits.std().item():.6f}")
        return logits


class Mish(nn.Module):
    """Mish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(x)


class ChannelAttentionModule(nn.Module):
    """Channel attention mechanism with a shared MLP."""

    def __init__(self, num_channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, int(num_channels // reduction))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden, bias=True), # Usando bias=True (padrão)
            nn.ReLU(),                                 # Usando ReLU não-inplace
            nn.Linear(hidden, num_channels, bias=True),  # Usando bias=True (padrão)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool_out = self.avg_pool(x).squeeze(-1)
        max_pool_out = self.max_pool(x).squeeze(-1)
        avg_mlp_out = self.mlp(avg_pool_out)
        max_mlp_out = self.mlp(max_pool_out)
        attn = torch.sigmoid(avg_mlp_out + max_mlp_out).unsqueeze(-1)
        return x * attn


class CMCBlock(nn.Module):
    """Convolution → Mish → Channel attention block."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            Mish(),
            ChannelAttentionModule(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLBlock(nn.Module):
    """MaxPool followed by LayerNorm along the temporal dimension."""

    def __init__(self, pool_kernel_size: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(pool_kernel_size)
        self.norm: Optional[nn.LayerNorm] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = x.transpose(1, 2)
        if self.norm is None or self.norm.normalized_shape != (x.size(-1),):
            self.norm = nn.LayerNorm(x.size(-1)).to(x.device)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class MCCNet(nn.Module):
    """Multi-scale CNN with channel attention from DA-Net."""

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CMCBlock(input_channels, 32, kernel_size=5, stride=1),
            MLBlock(pool_kernel_size=2),
            CMCBlock(32, 64, kernel_size=3, stride=1),
            MLBlock(pool_kernel_size=2),
            CMCBlock(64, 128, kernel_size=2, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class DANet(nn.Module):
    """Dual Attention Network for arrhythmia classification."""

    def __init__(
        self,
        input_channels: int = 1,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.mcc_net = MCCNet(input_channels)
        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=encoder_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.enc_proj = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.bridge = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.decoder = nn.LSTM(
            input_size=decoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            batch_first=True,
        )
        self.attn_combine = nn.Linear(
             decoder_hidden_dim + encoder_hidden_dim * 2, decoder_hidden_dim
        )
        self.classifier = nn.Linear(decoder_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mcc_net(x)
        features = features.transpose(1, 2)

        enc_out, (h_n, c_n) = self.encoder(features)

        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        c_cat = torch.cat([c_n[-2], c_n[-1]], dim=1)

        dec_h0 = torch.tanh(self.bridge(h_cat)).unsqueeze(0)
        dec_c0 = torch.tanh(self.bridge(c_cat)).unsqueeze(0)

        dec_input = torch.zeros(x.size(0), 1, self.decoder.input_size, device=x.device)
        dec_out, _ = self.decoder(dec_input, (dec_h0, dec_c0))

        enc_proj = self.enc_proj(enc_out)  # [B, T, D_dec]
        attn_weights = torch.bmm(enc_proj, dec_out.transpose(1, 2)).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)

        combined_out = torch.tanh(self.attn_combine(torch.cat([dec_out.squeeze(1), context], dim=1)))

        return self.classifier(combined_out)


def get_model(cfg: dict) -> nn.Module:
    name = (cfg.get("model_name") or cfg.get("name") or "").lower()

    if name in {"mousaviseq2seq", "mousavi"}:
        return MousaviSeq2Seq(
            cnn=cfg.get("cnn"),
            encoder=cfg.get("encoder"),
            decoder=cfg.get("decoder"),
            classifier=cfg.get("classifier"),
            enc_hidden=cfg.get("enc_hidden"),
            enc_layers=cfg.get("enc_layers"),
            bidirectional=cfg.get("bidirectional"),
            dec_hidden=cfg.get("dec_hidden"),
            dec_layers=cfg.get("dec_layers"),
            embedding_dim=cfg.get("embedding_dim"),
            num_classes=cfg.get("num_classes"),
            teacher_forcing=cfg.get("teacher_forcing"),
        )

    if name in {"danet"}:
        return DANet(
            input_channels=cfg.get("input_channels", 1),
            encoder_hidden_dim=cfg.get("encoder_hidden_dim", 256),
            decoder_hidden_dim=cfg.get("decoder_hidden_dim", 256),
            num_classes=cfg.get("num_classes", 3),
        )

    use_pe = cfg.get("use_pe") or cfg.get("use_positional_encoding", False)
    in_channels = cfg.get("in_channels", cfg.get("input_dim", 0))
    if use_pe:
        in_channels += cfg.get("pe_dim", 0)

    if name in {"gnntransformermodel", "gnn_transformer"}:
        return GNNTransformerModel(
            gnn_encoder_config=cfg["gnn_encoder_config"],
            transformer_config=cfg["transformer_config"],
            classifier_config=cfg["classifier_config"],
            debug=cfg.get("debug", False),
        )
    if name in {"gineseq2seq", "gine_seq2seq"}:
        return GINESeq2Seq(
            gine_config=cfg["gine_config"],
            encoder=cfg.get("encoder"),
            decoder=cfg.get("decoder"),
            classifier=cfg.get("classifier"),
            teacher_forcing=cfg.get("teacher_forcing"),
        )
    if name in {"gcn", "gcnmodel"}:
        return GCNModel(
            in_channels=in_channels,
            hidden_channels=cfg["hidden_channels"],
            num_classes=cfg["num_classes"],
            use_pe=use_pe,
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
            dropout=cfg.get("dropout", 0.0),
        )
    if name in {"gcn2", "gcn7", "gcn60", "gcn120", "gcn240"}:
        return GCNVariant(
            in_channels=in_channels,
            num_classes=cfg["num_classes"],
            arch=name,
            use_pe=use_pe,
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
            dropout=cfg.get("dropout", 0.0),
        )
    if name in {"gat", "gatmodel"}:
        return GATModel(
            input_dim=in_channels,
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["num_classes"],
            num_layers=cfg["num_layers"],
            heads=cfg["heads"],
            dropout=cfg.get("dropout", 0.1),
            use_pe=use_pe,
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
        )
    if name in {"cnn", "cnnmodel"}:
        return CNNModel(
            input_length=cfg.get("input_length", 240),
            num_classes=cfg["num_classes"],
        )
    if name in {"simplecnn", "simple_cnn"}:
        return SimpleCNN(
            input_channels=cfg.get("input_channels", 1),
            num_classes=cfg["num_classes"],
        )
    if name in {"resnet1dmodel", "resnet1d", "resnet"}:
        return ResNet1DModel(
            input_channels=cfg.get("input_channels", 1),
            num_classes=cfg["num_classes"],
            layers_config=cfg["layers_config"],
        )
    if name in {"hannunresnet", "hannun"}:
        return HannunResNet(
            input_channels=cfg.get("input_channels", 1),
            num_classes=cfg["num_classes"],
        )
    if name in {"gin", "ginmodel"}:
        return GINModel(
            in_channels=in_channels,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 5),
            num_classes=cfg["num_classes"],
            eps=cfg.get("eps", 0.0),
            train_eps=cfg.get("train_eps", True),
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
            dropout=cfg.get("dropout", 0.0),
        )
    if name in {"gine", "ginemodel"}:
        return GINEModel(
            in_channels=in_channels,
            edge_dim=cfg.get("edge_dim", 0),
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 5),
            num_classes=cfg["num_classes"],
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
            dropout=cfg.get("dropout", 0.0),
        )
    if name in {"transformerconv", "trconv"}:
        return TransformerConvModel(
            in_channels=in_channels,
            hidden_channels=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 4),
            heads=cfg.get("heads", 4),
            num_classes=cfg["num_classes"],
            dropout=cfg.get("dropout", 0.1),
            readout=cfg.get("readout", "mean"),
            readout_params=cfg.get("readout_params", {}),
        )
    raise ValueError(f"Unknown model name: {name}")
