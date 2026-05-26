# -*- coding: utf-8 -*-
# scripts/extract_embeddings.py (versão corrigida e aprimorada)

import argparse
import os
from typing import List

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm # Adicionar tqdm para a barra de progresso

from src.data_loader import ECGGraphDataset
from src.models import FoundationGIN


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings using a pre-trained GIN")
    parser.add_argument("--constructor", choices=["vg", "hvg"], required=True,
                        help="Graph construction method to use")
    parser.add_argument("--split", choices=["train", "test"], required=True,
                        help="Dataset split to process")
    parser.add_argument("--outdir", default="embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--batch-size", type=int, default=32) # Reduzido para segurança
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Usando dispositivo: {device}")

    print(f"Carregando dataset para: split='{args.split}', constructor='{args.constructor}'")
    dataset = ECGGraphDataset(
        root="Data",
        split=args.split.capitalize(),
        beat_size_before=100,
        beat_size_after=180,
        graph_constructor=args.constructor,
        feature_set="rr",
    )

    graphs: List[Data] = [item["graph"] for item in dataset]
    if not graphs:
        raise ValueError("O dataset está vazio. Verifique o caminho e a configuração.")
    
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    print("Carregando modelo de fundação GIN...")
    model = FoundationGIN().to(device)
    model.eval()
    foundation_dim = model.in_dim

    in_dim = graphs[0].num_node_features
    print(f"Dimensão das features do grafo: {in_dim}")
    print(f"Dimensão esperada pelo modelo de fundação: {foundation_dim}")
    
    if in_dim == 0:
        raise ValueError("Os grafos não têm features de nó. Adicione features (ex: 'rr') na configuração do dataset.")

    projector = nn.Linear(in_dim, foundation_dim).to(device)
    projector.eval()

    all_embs = []
    all_labels = []

    print(f"Iniciando extração de {len(graphs)} embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extraindo embeddings ({args.split})"):
            batch = batch.to(device)
            batch.x = projector(batch.x)
            
            emb = model(batch)
            
            all_embs.append(emb.cpu())
            all_labels.append(batch.y.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"Extração concluída. Shape dos embeddings: {embeddings.shape}")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = f"{args.constructor}_{args.split}"
    emb_path = os.path.join(args.outdir, f"{prefix}_embeddings.pt")
    lbl_path = os.path.join(args.outdir, f"{prefix}_labels.pt")

    print(f"Salvando embeddings em: {emb_path}")
    torch.save(embeddings, emb_path)
    print(f"Salvando rótulos em: {lbl_path}")
    torch.save(labels, lbl_path)
    print("Salvo com sucesso!")

if __name__ == "__main__":
    main()