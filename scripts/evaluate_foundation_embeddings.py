import argparse
import os

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifiers on GIN embeddings")
    parser.add_argument("--constructor", choices=["vg", "hvg"], required=True,
                        help="Which embeddings to use")
    parser.add_argument("--emb-dir", default="embeddings",
                        help="Directory with saved embeddings")
    args = parser.parse_args()

    def load_split(split: str):
        prefix = f"{args.constructor}_{split}"
        emb = torch.load(os.path.join(args.emb_dir, f"{prefix}_embeddings.pt"))
        labels = torch.load(os.path.join(args.emb_dir, f"{prefix}_labels.pt"))
        return emb.numpy(), labels.numpy()

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    classifiers = {
        "logreg": LogisticRegression(max_iter=1000),
        "svm": SVC(kernel="linear"),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"=== {name} ===")
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
